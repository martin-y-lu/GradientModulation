import torch
import torch.nn as nn
import torch.nn.functional as F

class GModReLUFunction(torch.autograd.Function):
    """
    Hyperbolic kernel
    f(x) = \frac{\lambda}{1 - x/k\lambda}

    Forward:
    y = R(x) = max(0,x)

    Backwards:
    \frac{\del J}{\del X} \approx 
    """

    @staticmethod
    def forward(ctx, input,l,k, kernel_type = ["standard"]):
        ctx.save_for_backward(input)
        ctx.l = l 
        ctx.k = k
        ctx.kernel_type = kernel_type
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        l, k, kernel_type = ctx.l, ctx.k, ctx.kernel_type
    
        # Standard ReLU mask
        positive_mask = (input > 0).float()
        if l == 0 or k == 0:
            kernel = torch.zeros_like(input)
        elif "nonscale" in kernel_type:
            kernel = l/ (1 + torch.abs(input)/(l*k))
        else:
            kernel = l/ (1 + torch.abs(input*grad_output)/(l*k))
        if "clip" in kernel_type:
            kernel = kernel*(input < 0).float()

        if "fixed" in kernel_type:
            grad_input = (positive_mask - torch.sign(input)*kernel)*grad_output
        else:
            grad_input = torch.where(
                grad_output > 0,
                positive_mask,
                positive_mask - torch.sign(input)*kernel)*grad_output
            
        return grad_input, None, None, None

class GModReLU(nn.Module):
    def __init__(self, l = 0.1,k = 5.5, kernel_type =  "standard"):
        """
        negative_grad_scale: float, how much gradient to inject in dead regions
        mode: 'positive_only' or 'full'
        """
        super().__init__()
        self.l = l
        self.k = k
        self.kernel_type = kernel_type

    def forward(self, input):
        return GModReLUFunction.apply(input, self.l, self.k, self.kernel_type)

class LGRLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, l, k, p = 1):
        ctx.l = l
        ctx.k = k
        ctx.p = p
        lin_output = input @ weight.T + bias
        ctx.save_for_backward(input, weight, bias, lin_output)
        return F.relu(lin_output)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, lin_output = ctx.saved_tensors
        l, k, p = ctx.l, ctx.k, ctx.p

        # Gradient mask and kernel
        relu_mask = (lin_output > 0).float()
        kernel = l / (1 + torch.abs(lin_output) / (l * k))

        # Modulated gradient through activation
        lin_grad_output = relu_mask*grad_output
        lin_eventual_output = - (grad_output <= 0).float() * torch.sign(lin_output) * kernel * grad_output
        grad_output = lin_grad_output + lin_eventual_output
        
        grad_input = (lin_grad_output + lin_eventual_output*p) @ weight
        grad_weight = grad_output.T @ input
        grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None

class LGRLinear(nn.Module):
    def __init__(self, in_features, out_features, l=0.01, k=5.0, p = 1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.l = l
        self.k = k
        self.p = p

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return LGRLinearFunction.apply(input, self.weight, self.bias, self.l, self.k, self.p)

class ALGRLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, weight_grad_sqr, bias_grad_sqr, l, k, p):
        ctx.l = l
        ctx.k = k
        ctx.p = p
        lin_output = input @ weight.T + bias
        ctx.save_for_backward(input, weight, bias, weight_grad_sqr, bias_grad_sqr, lin_output)
        return F.relu(lin_output)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, weight_grad_sqr, bias_grad_sqr, lin_output = ctx.saved_tensors
        l, k, p = ctx.l, ctx.k, ctx.p

        # Gradient mask and kernel
        relu_mask = (lin_output > 0).float()
        kernel = l / (1 + torch.abs(lin_output) / (l * k))

        # Modulated gradient through activation
        lin_grad_output = relu_mask*grad_output
        lin_eventual_output = - (grad_output <= 0) * torch.sign(lin_output) * kernel * grad_output
        grad_output = lin_grad_output + lin_eventual_output
        
        grad_input = (lin_grad_output + lin_eventual_output*p) @ weight
        #Note weight and bias grad_sqrs is broadcast across batch dimension
        grad_weight = lin_grad_output.T @ input + (lin_eventual_output.T @ input)*weight_grad_sqr
        
        grad_bias = lin_grad_ouput.sum(dim=0) + lin_eventual_output.sum(dim = 0)*bias_grad_sqr

        return grad_input, grad_weight, grad_bias, None, None, None, None, None
        
class ALGRLinear(nn.Module):
    def __init__(self, in_features, out_features, l=0.01, k=5.0, p=1.0, bias=True):
        super().__init__()
        self.l = l
        self.k = k
        self.p = p
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, weight_grad_sqr=None, bias_grad_sqr=None):
        # If none provided, fall back to ones (i.e. no adaptation)
        if weight_grad_sqr is None:
            weight_grad_sqr = torch.ones_like(self.weight)
        if bias_grad_sqr is None and self.bias is not None:
            bias_grad_sqr = torch.ones_like(self.bias)
        elif self.bias is None:
            bias_grad_sqr = torch.zeros(1)  # dummy value

        return ALGRLinearFunction.apply(
            input, self.weight, self.bias,
            weight_grad_sqr, bias_grad_sqr,
            self.l, self.k, self.p
        )

from torch.nn import Parameter
from torch.autograd import Function

class LGRConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, l, k, p = 1.0):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.l = l
        ctx.k = k
        ctx.p = p

        lin_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        ctx.lin_output = lin_output  # used in backward
        return F.relu(lin_output)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        lin_output = ctx.lin_output
        l, k, p = ctx.l, ctx.k, ctx.p

        # Gradient mask and kernel
        relu_mask = (lin_output > 0).float()
        kernel = l / (1 + torch.abs(lin_output) / (l * k))

        # Modulated gradient through activation
        lin_grad_output = relu_mask*grad_output
        lin_eventual_output = - (grad_output <= 0).float() * torch.sign(lin_output) * kernel * grad_output
        grad_output = lin_grad_output + lin_eventual_output

        # Compute gradients using autograd.grad-compatible ops
        grad_input = F.grad.conv2d_input(
            input.shape, weight, lin_grad_output + p*lin_eventual_output,
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        )
        grad_weight = F.grad.conv2d_weight(
            input, weight.shape, grad_output,
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        )
        grad_bias = grad_output.sum(dim=(0, 2, 3)) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None




class LGRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, l=0.01, k=5.0, p = 0.0):
        super().__init__()
        self.weight = Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = Parameter(torch.empty(out_channels)) if bias else None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.l = l
        self.k = k
        self.p = p

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return LGRConv2dFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
            self.l, self.k, self.p
        )

class ALGRConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, weight_grad_sqr, bias_grad_sqr, l, k, p = 1.0):
        ctx.save_for_backward(input, weight, bias, weight_grad_sqr, bias_grad_sqr)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.l = l
        ctx.k = k
        ctx.p = p

        lin_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        ctx.lin_output = lin_output  # used in backward
        return F.relu(lin_output)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, weight_grad_sqr, bias_grad_sqr = ctx.saved_tensors
        lin_output = ctx.lin_output
        l, k, p = ctx.l, ctx.k

        # Gradient mask and kernel
        relu_mask = (lin_output > 0).float()
        kernel = l / (1 + torch.abs(lin_output) / (l * k))

        # Modulated gradient through activation
        lin_grad_output = relu_mask*grad_output
        lin_eventual_output = - (grad_output <= 0) * torch.sign(lin_output) * kernel * grad_output
        grad_output = lin_grad_output + lin_eventual_output

        # Compute gradients using autograd.grad-compatible ops
        grad_input = F.grad.conv2d_input(
            input.shape, weight, lin_grad_output + p*lin_eventual_output,
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        )
        grad_weight = F.grad.conv2d_weight(
            input, weight.shape, lin_grad_output,
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        ) +  F.grad.conv2d_weight(
            input, weight.shape, lin_eventual_output,
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        )*weight_grad_sqr
        
        grad_bias = lin_grad_output.sum(dim=(0, 2, 3)) + lin_eventual_output.sum(dim=(0, 2, 3))*bias_grad_sqr if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


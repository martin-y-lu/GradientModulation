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
    def forward(ctx, input,l,k, kernel_type = "standard"):
        ctx.save_for_backward(input)
        ctx.l = l 
        ctx.k = k
        ctx.kernel_type = kernel_type
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        l, k, kernel_type = ctx.l, ctx.k, ctx.kernel_type
        if l == 0 or k == 0:
            grad_input = (input > 0).float() * grad_output
        else:
            # Standard ReLU mask
            positive_mask = (input > 0).float()
            if kernel_type == "nonscale" or kernel_type == "nonscalefixed":
                kernel = l/ (1 + torch.abs(input)/(l*k))
            else:
                kernel = l/ (1 + torch.abs(input*grad_output)/(l*k))

            if kernel_type == "fixed" or kernel_type == "nonscalefixed":
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
    def forward(ctx, input, weight, bias, l, k):
        ctx.l = l
        ctx.k = k
        lin_output = input @ weight.T + bias
        ctx.save_for_backward(input, weight, bias, lin_output)
        return F.relu(lin_output)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, lin_output = ctx.saved_tensors
        l, k = ctx.l, ctx.k

        # Gradient mask and kernel
        relu_mask = (lin_output > 0).float()
        kernel = l / (1 + torch.abs(lin_output) / (l * k))

        # Modulated gradient through activation
        lin_grad_output = torch.where(
            grad_output > 0,
            relu_mask,
            relu_mask - torch.sign(lin_output) * kernel
        ) * grad_output

        grad_input = lin_grad_output @ weight
        grad_weight = lin_grad_output.T @ input
        grad_bias = lin_grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None

class LGRLinear(nn.Module):
    def __init__(self, in_features, out_features, l=0.01, k=5.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.l = l
        self.k = k

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return LGRLinearFunction.apply(input, self.weight, self.bias, self.l, self.k)


from torch.nn import Parameter
from torch.autograd import Function

class LGRConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, l, k):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.l = l
        ctx.k = k

        lin_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        ctx.lin_output = lin_output  # used in backward
        return F.relu(lin_output)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        lin_output = ctx.lin_output
        l, k = ctx.l, ctx.k

        # ReLU mask + kernel
        relu_mask = (lin_output > 0).float()
        kernel = l / (1 + torch.abs(lin_output) / (l * k))

        lin_grad_output = torch.where(
            grad_output > 0,
            relu_mask,
            relu_mask - torch.sign(lin_output) * kernel
        ) * grad_output

        # Compute gradients using autograd.grad-compatible ops
        grad_input = F.grad.conv2d_input(
            input.shape, weight, lin_grad_output,
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        )
        grad_weight = F.grad.conv2d_weight(
            input, weight.shape, lin_grad_output,
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        )
        grad_bias = lin_grad_output.sum(dim=(0, 2, 3)) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class LGRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, l=0.01, k=5.0):
        super().__init__()
        self.weight = Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = Parameter(torch.empty(out_channels)) if bias else None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.l = l
        self.k = k

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return LGRConv2dFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
            self.l, self.k
        )


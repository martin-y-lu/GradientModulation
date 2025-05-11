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
    def forward(ctx, input,l,k):
        ctx.save_for_backward(input)
        ctx.l = l 
        ctx.k = k
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        l, k = ctx.l, ctx.k
        if l == 0 or k == 0:
            grad_input = (input > 0).float() * grad_output
        else:
            # Standard ReLU mask
            positive_mask = (input > 0).float()
            kernel = l/ (1 + torch.abs(input*grad_output)/(l*k))
            grad_input = torch.where(
                grad_output > 0,
                positive_mask,
                positive_mask - torch.sign(input)*kernel)*grad_output
        return grad_input, None, None

class GModReLU(nn.Module):
    def __init__(self, l = 0.1,k = 5.5):
        """
        negative_grad_scale: float, how much gradient to inject in dead regions
        mode: 'positive_only' or 'full'
        """
        super().__init__()
        self.l = l
        self.k = k

    def forward(self, input):
        return GModReLUFunction.apply(input, self.l, self.k)
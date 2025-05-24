#2nd draft of balanced eventuals; balancing the eventual with the gradient, and clipping to only do eventuals for input< 0
import torch
import torchvision
class GEDReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, S_n_n, S_n_p, Gi, l, k1, k2, p = 1):
        ctx.l = l
        ctx.k1 = k1 
        ctx.k2 = k2
        ctx.p = p
        ctx.save_for_backward(input, S_n_n, S_n_p, Gi)
        return F.relu(input)
    # @staticmethod
    # def forward(ctx, input):
    #     # ctx.l = l
    #     # ctx.k = k
    #     # ctx.p = p
    #     ctx.save_for_backward(input)
    #     return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, S_n_n, S_n_p, Gi = ctx.saved_tensors

        # print("---input shape:", input.shape)
        # print("S_n_n shape:", S_n_n.shape)
        # print("S_n_p shape:", S_n_p.shape)
        # print("S_p_n shape:", S_p_n.shape)
        # print("S_p_p shape:", S_p_p.shape)
        # print("grad_output shape:", grad_output.shape)


        l, k1, k2, p = ctx.l, ctx.k1, ctx.k2, ctx.p

        S_n_n = S_n_n.unsqueeze(0)  # [1, C, H, W]
        S_n_p = S_n_p.unsqueeze(0)
        Gi = Gi.unsqueeze(0)
        
        # Gradient mask and kernel
        relu_mask = (input > 0).float()
        kernel = torch.where(input*grad_output > 0,
            torch.zeros_like(input) if l*k1 == 0 else l / (1 + torch.abs(input) / (l * k1)),
            torch.zeros_like(input) if l*k2 == 0 else l / (1 + torch.abs(input) / (l * k2)),
        )

        # Modulated gradient through activation
        grad_input = relu_mask*grad_output
        eventual_input = (input < 0).float() * kernel * grad_output

        eps = 1e-12
        # Apply gating logic
        gated_eventual_input = torch.where((eventual_input <= 0)| (S_n_n + S_n_p <= 0),
                    eventual_input,
                    eventual_input * (-S_n_n / (S_n_p + eps)),
                )

        S_n = S_n_n + S_n_p
        S_n_c = S_n*(S_n< 0).float()
        s = min(torch.sum(Gi*S_n_c)/(torch.sum(Gi*Gi)+ eps),0)
        
        gated_grad_input = grad_input *(1 - s)
        # gated_grad_input = grad_input *( 1 + torch.where((S_n_n + S_n_p < 0) & (Gi > 0), -(S_n_n + S_n_p)/Gi, 0))

        grad_S_n_n = (((input < 0) & (eventual_input < 0)).float() * eventual_input).sum(dim = 0)
        grad_S_n_p = (((input < 0) & (eventual_input > 0)).float() * eventual_input).sum(dim = 0)

        return gated_grad_input + gated_eventual_input, grad_S_n_n, grad_S_n_p, grad_input , None, None, None, None


import torch
import torch.nn as nn
import torch.nn.functional as F

class GEDReLU(nn.Module):
    def __init__(self, shape = None, l=0.01, k1=1, k2 = 1, p=1.0):
        super().__init__()
        self.l = l
        self.k1 = k1
        self.k2 = k2
        self.p = p
        self.is_GED = True

        if shape != None:
            self.S_n_n = nn.Parameter(torch.zeros(shape), requires_grad=True)
            self.S_n_p = nn.Parameter(torch.zeros(shape), requires_grad=True)
            self.Gi = nn.Parameter(torch.zeros(shape), requires_grad=True)
            self.S_n_n.is_GED = True
            self.S_n_p.is_GED = True
            self.Gi.is_GED = True
        else:
            self.S_n_n = None
            

    def forward(self, input):
        # # Lazy init on first call
        if self.S_n_n is None:
            shape = input.shape[1:]  # Exclude batch dim
            device = input.device
            dtype = input.dtype
            self.S_n_n = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype), requires_grad=True)
            self.S_n_p = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype), requires_grad=True)
            self.Gi = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype), requires_grad=True)
            self.S_n_n.is_GED = True
            self.S_n_p.is_GED = True
            self.Gi.is_GED = True
        return GEDReLUFunction.apply(input, self.S_n_n, self.S_n_p, self.Gi, self.l, self.k1, self.k2, self.p)
        # return GEDReLUFunction.apply(input)

    def update_s(self, beta=0.9):
        """
        Manual update rule, called every epoch.
        Implements EMA-like update:
        S := beta * S + (1 - beta) * grad_S
        """
        print("Called Update S:",self)
        for param in [self.S_n_n, self.S_n_p, self.Gi]:
            if param.grad is not None:
                param.data.mul_(beta).add_((1 - beta) * param.grad.data)
                param.grad.detach_()
                param.grad.zero_()

    def get_s_buffers(self):
        return {
            "S_n_n": self.S_n_n,
            "S_n_p": self.S_n_p,
            "S_p_n": self.S_p_n,
            "S_p_p": self.S_p_p
        }
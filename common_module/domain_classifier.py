import torch
import torch.nn as nn

from config.train_cfg import cfg
from common_module.conv2d import Conv2D_ReLU, Conv2D

class Gradient_Reversal(torch.autograd.Function):
    def __init__(self):
        super(Gradient_Reversal, self).__init__()

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        _, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - alpha * grad_input, None

class GRL(nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.revgrad = Gradient_Reversal.apply

    def forward(self, x):
        return self.revgrad(x, self.alpha)

# ----------------------------------------------------------------------------------------------------
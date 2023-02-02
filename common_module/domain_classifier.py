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

class Img_Level_Classifier(nn.Module):
    def __init__(self, in_channels):
        super(Img_Level_Classifier, self).__init__()
        self.grl = GRL()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = Conv2D_ReLU(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2D_ReLU(in_channels//2, in_channels//4, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        f_p2 = features["p2"]
        f_p3 = features["p3"]
        f_p4 = features["p4"]
        f_p5 = features["p5"]
        
        f_p2 = self.conv2(self.conv1(f_p2))
        f_p2 = torch.mean(f_p2, dim=1)
        f_p2 = self.sigmoid(f_p2)
        f_p2_score = torch.mean(f_p2, dim=[1, 2]).unsqueeze(1)

        f_p3 = self.conv2(self.conv1(f_p3))
        f_p3 = torch.mean(f_p3, dim=1)
        f_p3 = self.sigmoid(f_p3)
        f_p3_score = torch.mean(f_p3, dim=[1, 2]).unsqueeze(1)

        f_p4 = self.conv2(self.conv1(f_p4))
        f_p4 = torch.mean(f_p4, dim=1)
        f_p4 = self.sigmoid(f_p4)
        f_p4_score = torch.mean(f_p4, dim=[1, 2]).unsqueeze(1)

        f_p5 = self.conv2(self.conv1(f_p5))
        f_p5 = torch.mean(f_p5, dim=1)
        f_p5 = self.sigmoid(f_p5)
        f_p5_score = torch.mean(f_p5, dim=[1, 2]).unsqueeze(1)

        img_level_score = (f_p2_score + f_p3_score + f_p4_score + f_p5_score) / 4.0
        return img_level_score
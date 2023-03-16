import torch
import torch.nn as nn

from config.train_test_cfg import cfg
from common_module.conv2d import Conv2D_ReLU

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
        f_p2 = self.grl(features["p2"])
        f_p3 = self.grl(features["p3"])
        f_p4 = self.grl(features["p4"])
        f_p5 = self.grl(features["p5"])
        
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

# ----------------------------------------------------------------------------------------------------

class Ins_Level_Classifier(nn.Module):
    def __init__(self, in_channels):
        super(Ins_Level_Classifier, self).__init__()
        self.grl = GRL()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = Conv2D_ReLU(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2D_ReLU(in_channels//2, in_channels//4, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        features = self.grl(features)

        features = self.conv2(self.conv1(features))
        features = torch.split(features, cfg.box_batch_size_per_img, dim=0)
        features = torch.stack(features, dim=0)
        features = torch.mean(features, dim=2)
        
        ins_level_score = self.sigmoid(features)
        ins_level_score = torch.mean(ins_level_score, dim=[1, 2, 3]).unsqueeze(1)
        return ins_level_score

# ----------------------------------------------------------------------------------------------------

class Daytime_Classifier(nn.Module):
    def __init__(self, in_channels):
        super(Daytime_Classifier, self).__init__()
        self.img_level_classifier = Img_Level_Classifier(in_channels)
        self.ins_level_classifier = Ins_Level_Classifier(in_channels)

        self.bce_loss_fn = nn.BCELoss()
        self.mse_loss_fn = nn.MSELoss()

    def forward(self, img_features, ins_features, targets):
        img_level_scores = self.img_level_classifier(img_features)
        ins_level_scores = self.ins_level_classifier(ins_features)

        losses = {}
        if self.training:
            assert targets is not None
            labels = torch.stack([t["daytime_class"] for t in targets], dim=0).to(torch.float).unsqueeze(1)
            loss_img_score, loss_ins_score, loss_consistency = self.compute_loss(img_level_scores, ins_level_scores, labels)
            losses = {
                "loss_daytime_img_score": loss_img_score, 
                "loss_daytime_ins_score": loss_ins_score, 
                "loss_daytime_consistency": loss_consistency
            }
            
        return losses

    def compute_loss(self, img_scores, ins_scores, labels):
        loss_img_score = self.bce_loss_fn(img_scores, labels)
        loss_ins_score = self.bce_loss_fn(ins_scores, labels)
        loss_consistency = self.mse_loss_fn(img_scores, ins_scores)

        return loss_img_score, loss_ins_score, loss_consistency

# ----------------------------------------------------------------------------------------------------

class Weather_Classifier(nn.Module):
    def __init__(self, in_channels):
        super(Weather_Classifier, self).__init__()
        self.img_level_classifier = Img_Level_Classifier(in_channels)
        self.ins_level_classifier = Ins_Level_Classifier(in_channels)

        self.bce_loss_fn = nn.BCELoss()
        self.mse_loss_fn = nn.MSELoss()

    def forward(self, img_features, ins_features, targets):
        img_level_scores = self.img_level_classifier(img_features)
        ins_level_scores = self.ins_level_classifier(ins_features)

        losses = {}
        if self.training:
            assert targets is not None
            labels = torch.stack([t["weather_class"] for t in targets], dim=0).to(torch.float).unsqueeze(1)
            loss_img_score, loss_ins_score, loss_consistency = self.compute_loss(img_level_scores, ins_level_scores, labels)
            losses = {
                "loss_weather_img_score": loss_img_score, 
                "loss_weather_ins_score": loss_ins_score, 
                "loss_weather_consistency": loss_consistency
            }

        return losses

    def compute_loss(self, img_scores, ins_scores, labels):
        loss_img_score = self.bce_loss_fn(img_scores, labels)
        loss_ins_score = self.bce_loss_fn(ins_scores, labels)
        loss_consistency = self.mse_loss_fn(img_scores, ins_scores)

        return loss_img_score, loss_ins_score, loss_consistency
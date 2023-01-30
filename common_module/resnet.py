import torch.nn as nn

from collections import OrderedDict
from .conv2d import Conv2D, Conv2D_ReLU

class Residual_Basic(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, downsample=None):
        super(Residual_Basic, self).__init__()
        self.conv1 = Conv2D_ReLU(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = Conv2D(out_features, out_features, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x  = self.relu(x + identity)
        return x
    
class Residual_BottleNeck(nn.Module):
    def __init__(self, in_features, width, out_features, kernel_size=3, stride=1, padding=1, downsample=None):
        super(Residual_BottleNeck, self).__init__()
        self.conv1 = Conv2D_ReLU(in_features, width, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv2D_ReLU(width, width, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = Conv2D(width, out_features, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.relu(x + identity)
        return x
    
# ----------------------------------------------------------------------------------------------------

class ResNet50(nn.Module):
    def __init__(self, in_channels=3, out_channels=2048):
        super(ResNet50, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = Conv2D_ReLU(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            Residual_BottleNeck(in_features=64, width=64, out_features=256, kernel_size=3, stride=1, padding=1, downsample=nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)),
            Residual_BottleNeck(in_features=256, width=64, out_features=256, kernel_size=3, stride=1, padding=1),
            Residual_BottleNeck(in_features=256, width=64, out_features=256, kernel_size=3, stride=1, padding=1),
        )
        self.layer2 = nn.Sequential(
            Residual_BottleNeck(in_features=256, width=128, out_features=512, kernel_size=3, stride=2, padding=1, downsample=nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2)),
            Residual_BottleNeck(in_features=512, width=128, out_features=512, kernel_size=3, stride=1, padding=1),
            Residual_BottleNeck(in_features=512, width=128, out_features=512, kernel_size=3, stride=1, padding=1),
            Residual_BottleNeck(in_features=512, width=128, out_features=512, kernel_size=3, stride=1, padding=1),
        )
        self.layer3 = nn.Sequential(
            Residual_BottleNeck(in_features=512, width=256, out_features=1024, kernel_size=3, stride=2, padding=1, downsample=nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2)),
            Residual_BottleNeck(in_features=1024, width=256, out_features=1024, kernel_size=3, stride=1, padding=1),
            Residual_BottleNeck(in_features=1024, width=256, out_features=1024, kernel_size=3, stride=1, padding=1),
            Residual_BottleNeck(in_features=1024, width=256, out_features=1024, kernel_size=3, stride=1, padding=1),
            Residual_BottleNeck(in_features=1024, width=256, out_features=1024, kernel_size=3, stride=1, padding=1),
            Residual_BottleNeck(in_features=1024, width=256, out_features=1024, kernel_size=3, stride=1, padding=1),
        )
        self.layer4 = nn.Sequential(
            Residual_BottleNeck(in_features=1024, width=512, out_features=self.out_channels, kernel_size=3, stride=2, padding=1, downsample=nn.Conv2d(in_channels=1024, out_channels=self.out_channels, kernel_size=1, stride=2)),
            Residual_BottleNeck(in_features=self.out_channels, width=512, out_features=self.out_channels, kernel_size=3, stride=1, padding=1),
            Residual_BottleNeck(in_features=self.out_channels, width=512, out_features=self.out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, mode=""):
        x = self.conv(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)     # (Batch, 256, W/4, H/4)
        c3 = self.layer2(c2)    # (Batch, 512, W/8, H/8)
        c4 = self.layer3(c3)    # (Batch, 1024, W/16, H/16)
        c5 = self.layer4(c4)    # (Batch, 2048, W/32, H/32)

        if mode == "fpn":
            return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])
        else:
            return c5
        
class ResNet50_FPN(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.out_channels = out_channels
        self.resnet50 = ResNet50()
        self.resnet50_out_channels = self.resnet50.out_channels

        self.conv_c2 = nn.Conv2d(self.resnet50_out_channels//8, self.resnet50_out_channels//8, kernel_size=1, stride=1, padding=0)
        self.conv_c3 = nn.Conv2d(self.resnet50_out_channels//4, self.resnet50_out_channels//8, kernel_size=1, stride=1, padding=0)
        self.conv_c4 = nn.Conv2d(self.resnet50_out_channels//2, self.resnet50_out_channels//8, kernel_size=1, stride=1, padding=0)
        self.conv_c5 = nn.Conv2d(self.resnet50_out_channels, self.resnet50_out_channels//8, kernel_size=1, stride=1, padding=0)
        
        self.conv_p2 = Conv2D_ReLU(self.resnet50_out_channels//8, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_p3 = Conv2D_ReLU(self.resnet50_out_channels//8, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_p4 = Conv2D_ReLU(self.resnet50_out_channels//8, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_p5 = Conv2D_ReLU(self.resnet50_out_channels//8, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        c2, c3, c4, c5 = list(self.resnet50(x, "fpn").values())
        
        m5 = self.conv_c5(c5)
        m4 = self.upsample(m5) + self.conv_c4(c4)
        m3 = self.upsample(m4) + self.conv_c3(c3)
        m2 = self.upsample(m3) + self.conv_c2(c2)

        p2 = self.conv_p2(m2)
        p3 = self.conv_p3(m3)
        p4 = self.conv_p4(m4)
        p5 = self.conv_p5(m5)

        return OrderedDict([("p2", p2), ("p3", p3), ("p4", p4), ("p5", p5)])
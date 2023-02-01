import torch
import torch.nn as nn






class Detect_Head(nn.Module):
    def __init__(self, feature_channels):
        super(Detect_Head, self).__init__()

    def forward(self, instance_features, labels, reg_targets):
        return
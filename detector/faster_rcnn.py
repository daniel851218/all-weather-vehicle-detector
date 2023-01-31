import os
import torch
import torch.nn as nn

from config.train_cfg import cfg
from common_module.resnet import ResNet50_FPN
from common_module.rpn import RPN

class Faster_RCNN(nn.Module):
    def __init__(self):
        super(Faster_RCNN, self).__init__()
        self.backbone = ResNet50_FPN()
        self.rpn = RPN(self.backbone.out_channels, len(cfg.aspect_ratio))
        self.roi = None
        self.cls_head = None
        self.reg_head = None

    def forward(self, imgs, targets):
        features = self.backbone(imgs)
        # img: torch.Tensor, shape=(batch_size, 3, 416, 640)
        # features: collections.OrderedDict,
        #           features["p2"]: shape=(batch_size, 512, 104, 160)
        #           features["p3"]: shape=(batch_size, 512,  52,  80)
        #           features["p4"]: shape=(batch_size, 512,  26,  40)
        #           features["p5"]: shape=(batch_size, 512,  13,  20)

        proposals, rpn_losses = self.rpn(imgs, features, targets)
        # proposals: list
        #            len: batch_size
        #            shape of each element: (post_nms_top_n, 4) = (2000, 4)
        #
        # rpn_losses: dict
        #             len: batch_size
        #             keys: "loss_rpn_score", "loss_rpn_box_reg"
        
        

        
        return
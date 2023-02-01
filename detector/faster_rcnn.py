import os
import torch
import torch.nn as nn

from config.train_cfg import cfg
from common_module.resnet import ResNet50_FPN
from common_module.rpn import RPN
from common_module.roi import RoI_Align
from common_module.detect_head import Detect_Head

class Faster_RCNN(nn.Module):
    def __init__(self):
        super(Faster_RCNN, self).__init__()
        self.backbone = ResNet50_FPN()
        self.rpn = RPN(self.backbone.out_channels, len(cfg.aspect_ratio))
        self.roi_align = RoI_Align()
        self.detect_head = Detect_Head(feature_channels=self.backbone.out_channels)

    def forward(self, imgs, targets):
        features = self.backbone(imgs)
        # img: torch.Tensor, shape = (batch_size, 3, 416, 640)
        # features: collections.OrderedDict,
        #           features["p2"]: shape = (batch_size, 512, 104, 160)
        #           features["p3"]: shape = (batch_size, 512,  52,  80)
        #           features["p4"]: shape = (batch_size, 512,  26,  40)
        #           features["p5"]: shape = (batch_size, 512,  13,  20)

        proposals, rpn_losses = self.rpn(imgs, features, targets)
        # proposals: list
        #            len: batch_size
        #            shape of each element: (post_nms_top_n, 4) = (2000, 4)
        #
        # rpn_losses: dict
        #             len: batch_size
        #             keys: "loss_rpn_score", "loss_rpn_box_reg"

        instance_features, labels, reg_targets, proposals = self.roi_align(features, proposals, [(cfg.img_h, cfg.img_w)]*len(proposals), targets)
        # instance_features: torch.Tensor, shape = (box_batch_size_per_img*batch_size, backbone.out_channels, roi_align_out_size)
        #                                        = (box_batch_size_per_img*batch_size, 512, 14, 14)
        #
        # labels: list
        #         len: batch_size
        #         shape of each element: (box_batch_size_per_img, ) = (512, )
        # 
        # reg_targets: list
        #              len: batch_size
        #              shape of each element: (box_batch_size_per_img, 4) = (512, 4)
        
        embedded_ins_features,cls_scores, bbox_reg, roi_losses, detection_result = self.detect_head(instance_features, labels, reg_targets, proposals)
        
        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)
        
        return self.eager_outputs(losses, detection_result)
    
    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses
        return detections
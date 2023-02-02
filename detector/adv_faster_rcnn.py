import os
import torch
import torch.nn as nn

from config.train_cfg import cfg
from common_module.resnet import ResNet50_FPN
from common_module.rpn import RPN
from common_module.roi import RoI_Align
from common_module.detect_head import Detect_Head

class Adversarial_Faster_RCNN(nn.Module):
    def __init__(self):
        super(Adversarial_Faster_RCNN, self).__init__()
        self.backbone = ResNet50_FPN()
        self.rpn = RPN(self.backbone.out_channels, len(cfg.aspect_ratio))
        self.roi_align = RoI_Align()
        self.detect_head = Detect_Head(feature_channels=self.backbone.out_channels)

        self.daytime_classifier = None
        self.weather_classifier = None

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
        # embedded_ins_features: torch.Tensor, shape = (box_batch_size_per_img*batch_size, 1024, 7, 7)
        # 
        # cls_scores: torch.Tensor, shape = (box_batch_size_per_img*batch_size, num_classes)
        # 
        # bbox_reg: torch.Tensor, shape = (box_batch_size_per_img*batch_size, num_classes*4)
        # 
        # roi_losses: dict
        #             keys: "loss_roi_cls", "loss_roi_box_reg"
        
        losses = {
            "loss_total": rpn_losses["loss_rpn_score"] + rpn_losses["loss_rpn_score"] + roi_losses["loss_roi_cls"] + roi_losses["loss_roi_box_reg"]
        }
        losses.update(rpn_losses)
        losses.update(roi_losses)
        
        return self.eager_outputs(losses, detection_result)
    
    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses
        return detections
    
    def save_model(self, epoch, ckpt_dir):
        print("Saving Model ...")
        weight_dir = "weights"
        if not os.path.isdir(os.path.join(ckpt_dir, weight_dir)):
            os.mkdir(os.path.join(ckpt_dir, weight_dir))

        torch.save(self.backbone.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_backbone.pth"))
        torch.save(self.rpn.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_rpn.pth"))
        torch.save(self.roi_align.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_roi.pth"))
        torch.save(self.detect_head.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_detect_head.pth"))

    def load_model(self, model_dir, epoch):
        print("Loading Model ...")
        assert len(model_dir) > 0

        backbone_weight = os.path.join(model_dir, f"{epoch}_backbone.pth")
        rpn_weight = os.path.join(model_dir, f"{epoch}_rpn.pth")
        roi_weight = os.path.join(model_dir, f"{epoch}_roi.pth")
        detect_head_weight = os.path.join(model_dir, f"{epoch}_detect_head.pth")

        print(f"{'Backbone Weight: ':>25s}" + backbone_weight)
        print(f"{'RPN Weight: ':>25s}" + rpn_weight)
        print(f"{'ROI Weight: ':>25s}" + roi_weight)
        print(f"{'Detection Head Weight: ':>25s}" + detect_head_weight + "\n")

        self.backbone.load_state_dict(torch.load(backbone_weight))
        self.rpn.load_state_dict(torch.load(rpn_weight))
        self.roi_align.load_state_dict(torch.load(roi_weight))
        self.detect_head.load_state_dict(torch.load(detect_head_weight))
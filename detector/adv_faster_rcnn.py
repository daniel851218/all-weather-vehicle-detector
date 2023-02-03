import os
import torch
import torch.nn as nn

from config.train_cfg import cfg
from common_module.resnet import ResNet50_FPN
from common_module.rpn import RPN
from common_module.roi import RoI_Align
from common_module.detect_head import Detect_Head
from common_module.domain_classifier import Daytime_Classifier, Weather_Classifier

class Adversarial_Faster_RCNN(nn.Module):
    def __init__(self):
        super(Adversarial_Faster_RCNN, self).__init__()
        self.backbone = ResNet50_FPN()
        self.rpn = RPN(self.backbone.out_channels, len(cfg.aspect_ratio))
        self.roi_align = RoI_Align()
        self.detect_head = Detect_Head(feature_channels=self.backbone.out_channels)

        self.daytime_classifier = Daytime_Classifier(self.backbone.out_channels)
        self.weather_classifier = Weather_Classifier(self.backbone.out_channels)

    def forward(self, imgs, targets):
        img_features = self.backbone(imgs)
        # img: torch.Tensor, shape = (batch_size, 3, 416, 640)
        # img_features: collections.OrderedDict,
        #           img_features["p2"]: shape = (batch_size, 512, 104, 160)
        #           img_features["p3"]: shape = (batch_size, 512,  52,  80)
        #           img_features["p4"]: shape = (batch_size, 512,  26,  40)
        #           img_features["p5"]: shape = (batch_size, 512,  13,  20)

        proposals, rpn_losses = self.rpn(imgs, img_features, targets)
        # proposals: list
        #            len: batch_size
        #            shape of each element: (post_nms_top_n, 4) = (2000, 4)
        #
        # rpn_losses: dict
        #             keys: "loss_rpn_score", "loss_rpn_box_reg"

        ins_features, labels, reg_targets, proposals = self.roi_align(img_features, proposals, [(cfg.img_h, cfg.img_w)]*len(proposals), targets)
        # ins_features: torch.Tensor, shape = (box_batch_size_per_img*batch_size, backbone.out_channels, roi_align_out_size)
        #                                        = (box_batch_size_per_img*batch_size, 512, 14, 14)
        #
        # labels: list
        #         len: batch_size
        #         shape of each element: (box_batch_size_per_img, ) = (512, )
        # 
        # reg_targets: list
        #              len: batch_size
        #              shape of each element: (box_batch_size_per_img, 4) = (512, 4)
        
        embedded_ins_features,cls_scores, bbox_reg, roi_losses, detection_result = self.detect_head(ins_features, labels, reg_targets, proposals)
        # embedded_ins_features: torch.Tensor, shape = (box_batch_size_per_img*batch_size, 1024, 7, 7)
        # 
        # cls_scores: torch.Tensor, shape = (box_batch_size_per_img*batch_size, num_classes)
        # 
        # bbox_reg: torch.Tensor, shape = (box_batch_size_per_img*batch_size, num_classes*4)
        # 
        # roi_losses: dict
        #             keys: "loss_roi_cls", "loss_roi_box_reg"

        if self.training:
            daytime_adv_losses = self.daytime_classifier(img_features, ins_features, targets)
            weather_adv_losses = self.weather_classifier(img_features, ins_features, targets)
        # daytime_adv_losses: dict
        #                     keys: "loss_daytime_img_score", "loss_daytime_ins_score", "loss_daytime_consistency"
        #
        # weather_adv_losses: dict
        #                     keys: "loss_weather_img_score", "loss_weather_ins_score", "loss_weather_consistency"

        losses = {}
        if self.training:
            losses["loss_total"] = rpn_losses["loss_rpn_score"] + rpn_losses["loss_rpn_box_reg"] + \
                                   roi_losses["loss_roi_cls"] + roi_losses["loss_roi_box_reg"] + \
                                   cfg.lambda_adv_daytime * (daytime_adv_losses["loss_daytime_img_score"] + daytime_adv_losses["loss_daytime_ins_score"] + daytime_adv_losses["loss_daytime_consistency"]) + \
                                   cfg.lambda_adv_weather * (weather_adv_losses["loss_weather_img_score"] + weather_adv_losses["loss_weather_ins_score"] + weather_adv_losses["loss_weather_consistency"])
            losses.update(rpn_losses)
            losses.update(roi_losses)
            losses.update(daytime_adv_losses)
            losses.update(weather_adv_losses)
        
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
        torch.save(self.daytime_classifier.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_daytime_classifier.pth"))
        torch.save(self.weather_classifier.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_weather_classifier.pth"))

    def load_model(self, model_dir, epoch):
        print("Loading Model ...")
        assert len(model_dir) > 0

        backbone_weight = os.path.join(model_dir, f"{epoch}_backbone.pth")
        rpn_weight = os.path.join(model_dir, f"{epoch}_rpn.pth")
        roi_weight = os.path.join(model_dir, f"{epoch}_roi.pth")
        detect_head_weight = os.path.join(model_dir, f"{epoch}_detect_head.pth")
        daytime_classifier_weight = os.path.join(model_dir, f"{epoch}_daytime_classifier.pth")
        weather_classifier_weight = os.path.join(model_dir, f"{epoch}_weather_classifier.pth")

        print(f"{'Backbone Weight: ':>40s}" + backbone_weight)
        print(f"{'RPN Weight: ':>40s}" + rpn_weight)
        print(f"{'ROI Weight: ':>40s}" + roi_weight)
        print(f"{'Detection Head Weight: ':>40s}" + detect_head_weight)
        print(f"{'Daytime Classifier Weight: ':>40s}" + daytime_classifier_weight)
        print(f"{'Weather Classifier Weight: ':>40s}" + weather_classifier_weight + "\n")

        self.backbone.load_state_dict(torch.load(backbone_weight))
        self.rpn.load_state_dict(torch.load(rpn_weight))
        self.roi_align.load_state_dict(torch.load(roi_weight))
        self.detect_head.load_state_dict(torch.load(detect_head_weight))
        self.daytime_classifier.load_state_dict(torch.load(daytime_classifier_weight))
        self.weather_classifier.load_state_dict(torch.load(weather_classifier_weight))
import os
import torch
import torch.nn as nn

from torchvision import ops
from common_module.pseudo_label_filter import filter_outlier
from config.train_test_cfg import cfg
from common_module.resnet import ResNet50_FPN
from common_module.rpn import RPN
from common_module.roi import RoI_Align
from common_module.detect_head import Detect_Head
from common_module.domain_classifier import Daytime_Classifier, Weather_Classifier

class Semi_Supervised_Adversarial_Faster_RCNN(nn.Module):
    def __init__(self):
        super(Semi_Supervised_Adversarial_Faster_RCNN, self).__init__()
        self.student_backbone = ResNet50_FPN()
        self.student_rpn = RPN(self.student_backbone.out_channels, len(cfg.aspect_ratio))
        self.student_roi_align = RoI_Align()      
        self.student_detect_head = Detect_Head(feature_channels=self.student_backbone.out_channels)
        self.student_daytime_classifier = Daytime_Classifier(self.student_backbone.out_channels)
        self.student_weather_classifier = Weather_Classifier(self.student_backbone.out_channels)


        self.teacher_backbone = ResNet50_FPN().requires_grad_(False)
        self.teacher_rpn = RPN(self.teacher_backbone.out_channels, len(cfg.aspect_ratio)).requires_grad_(False)
        self.teacher_roi_align = RoI_Align().requires_grad_(False)
        self.teacher_detect_head = Detect_Head(feature_channels=self.teacher_backbone.out_channels).requires_grad_(False)
        
        self.load_model(cfg.pre_train_model_path, cfg.pre_train_model_epoch)

    def forward(self, bdd_imgs, bdd_targets, driving_video_imgs_1, driving_video_imgs_2, driving_video_targets, ratio, delta):
        losses = {}
        if self.training:
            sup_loss = self.forward_supervised(bdd_imgs, bdd_targets)
            unsup_loss = self.forward_unsupervised(driving_video_imgs_1, driving_video_imgs_2, driving_video_targets, ratio, delta)

            losses["loss_sup"] = sup_loss
            losses["loss_unsup"] = unsup_loss
            losses["loss_total"] = sup_loss["loss_rpn_score"] + sup_loss["loss_rpn_box_reg"] + sup_loss["loss_roi_cls"] + sup_loss["loss_roi_box_reg"] + \
                                   cfg.lambda_unsup * (unsup_loss["loss_rpn_score"] + unsup_loss["loss_rpn_box_reg"] + unsup_loss["loss_roi_cls"] + unsup_loss["loss_roi_box_reg"]) + \
                                   cfg.lambda_adv_daytime * (unsup_loss["loss_daytime_img_score"] + unsup_loss["loss_daytime_ins_score"] + unsup_loss["loss_daytime_consistency"]) + \
                                   cfg.lambda_adv_weather * (unsup_loss["loss_weather_img_score"] + unsup_loss["loss_weather_ins_score"] + unsup_loss["loss_weather_consistency"])
            
            return losses
        else:
            features = self.teacher_backbone(bdd_imgs)
            proposals, rpn_losses = self.teacher_rpn(bdd_imgs, features, bdd_targets)
            instance_features, labels, reg_targets, proposals = self.teacher_roi_align(features, proposals, [(cfg.img_h, cfg.img_w)]*len(proposals), bdd_targets)
            embedded_ins_features,cls_scores, bbox_reg, roi_losses, detection = self.teacher_detect_head(instance_features, labels, reg_targets, proposals)
            
            return detection
    
    def forward_supervised(self, imgs, targets):
        features = self.student_backbone(imgs)
        proposals, rpn_losses = self.student_rpn(imgs, features, targets)
        instance_features, labels, reg_targets, proposals = self.student_roi_align(features, proposals, [(cfg.img_h, cfg.img_w)]*len(proposals), targets)
        _,_, _, roi_losses, _ = self.student_detect_head(instance_features, labels, reg_targets, proposals)

        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)

        return losses
    
    def forward_unsupervised(self, imgs_1, imgs_2, targets, ratio, delta):
        empty_targets = ({},) * imgs_1.shape[0]

        # Use teacher model generates pseudo-labels
        self.teacher_backbone.eval()
        self.teacher_rpn.eval()
        self.teacher_roi_align.eval()
        self.teacher_detect_head.eval()
        
        features_t = self.teacher_backbone(imgs_1)
        proposals_t, _ = self.teacher_rpn(imgs_1, features_t, empty_targets)
        instance_features_t, labels_t, reg_targets_t, proposals_t = self.teacher_roi_align(features_t, proposals_t, [(cfg.img_h, cfg.img_w)]*len(proposals_t), empty_targets)
        _, _, _, _, detection_t = self.teacher_detect_head(instance_features_t, labels_t, reg_targets_t, proposals_t)

        # filter low quality pseudo-labels
        for i, d in enumerate(detection_t):
            low_conf_filter = d["scores"] > cfg.test_weak_conf_thresh
            d["boxes"] = d["boxes"][low_conf_filter]
            d["labels"] = d["labels"][low_conf_filter]
            d["scores"] = d["scores"][low_conf_filter]

            nms_filter = ops.batched_nms(d["boxes"], d["scores"], d["labels"], cfg.test_iou_thresh)
            d["boxes"] = d["boxes"][nms_filter]
            d["labels"] = d["labels"][nms_filter]
            d["scores"] = d["scores"][nms_filter]
            
            d["boxes"], d["labels"], d["scores"] = filter_outlier(d["boxes"], d["labels"], d["scores"], ratio[i], delta[i])

            d["boxes"], d["labels"] = d["boxes"].to(cfg.device), d["labels"].to(cfg.device)
            del(d["scores"])

        # train student model
        features_s = self.student_backbone(imgs_2)
        proposals_s, rpn_losses = self.student_rpn(imgs_2, features_s, detection_t)
        instance_features_s, labels_s, reg_targets_s, proposals_s = self.student_roi_align(features_s, proposals_s, [(cfg.img_h, cfg.img_w)]*len(proposals_s), detection_t)
        _, _, _, roi_losses, _ = self.student_detect_head(instance_features_s, labels_s, reg_targets_s, proposals_s)

        daytime_adv_losses = self.student_daytime_classifier(features_s, instance_features_s, targets)
        weather_adv_losses = self.student_weather_classifier(features_s, instance_features_s, targets)

        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)
        losses.update(daytime_adv_losses)
        losses.update(weather_adv_losses)

        return losses
    
    def save_model(self, epoch, ckpt_dir):
        print("Saving Model ...")
        weight_dir = "weights"
        if not os.path.isdir(os.path.join(ckpt_dir, weight_dir)):
            os.mkdir(os.path.join(ckpt_dir, weight_dir))

        torch.save(self.teacher_backbone.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_backbone.pth"))
        torch.save(self.teacher_rpn.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_rpn.pth"))
        torch.save(self.teacher_roi_align.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_roi.pth"))
        torch.save(self.teacher_detect_head.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_detect_head.pth"))

        torch.save(self.student_daytime_classifier.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_daytime_classifier.pth"))
        torch.save(self.student_weather_classifier.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_weather_classifier.pth"))
        
    def load_model(self, model_dir, epoch):
        print("Loading Model ...")
        assert len(model_dir) > 0

        backbone_weight = os.path.join(model_dir, f"{epoch}_backbone.pth")
        rpn_weight = os.path.join(model_dir, f"{epoch}_rpn.pth")
        roi_weight = os.path.join(model_dir, f"{epoch}_roi.pth")
        detect_head_weight = os.path.join(model_dir, f"{epoch}_detect_head.pth")
        daytime_classifier_weight = os.path.join(model_dir, f"{epoch}_daytime_classifier.pth")
        weather_classifier_weight = os.path.join(model_dir, f"{epoch}_weather_classifier.pth")

        print(f"{'Backbone Weight: ' :>35}" + backbone_weight)
        print(f"{'RPN Weight: ' :>35}" + rpn_weight)
        print(f"{'ROI Weight: ' :>35}" + roi_weight)
        print(f"{'Detection Head Weight: ' :>35}" + detect_head_weight)
        print(f"{'Daytime Classifier Weight: ' :>35s}" + daytime_classifier_weight)
        print(f"{'Weather Classifier Weight: ' :>35s}" + weather_classifier_weight + "\n")

        self.student_backbone.load_state_dict(torch.load(backbone_weight))
        self.student_rpn.load_state_dict(torch.load(rpn_weight))
        self.student_roi_align.load_state_dict(torch.load(roi_weight))
        self.student_detect_head.load_state_dict(torch.load(detect_head_weight))
        self.student_daytime_classifier.load_state_dict(torch.load(daytime_classifier_weight))
        self.student_weather_classifier.load_state_dict(torch.load(weather_classifier_weight))

        self.teacher_backbone.load_state_dict(torch.load(backbone_weight))
        self.teacher_rpn.load_state_dict(torch.load(rpn_weight))
        self.teacher_roi_align.load_state_dict(torch.load(roi_weight))
        self.teacher_detect_head.load_state_dict(torch.load(detect_head_weight))
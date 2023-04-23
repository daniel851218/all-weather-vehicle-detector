import os
import torch
import torch.nn as nn

from torchvision import ops
from common_module.outlier_filter import filter_outlier
from config.train_test_cfg import cfg
from common_module.resnet import ResNet50_FPN
from common_module.rpn import RPN
from common_module.roi import RoI_Align
from common_module.detect_head import Detect_Head
from common_module.domain_classifier import Daytime_Classifier, Weather_Classifier

class Semi_Supervised_Faster_RCNN_Stage_1(nn.Module):
    def __init__(self):
        super(Semi_Supervised_Faster_RCNN_Stage_1, self).__init__()
        self.backbone = ResNet50_FPN()
        self.rpn = RPN(self.backbone.out_channels, len(cfg.aspect_ratio))
        self.roi_align = RoI_Align()

        self.primary_detect_head = Detect_Head(feature_channels=self.backbone.out_channels)
        self.auxiliary_detect_head = Detect_Head(feature_channels=self.backbone.out_channels)

        self.daytime_classifier = Daytime_Classifier(self.backbone.out_channels)
        self.weather_classifier = Weather_Classifier(self.backbone.out_channels)

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.SmoothL1Loss()

    def forward(self, src_imgs, src_targets, tgt_imgs, tgt_targets):
        losses = {}
        if self.training:
            src_embedded_ins_features_p, src_embedded_ins_features_a, src_losses, _ = self.domain_forward(src_imgs, src_targets, domain="source")
            tgt_embedded_ins_features_p, tgt_embedded_ins_features_a, tgt_losses, tgt_detection = self.domain_forward(tgt_imgs, tgt_targets, domain="target")

            pos_idx, neg_idx = self.divide_pos_neg_set(src_embedded_ins_features_p.detach(), tgt_embedded_ins_features_p.detach())
            feature_consistency_loss = self.compute_feature_consistency_loss(src_embedded_ins_features_a.detach(), tgt_embedded_ins_features_a, pos_idx, neg_idx)
            
            losses["loss_primary"] = src_losses
            losses["loss_auxiliary"] = tgt_losses
            losses["loss_feature_consistency"] = feature_consistency_loss

            primary_total_loss = src_losses["loss_rpn_score"] + src_losses["loss_rpn_box_reg"] + \
                                 src_losses["loss_roi_cls"] + src_losses["loss_roi_box_reg"] + \
                                 cfg.lambda_cls_consistency * src_losses["loss_roi_cls_consistency"] + \
                                 cfg.lambda_reg_consistency * src_losses["loss_roi_reg_consistency"] + \
                                 cfg.lambda_adv_daytime * (src_losses["loss_daytime_img_score"] + src_losses["loss_daytime_ins_score"] + src_losses["loss_daytime_consistency"]) + \
                                 cfg.lambda_adv_weather * (src_losses["loss_weather_img_score"] + src_losses["loss_weather_ins_score"] + src_losses["loss_weather_consistency"])
            
            auxiliary_total_loss = tgt_losses["loss_rpn_score"] + tgt_losses["loss_rpn_box_reg"] + \
                                   tgt_losses["loss_roi_cls"] + tgt_losses["loss_roi_box_reg"] + \
                                   cfg.lambda_cls_consistency * tgt_losses["loss_roi_cls_consistency"] + \
                                   cfg.lambda_reg_consistency * tgt_losses["loss_roi_reg_consistency"] + \
                                   cfg.lambda_adv_daytime * (tgt_losses["loss_daytime_img_score"] + tgt_losses["loss_daytime_ins_score"] + tgt_losses["loss_daytime_consistency"]) + \
                                   cfg.lambda_adv_weather * (tgt_losses["loss_weather_img_score"] + tgt_losses["loss_weather_ins_score"] + tgt_losses["loss_weather_consistency"])
            
            losses["loss_total"] = primary_total_loss + auxiliary_total_loss + cfg.lambda_feature_consistency * feature_consistency_loss

        else:
            tgt_embedded_ins_features_p, tgt_embedded_ins_features_a, tgt_losses, tgt_detection = self.domain_forward(tgt_imgs, tgt_targets, domain="target")
        
        return self.eager_outputs(losses, tgt_detection)
    
    def domain_forward(self, imgs, targets, domain):
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
        
        if domain == "source":
            embedded_ins_features_p, cls_scores_p, bbox_reg_p, roi_losses_p, detection_p = self.primary_detect_head(ins_features, labels, reg_targets, proposals)

            with torch.no_grad():
                embedded_ins_features_a, cls_scores_a, bbox_reg_a, roi_losses_a, detection_a = self.auxiliary_detect_head(ins_features, None, None, proposals)
            
            ssl_gt_label = torch.argmax(torch.softmax(cls_scores_a, dim=1), dim=1)
            cls_consistency_loss = self.cls_loss_fn(cls_scores_p, ssl_gt_label)
            reg_consistency_loss = self.reg_loss_fn(bbox_reg_p, bbox_reg_a)
        elif domain == "target":
            if self.training:
                with torch.no_grad():
                    embedded_ins_features_p, cls_scores_p, bbox_reg_p, roi_losses_p, detection_p = self.primary_detect_head(ins_features, None, None, proposals)
                    ssl_gt_label = torch.argmax(torch.softmax(cls_scores_p, dim=1), dim=1)
                
                embedded_ins_features_a, cls_scores_a, bbox_reg_a, roi_losses_a, detection_a = self.auxiliary_detect_head(ins_features, labels, reg_targets, proposals)

                cls_consistency_loss = self.cls_loss_fn(cls_scores_a, ssl_gt_label)
                reg_consistency_loss = self.reg_loss_fn(bbox_reg_a, bbox_reg_p)
            else:
                embedded_ins_features_a, cls_scores_a, bbox_reg_a, roi_losses_a, detection_a = self.auxiliary_detect_head(ins_features, labels, reg_targets, proposals)
        else:
            raise ValueError("domain only can be source or target.")
        
        # embedded_ins_features: torch.Tensor, shape = (box_batch_size_per_img*batch_size, 1024, 7, 7)
        # 
        # cls_scores: torch.Tensor, shape = (box_batch_size_per_img*batch_size, num_classes)
        # 
        # bbox_reg: torch.Tensor, shape = (box_batch_size_per_img*batch_size, num_classes*4)
        # 
        # roi_losses: dict
        #             keys: "loss_roi_cls", "loss_roi_box_reg"
        #
        # detection: list
        #                     keys of each elements: boxes, labels, scores

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
            is_labeled = True if "boxes" in targets[0].keys() and "labels" in targets[0].keys() else False
            if is_labeled:
                losses.update(rpn_losses)
                
                if domain == "source":
                    losses.update(roi_losses_p)
                else:
                    losses.update(roi_losses_a)

                losses["loss_roi_cls_consistency"] = cls_consistency_loss
                losses["loss_roi_reg_consistency"] = reg_consistency_loss

                losses.update(daytime_adv_losses)
                losses.update(weather_adv_losses)
            else:
                losses["loss_roi_cls_consistency"] = cls_consistency_loss
                losses["loss_roi_reg_consistency"] = reg_consistency_loss

                losses.update(daytime_adv_losses)
                losses.update(weather_adv_losses)
        
        if domain == "source":
            return embedded_ins_features_p, embedded_ins_features_a, losses, detection_p
        else:
            if self.training:
                return embedded_ins_features_p, embedded_ins_features_a, losses, detection_a
            else:
                return None, embedded_ins_features_a, losses, detection_a

    def divide_pos_neg_set(self, src_embedded_ins_features, tgt_embedded_ins_features):
        # compute cosine
        src_embedded_ins_features = torch.mean(src_embedded_ins_features, dim=[2, 3]).T
        tgt_embedded_ins_features = torch.mean(tgt_embedded_ins_features, dim=[2, 3])

        inner_product = torch.mm(tgt_embedded_ins_features, src_embedded_ins_features)
        norm_src = torch.norm(src_embedded_ins_features, dim=0, keepdim=True).expand_as(inner_product)
        norm_tgt = torch.norm(tgt_embedded_ins_features, dim=1, keepdim=True).expand_as(inner_product)
        cos_similarity = inner_product / norm_src / norm_tgt
        
        # take top-k entry as positive set, and the others are negative set
        sorted_idx = torch.argsort(cos_similarity, dim=1, descending=True)
        top_k = int(len(sorted_idx) * cfg.cos_similarity_top_n_ratio)
        
        pos_idx = sorted_idx[:, :top_k]
        neg_idx = sorted_idx[:, top_k:]

        return pos_idx, neg_idx
    
    def compute_feature_consistency_loss(self, src_embedded_ins_features, tgt_embedded_ins_features, pos_idx, neg_idx):
        src_embedded_ins_features = torch.mean(src_embedded_ins_features, dim=[2, 3])
        src_embedded_ins_features = src_embedded_ins_features / torch.norm(src_embedded_ins_features, dim=1, keepdim=True)
        tgt_embedded_ins_features = torch.mean(tgt_embedded_ins_features, dim=[2, 3])
        tgt_embedded_ins_features = tgt_embedded_ins_features / torch.norm(tgt_embedded_ins_features, dim=1, keepdim=True)
        
        pos_sample = src_embedded_ins_features[pos_idx]
        neg_sample = src_embedded_ins_features[neg_idx]

        pos_similarity = torch.sum(tgt_embedded_ins_features.unsqueeze(1) * pos_sample, dim=2, keepdim=True)
        pos_similarity = torch.sum(pos_similarity.exp(), dim=1)
        

        neg_similarity = torch.sum(tgt_embedded_ins_features.unsqueeze(1) * neg_sample, dim=2, keepdim=True)
        neg_similarity = torch.sum(neg_similarity.exp(), dim=1)
        
        loss = -torch.mean((pos_similarity / (pos_similarity + neg_similarity)).log())
        return loss
    
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
        torch.save(self.primary_detect_head.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_primary_detect_head.pth"))
        torch.save(self.auxiliary_detect_head.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_auxiliary_detect_head.pth"))
        torch.save(self.daytime_classifier.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_daytime_classifier.pth"))
        torch.save(self.weather_classifier.state_dict(), os.path.join(ckpt_dir, weight_dir, f"{epoch}_weather_classifier.pth"))

    def load_model(self, model_dir, epoch):
        print("Loading Model ...")
        assert len(model_dir) > 0

        backbone_weight = os.path.join(model_dir, f"{epoch}_backbone.pth")
        rpn_weight = os.path.join(model_dir, f"{epoch}_rpn.pth")
        roi_weight = os.path.join(model_dir, f"{epoch}_roi.pth")
        primary_detect_head_weight = os.path.join(model_dir, f"{epoch}_primary_detect_head.pth")
        auxiliary_detect_head_weight = os.path.join(model_dir, f"{epoch}_auxiliary_detect_head.pth")
        daytime_classifier_weight = os.path.join(model_dir, f"{epoch}_daytime_classifier.pth")
        weather_classifier_weight = os.path.join(model_dir, f"{epoch}_weather_classifier.pth")

        print(f"{'Backbone Weight: ':>60s}" + backbone_weight)
        print(f"{'RPN Weight: ':>60s}" + rpn_weight)
        print(f"{'ROI Weight: ':>60s}" + roi_weight)
        print(f"{'Primary Detection Head Weight: ':>60s}" + primary_detect_head_weight)
        print(f"{'Auxiliary Detection Head Weight':>60s}" + auxiliary_detect_head_weight)
        print(f"{'Daytime Classifier Weight: ':>60s}" + daytime_classifier_weight)
        print(f"{'Weather Classifier Weight: ':>60s}" + weather_classifier_weight + "\n")

        self.backbone.load_state_dict(torch.load(backbone_weight))
        self.rpn.load_state_dict(torch.load(rpn_weight))
        self.roi_align.load_state_dict(torch.load(roi_weight))
        self.primary_detect_head.load_state_dict(torch.load(primary_detect_head_weight))
        self.auxiliary_detect_head.load_state_dict(torch.load(auxiliary_detect_head_weight))
        self.daytime_classifier.load_state_dict(torch.load(daytime_classifier_weight))
        self.weather_classifier.load_state_dict(torch.load(weather_classifier_weight))

# ----------------------------------------------------------------------------------------------------

class Semi_Supervised_Faster_RCNN_Stage_2(nn.Module):
    def __init__(self):
        super(Semi_Supervised_Faster_RCNN_Stage_2, self).__init__()
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
            low_conf_filter = d["scores"] > cfg.test_conf_thresh
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
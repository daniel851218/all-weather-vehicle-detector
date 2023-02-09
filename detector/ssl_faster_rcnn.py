import os
import torch
import torch.nn as nn

from config.train_cfg import cfg
from common_module.resnet import ResNet50_FPN
from common_module.rpn import RPN
from common_module.roi import RoI_Align
from common_module.detect_head import Detect_Head
from common_module.domain_classifier import Daytime_Classifier, Weather_Classifier

class Semi_Supervised_Faster_RCNN(nn.Module):
    def __init__(self):
        super(Semi_Supervised_Faster_RCNN, self).__init__()
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
            src_embedded_ins_features_p, src_embedded_ins_features_a, src_losses, src_detection_result = self.domain_forward(src_imgs, src_targets, domain="source")
            tgt_embedded_ins_features_p, tgt_embedded_ins_features_a, tgt_losses, tgt_detection_result = self.domain_forward(tgt_imgs, tgt_targets, domain="target")

            pos_idx, neg_idx = self.divide_pos_neg_set(src_embedded_ins_features_p.detach(), tgt_embedded_ins_features_p.detach())
            feature_consistency_loss = self.compute_feature_consistency_loss(src_embedded_ins_features_a.detach(), tgt_embedded_ins_features_a, pos_idx, neg_idx)
            
            losses["loss_primary"] = src_losses
            losses["loss_auxiliary"] = tgt_losses
            losses["loss_feature_consistency"] = feature_consistency_loss
        else:
            tgt_embedded_ins_features_p, tgt_embedded_ins_features_a, tgt_losses, tgt_detection_result = self.domain_forward(tgt_imgs, tgt_targets, domain="target")
        
        return self.eager_outputs(losses, tgt_detection_result)
    
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
            embedded_ins_features_p, cls_scores_p, bbox_reg_p, roi_losses_p, detection_result_p = self.primary_detect_head(ins_features, labels, reg_targets, proposals)

            with torch.no_grad():
                embedded_ins_features_a, cls_scores_a, bbox_reg_a, roi_losses_a, detection_result_a = self.auxiliary_detect_head(ins_features, None, None, proposals)
            
            ssl_gt_label = torch.argmax(torch.softmax(cls_scores_a, dim=1), dim=1)
            cls_consistency_loss = self.cls_loss_fn(cls_scores_p, ssl_gt_label)
            reg_consistency_loss = self.reg_loss_fn(bbox_reg_p, bbox_reg_a)
        elif domain == "target":
            with torch.no_grad():
                embedded_ins_features_p, cls_scores_p, bbox_reg_p, roi_losses_p, detection_result_p = self.primary_detect_head(ins_features, None, None, proposals)
            
            embedded_ins_features_a, cls_scores_a, bbox_reg_a, roi_losses_a, detection_result_a = self.auxiliary_detect_head(ins_features, labels, reg_targets, proposals)

            ssl_gt_label = torch.argmax(torch.softmax(cls_scores_p, dim=1), dim=1)
            cls_consistency_loss = self.cls_loss_fn(cls_scores_a, ssl_gt_label)
            reg_consistency_loss = self.reg_loss_fn(bbox_reg_a, bbox_reg_p)
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
        # detection_result: list
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
            return embedded_ins_features_p, embedded_ins_features_a, losses, detection_result_p
        else:
            return embedded_ins_features_p, embedded_ins_features_a, losses, detection_result_a
        
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
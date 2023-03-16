import torch
import torch.nn as nn
from torch.jit.annotations import List, Dict

from config.train_test_cfg import cfg
from common_module import boxes_utils
from common_module.resnet import Residual_Basic

class Classification_Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classification_Head, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        
        return x
    
class Regression_Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Regression_Head, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = self.fc1(x)
        
        return x

class Detect_Head(nn.Module):
    def __init__(self, feature_channels):
        super(Detect_Head, self).__init__()
        self.embedded_layer = Residual_Basic(in_features=feature_channels, out_features=1024, kernel_size=3, stride=1, padding=1, downsample=nn.Conv2d(in_channels=feature_channels, out_channels=1024, kernel_size=1, stride=1))
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.cls_head = Classification_Head(in_channels=1024, num_classes=cfg.num_classes)
        self.reg_head = Regression_Head(in_channels=1024, num_classes=cfg.num_classes)

        self.cls_loss_fn = nn.CrossEntropyLoss()

        self.box_coder = boxes_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

    def forward(self, instance_features, labels, reg_targets, proposals):
        embedded_ins_features = self.embedded_layer(instance_features)
        instance_features = self.flatten(embedded_ins_features)
        
        cls_scores = self.cls_head(instance_features)
        bbox_reg = self.reg_head(instance_features)
        # cls_scores: torch.Tensor, shape = (batch_size*box_batch_size_per_img, num_classes) = (batch_size*box_batch_size_per_img, 7)
        # bbox_reg: torch.Tensor, shape = (batch_size*box_batch_size_per_img, num_classes * 4) = (batch_size*box_batch_size_per_img, 28)
        
        losses = {}
        detection_result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        if self.training:
            if labels is not None and reg_targets is not None:
                cls_loss, reg_loss = self.compute_loss(cls_scores, bbox_reg, labels, reg_targets)
                losses = {"loss_roi_cls": cls_loss, "loss_roi_box_reg": reg_loss}
        else:
            boxes, scores, labels = self.postprocess_detections(cls_scores, bbox_reg, proposals)
            num_imgs = len(boxes)
            for i in range(num_imgs):
                detection_result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        
        return embedded_ins_features,cls_scores, bbox_reg, losses, detection_result
    
    def compute_loss(self, cls_scores, bbox_reg, labels, reg_targets):
        labels = torch.cat(labels, dim=0)
        reg_targets = torch.cat(reg_targets, dim=0)
        
        cls_loss = self.cls_loss_fn(cls_scores, labels)

        sampled_pos_idxs_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_idxs_subset]

        N, _ = cls_scores.shape
        bbox_reg = bbox_reg.reshape(N, -1, 4)
        reg_loss = boxes_utils.smooth_l1_loss(bbox_reg[sampled_pos_idxs_subset, labels_pos], 
                                              reg_targets[sampled_pos_idxs_subset], 
                                              beta=1.0 / 9.0,
                                              size_average=False) / labels.numel()

        return cls_loss, reg_loss
    
    def postprocess_detections(self, cls_scores, bbox_reg, proposals):
        boxes_per_image = [p.shape[0] for p in proposals]

        pred_boxes = self.box_coder.decode(bbox_reg, proposals)
        pred_scores = torch.softmax(cls_scores, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes, all_scores, all_labels = [], [], []
        for boxes, scores in zip(pred_boxes_list, pred_scores_list):
            boxes = boxes_utils.clip_boxes_to_image(boxes, (cfg.img_h, cfg.img_w))

            # Create labels for each prediction
            labels = torch.arange(cfg.num_classes).to(cfg.device)
            labels = labels.view(1, -1).expand_as(scores)

            # Remove prediction with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # Batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # Remove low scoring boxes
            idxs = torch.nonzero(scores > cfg.box_score_thresh).squeeze(1)
            boxes, scores, labels = boxes[idxs], scores[idxs], labels[idxs]

            # Remove empty boxes
            keep = boxes_utils.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # Non-Maximun Suppression, independently done per class
            keep = boxes_utils.batched_nms(boxes, scores, labels, cfg.box_nms_thresh)

            # Keep only top-k scoring predictions
            keep = keep[:cfg.box_detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels
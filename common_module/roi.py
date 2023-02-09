import torch
import torch.nn as nn
import torchvision

from config.train_cfg import cfg
from common_module import boxes_utils

class RoI_Align(nn.Module):
    def __init__(self):
        super(RoI_Align, self).__init__()
        self.proposal_matcher = boxes_utils.Matcher(cfg.box_fg_iou_thresh, cfg.box_bg_iou_thresh, False)
        self.fg_bg_sampler = boxes_utils.Balanced_Pos_Neg_Sampler(cfg.box_batch_size_per_img, cfg.box_positive_fraction)
        self.box_coder = boxes_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.roi_align = torchvision.ops.MultiScaleRoIAlign(featmap_names=["p2", "p3", "p4", "p5"], output_size=cfg.roi_align_out_size, sampling_ratio=cfg.roi_sample_ratio)

    def forward(self, features, proposals, img_shapes, targets=None):
        is_labeled = True if "boxes" in targets[0].keys() and "labels" in targets[0].keys() else False
        if targets is not None and is_labeled:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"

        if self.training:
            if is_labeled:
                proposals, labels, reg_targets = self.select_training_samples(proposals, targets)
            # proposals: list
            #            len: batch_size
            #            shape of each element: (box_batch_size_per_img, 4) = (512, 4)
            #
            # labels: list
            #         len: batch_size
            #         shape of each element: (box_batch_size_per_img, ) = (512, )
            #
            # reg_targets: list
            #              len: batch_size
            #              shape of each element: (box_batch_size_per_img, 4) = (512, 4)
            else:
                for i in range(len(proposals)):
                    proposals[i] = proposals[i][:cfg.box_batch_size_per_img, :]
                labels, reg_targets = None, None
        else:
            labels, reg_targets = None, None

        instance_features = self.roi_align(features, proposals, img_shapes)
        # instance_features: torch.Tensor, shape = (box_batch_size_per_img*batch_size, backbone.out_channels, roi_align_out_size)
        
        return instance_features, labels, reg_targets, proposals
    
    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """
        Get the matched gt_bbox for every anchors, and set positive/negative samples
        """
        matched_idxs = []
        labels = []
        for proposals_per_img, gt_boxes_per_img, gt_labels_per_img in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_per_img.numel() == 0:
                clamped_matched_idxs_per_img = torch.zeros((proposals_per_img.shape[0],), dtype=torch.int64).to(cfg.device)
                labels_per_img = torch.zeros((proposals_per_img.shape[0],), dtype=torch.int64).to(cfg.device)
            else:
                match_quality_matrix = boxes_utils.box_iou(gt_boxes_per_img, proposals_per_img)
                matched_idxs_per_img = self.proposal_matcher(match_quality_matrix)
                clamped_matched_idxs_per_img = matched_idxs_per_img.clamp(min=0)
                labels_per_img = gt_labels_per_img[clamped_matched_idxs_per_img].to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_idxs = matched_idxs_per_img == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_img[bg_idxs] = 0

                # label ignore proposals (between low and high threshold)
                ignore_idxs = matched_idxs_per_img == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_img[ignore_idxs] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_per_img)
            labels.append(labels_per_img)

        return matched_idxs, labels
    
    def subsample(self, labels):
        sampled_pos_idxs, sampled_neg_idxs = self.fg_bg_sampler(labels)
        sampled_idxs = []
        for pos_idxs_img, neg_idxs_img in zip(sampled_pos_idxs, sampled_neg_idxs):
            img_sampled_idxs = torch.nonzero(pos_idxs_img | neg_idxs_img).squeeze(1)
            sampled_idxs.append(img_sampled_idxs)

        return sampled_idxs

    def select_training_samples(self, proposals, targets):
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])
        dtype = proposals[0].dtype

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # Append ground-truth bboxes to proposal for training
        proposals = [torch.cat([proposal, gt_box])for proposal, gt_box in zip(proposals, gt_boxes)]

        # Get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # Sample a fixed proportion of positive-negative proposals
        sampled_idxs = self.subsample(labels)

        matched_gt_boxes = []
        num_imgs = len(proposals)
        for img_id in range(num_imgs):
            img_sampled_idxs = sampled_idxs[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_idxs]
            labels[img_id] = labels[img_id][img_sampled_idxs]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_idxs]

            gt_boxes_per_img = gt_boxes[img_id]
            if gt_boxes_per_img.numel() == 0:
                gt_boxes_per_img = torch.zeros((1, 4), dtype=dtype).to(cfg.device)
            
            matched_gt_boxes.append(gt_boxes_per_img[matched_idxs[img_id]])
        
        reg_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, reg_targets
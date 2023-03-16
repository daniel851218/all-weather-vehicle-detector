import torch
import torch.nn as nn

from config.train_test_cfg import cfg
from common_module import boxes_utils
from common_module.conv2d import Conv2D_ReLU

class AnchorsGenerator(nn.Module):
    def __init__(self):
        super(AnchorsGenerator, self).__init__()
        self.anchor_sizes = torch.as_tensor(cfg.anchor_size).to(cfg.device)
        self.aspect_ratios = torch.as_tensor(cfg.aspect_ratio).to(cfg.device)

    def forward(self, imgs, feature_maps):
        img_size = imgs.shape[-2:]      # (img_h, img_w) = (416, 640)
        grid_sizes = [f[0].shape[-2:] for f in feature_maps]        # (feature_w, feature_h), [(104, 160), (52, 80), (26, 40), (13, 20)]
        strides = [(int(img_size[0] / g[0]), int(img_size[1] / g[1])) for g in grid_sizes]      # [(4, 4), (8, 8), (16, 16), (32, 32)]
        cell_anchors = self.set_cell_anchors()
        
        anchors_per_image = [self.grid_anchors(grid_size, stride, cell_anchor) for grid_size, stride, cell_anchor in zip(grid_sizes, strides, cell_anchors)]
        anchors_per_image = torch.cat(anchors_per_image, dim=0)
        anchors = [anchors_per_image] * imgs.shape[0]

        return anchors
     
    def set_cell_anchors(self):
        '''
        generate anchor template
        '''
        h_ratios = torch.sqrt(self.aspect_ratios).reshape(-1, 1)
        w_ratios = 1.0 / h_ratios.reshape(-1, 1)
        ratios = torch.cat([-w_ratios, -h_ratios, w_ratios, h_ratios], dim=1)

        anchor_template = []
        for size in self.anchor_sizes:
            anchor_template.append(((size * ratios) / 2).round())
        
        return anchor_template
    
    def grid_anchors(self, grid_size, stride, cell_anchor):
        '''
        - grid_sizes: (grid_h, grid_w)
        - strides: downsample_rate
        '''
        grid_h, grid_w = grid_size
        stride_h, stride_w = stride

        shift_x = torch.arange(0, grid_w, dtype=torch.int32, device=torch.device(cfg.device)) * stride_w
        shift_y = torch.arange(0, grid_h, dtype=torch.int32, device=torch.device(cfg.device)) * stride_h
        
        shift_ys, shift_xs = torch.meshgrid(shift_y, shift_x)
        shift_xs = shift_xs.reshape(-1)
        shift_ys = shift_ys.reshape(-1)

        shifts = torch.stack([shift_xs, shift_ys, shift_xs, shift_ys], dim=1)
        anchors = (shifts.reshape(-1, 1, 4) + cell_anchor.reshape(1, -1, 4)).reshape(-1, 4)

        return anchors

class RPN_Head(nn.Module):
    def __init__(self, in_channels, num_anchors):
        '''
        - background/foreground classification
        - bounding box regression
        '''
        super(RPN_Head, self).__init__()
        self.conv = Conv2D_ReLU(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1, padding=0)      # classify foreground or background
        self.reg_layer = nn.Conv2d(in_channels, 4*num_anchors, kernel_size=1, stride=1, padding=0)    # bounding box regression
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        cls_scores = []
        bbox_reg = []

        for f in features:
            f = self.conv(f)
            cls_scores.append(self.sigmoid(self.cls_layer(f)))
            bbox_reg.append(self.reg_layer(f))

        return cls_scores, bbox_reg

class RPN(nn.Module):
    def __init__(self, backbone_out_channels, num_anchors):
        super(RPN, self).__init__()
        self.rpn_head = RPN_Head(backbone_out_channels, num_anchors)
        self.anchors_generator = AnchorsGenerator()
        self.box_coder = boxes_utils.BoxCoder()
        self.proposal_matcher = boxes_utils.Matcher(cfg.fg_iou_thresh, cfg.bg_iou_thresh, True)
        self.fg_bg_sampler = boxes_utils.Balanced_Pos_Neg_Sampler(cfg.batch_size_per_img, cfg.positive_fraction)
        self.min_size = 1e-3

    def forward(self, imgs, features, targets):
        features = list(features.values())      # OrderedDict to list [p2, p3, p4, p5]
        fg_bg_scores, pred_bbox_deltas = self.rpn_head(features)
        # fg_bg_scores: list, shape = [(B, num_anchors, 104, 160), (B, num_anchors, 52, 80), (B, num_anchors, 26, 40), (B, num_anchors, 13, 20)]
        # pred_bbox_deltas: list, shape = [(B, 4*num_anchors, 104, 160), (B, 4*num_anchors, 52, 80), (B, 4*num_anchors, 26, 40), (B, 4*num_anchors, 13, 20)]
        
        anchors = self.anchors_generator(imgs, features)
        # anchors: list,
        # len(anchors): batch_size 
        # shape of each element is (66330, 4), 66330 = 3 * (13*20 + 26*40 + 52*80 + 104*160)

        num_anchors_per_level = [o[0].numel() for o in fg_bg_scores]
        # [49920, 12480, 3120, 780] = [3*104*1660, 3*52*80, 3*26*40, 3*13*20]

        fg_bg_scores, pred_bbox_deltas = boxes_utils.concat_box_prediction_layers(fg_bg_scores, pred_bbox_deltas)
        # fg_bg_scores: torch.Tensor, shape = (batch_size*66330, 1)
        # pred_bbox_deltas: torch.Tensor, shape = (batch_size*66330, 4)

        num_imgs = len(imgs)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.reshape(num_imgs, -1, 4)
        # proposals: torch.Tensor, shape = (batch_size, 66330, 4)

        # remove small bboxes, nms process, get post_nms_top_n target
        proposals, scores = self.filter_proposals(proposals, fg_bg_scores, [(cfg.img_h, cfg.img_w)]*num_imgs, num_anchors_per_level)
        # proposals: list
        # scores: list

        losses = {}
        is_labeled = True if "boxes" in targets[0].keys() and "labels" in targets[0].keys() else False
        if self.training and is_labeled:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)

            # encode parameters based on the bboxes and anchors
            reg_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_rpn_score, loss_rpn_box_reg = self.compute_loss(fg_bg_scores, pred_bbox_deltas, labels, reg_targets)
            losses = {"loss_rpn_score": loss_rpn_score, "loss_rpn_box_reg": loss_rpn_box_reg}
        
        return proposals, losses
        
    def get_top_n_idx(self, fg_bg_scores, num_anchors_per_level):
        '''
        Get the top pre_nms_top_n anchor index in predicted feature_maps based on scores
        '''
        result = []
        offset = 0
        for scores_single_level in fg_bg_scores.split(num_anchors_per_level, 1):
            num_anchors = scores_single_level.shape[1]
            pre_nms_top_n = min(cfg.pre_nms_top_n, num_anchors)

            # Returns the k largest elements of the given input tensor along a given dimension
            _, top_n_idx = scores_single_level.topk(pre_nms_top_n, dim=1)
            result.append(top_n_idx + offset)
            offset += num_anchors

        return torch.cat(result, dim=1)

    def filter_proposals(self, proposals, fg_bg_scores, image_sizes, num_anchors_per_level):
        '''
        Remove small bboxes, nms process, get post_nms_top_n target
        - proposals: predicted bbox coordinates [batch, sum_levels_num_anchors, 4]
        - fg_bg_scores: predicted foreground/background scores [batch*sum_levels_num_anchors, 1]
        - image_sizes: image size (h, w)
        - num_anchors_per_level: number of anchors per feature maps
        '''
        num_imgs = len(image_sizes)
        
        
        fg_bg_scores = fg_bg_scores.detach().reshape(num_imgs, -1)      # Don't backprop
        
        levels = [torch.full((n, ), idx, dtype=torch.int64) for idx, n in enumerate(num_anchors_per_level)]     # n = 49920, 12480, 3120, 780
        levels = torch.cat(levels, dim=0)
        levels = levels.reshape(1, -1).expand_as(fg_bg_scores)
        
        top_n_idx = self.get_top_n_idx(fg_bg_scores, num_anchors_per_level)
        # top_n_idx: torch.Tensor, shape=(batch_size, 2000+2000+2000+780)

        batch_idx = torch.arange(num_imgs).reshape(-1, 1).to(cfg.device)

        fg_bg_scores = fg_bg_scores[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, fg_bg_scores, levels, image_sizes):     # For each image
            # Adjust predicted bbox, make boxes outside of the image in image
            boxes = boxes_utils.clip_boxes_to_image(boxes, img_shape)

            # Remove boxes which contains at least one side smaller than min_size
            keep = boxes_utils.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # Non-Maximum Suppression, independently done per level
            keep = boxes_utils.batched_nms(boxes, scores, lvl, cfg.nms_thresh)
            keep = keep[: cfg.post_nms_top_n]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores

    def assign_targets_to_anchors(self, anchors, targets):
        '''
        Get the best match gt for anchors, divided into bg samples, fg samples and unused samples
        '''
        labels = []
        matched_gt_boxes = []
        for anchors_per_img, targets_per_img in zip(anchors, targets):
            gt_boxes = targets_per_img["boxes"]
            if gt_boxes.numel() == 0:
                matched_gt_boxes_per_image = torch.zeros(anchors_per_img.shape, dtype=torch.float32, device=torch.device(cfg.device))
                labels_per_image = torch.zeros((anchors_per_img.shape[0],), dtype=torch.float32, device=torch.device(cfg.device))
            else:
                match_quality_matrix = boxes_utils.box_iou(gt_boxes, anchors_per_img)
                matched_idx = self.proposal_matcher(match_quality_matrix)

                # get the targets corresponding GT for each proposal
                matched_gt_boxes_per_image = gt_boxes[matched_idx.clamp(min=0)]
                labels_per_image = (matched_idx >= 0).to(dtype=torch.float32)

                # background (negative examples)
                bg_indices = matched_idx == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idx == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        
        return labels, matched_gt_boxes
    
    def compute_loss(self, scores, pred_bbox_deltas, labels, reg_targets):
        '''
        compute RPN loss, include classification loss(foreground and background), bbox regression loss
        '''
        sampled_pos_idxs, sampled_neg_idxs = self.fg_bg_sampler(labels)
        sampled_pos_idxs = torch.nonzero(torch.cat(sampled_pos_idxs, dim=0)).squeeze(1)
        sampled_neg_idxs = torch.nonzero(torch.cat(sampled_neg_idxs, dim=0)).squeeze(1)

        sampled_idxs = torch.cat([sampled_pos_idxs, sampled_neg_idxs], dim=0)
        scores = scores.flatten()

        labels = torch.cat(labels, dim=0)
        reg_targets = torch.cat(reg_targets, dim=0)

        # bbox regression loss
        box_loss = boxes_utils.smooth_l1_loss(pred_bbox_deltas[sampled_pos_idxs], reg_targets[sampled_pos_idxs],
                                  beta=1.0/9.0, size_average=False) / (sampled_idxs.numel())
        
        # classification loss
        score_loss = nn.BCEWithLogitsLoss()(scores[sampled_idxs], labels[sampled_idxs])

        return score_loss, box_loss
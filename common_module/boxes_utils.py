import math
import torch
import torchvision

def clip_boxes_to_image(boxes, size):
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size
    
    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim).reshape(boxes.shape)
    return clipped_boxes

def remove_small_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    
    keep = (ws >= min_size) & (hs >= min_size)
    keep = keep.nonzero().squeeze(1)
    return keep

def batched_nms(boxes, scores, idxs, iou_threshold):
    '''
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS will not be applied between elements of different categories.
    '''
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # strategy: in order to perform NMS independently per level.
    # we add an offset to all the boxes. The offset is dependent
    # only on the level idx, and is large enough so that boxes
    # from different level do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)
    
    return keep

def box_area(boxes):
    '''
    Calculate box area
    '''
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    '''
    Calculate IoU betweent two boxes
    boxes format: (x1, y1, x2, y2)
    '''
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)             # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]       # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)

    return iou

def permute_and_flatten(layer, N, C, H, W):
    '''
    Adjust tensor order，and reshape
    - layer: classification or bboxes parameters
    - N: batch size
    - C: number of classes or bounding box coordinates
    - H: height
    - W: width
    '''
    layer = layer.reshape(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    
    return layer

def concat_box_prediction_layers(box_score_multi_level, box_reg_multi_level):
    '''
    Adjust box classification and bounding box regression parameters order and reshape
    '''
    box_score_flatten = []
    box_reg_flatten = []

    for box_score_per_level, box_reg_per_level in zip(box_score_multi_level, box_reg_multi_level):
        N, AxC, H, W = box_score_per_level.shape
        Ax4 = box_reg_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A

        box_score_per_level = permute_and_flatten(box_score_per_level, N, C, H, W)
        box_score_flatten.append(box_score_per_level)

        box_reg_per_level = permute_and_flatten(box_reg_per_level, N, 4, H, W)
        box_reg_flatten.append(box_reg_per_level)

    box_score = torch.cat(box_score_flatten, dim=1).flatten(0, -2)
    box_reg = torch.cat(box_reg_flatten, dim=1).reshape(-1, 4)

    return box_score, box_reg

def smooth_l1_loss(pred, target, beta=1.0/9.0, size_average=True):
    n = torch.abs(pred - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    
    return loss.sum()

# ----------------------------------------------------------------------------------------------------

class BoxCoder():
    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0), bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip      # log(img_size_max / anchor_size_min)

    def decode_single(self, box_reg, anchors):
        '''
        From a set of original boxes and encoded relative box offsets, get the decoded boxes.
        - box_reg: encoded boxes (bounding box regression parameters)
        '''
        anchors = anchors.to(box_reg.device)

        # x_min, y_min, x_max, y_max
        widths = anchors[:, 2:3] - anchors[:, 0:1]      # anchor width
        heights = anchors[:, 3:4] - anchors[:, 1:2]     # anchor height
        center_x = anchors[:, 0:1] + 0.5 * widths     # anchor center x coordinate
        center_y = anchors[:, 1:2] + 0.5 * heights    # anchor center y coordinate

        wx, wy, ww, wh = self.weights
        dx = box_reg[:, 0::4] / wx                  # predicated anchors center x regression parameters
        dy = box_reg[:, 1::4] / wy                  # predicated anchors center y regression parameters
        dw = box_reg[:, 2::4] / ww                  # predicated anchors width regression parameters
        dh = box_reg[:, 3::4] / wh                  # predicated anchors height regression parameters

        # limit max value, prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_center_x = dx * widths + center_x
        pred_center_y = dy * heights + center_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_x_min = pred_center_x - 0.5 * pred_w
        pred_y_min = pred_center_y - 0.5 * pred_h
        pred_x_max = pred_center_x + 0.5 * pred_w
        pred_y_max = pred_center_y + 0.5 * pred_h
        
        pred_boxes = torch.stack([pred_x_min, pred_y_min, pred_x_max, pred_y_max], dim=2)
        return pred_boxes

    def decode(self, box_reg, anchors):
        '''
        Decode regression parameters
        - box_reg: bounding box regression parameters
        - anchors: anchors which are generated by AnchorGenerator
        '''
        anchors_per_image = [a.shape[0] for a in anchors]
        anchors_sum = sum(anchors_per_image)
        concate_anchors = torch.cat(anchors, dim=0)         # shape = (batch_size*66330, 4)

        pred_boxes = self.decode_single(box_reg, concate_anchors)
        return pred_boxes
    
    def encode_boxes(self, reference_boxes, anchors, weights):
        '''
        Encode a set of proposals with respect to some reference boxes
        '''
        wx = weights[0]
        wy = weights[1]
        ww = weights[2]
        wh = weights[3]

        # Returns a new tensor with a dimension of size one inserted at the specified position.
        anchors_x1 = anchors[:, 0].unsqueeze(1)
        anchors_y1 = anchors[:, 1].unsqueeze(1)
        anchors_x2 = anchors[:, 2].unsqueeze(1)
        anchors_y2 = anchors[:, 3].unsqueeze(1)

        reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
        reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
        reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
        reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

        # parse widths and heights
        ex_widths = anchors_x2 - anchors_x1
        ex_heights = anchors_y2 - anchors_y1

         # center point
        ex_center_x = anchors_x1 + 0.5 * ex_widths
        ex_center_y = anchors_y1 + 0.5 * ex_heights

        gt_widths = reference_boxes_x2 - reference_boxes_x1
        gt_heights = reference_boxes_y2 - reference_boxes_y1
        gt_center_x = reference_boxes_x1 + 0.5 * gt_widths
        gt_center_y = reference_boxes_y1 + 0.5 * gt_heights

        targets_dx = wx * (gt_center_x - ex_center_x) / (ex_widths + 1e-10)
        targets_dy = wy * (gt_center_y - ex_center_y) / (ex_heights + 1e-10)
        targets_dw = ww * torch.log(gt_widths / (ex_widths + 1e-10) + 1e-10)
        targets_dh = wh * torch.log(gt_heights / (ex_heights + 1e-10) + 1e-10)

        targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def encode_single(self, reference_boxes, anchors):
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = self.encode_boxes(reference_boxes, anchors, weights)

        return targets

    def encode(self, reference_boxes, anchors):
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        anchors = torch.cat(anchors, dim=0)

        # targets_dx, targets_dy, targets_dw, targets_dh
        targets = self.encode_single(reference_boxes, anchors)

        return targets.split(boxes_per_image, dim=0)

# ----------------------------------------------------------------------------------------------------

class Matcher(object):
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def set_low_quality_matches(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.

         # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None])
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]

    def __call__(self, match_quality_matrix):
        '''
        Calculate maximum iou between anchors and gt boxes, save index，
        iou < low_threshold: -1     (seen as background)
        iou > high_threshold: 1     (seen as foreground)
        low_threshold<=iou<high_threshold: -2   (ignore)
        ----------------------------------------------------------------
        match_quality_matrix is M (gt) x N (predicted)
        '''
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images during training")
            else:
                raise ValueError("No proposal boxes available for one of the images during training")
            
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matched_idx = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matched_idx.clone()
        else:
            all_matches = None

        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)

        matched_idx[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1
        matched_idx[between_thresholds] = self.BETWEEN_THRESHOLDS    # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches(matched_idx, all_matches, match_quality_matrix)

        return matched_idx
    
# ----------------------------------------------------------------------------------------------------

class Balanced_Pos_Neg_Sampler(object):
    def __init__(self, batch_size_per_img, positive_fraction):
        self.batch_size_per_img = batch_size_per_img
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        '''
        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        '''
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_img in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_img >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_img == 0).squeeze(1)

            num_pos = int(self.batch_size_per_img * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)

            num_neg = self.batch_size_per_img - num_pos
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm_pos = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm_neg = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            pos_idx_per_img = positive[perm_pos]
            neg_idx_per_img = negative[perm_neg]

            # create binary mask from indices
            pos_idx_per_img_mask = torch.zeros_like(matched_idxs_per_img, dtype=torch.uint8)
            neg_idx_per_img_mask = torch.zeros_like(matched_idxs_per_img, dtype=torch.uint8)

            pos_idx_per_img_mask[pos_idx_per_img] = 1
            neg_idx_per_img_mask[neg_idx_per_img] = 1

            pos_idx.append(pos_idx_per_img_mask)
            neg_idx.append(neg_idx_per_img_mask)
        
        return pos_idx, neg_idx
import torch
from config.train_test_cfg import cfg

def filter_outlier(boxes, labels, scores, restored_ratio, restored_delta):
    restore_ratio = 1. / restored_ratio
    delta_x, delta_y = restored_delta

    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    for i in range(len(cfg.obj_filter_thresh)):
        score = scores[labels == (i+1)]

        x1 = boxes[labels == (i+1)][:, 0]
        y1 = boxes[labels == (i+1)][:, 1]
        x2 = boxes[labels == (i+1)][:, 2]
        y2 = boxes[labels == (i+1)][:, 3]

        x1 = (x1 - delta_x) * restore_ratio
        y1 = (y1 - delta_y) * restore_ratio
        x2 = (x2 - delta_x) * restore_ratio
        y2 = (y2 - delta_y) * restore_ratio

        width = x2 - x1
        height = y2 - y1
        ratio = width / height
        area = width * height

        width_thresh = cfg.obj_filter_thresh[i]["width"]
        height_thresh = cfg.obj_filter_thresh[i]["height"]
        ratio_thresh_Q1 = cfg.obj_filter_thresh[i]["ratio"][0]
        ratio_thresh_Q3 = cfg.obj_filter_thresh[i]["ratio"][0]
        area_thresh = cfg.obj_filter_thresh[i]["area"]

        width_mask = width < width_thresh
        height_mask = height < height_thresh
        
        ratio_mask_Q1 = ratio < ratio_thresh_Q1
        ratio_mask_Q3 = ratio > ratio_thresh_Q3
        ratio_mask = torch.logical_or(ratio_mask_Q1, ratio_mask_Q3)
        
        area_mask = area < area_thresh

        score_mask = score > cfg.test_strong_conf_thresh

        mask = width_mask.to(torch.int) + height_mask.to(torch.int) + 1.5 * ratio_mask.to(torch.int) + 1.5 *  area_mask.to(torch.int) - 10 * score_mask
        mask = torch.logical_not(mask >= 3)

        filtered_boxes += boxes[labels == (i+1)][mask].to(torch.int).tolist()
        filtered_labels += labels[labels == (i+1)][mask].tolist()
        filtered_scores += scores[labels == (i+1)][mask].tolist()
        
    return torch.tensor(filtered_boxes, dtype=torch.float), torch.tensor(filtered_labels, dtype=torch.int), torch.tensor(filtered_scores)
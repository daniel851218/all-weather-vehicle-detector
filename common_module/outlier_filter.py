import torch

obj_filter_thresh = [
    # Car
    {
        "width": 24.0,
        "height": 20.0,
        "ratio": 1.0,
        "area": 483.0
    },

    # Truck
    {
        "width": 39.0,
        "height": 29.0,
        "ratio": 0.9666666666666668,
        "area": 1110.0
    },

    # Pedestrian
    {
        "width": 12.0,
        "height": 31.0,
        "ratio": 0.3333333333333333,
        "area": 364.0
    },

    # Bus
    {
        "width": 40.0,
        "height": 29.0,
        "ratio": 0.984126984126984,
        "area": 1113.0
    },

    # Bicycle
    {
        "width": 20.0,
        "height": 32.0,
        "ratio": 0.4545454545454545,
        "area": 648.0
    },

    # Motorcycle
    {
        "width": 20.0,
        "height": 27.0,
        "ratio": 0.5060240963855421,
        "area": 528.0
    },
]

def filter_outlier(boxes, labels, scores, restored_ratio, restored_delta):
    restore_ratio = 1. / restored_ratio
    delta_x, delta_y = restored_delta

    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    for i in range(len(obj_filter_thresh)):
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

        width_thresh = obj_filter_thresh[i]["width"]
        height_thresh = obj_filter_thresh[i]["height"]
        ratio_thresh = obj_filter_thresh[i]["ratio"]
        area_thresh = obj_filter_thresh[i]["area"]

        width_mask = width < width_thresh
        height_mask = height < height_thresh
        ratio_mask = ratio < ratio_thresh
        area_mask = area < area_thresh

        mask = width_mask.to(torch.int) + height_mask.to(torch.int) + ratio_mask.to(torch.int) + area_mask.to(torch.int)
        mask = torch.logical_not(mask > 2)

        filtered_boxes += boxes[labels == (i+1)][mask].tolist()
        filtered_labels += labels[labels == (i+1)][mask].tolist()
        filtered_scores += scores[labels == (i+1)][mask].tolist()
        
    return filtered_boxes, filtered_labels, filtered_scores
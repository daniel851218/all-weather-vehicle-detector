import os

class Config():
    device = "cpu"

    # image size
    img_w = 640
    img_h = 416

    # anchor parameters
    anchor_size = [32.0, 64.0, 128.0, 256.0]
    aspect_ratio = [0.5, 1.0, 2.0]

    # RPN parameters
    pre_nms_top_n = 2000
    post_nms_top_n = 2000
    nms_thresh = 0.7
    fg_iou_thresh = 0.7
    bg_iou_thresh = 0.3
    batch_size_per_img = 256
    positive_fraction = 0.5

cfg = Config()
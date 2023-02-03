import os

class Config():
    dataset = "SHIFT"
    ckpt_dir = os.path.join("runs", dataset, "adv_faster_rcnn")

    # use pre-trained weight or not
    is_pre_train = False
    pre_train_model_epoch = None
    pre_train_model_path = None
    
    device = "cuda"
    batch_size = 8
    start_epoch = 0
    epochs = 100
    max_patience = 10

    # optimizer parameters
    lr = 5e-3
    momentum = 0.9
    weight_decay = 1e-3

    # lr_scheduler
    lr_gamma = 0.5
    lr_dec_step_size = 1
    lr_min = 5e-7

    # target objects
    num_classes = 6 + 1       # foreground + background
    obj_classes = {           # 0 for background
        1: "Car",
        2: "Truck",
        3: "Pedestrian",
        4: "Bus",
        5: "Bicycle",
        6: "Motorcycle"}

    # image size
    img_w = 640
    img_h = 416

    # adversarial training parameters
    lambda_adv_daytime = 0.1
    lambda_adv_weather = 0.15

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

    # RoI parameters
    box_fg_iou_thresh = 0.5
    box_bg_iou_thresh = 0.5
    box_batch_size_per_img = 512
    box_positive_fraction = 0.25
    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 100

    roi_align_out_size = [14, 14]
    roi_sample_ratio = 2

cfg = Config()
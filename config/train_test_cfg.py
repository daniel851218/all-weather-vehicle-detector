import os

class Config():
    # =====================================================================================================
    # |                                        Training Parameters                                        |
    # =====================================================================================================
    dataset = "BDD"
    ckpt_dir = os.path.join("runs", dataset, "")
    
    # Use pre-trained weight or not
    is_pre_train = False
    pre_train_model_epoch = None
    pre_train_model_path = os.path.join("best_weights", "", "", "", "weights")
    
    # Basic setting
    device = "cuda"
    batch_size = 3
    grad_accumulate_step = 3
    start_epoch = 0
    epochs = 100
    max_patience = 30

    # optimizer parameters
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-4

    # lr_scheduler
    lr_gamma = 0.5
    lr_dec_step_size = 2
    lr_min = 1e-15

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
    lambda_adv_weather = 0.2

    # semi-supervised training parameters
    lambda_unsup = 1.75
    step_ema = 20
    alpha_ema = 0.995

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

    roi_align_out_size = [7, 7]
    roi_sample_ratio = 2

    # threshold of pseudo labels
    obj_filter_thresh = [
    # Car
    {
        "width": 24.0,
        "height": 20.0,
        "ratio": [1.0, 1.638],
        "area": 483.0
    },

    # Truck
    {
        "width": 39.0,
        "height": 29.0,
        "ratio": [0.9666666666666668, 1.85],
        "area": 1110.0
    },

    # Pedestrian
    {
        "width": 12.0,
        "height": 31.0,
        "ratio": [0.3333, 0.4783],
        "area": 364.0
    },

    # Bus
    {
        "width": 40.0,
        "height": 29.0,
        "ratio": [0.9841, 1.9231],
        "area": 1113.0
    },

    # Bicycle
    {
        "width": 20.0,
        "height": 32.0,
        "ratio": [0.4545, 0.9574],
        "area": 648.0
    },

    # Motorcycle
    {
        "width": 20.0,
        "height": 27.0,
        "ratio": [0.5060, 1.1471],
        "area": 528.0
    },
]

    # =====================================================================================================
    # |                                         Testing Parameters                                        |
    # =====================================================================================================
    test_img_type = "all"
    test_imgs_dir = "test_images"
    result_imgs_dir = os.path.join(test_imgs_dir, "result")
    filter_enable = False

    if filter_enable:
        test_iou_thresh = 0.25
        test_weak_conf_thresh = 0.75
    else:
        test_iou_thresh = 0.5
        test_weak_conf_thresh = 0.5
    
    test_strong_conf_thresh = 0.9

    obj_color = {
        1: 0x0000FF,    # Red
        2: 0x00FF00,    # Lime
        3: 0x008CFF,    # Dark Orange
        4: 0xFF00FF,    # Fuchsia
        5: 0x00FFFF,    # Yellow
        6: 0xFFFF00     # Aqua
    }

cfg = Config()
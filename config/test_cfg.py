import os

class Config():
    # Testing
    dataset = "SHIFT"
    img_type = "all"
    batch_size = 2
    best_epoch = None
    load_model_dir = os.path.join("best_weights", "SHIFT", "", "", "weights")
    
    # Inference
    test_imgs_dir = "test_images"
    result_imgs_dir = os.path.join(test_imgs_dir, "result")
    img_w = 640
    img_h = 416
    conf_thresh = 0.5
    iou_thresh = 0.5

    device = "cuda"
    num_classes = 6 + 1       # foreground + background
    obj_class = {           # 0 for background
        1: "Car",
        2: "Truck",
        3: "Pedestrian",
        4: "Bus",
        5: "Bicycle",
        6: "Motorcycle"}

    obj_color = {
        1: 0x0000FF,    # Red
        2: 0x00FF00,    # Lime
        3: 0xFF0000,    # Blue
        4: 0xFF00FF,    # Fuchsia
        5: 0x00FFFF,    # Yellow
        6: 0xFFFF00     # Aqua
    }
    
cfg = Config()
import os
from config.train_cfg import cfg

class Load_Data_Config():
    shift_txt_file_train = os.path.join("dataset", "SHIFT", "train_shift.txt")

    if cfg.test_img_type == "all":
        shift_txt_file_val = os.path.join("dataset", "SHIFT", "val_shift.txt")
    elif cfg.test_img_type == "daytime_normal":
        shift_txt_file_val = os.path.join("dataset", "SHIFT", "val_shift_daytime_normal.txt")
    elif cfg.test_img_type == "daytime_rainy":
        shift_txt_file_val = os.path.join("dataset", "SHIFT", "val_shift_daytime_rainy.txt")
    elif cfg.test_img_type == "night_normal":
        shift_txt_file_val = os.path.join("dataset", "SHIFT", "val_shift_night_normal.txt")
    elif cfg.test_img_type == "night_rainy":
        shift_txt_file_val = os.path.join("dataset", "SHIFT", "val_shift_night_rainy.txt")
    else:
        raise ValueError("image type is not correct !")

    bdd_txt_file_train = os.path.join("dataset", "BDD100K", "train_bdd.txt")

    if cfg.test_img_type == "all":
        bdd_txt_file_val = os.path.join("dataset", "BDD100K", "val_bdd.txt")
    elif cfg.test_img_type == "daytime_normal":
        bdd_txt_file_val = os.path.join("dataset", "BDD100K", "val_bdd_daytime_normal.txt")
    elif cfg.test_img_type == "daytime_rainy":
        bdd_txt_file_val = os.path.join("dataset", "BDD100K", "val_bdd_daytime_rainy.txt")
    elif cfg.test_img_type == "night_normal":
        bdd_txt_file_val = os.path.join("dataset", "BDD100K", "val_bdd_night_normal.txt")
    elif cfg.test_img_type == "night_rainy":
        bdd_txt_file_val = os.path.join("dataset", "BDD100K", "val_bdd_night_rainy.txt")
    else:
        raise ValueError("image type is not correct !")

    driving_video_txt_file = os.path.join("dataset", "Driving_Video", "train_driving_video.txt")

    img_w = 640
    img_h = 416

data_cfg = Load_Data_Config()
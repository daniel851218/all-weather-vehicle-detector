import os
from config.train_test_cfg import cfg

SHIFT_PATH = os.path.join("dataset", "SHIFT")
BDD_PATH = os.path.join("dataset", "BDD100K")
DRIVING_VIDEO_PATH = os.path.join("dataset", "Driving_Video")

class Load_Data_Config():
    # ============================================= SHIFT Dataset =============================================
    shift_txt_file_train = os.path.join(SHIFT_PATH, "train_shift.txt")

    if cfg.test_img_type == "all":
        shift_txt_file_val = os.path.join(SHIFT_PATH, "val_shift.txt")
    elif cfg.test_img_type == "daytime_normal":
        shift_txt_file_val = os.path.join(SHIFT_PATH, "val_shift_daytime_normal.txt")
    elif cfg.test_img_type == "daytime_rainy":
        shift_txt_file_val = os.path.join(SHIFT_PATH, "val_shift_daytime_rainy.txt")
    elif cfg.test_img_type == "night_normal":
        shift_txt_file_val = os.path.join(SHIFT_PATH, "val_shift_night_normal.txt")
    elif cfg.test_img_type == "night_rainy":
        shift_txt_file_val = os.path.join(SHIFT_PATH, "val_shift_night_rainy.txt")
    else:
        raise ValueError("image type is not correct !")

    # ============================================ BDD100K Dataset ============================================
    bdd_txt_file_train = os.path.join(BDD_PATH, "train_bdd.txt")

    if cfg.test_img_type == "all":
        bdd_txt_file_val = os.path.join(BDD_PATH, "val_bdd.txt")
    elif cfg.test_img_type == "daytime_normal":
        bdd_txt_file_val = os.path.join(BDD_PATH, "val_bdd_daytime_normal.txt")
    elif cfg.test_img_type == "daytime_rainy":
        bdd_txt_file_val = os.path.join(BDD_PATH, "val_bdd_daytime_rainy.txt")
    elif cfg.test_img_type == "night_normal":
        bdd_txt_file_val = os.path.join(BDD_PATH, "val_bdd_night_normal.txt")
    elif cfg.test_img_type == "night_rainy":
        bdd_txt_file_val = os.path.join(BDD_PATH, "val_bdd_night_rainy.txt")
    else:
        raise ValueError("image type is not correct !")

    # ========================================= Driving Video Dataset =========================================
    driving_video_txt_file = os.path.join(DRIVING_VIDEO_PATH, "daytime_normal_China_1.txt")

data_cfg = Load_Data_Config()
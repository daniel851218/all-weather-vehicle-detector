import os

class Load_Data_Config():
    shift_txt_file_train = os.path.join("dataset", "SHIFT", "train_shift.txt")
    shift_txt_file_val = os.path.join("dataset", "SHIFT", "val_shift.txt")

    bdd_txt_file_train = os.path.join("dataset", "BDD100K", "train_bdd.txt")
    bdd_txt_file_val = os.path.join("dataset", "BDD100K", "val_bdd.txt")

    driving_video_txt_file = os.path.join("dataset", "Driving_Video", "train_driving_video.txt")

    img_w = 640
    img_h = 416

data_cfg = Load_Data_Config()
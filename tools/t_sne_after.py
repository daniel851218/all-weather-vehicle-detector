# python -m tools.t_sne_after

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from sklearn import manifold
from PIL import Image
from torchvision import transforms

from detector.faster_rcnn import Faster_RCNN
from config.train_test_cfg import cfg

IS_BDD = False
if IS_BDD:
    DAYTIME_NORMAL_DIR = "/home/rvl/Desktop/YiChao/BDD100K/val/daytime_normal"
    DAYTIME_RAINY_DIR = "/home/rvl/Desktop/YiChao/BDD100K/val/daytime_rainy"
    NIGHT_NORMAL_DIR = "/home/rvl/Desktop/YiChao/BDD100K/val/night_normal"
    NIGHT_RAINY_DIR = "/home/rvl/Desktop/YiChao/BDD100K/val/night_rainy"
else:
    DAYTIME_NORMAL_DIR = "/home/rvl/Desktop/YiChao/SHIFT_DATASET/test/classified/daytime_normal"
    DAYTIME_RAINY_DIR = "/home/rvl/Desktop/YiChao/SHIFT_DATASET/test/classified/daytime_rainy"
    NIGHT_NORMAL_DIR = "/home/rvl/Desktop/YiChao/SHIFT_DATASET/test/classified/night_normal"
    NIGHT_RAINY_DIR = "/home/rvl/Desktop/YiChao/SHIFT_DATASET/test/classified/night_rainy"

TYPE_DAYTIME_NORMAL = 0
TYPE_DAYTIME_RAINY = 1
TYPE_NIGHT_NORMAL = 2
TYPE_NIGHT_RAINY = 3

NUM_SAMPLE = 281
COLORS = ["blue", "orange", "green", "red"]
PLOT_LABEL = ["daytime normal", "daytime rainy", "night normal", "night rainy"]

def resize_img(img):
    '''
    img: PIL image
    '''
    h, w = img.height, img.width
    ratio_w = cfg.img_w / w
    ratio_h = cfg.img_h / h
    ratio = min(ratio_w, ratio_h)

    new_h = int(h * ratio)
    new_w = int(w * ratio)
    new_img = img.resize((new_w, new_h))

    return new_img, ratio

def pad_image(img):
    '''
    img: Tensor (channel, h, w)
    '''
    new_img = torch.zeros((3, cfg.img_h, cfg.img_w), dtype=img.dtype) + 0.5

    ori_h, ori_w = img.shape[1], img.shape[2]

    delta_h = int(new_img.shape[1] - ori_h)
    delta_w = int(new_img.shape[2] - ori_w)

    x1 = int(delta_w / 2)
    y1 = int(delta_h / 2)

    new_img[:, y1:y1+ori_h, x1:x1+ori_w] = img
    return new_img, (x1, y1)

def pre_process(img_file):
    '''
    img: PIL image
    '''
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_PIL = Image.open(img_file)

    img, ratio = resize_img(img_PIL)
    img_tensor = T(img)
    img_tensor, delta = pad_image(img_tensor)

    return img_PIL, img_tensor.to(cfg.device), ratio, delta

def restore_bbox_size(box, ratio, delta):
    x1, y1, x2, y2 = box
    restore_ratio = 1. / ratio
    delta_x, delta_y = delta

    x1 = int((x1 - delta_x) * restore_ratio)
    y1 = int((y1 - delta_y) * restore_ratio)
    x2 = int((x2 - delta_x) * restore_ratio)
    y2 = int((y2 - delta_y) * restore_ratio)

    return x1, y1, x2, y2

if __name__ == "__main__":
    if IS_BDD:
        print("Visualize BDD100K Dataset\n")
        daytime_normal_list = glob(os.path.join(DAYTIME_NORMAL_DIR, "*.jpg"))
        daytime_rainy_list = glob(os.path.join(DAYTIME_RAINY_DIR, "*.jpg"))
        night_normal_list = glob(os.path.join(NIGHT_NORMAL_DIR, "*.jpg"))
        night_rainy_list = glob(os.path.join(NIGHT_RAINY_DIR, "*.jpg"))
    else:
        print("Visualize SHIFT Dataset\n")
        daytime_normal_list = glob(os.path.join(DAYTIME_NORMAL_DIR, "*", "*.jpg"), recursive=True)
        daytime_rainy_list = glob(os.path.join(DAYTIME_RAINY_DIR, "*", "*.jpg"), recursive=True)
        night_normal_list = glob(os.path.join(NIGHT_NORMAL_DIR, "*", "*.jpg"), recursive=True)
        night_rainy_list = glob(os.path.join(NIGHT_RAINY_DIR, "*", "*.jpg"), recursive=True)

    random.shuffle(daytime_normal_list)
    daytime_normal_list = daytime_normal_list[:NUM_SAMPLE]
    
    random.shuffle(daytime_rainy_list)
    daytime_rainy_list = daytime_rainy_list[:NUM_SAMPLE]
    
    random.shuffle(night_normal_list)
    night_normal_list = night_normal_list[:NUM_SAMPLE]

    random.shuffle(night_rainy_list)
    night_rainy_list = night_rainy_list[:NUM_SAMPLE]

    obj_detector = Faster_RCNN().to(cfg.device)
    obj_detector.load_model(cfg.pre_train_model_path, cfg.pre_train_model_epoch)
    obj_detector.eval()

    features = []
    lbls = []
    with torch.no_grad():
        for img_file in tqdm(daytime_normal_list):
            img_PIL, img_tensor, ratio, delta = pre_process(img_file)
            feature_maps = obj_detector.backbone(img_tensor.unsqueeze(0))
            features.append(feature_maps["p5"].reshape(-1).to("cpu").numpy())
            lbls.append(TYPE_DAYTIME_NORMAL)

        for img_file in tqdm(daytime_rainy_list):
            img_PIL, img_tensor, ratio, delta = pre_process(img_file)
            feature_maps = obj_detector.backbone(img_tensor.unsqueeze(0))
            features.append(feature_maps["p5"].reshape(-1).to("cpu").numpy())
            lbls.append(TYPE_DAYTIME_RAINY)

        for img_file in tqdm(night_normal_list):
            img_PIL, img_tensor, ratio, delta = pre_process(img_file)
            feature_maps = obj_detector.backbone(img_tensor.unsqueeze(0))
            features.append(feature_maps["p5"].reshape(-1).to("cpu").numpy())
            lbls.append(TYPE_NIGHT_NORMAL)

        for img_file in tqdm(night_rainy_list):
            img_PIL, img_tensor, ratio, delta = pre_process(img_file)
            feature_maps = obj_detector.backbone(img_tensor.unsqueeze(0))
            features.append(feature_maps["p5"].reshape(-1).to("cpu").numpy())
            lbls.append(TYPE_NIGHT_RAINY)
    
    features = np.array(features)
    lbls = np.array(lbls)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    t_SNE = manifold.TSNE(n_components=2, init="pca", random_state=111, n_iter=3000)
    result = t_SNE.fit_transform(features)
    result_max, result_min = result.max(0), result.min(0)
    result = (result - result_min) / (result_max - result_min)

    ax.scatter(result[:, 0][lbls == TYPE_DAYTIME_NORMAL], result[:, 1][lbls == TYPE_DAYTIME_NORMAL], label=PLOT_LABEL[TYPE_DAYTIME_NORMAL], color=COLORS[TYPE_DAYTIME_NORMAL], s=3)
    ax.scatter(result[:, 0][lbls == TYPE_DAYTIME_RAINY], result[:, 1][lbls == TYPE_DAYTIME_RAINY], label=PLOT_LABEL[TYPE_DAYTIME_RAINY], color=COLORS[TYPE_DAYTIME_RAINY], s=3)
    ax.scatter(result[:, 0][lbls == TYPE_NIGHT_NORMAL], result[:, 1][lbls == TYPE_NIGHT_NORMAL], label=PLOT_LABEL[TYPE_NIGHT_NORMAL], color=COLORS[TYPE_NIGHT_NORMAL], s=3)
    ax.scatter(result[:, 0][lbls == TYPE_NIGHT_RAINY], result[:, 1][lbls == TYPE_NIGHT_RAINY], label=PLOT_LABEL[TYPE_NIGHT_RAINY], color=COLORS[TYPE_NIGHT_RAINY], s=3)

    plt.legend(loc='upper left')
    plt.show()
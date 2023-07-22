# python -m tools.t_sne_before

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from sklearn import manifold

IS_BDD = True
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

    imgs = []
    lbls = []
    for file_name in tqdm(daytime_normal_list):
        img = cv2.imread(file_name)
        h, w, channel = img.shape
        img = cv2.resize(img, (h//2, w//2))
        img = img / 255
        imgs.append(img)
        lbls.append(TYPE_DAYTIME_NORMAL)

    for file_name in tqdm(daytime_rainy_list):
        img = cv2.imread(file_name)
        h, w, channel = img.shape
        img = cv2.resize(img, (h//2, w//2))
        img = img / 255
        imgs.append(img)
        lbls.append(TYPE_DAYTIME_RAINY)

    for file_name in tqdm(night_normal_list):
        img = cv2.imread(file_name)
        h, w, channel = img.shape
        img = cv2.resize(img, (h//2, w//2))
        img = img / 255
        imgs.append(img)
        lbls.append(TYPE_NIGHT_NORMAL)

    for file_name in tqdm(night_rainy_list):
        img = cv2.imread(file_name)
        h, w, channel = img.shape
        img = cv2.resize(img, (h//2, w//2))
        img = img / 255
        imgs.append(img)
        lbls.append(TYPE_NIGHT_RAINY)

    lbls = np.array(lbls)
    imgs = np.array(imgs)
    imgs = imgs.reshape(lbls.shape[0], -1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    t_SNE = manifold.TSNE(n_components=2, init="pca", random_state=111, n_iter=3000)
    result = t_SNE.fit_transform(imgs)
    result_max, result_min = result.max(0), result.min(0)
    result = (result - result_min) / (result_max - result_min)

    ax.scatter(result[:, 0][lbls == TYPE_DAYTIME_NORMAL], result[:, 1][lbls == TYPE_DAYTIME_NORMAL], label=PLOT_LABEL[TYPE_DAYTIME_NORMAL], color=COLORS[TYPE_DAYTIME_NORMAL], s=3)
    ax.scatter(result[:, 0][lbls == TYPE_DAYTIME_RAINY], result[:, 1][lbls == TYPE_DAYTIME_RAINY], label=PLOT_LABEL[TYPE_DAYTIME_RAINY], color=COLORS[TYPE_DAYTIME_RAINY], s=3)
    ax.scatter(result[:, 0][lbls == TYPE_NIGHT_NORMAL], result[:, 1][lbls == TYPE_NIGHT_NORMAL], label=PLOT_LABEL[TYPE_NIGHT_NORMAL], color=COLORS[TYPE_NIGHT_NORMAL], s=3)
    ax.scatter(result[:, 0][lbls == TYPE_NIGHT_RAINY], result[:, 1][lbls == TYPE_NIGHT_RAINY], label=PLOT_LABEL[TYPE_NIGHT_RAINY], color=COLORS[TYPE_NIGHT_RAINY], s=3)

    plt.legend(loc='upper left')
    plt.show()
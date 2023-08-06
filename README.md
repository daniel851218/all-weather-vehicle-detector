# All-Weather Vehicle Detection Based on Adversarial and Semi-Supervised Learning
<!---------------------------------------------------------------------------------------------------->

## Introduction

In order to improve the performance of object detector in the **adverse conditions**. In this work, we classify all images in the dataset into 4 domains according to the **lighting** and **weather** types.

Based on **Faster R-CNN**, we add two domain classifiers to perform **Adversarial Learning**, one is **daytime classifier**, and the other is **weather classifier**.

Considering that the number of images belonging to certain weather conditions in the existing datasets is **not sufficient**, collecting additional images and labeling them can be a **time-consuming** and **labor-intensive** task. To address this issue, We further add **Teacher-Student Framework** to perform **Semi-Supervised Learning**, so that the object detect can be trained by the images without labels.

---

<!---------------------------------------------------------------------------------------------------->

## Experiment Environment

### System Hardware
- CPU：Intel i7-13700K
- GPU：Nvidia GeForce RTX 4090

### System Software
- Ubuntu 20.04
- CUDA：11.1
- CuDNN：8.0.5
- Python：3.8.11

### Python Packages
- torch：1.8.2+cu111
- torchvision：0.9.2+cu111
- tensorboard：2.12.2
- numpy：1.23.2
- Pillow：9.5.0
- matplotlib：3.7.1
- imgaug：0.4.0

---

<!---------------------------------------------------------------------------------------------------->

## Dataset

### SHIFT Dataset
- [SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation](https://www.vis.xyz/shift/)
- Consist of **synthetic** images.
- Be used to do the experiment of Adversarial Learning.
- According to the following table, classify the images into **Daytime Normal**, **Daytime Rainy**, **Night Normal** and **Night Rainy**.

|      DAYTIME      |      NIGHT     |     NORMAL     |    RAINY   |
|:-----------------:|:--------------:|:--------------:|:----------:|
| morning/afternoon | sunrise/sunset |      clear     | small rain |
|        noon       |      night     |  slight cloudy |  mid rain  |
|     dawn/dusk     |   dark night   | partial cloudy | heavy rain |
|                   |                |    overcast    |            |

### BDD100K Dataset
- [BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning](https://www.vis.xyz/bdd100k/)
- Consist of **real** images.
- Be used to do the experiment of Adversarial Learning and Semi-Supervised Learning.
- According to the following table, classify the images into **Daytime Normal**, **Daytime Rainy**, **Night Normal** and **Night Rainy**.

|  DAYTIME  | NIGHT |     NORMAL    | RAINY |
|:---------:|:-----:|:-------------:|:-----:|
|  daytime  | night |     clear     | rainy |
| dawn/dusk |       |    overcast   |       |
|           |       | partly cloudy |       |

### Private Dataset
- Consist of **real** images which are collected from YouTube and filmed using GoPro.
- Be used to do the experiment of Semi-Supervised Learning.

---

<!---------------------------------------------------------------------------------------------------->

## Training Stage
`train_test_cfg.py` contain all the training parameters.
`load_data_cfg.py` contain the path of txt file that saves all the training/testing image path.

- Faster R-CNN：run `python train_faster_rcnn.py`
- Adversarial Faster R-CNN：run `python train_adv_faster_rcnn.py`
- Semi-Supervised Adversarial Faster R-CNN：run `python train_ssl_faster_rcnn.py`

---

<!---------------------------------------------------------------------------------------------------->

## Testing Stage

- Test：run `python test_faster_rcnn.py`
- Inference：run `python inference.py`
- Draw Ground Truth：run `python -m tools.draw_gt_box`
- Show t-SNE result：run `python -m tools.t_sne_before` and `python -m tools.t_sne_after`

---

<!---------------------------------------------------------------------------------------------------->

## Model Weights

- Faster R-CNN
  - SHIFT：[download](https://drive.google.com/drive/folders/1K7K5xtNVBjS8BnflnYhxO6f85-M13huX?usp=drive_link)
  - BDD100K：[download](https://drive.google.com/drive/folders/1P5qD3l20osYu6LNJPX2bPTLMRvwZP9hm?usp=drive_link)
- Adversarial Faster R-CNN
  - SHIFT：[download](https://drive.google.com/drive/folders/1ZzDF7fCA-5kxNkX8H8ENr9Wl1qETsF6D?usp=drive_link)
  - BDD100K：[download](https://drive.google.com/drive/folders/1MHpgMAUbedmkHMUGHH1nxUvscu8SAnlj?usp=drive_link)
- Semi-Supervised Adversarial Faster R-CNN：[download](https://drive.google.com/drive/folders/1OmXHI8grZGsEzSZs7qjLSJWdxtLaHtSi?usp=drive_link)

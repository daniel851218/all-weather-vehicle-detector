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
<style> table{ margin: auto; } </style>

## Dataset

### SHIFT Dataset
- [SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation](https://www.vis.xyz/shift/)
- Consist of **synthetic** images.
- Be used to do the experiment of Adversarial Learning.
- According to the following table, classify the images into **Daytime Normal**, **Daytime Rainy**, **Night Normal** and **Night Rainy**.

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#9ABAD9;border-spacing:0;}
.tg td{background-color:#EBF5FF;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#444;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#409cff;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#fff;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-f1yk{font-family:"Times New Roman", Times, serif !important;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-f1yk">DAYTIME</th>
    <th class="tg-f1yk">NIGHT</th>
    <th class="tg-f1yk">NORMAL</th>
    <th class="tg-f1yk">RAINY</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-f1yk">morning/afternoon</td>
    <td class="tg-f1yk">sunrise/sunset</td>
    <td class="tg-f1yk">clear</td>
    <td class="tg-f1yk">small rain</td>
  </tr>
  <tr>
    <td class="tg-f1yk">noon</td>
    <td class="tg-f1yk">night</td>
    <td class="tg-f1yk">slight cloudy</td>
    <td class="tg-f1yk">mid rain</td>
  </tr>
  <tr>
    <td class="tg-f1yk">dawn/dusk</td>
    <td class="tg-f1yk">dark night</td>
    <td class="tg-f1yk">partial cloudy</td>
    <td class="tg-f1yk">heavy rain</td>
  </tr>
  <tr>
    <td class="tg-f1yk"></td>
    <td class="tg-f1yk"></td>
    <td class="tg-f1yk">overcast</td>
    <td class="tg-f1yk"></td>
  </tr>
</tbody>
</table>

### BDD100K Dataset
- [BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning](https://www.vis.xyz/bdd100k/)
- Consist of **real** images.
- Be used to do the experiment of Adversarial Learning and Semi-Supervised Learning.
- According to the following table, classify the images into **Daytime Normal**, **Daytime Rainy**, **Night Normal** and **Night Rainy**.

<table class="tg">
<thead>
  <tr>
    <th class="tg-f1yk">DAYTIME</th>
    <th class="tg-f1yk">NIGHT</th>
    <th class="tg-f1yk">NORMAL</th>
    <th class="tg-f1yk">RAINY</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-f1yk">daytime</td>
    <td class="tg-f1yk">night</td>
    <td class="tg-f1yk">clear</td>
    <td class="tg-f1yk">rainy</td>
  </tr>
  <tr>
    <td class="tg-f1yk">dawn/dusk</td>
    <td class="tg-f1yk"></td>
    <td class="tg-f1yk">overcast</td>
    <td class="tg-f1yk"></td>
  </tr>
  <tr>
    <td class="tg-f1yk"></td>
    <td class="tg-f1yk"></td>
    <td class="tg-f1yk">partly cloudy</td>
    <td class="tg-f1yk"></td>
  </tr>
</tbody>
</table>

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

<!---------------------------------------------------------------------------------------------------->
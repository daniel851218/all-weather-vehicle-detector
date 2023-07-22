import os
import time
import torch

from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision import transforms, ops

from detector.faster_rcnn import Faster_RCNN
from config.train_test_cfg import cfg
from common_module.pseudo_label_filter import filter_outlier

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

def time_sync():
    if cfg.device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

if __name__ == "__main__":
    img_file_list = glob(os.path.join(cfg.test_imgs_dir, "*.jpg"))
    
    obj_detector = Faster_RCNN().to(cfg.device)
    obj_detector.load_model(cfg.pre_train_model_path, cfg.pre_train_model_epoch)
    obj_detector.eval()

    time_pre_process = []
    time_inference = []
    time_nms = []

    for img_file in tqdm(img_file_list):
        t1 = time_sync()
        img_PIL, img_tensor, ratio, delta = pre_process(img_file)
        t2 = time_sync()
        time_pre_process.append(t2 - t1)

        with torch.no_grad():
            detections = obj_detector(img_tensor.unsqueeze(0), [{}])[0]
            t3 = time_sync()
            time_inference.append(t3 - t2)

            pred_bboxes = detections["boxes"]
            pred_labels = detections["labels"]
            pred_scores = detections["scores"]

            low_conf_filter = pred_scores > cfg.test_conf_thresh
            pred_bboxes = pred_bboxes[low_conf_filter]
            pred_labels = pred_labels[low_conf_filter]
            pred_scores = pred_scores[low_conf_filter]

            nms_filter = ops.batched_nms(pred_bboxes, pred_scores, pred_labels, cfg.test_iou_thresh)
            t4 = time_sync()
            time_nms.append(t4 - t3)

            if cfg.filter_enable:
                pred_bboxes, pred_labels, pred_scores = filter_outlier(pred_bboxes[nms_filter], pred_labels[nms_filter], pred_scores[nms_filter], 1, (0, 0))
                pred_bboxes = pred_bboxes.tolist()
                pred_labels = pred_labels.tolist()
                pred_scores = pred_scores.tolist()
            else:
                pred_bboxes = pred_bboxes[nms_filter].tolist()
                pred_labels = pred_labels[nms_filter].tolist()
                pred_scores = pred_scores[nms_filter].tolist()

            for box, lbl, conf in zip(pred_bboxes, pred_labels, pred_scores):
                x1, y1, x2, y2 = restore_bbox_size(box, ratio, delta)
                draw_rectangle = ImageDraw.ImageDraw(img_PIL)
                draw_rectangle.rectangle(((x1, y1), (x2, y2)), fill=None, outline=cfg.obj_color[lbl], width=2)

            img_PIL.save(os.path.join(cfg.result_imgs_dir, img_file.split(os.sep)[-1]))

    avg_time_pre_process = sum(time_pre_process) / len(time_pre_process)
    avg_time_inference = sum(time_inference) / len(time_inference)
    avg_time_nms = sum(time_nms) / len(time_nms)
    time_total = avg_time_pre_process + avg_time_inference + avg_time_nms
    
    print(f"{'Pre-process: ':>15s}{avg_time_pre_process:.4f} s")
    print(f"{'Inference: ':>15s}{avg_time_inference:.4f} s")
    print(f"{'NMS: ':>15s}{avg_time_nms:.4f} s")
    print(f"{'Total: ':>15s}{time_total:.4f} s\n")

    print(f"{'FPS: '}{1./time_total:.4f} on {torch.cuda.get_device_name()}")
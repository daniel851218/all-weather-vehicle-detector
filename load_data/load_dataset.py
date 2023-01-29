import os
import json
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from load_data.img_aug import ImgAugTransform
from config.load_data_cfg import data_cfg

class Base_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.img_aug_transform = ImgAugTransform()

    def get_file_list(self, file_name):
        '''
        file_name: str
        '''
        with open(file_name, "r") as file:
            return file.readlines()
        
    def resize_img(self, img, img_w, img_h):
        '''
        img: PIL image
        '''
        h, w = img.height, img.width
        ratio_w = img_w / w
        ratio_h = img_h / h
        ratio = min(ratio_w, ratio_h)

        new_h = int(h * ratio)
        new_w = int(w * ratio)
        new_img = img.resize((new_w, new_h))

        return new_img, ratio
    
    def pad_image(self, img, img_w, img_h):
        '''
        img: Tensor (channel, h, w)
        img_w: int
        img_h: int
        '''
        new_img = torch.zeros((3, img_h, img_w), dtype=img.dtype) + 0.5

        ori_h, ori_w = img.shape[1], img.shape[2]

        delta_h = int(new_img.shape[1] - ori_h)
        delta_w = int(new_img.shape[2] - ori_w)

        x1 = int(delta_w / 2)
        y1 = int(delta_h / 2)

        new_img[:, y1:y1+ori_h, x1:x1+ori_w] = img
        return new_img, (x1, y1)
    
    def resize_bbx(self, json_data, ratio, delta):
        '''
        json_data: dict
        '''
        num_obj = len(json_data)
        boxes = torch.zeros((num_obj, 4), dtype=torch.float)
        classes = torch.zeros((num_obj), dtype=torch.int)
        delta_x, delta_y = delta
        for i, obj in enumerate(json_data.values()):
            boxes[i, 0] = int((obj["x1"] * ratio) + delta_x)
            boxes[i, 1] = int((obj["y1"] * ratio) + delta_y)
            boxes[i, 2] = int((obj["x2"] * ratio) + delta_x)
            boxes[i, 3] = int((obj["y2"] * ratio) + delta_y)
            classes[i] = obj["category"]
            
        return boxes, classes
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    
# ----------------------------------------------------------------------------------------------------

class SHIFT_Dataset(Base_Dataset):
    def __init__(self, is_train=True):
        super(SHIFT_Dataset, self).__init__()
        self.is_train = is_train
        self.file_list = self.get_file_list(data_cfg.shift_txt_file_train) if is_train else self.get_file_list(data_cfg.shift_txt_file_val)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_file = self.file_list[idx].strip()
        json_file = img_file.replace("jpg", "json")
        
        if not (os.path.isfile(img_file) and os.path.isfile(json_file)):
            raise ValueError(f"File does not exist.")
        
        with open(json_file, "r") as file:
            json_data = json.loads(file.read())

        img = Image.open(img_file)
        img, ratio = self.resize_img(img, data_cfg.img_w, data_cfg.img_h)
        img = self.img_aug_transform(img) if self.is_train else self.to_tensor(img)
        img = self.normalize(img)
        img, delta = self.pad_image(img, data_cfg.img_w, data_cfg.img_h)
        boxes, obj_classes = self.resize_bbx(json_data, ratio, delta)

        daytime_class = 1 if "daytime" in img_file.split(os.sep)[-3].split("_") else 0
        weather_class = 1 if "normal" in img_file.split(os.sep)[-3].split("_") else 0
        daytime_class = torch.tensor((daytime_class))
        weather_class = torch.tensor((weather_class))

        img = img
        target = {
            "boxes": boxes, 
            "labels": obj_classes, 
            "daytime_class": daytime_class, 
            "weather_class": weather_class, 
            }

        return img, target
    
# ----------------------------------------------------------------------------------------------------







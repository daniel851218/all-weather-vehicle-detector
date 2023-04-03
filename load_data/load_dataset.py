import os
import json
import torch

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

class BDD_Dataset(Base_Dataset):
    def __init__(self, is_train=True):
        super(BDD_Dataset, self).__init__()
        self.is_train = is_train
        self.file_list = self.get_file_list(data_cfg.bdd_txt_file_train) if is_train else self.get_file_list(data_cfg.bdd_txt_file_val)

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

        daytime_class = 1 if "daytime" in img_file.split(os.sep)[-2].split("_") else 0
        weather_class = 1 if "normal" in img_file.split(os.sep)[-2].split("_") else 0
        daytime_class = torch.tensor((daytime_class))
        weather_class = torch.tensor((weather_class))

        target = {
            "boxes": boxes, 
            "labels": obj_classes, 
            "daytime_class": daytime_class, 
            "weather_class": weather_class
            }

        return img, target

# ----------------------------------------------------------------------------------------------------

class Driving_Video_Dataset(Base_Dataset):
    def __init__(self):
        super(Driving_Video_Dataset, self).__init__()
        self.file_list = self.get_file_list(data_cfg.driving_video_txt_file)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_file = self.file_list[idx].strip()

        if not os.path.isfile(img_file):
            raise ValueError(f"File does not exist.")
        
        img = Image.open(img_file)
        img, ratio = self.resize_img(img, data_cfg.img_w, data_cfg.img_h)

        img_aug_1 = self.img_aug_transform(img)
        img_aug_1 = self.normalize(img_aug_1)
        img_aug_1, delta = self.pad_image(img_aug_1, data_cfg.img_w, data_cfg.img_h)

        img_aug_2 = self.img_aug_transform(img)
        img_aug_2 = self.normalize(img_aug_2)
        img_aug_2, delta = self.pad_image(img_aug_2, data_cfg.img_w, data_cfg.img_h)
        
        daytime_class = 1 if "daytime" in img_file.split(os.sep)[-3].split("_") else 0
        weather_class = 1 if "normal" in img_file.split(os.sep)[-3].split("_") else 0
        daytime_class = torch.tensor((daytime_class))
        weather_class = torch.tensor((weather_class))

        target = {
            "daytime_class": daytime_class, 
            "weather_class": weather_class, 
            }
        
        return img_aug_1, img_aug_2, target

# ----------------------------------------------------------------------------------------------------

class Mixed_Labeled_Dataset(Base_Dataset):
    def __init__(self):
        super(Mixed_Labeled_Dataset, self).__init__()
        self.shift_file_list = self.get_file_list(data_cfg.shift_txt_file_train)
        self.bdd_file_list = self.get_file_list(data_cfg.bdd_txt_file_train)

    def __len__(self):
        return min(len(self.shift_file_list), len(self.bdd_file_list))
    
    def __getitem__(self, idx):
        shift_img_file = self.shift_file_list[idx].strip()
        shift_json_file = shift_img_file.replace("jpg", "json")
        if not (os.path.isfile(shift_img_file) and os.path.isfile(shift_json_file)):
            raise ValueError(f"File does not exist.")
        
        with open(shift_json_file, "r") as file:
            shift_json_data = json.loads(file.read())

        bdd_img_file = self.bdd_file_list[idx].strip()
        bdd_json_file = bdd_img_file.replace("jpg", "json")
        if not (os.path.isfile(bdd_img_file) and os.path.isfile(bdd_json_file)):
            raise ValueError(f"File does not exist.")
        
        with open(bdd_json_file, "r") as file:
            bdd_json_data = json.loads(file.read())
        
        shift_img = Image.open(shift_img_file)
        shift_img, shift_ratio = self.resize_img(shift_img, data_cfg.img_w, data_cfg.img_h)
        shift_img = self.img_aug_transform(shift_img)
        shift_img = self.normalize(shift_img)
        shift_img, shift_delta = self.pad_image(shift_img, data_cfg.img_w, data_cfg.img_h)
        shift_boxes, shift_obj_classes = self.resize_bbx(shift_json_data, shift_ratio, shift_delta)

        bdd_img = Image.open(bdd_img_file)
        bdd_img, bdd_ratio = self.resize_img(bdd_img, data_cfg.img_w, data_cfg.img_h)
        bdd_img = self.img_aug_transform(bdd_img)
        bdd_img = self.normalize(bdd_img)
        bdd_img, bdd_delta = self.pad_image(bdd_img, data_cfg.img_w, data_cfg.img_h)
        bdd_boxes, bdd_obj_classes = self.resize_bbx(bdd_json_data, bdd_ratio, bdd_delta)

        shift_daytime_class = 1 if "daytime" in shift_img_file.split(os.sep)[-3].split("_") else 0
        shift_weather_class = 1 if "normal" in shift_img_file.split(os.sep)[-3].split("_") else 0
        shift_daytime_class = torch.tensor((shift_daytime_class))
        shift_weather_class = torch.tensor((shift_weather_class))

        bdd_daytime_class = 1 if "daytime" in bdd_img_file.split(os.sep)[-2].split("_") else 0
        bdd_weather_class = 1 if "normal" in bdd_img_file.split(os.sep)[-2].split("_") else 0
        bdd_daytime_class = torch.tensor((bdd_daytime_class))
        bdd_weather_class = torch.tensor((bdd_weather_class))

        shift_target = {
            "boxes": shift_boxes, 
            "labels": shift_obj_classes, 
            "daytime_class": shift_daytime_class, 
            "weather_class": shift_weather_class
            }
        
        bdd_target = {
            "boxes": bdd_boxes, 
            "labels": bdd_obj_classes, 
            "daytime_class": bdd_daytime_class, 
            "weather_class": bdd_weather_class
            }

        return shift_img, shift_target, bdd_img, bdd_target

# ----------------------------------------------------------------------------------------------------

class Mixed_Real_Dataset(Base_Dataset):
    def __init__(self):
        super(Mixed_Real_Dataset, self).__init__()
        self.bdd_file_list = self.get_file_list(data_cfg.bdd_txt_file_train)
        self.driving_video_file_list = self.get_file_list(data_cfg.driving_video_txt_file)

    def __len__(self):
        return max(len(self.bdd_file_list), len(self.driving_video_file_list))
    
    def __getitem__(self, idx):
        bdd_idx = torch.randint(0, len(self.bdd_file_list), (1,)) if idx > (len(self.bdd_file_list)-1) else idx
        driving_video_idx = torch.randint(0, len(self.driving_video_file_list), (1,)) if idx > (len(self.driving_video_file_list)-1) else idx
        
        bdd_img_file = self.bdd_file_list[bdd_idx].strip()
        bdd_json_file = bdd_img_file.replace("jpg", "json")
        if not os.path.isfile(bdd_img_file) or not os.path.isfile(bdd_json_file):
            raise ValueError(f"File does not exist.")
        
        with open(bdd_json_file, "r") as file:
            bdd_json_data = json.loads(file.read())

        driving_video_img_file = self.driving_video_file_list[driving_video_idx].strip()
        if not os.path.isfile(driving_video_img_file):
            raise ValueError(f"File does not exist.")

        bdd_img = Image.open(bdd_img_file)
        bdd_img, bdd_ratio = self.resize_img(bdd_img, data_cfg.img_w, data_cfg.img_h)
        bdd_img = self.img_aug_transform(bdd_img)
        bdd_img = self.normalize(bdd_img)
        bdd_img, bdd_delta = self.pad_image(bdd_img, data_cfg.img_w, data_cfg.img_h)
        bdd_boxes, bdd_obj_classes = self.resize_bbx(bdd_json_data, bdd_ratio, bdd_delta)
        
        driving_video_img = Image.open(driving_video_img_file)
        driving_video_img, driving_video_ratio = self.resize_img(driving_video_img, data_cfg.img_w, data_cfg.img_h)

        driving_video_img_1 = self.img_aug_transform(driving_video_img)
        driving_video_img_1 = self.normalize(driving_video_img_1)
        driving_video_img_1, driving_video_delta = self.pad_image(driving_video_img_1, data_cfg.img_w, data_cfg.img_h)

        driving_video_img_2 = self.img_aug_transform(driving_video_img)
        driving_video_img_2 = self.normalize(driving_video_img_2)
        driving_video_img_2, driving_video_delta = self.pad_image(driving_video_img_2, data_cfg.img_w, data_cfg.img_h)

        driving_video_daytime_class = 1 if "daytime" in driving_video_img_file.split(os.sep)[-3].split("_") else 0
        driving_video_weather_class = 1 if "normal" in driving_video_img_file.split(os.sep)[-3].split("_") else 0
        driving_video_daytime_class = torch.tensor((driving_video_daytime_class))
        driving_video_weather_class = torch.tensor((driving_video_weather_class))

        bdd_target = {
            "boxes": bdd_boxes, 
            "labels": bdd_obj_classes, 
            }
        
        driving_video_target = {
            "daytime_class": driving_video_daytime_class, 
            "weather_class": driving_video_weather_class
        }
        
        return bdd_img, bdd_target, driving_video_img_1, driving_video_img_2, driving_video_target, driving_video_ratio, driving_video_delta
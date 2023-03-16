import os
import torch

from tqdm import tqdm
from colorama import Fore
from torch.utils.data import DataLoader

from config.train_test_cfg import cfg
from load_data.load_dataset import SHIFT_Dataset, BDD_Dataset
from common_module.eval_utils import calc_mAP, get_class_TP_FP
from detector.faster_rcnn import Faster_RCNN

def to_CUDA(imgs, targets):
    imgs = imgs.to("cuda")
    for t in targets:
        t["boxes"] = t["boxes"].to("cuda")
        t["labels"] = t["labels"].to("cuda")
        t["daytime_class"] = t["daytime_class"].to("cuda")
        t["weather_class"] = t["weather_class"].to("cuda")

    return imgs, targets

def evaluate(model, data_loader):
    model.eval()
    class_TP_FP = [torch.tensor([]) for _ in range((cfg.num_classes - 1))]
    class_score = [torch.tensor([]) for _ in range((cfg.num_classes - 1))]
    num_gt_boxes = torch.zeros((cfg.num_classes-1))
    
    with torch.no_grad():
        for imgs, targets in tqdm(data_loader):
            imgs = torch.stack(imgs, dim=0)

            if cfg.device == "cuda" and torch.cuda.is_available():
                imgs, targets = to_CUDA(imgs, targets)
            
            detections = model(imgs, targets)
            batch_class_TP_FP, batch_class_score, batch_class_num_gt = get_class_TP_FP(targets, detections, 0.5)

            for c in range(cfg.num_classes-1):
                class_TP_FP[c] = torch.cat([class_TP_FP[c].to(torch.bool), batch_class_TP_FP[c].to(torch.bool)])
                class_score[c] = torch.cat([class_score[c].to(torch.float), batch_class_score[c].to(torch.float)])
            num_gt_boxes += batch_class_num_gt

    AP_each_class, P_each_class, R_each_class = calc_mAP(class_TP_FP, class_score, num_gt_boxes)
    return AP_each_class, P_each_class, R_each_class, num_gt_boxes

def print_result(AP_each_class, P_each_class, R_each_class, num_gt_boxes):
    mAP = sum(AP_each_class) / len(AP_each_class)
    print(f"\nImage Type: {cfg.test_img_type}")
    print(f"\n{'Class':^15s}|{'Labels':^15s}|{'P':^15s}|{'R':^15s}|{'mAP@.5':^15s}")
    print("=" * 79)
    for i in range(len(AP_each_class)):
        print(f"{cfg.obj_classes[i+1]:^15s}|{int(num_gt_boxes[i]):^15d}|{P_each_class[i]:^15f}|{R_each_class[i]:^15f}|{AP_each_class[i]:^15f}")
        print("-" * 79)
    print(f"{'':15s} {'':15s} {'':15s} {'':15s}|{mAP:^15f}\n")

if __name__ == "__main__":
    # =========================== Initialize ===========================
    print(Fore.GREEN + "="*100 +  "\nInitialize ...\n" + Fore.RESET)
    if cfg.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # =========================== Load Data ===========================
    print(Fore.GREEN + "Load Data ..." + Fore.RESET)
    if cfg.dataset == "SHIFT":
        val_dataset = SHIFT_Dataset(is_train=False)
    elif cfg.dataset == "BDD":
        val_dataset = BDD_Dataset(is_train=False)
    else:
        raise ValueError("dataset is not exist !")
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=os.cpu_count(), collate_fn=val_dataset.collate_fn)
    print(f"\tValidation Data: {len(val_dataset)}", )

    # =========================== Create Model ===========================
    print(Fore.GREEN + "Create Model ...\n" + Fore.RESET)
    obj_detector = Faster_RCNN().to(cfg.device)
    obj_detector.load_model(cfg.pre_train_model_path, cfg.pre_train_model_epoch)

    # =========================== Testing ===========================
    print(Fore.GREEN + "Testing ...\n" + "="*100 + Fore.RESET)  
    AP_each_class, P_each_class, R_each_class, num_gt_boxes = evaluate(obj_detector, val_dataloader)
    print_result(AP_each_class, P_each_class, R_each_class, num_gt_boxes)
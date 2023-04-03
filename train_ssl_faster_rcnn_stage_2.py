import os
import time
import shutil
import torch
import torch.optim as optim

from tqdm import tqdm
from colorama import Fore
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.train_test_cfg import cfg
from load_data.load_dataset import BDD_Dataset, Mixed_Real_Dataset
from common_module.eval_utils import calc_mAP, get_class_TP_FP
from detector.ssl_faster_rcnn import Semi_Supervised_Faster_RCNN_Stage_2

def to_CUDA(imgs, targets):
    imgs = imgs.to("cuda")
    for t in targets:
        for key in t.keys():
            t[key] = t[key].to("cuda") if t[key] is not None else t[key]

    return imgs, targets

def update_weight_ema(model, epoch):
    alpha = cfg.alpha_ema

    for student_param, teacher_param in zip(model.student_backbone.parameters(), model.teacher_backbone.parameters()):
        teacher_param.data = alpha*teacher_param.data + (1-alpha)*student_param.data
    
    for student_param, teacher_param in zip(model.student_rpn.parameters(), model.teacher_rpn.parameters()):
        teacher_param.data = alpha*teacher_param.data + (1-alpha)*student_param.data

    for student_param, teacher_param in zip(model.student_roi_align.parameters(), model.teacher_roi_align.parameters()):
        teacher_param.data = alpha*teacher_param.data + (1-alpha)*student_param.data

    for student_param, teacher_param in zip(model.student_detect_head.parameters(), model.teacher_detect_head.parameters()):
        teacher_param.data = alpha*teacher_param.data + (1-alpha)*student_param.data

def train_one_epoch(model, optimizer, data_loader, writer, epoch, iter_per_batch, log_loss, log_freq):
    model.train()
    count = 0
    pbar = tqdm(data_loader)

    for bdd_imgs, bdd_targets, driving_video_imgs_1, driving_video_imgs_2, driving_video_targets, ratio, delta in pbar:
        bdd_imgs = torch.stack(bdd_imgs, dim=0)
        driving_video_imgs_1 = torch.stack(driving_video_imgs_1, dim=0)
        driving_video_imgs_2 = torch.stack(driving_video_imgs_2, dim=0)

        if cfg.device == "cuda" and torch.cuda.is_available():
            bdd_imgs, bdd_targets = to_CUDA(bdd_imgs, bdd_targets)
            driving_video_imgs_1, _ = to_CUDA(driving_video_imgs_1, driving_video_targets)
            driving_video_imgs_2, _ = to_CUDA(driving_video_imgs_2, {})

        loss_dict = model(bdd_imgs, bdd_targets, driving_video_imgs_1, driving_video_imgs_2, driving_video_targets, ratio, delta)
        optimizer.zero_grad()
        loss_dict["loss_total"].backward()
        optimizer.step()
        update_weight_ema(model, epoch)

        for k in loss_dict["loss_sup"].keys():
            log_loss["loss_sup"][k].append(loss_dict["loss_sup"][k].item())
        
        for k in loss_dict["loss_unsup"].keys():
            log_loss["loss_unsup"][k].append(loss_dict["loss_unsup"][k].item())

        log_loss["loss_total"].append(loss_dict["loss_total"])

        pbar.set_description(Fore.BLUE + "loss_total: " + Fore.RESET + f"{loss_dict['loss_total']:2.3f}")
        iters = epoch * iter_per_batch + count
        if iters % log_freq == 0:
            avg_loss_total = sum(log_loss["loss_total"]) / len(log_loss["loss_total"])

            avg_loss_sup_rpn_score = sum(log_loss["loss_sup"]["loss_rpn_score"]) / len(log_loss["loss_sup"]["loss_rpn_score"])
            avg_loss_sup_rpn_box_reg = sum(log_loss["loss_sup"]["loss_rpn_box_reg"]) / len(log_loss["loss_sup"]["loss_rpn_box_reg"])
            avg_loss_sup_roi_cls = sum(log_loss["loss_sup"]["loss_roi_cls"]) / len(log_loss["loss_sup"]["loss_roi_cls"])
            avg_loss_sup_roi_box_reg = sum(log_loss["loss_sup"]["loss_roi_box_reg"]) / len(log_loss["loss_sup"]["loss_roi_box_reg"])

            avg_loss_unsup_rpn_score = sum(log_loss["loss_unsup"]["loss_rpn_score"]) / len(log_loss["loss_unsup"]["loss_rpn_score"])
            avg_loss_unsup_roi_cls = sum(log_loss["loss_unsup"]["loss_roi_cls"]) / len(log_loss["loss_unsup"]["loss_roi_cls"])

            avg_loss_daytime = sum(log_loss["loss_unsup"]["loss_daytime_img_score"]) / len(log_loss["loss_unsup"]["loss_daytime_img_score"]) + \
                               sum(log_loss["loss_unsup"]["loss_daytime_ins_score"]) / len(log_loss["loss_unsup"]["loss_daytime_ins_score"]) + \
                               sum(log_loss["loss_unsup"]["loss_daytime_consistency"]) / len(log_loss["loss_unsup"]["loss_daytime_consistency"])
            avg_loss_weather = sum(log_loss["loss_unsup"]["loss_weather_img_score"]) / len(log_loss["loss_unsup"]["loss_weather_img_score"]) + \
                               sum(log_loss["loss_unsup"]["loss_weather_ins_score"]) / len(log_loss["loss_unsup"]["loss_weather_ins_score"]) + \
                               sum(log_loss["loss_unsup"]["loss_weather_consistency"]) / len(log_loss["loss_unsup"]["loss_weather_consistency"])
            

            writer.add_scalar("loss_total", avg_loss_total, iters)
            
            writer.add_scalar("loss_sup_rpn_score", avg_loss_sup_rpn_score, iters)
            writer.add_scalar("loss_sup_rpn_box_reg", avg_loss_sup_rpn_box_reg, iters)
            writer.add_scalar("loss_sup_roi_cls", avg_loss_sup_roi_cls, iters)
            writer.add_scalar("loss_sup_roi_box_reg", avg_loss_sup_roi_box_reg, iters)

            writer.add_scalar("loss_unsup_rpn_score", avg_loss_unsup_rpn_score, iters)
            writer.add_scalar("loss_unsup_roi_cls", avg_loss_unsup_roi_cls, iters)

            writer.add_scalar("loss_daytime", avg_loss_daytime, iters)
            writer.add_scalar("loss_weather", avg_loss_weather, iters)

        count += 1

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

            detections = model(imgs, targets, None, None, None)

            batch_class_TP_FP, batch_class_score, batch_class_num_gt = get_class_TP_FP(targets, detections, 0.5)

            for c in range(cfg.num_classes-1):
                class_TP_FP[c] = torch.cat([class_TP_FP[c].to(torch.bool), batch_class_TP_FP[c].to(torch.bool)])
                class_score[c] = torch.cat([class_score[c].to(torch.float), batch_class_score[c].to(torch.float)])
            num_gt_boxes += batch_class_num_gt

    AP_each_class, _, _ = calc_mAP(class_TP_FP, class_score, num_gt_boxes)
    return sum(AP_each_class) / len(AP_each_class)

if __name__ == "__main__":
    # ============================ Initialize ============================
    print(Fore.GREEN + "="*100 +  "\nInitialize ...\n" + Fore.RESET)
    cur_time = time.localtime(time.time())
    time_str = f"{cur_time.tm_mon:02d}{cur_time.tm_mday:02d}_{cur_time.tm_hour:02d}{cur_time.tm_min:02d}"
    ckpt_dir = os.path.join(cfg.ckpt_dir, time_str)
    writer = SummaryWriter(ckpt_dir)
    shutil.copy(os.path.join("config", "train_test_cfg.py"), os.path.join(ckpt_dir, "train_test_cfg.py"))

    if cfg.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ============================= Load Data ============================

    print(Fore.GREEN + "Load Data ..." + Fore.RESET)
    train_dataset = Mixed_Real_Dataset()
    val_dataset = BDD_Dataset(is_train=False)

    print(f"\tMixed Labeled Data: {len(train_dataset)}")
    print(f"\tBDD Validation Data: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True ,num_workers=os.cpu_count(), collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=os.cpu_count(), collate_fn=val_dataset.collate_fn)

    # =========================== Create Model ===========================
    print(Fore.GREEN + "Create Model ...\n" + Fore.RESET)
    obj_detector = Semi_Supervised_Faster_RCNN_Stage_2().to(cfg.device)
    
    optimizer = optim.SGD(obj_detector.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=cfg.lr_gamma, patience=cfg.lr_dec_step_size, min_lr=cfg.lr_min)

    # ============================= Training =============================
    print(Fore.GREEN + "Training ...\n" + "="*100 + Fore.RESET)
    best_mAP_50 = 0

    log_loss_sup = {
        "loss_rpn_score": [],
        "loss_rpn_box_reg": [],
        "loss_roi_cls": [],
        "loss_roi_box_reg": []
    }

    log_loss_unsup = {
        "loss_rpn_score": [],
        "loss_roi_cls": [],
        "loss_daytime_img_score": [],
        "loss_daytime_ins_score": [],
        "loss_daytime_consistency": [],
        "loss_weather_img_score": [],
        "loss_weather_ins_score": [],
        "loss_weather_consistency": [],
    }

    log_loss = {
        "loss_total": [],
        "loss_sup": log_loss_sup,
        "loss_unsup": log_loss_unsup,
    }
    
    patience = 0
    for epoch in range(cfg.start_epoch, cfg.epochs):
        print(f"\nEpoch: {epoch:03d}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        iter_per_batch = len(train_dataset) // cfg.batch_size + 1 if len(train_dataset) % cfg.batch_size else len(train_dataset) // cfg.batch_size
        train_one_epoch(obj_detector, optimizer, train_dataloader, writer, epoch, iter_per_batch, log_loss, log_freq=100)

        mAP_50 = evaluate(obj_detector, val_dataloader)
        print(f"mAP_50: {mAP_50}")
        if best_mAP_50 < mAP_50:
            print(Fore.RED + "Best mAP_50: " + Fore.RESET + f"{best_mAP_50} \u2192 {mAP_50}")
            writer.add_text("mAP_50", "Best mAP_50: " + f"{best_mAP_50} \u2192 {mAP_50}", epoch)

            best_mAP_50 = mAP_50
            patience = 0
            obj_detector.save_model(epoch, ckpt_dir)

        lr_scheduler.step(best_mAP_50)
        patience += 1
        if patience > cfg.max_patience:
            break

    # ============================== Finish ==============================
    print(Fore.GREEN + "Finish Training ...\n" + Fore.RESET)
    print(f"Train {epoch} epochs")
    print(f"Best mAP_50: {best_mAP_50}\n")
import torch
import matplotlib.pyplot as plt

from config.test_cfg import cfg
from common_module.boxes_utils import box_iou

def get_class_TP_FP(targets, detections, iou_thresh):
    batch_class_TP_FP = [[] for _ in range((cfg.num_classes - 1))]
    batch_class_score = [[] for _ in range((cfg.num_classes - 1))]
    batch_class_num_gt = [0 for _ in range((cfg.num_classes - 1))]

    pred_boxes = [d["boxes"] for d in detections]
    pred_labels = [d["labels"] for d in detections]
    pred_scores = [d["scores"] for d in detections]

    gt_boxes = [t["boxes"].to(torch.float) for t in targets]
    gt_labels = [t["labels"] for t in targets]

    for c in range(1, cfg.num_classes):         # for each class (Warning: class label start from 1 !!!)
        for i in range(len(gt_boxes)):          # for each image
            gt_box = gt_boxes[i]
            gt_label = gt_labels[i]
            
            pred_box = pred_boxes[i]
            pred_label = pred_labels[i]
            pred_score = pred_scores[i]

            gt_box_class = gt_box[gt_label == c]
            pred_box_class = pred_box[pred_label == c]
            pred_score_class = pred_score[pred_label == c]
            
            batch_class_num_gt[c-1] += len(gt_label[gt_label == c])
            
            if(len(pred_box_class) == 0):
                continue
            elif(len(gt_box_class) == 0):
                match_quality_matrix = torch.zeros((1, len(pred_box_class)), dtype=torch.float, device=cfg.device)
            else:
                match_quality_matrix = box_iou(gt_box_class, pred_box_class)

            max_iou, max_idx = torch.max(match_quality_matrix, dim=0)
            TP_FP_class = max_iou >= iou_thresh

            # avoid one gt_box match two or more pred_box
            gt_box_has_matched = []
            for i in range(len(max_idx)):
                gt_box_idx = max_idx[i]
                if TP_FP_class[i] == True:
                    if gt_box_idx not in gt_box_has_matched:                
                        gt_box_has_matched.append(gt_box_idx)
                    else:
                        TP_FP_class[i] = False
            
            batch_class_TP_FP[c-1].append(TP_FP_class)
            batch_class_score[c-1].append(pred_score_class)

    for c in range(cfg.num_classes-1):
        if len(batch_class_TP_FP[c]) == 0:
            batch_class_TP_FP[c] = torch.tensor(batch_class_TP_FP[c]).to("cpu")
            batch_class_score[c] = torch.tensor(batch_class_score[c]).to("cpu")
        else:
            batch_class_TP_FP[c] = torch.cat(batch_class_TP_FP[c], dim=0).to("cpu")
            batch_class_score[c] = torch.cat(batch_class_score[c], dim=0).to("cpu")            

    return batch_class_TP_FP, batch_class_score, torch.tensor(batch_class_num_gt)

def plot_PR_curve(precision_list, recall_list, obj_class):
    plt.plot(recall_list, precision_list)
    plt.title(f"{obj_class} PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

def calc_mAP(class_TP_FP, class_score, num_gt_boxes, is_plot=False):
    AP_each_class = []
    P_each_class = []
    R_each_class = []
    epsilon = 1e-10
    # sorting by class_score
    for c in range(cfg.num_classes-1):
        class_score[c], sorted_idx = torch.sort(class_score[c], descending=True)
        class_TP_FP[c] = class_TP_FP[c][sorted_idx]
        num_gt = num_gt_boxes[c]

        precision = [1]
        recall = [0]
        f1_score = [0]
        tp_sum = 0
        for i, tp in enumerate(class_TP_FP[c]):
            tp_sum += tp
            precision.append(tp_sum / ((i+1) + epsilon))
            recall.append(tp_sum / (num_gt + epsilon))
            f1_score.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1] + epsilon))
            
        precision = torch.tensor(precision)
        recall = torch.tensor(recall)
        f1_score = torch.tensor(f1_score)

        f1_max_idx = torch.argmax(f1_score)
        AP_each_class.append(torch.trapz(precision, recall).item())
        P_each_class.append(precision[f1_max_idx])
        R_each_class.append(recall[f1_max_idx])

        if is_plot:
            plot_PR_curve(precision, recall, cfg.obj_classes[c+1])

    return AP_each_class, P_each_class, R_each_class
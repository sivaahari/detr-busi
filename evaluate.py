import torch
from torch.utils.data import DataLoader
from datasets.busi import BUSIDataset
from models.detr import DETR
from utils.box_ops import compute_iou
import configs.config as cfg


def evaluate(split: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BUSIDataset(split=split)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Evaluating on {split} split: {len(dataset)} samples\n")

    model = DETR(num_classes=cfg.NUM_CLASSES, num_queries=cfg.NUM_QUERIES).to(device)
    model.load_state_dict(
        torch.load(cfg.SAVE_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    # per-class accumulators
    stats = {i: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0} for i in range(2)}

    with torch.no_grad():
        for images, gt_bboxes, gt_labels in loader:
            images   = images.to(device)
            gt_label = gt_labels[0].item()
            gt_box   = gt_bboxes[0].cpu()

            logits, boxes = model(images)

            probs      = torch.softmax(logits[0], dim=-1)
            scores     = probs[:, :2].max(dim=1).values
            best_idx   = scores.argmax()
            pred_class = probs[best_idx, :2].argmax().item()
            pred_box   = boxes[0][best_idx].cpu()

            iou      = compute_iou(pred_box, gt_box)
            detected = iou >= cfg.IOU_THRESHOLD

            if detected and pred_class == gt_label:
                stats[gt_label]["tp"]      += 1
                stats[gt_label]["iou_sum"] += iou
            elif detected and pred_class != gt_label:
                stats[pred_class]["fp"] += 1
                stats[gt_label]["fn"]   += 1
                stats[gt_label]["iou_sum"] += iou
            else:
                stats[gt_label]["fn"] += 1

    # ── results table ─────────────────────────────────────────────────────────
    print(f"{'Class':<12} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 50)

    all_tp = all_fp = all_fn = 0
    all_iou_sum = 0.0

    for cls_idx, name in enumerate(cfg.CLASS_NAMES):
        tp = stats[cls_idx]["tp"]
        fp = stats[cls_idx]["fp"]
        fn = stats[cls_idx]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        mean_iou  = stats[cls_idx]["iou_sum"] / tp if tp > 0 else 0.0

        print(f"{name:<12} {mean_iou:>8.4f} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f}")

        all_tp      += tp
        all_fp      += fp
        all_fn      += fn
        all_iou_sum += stats[cls_idx]["iou_sum"]

    print("-" * 50)
    overall_p   = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_r   = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1  = (2 * overall_p * overall_r / (overall_p + overall_r)
                   if (overall_p + overall_r) > 0 else 0.0)
    overall_iou = all_iou_sum / all_tp if all_tp > 0 else 0.0

    print(f"{'Mean':<12} {overall_iou:>8.4f} {overall_p:>10.4f} "
          f"{overall_r:>8.4f} {overall_f1:>8.4f}")


if __name__ == "__main__":
    evaluate()

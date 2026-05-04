import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.busi import BUSIDataset
from models.detr import DETR
from utils.box_ops import compute_iou
import configs.config as cfg


def compute_dice(pred, target, num_classes):
    """Compute Dice score per class."""
    dice_scores = []
    pred_cls = torch.argmax(pred, dim=0)
    for c in range(num_classes):
        pred_c = (pred_cls == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice.item())
    return dice_scores


def evaluate(split: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BUSIDataset(split=split)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Evaluating on {split} split: {len(dataset)} samples\n")

    model = DETR(num_classes=cfg.NUM_CLASSES, num_queries=cfg.NUM_QUERIES, use_segmentation=True).to(device)
    model.load_state_dict(
        torch.load(cfg.SAVE_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    # per-class accumulators for detection
    det_stats = {i: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0} for i in range(2)}
    
    # per-class accumulators for segmentation
    seg_dice_sum = {0: 0.0, 1: 0.0, 2: 0.0}
    seg_count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            gt_mask = batch[1].long().to(device)
            gt_label = batch[3][0].item()
            gt_box = batch[2][0].cpu()

            logits, boxes, seg_mask = model(images)
            
            # Resize seg_mask to match gt_mask
            seg_mask = F.interpolate(seg_mask, size=gt_mask.shape[1:], 
                                    mode='bilinear', align_corners=False)

            # Detection metrics
            probs      = torch.softmax(logits[0], dim=-1)
            scores     = probs[:, :2].max(dim=1).values
            best_idx   = scores.argmax()
            pred_class = probs[best_idx, :2].argmax().item()
            pred_box   = boxes[0][best_idx].cpu()

            iou      = compute_iou(pred_box, gt_box)
            detected = iou >= cfg.IOU_THRESHOLD

            if detected and pred_class == gt_label:
                det_stats[gt_label]["tp"]      += 1
                det_stats[gt_label]["iou_sum"] += iou
            elif detected and pred_class != gt_label:
                det_stats[pred_class]["fp"] += 1
                det_stats[gt_label]["fn"]   += 1
                det_stats[gt_label]["iou_sum"] += iou
            else:
                det_stats[gt_label]["fn"] += 1
            
            # Segmentation metrics (Dice per class)
            dice_scores = compute_dice(seg_mask[0], gt_mask, cfg.NUM_CLASSES)
            for c in range(3):
                seg_dice_sum[c] += dice_scores[c]
            seg_count += 1

    # ── Detection results table ─────────────────────────────────────────────────
    print("=" * 70)
    print("DETECTION METRICS")
    print("=" * 70)
    print(f"{'Class':<12} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 50)

    all_tp = all_fp = all_fn = 0
    all_iou_sum = 0.0

    for cls_idx, name in enumerate(cfg.CLASS_NAMES):
        tp = det_stats[cls_idx]["tp"]
        fp = det_stats[cls_idx]["fp"]
        fn = det_stats[cls_idx]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        mean_iou  = det_stats[cls_idx]["iou_sum"] / tp if tp > 0 else 0.0

        print(f"{name:<12} {mean_iou:>8.4f} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f}")

        all_tp      += tp
        all_fp      += fp
        all_fn      += fn
        all_iou_sum += det_stats[cls_idx]["iou_sum"]

    print("-" * 50)
    overall_p   = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_r   = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1  = (2 * overall_p * overall_r / (overall_p + overall_r)
                   if (overall_p + overall_r) > 0 else 0.0)
    overall_iou = all_iou_sum / all_tp if all_tp > 0 else 0.0

    print(f"{'Mean':<12} {overall_iou:>8.4f} {overall_p:>10.4f} "
          f"{overall_r:>8.4f} {overall_f1:>8.4f}")
    
    # ── Segmentation results table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SEGMENTATION METRICS (Dice Score)")
    print("=" * 70)
    print(f"{'Class':<12} {'Dice':>8}")
    print("-" * 30)
    
    for cls_idx, name in enumerate(["Benign", "Malignant", "Background"]):
        dice = seg_dice_sum[cls_idx] / seg_count
        print(f"{name:<12} {dice:>8.4f}")
    
    mean_dice = (seg_dice_sum[0] + seg_dice_sum[1]) / (2 * seg_count)
    print("-" * 30)
    print(f"{'Foreground':<12} {mean_dice:>8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()

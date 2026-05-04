import torch
import cv2
import numpy as np
import torch.nn.functional as F
import configs.config as cfg


def visualize_prediction(image, pred_logits, pred_boxes, gt_bbox=None):
    """
    Overlay predicted bbox (and optionally GT bbox) on the image.

    image      : (2, H, W) tensor — channel 0 is the grayscale image
    pred_logits: (num_queries, num_classes) tensor
    pred_boxes : (num_queries, 4) tensor — normalized [x_min, y_min, x_max, y_max]
    gt_bbox    : (4,) tensor — normalized GT box (optional, drawn in yellow)
    """
    base = image[0].cpu().numpy()
    base = (base * 255).astype(np.uint8)
    canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]

    # ── pick best prediction ──────────────────────────────────────────────────
    probs    = torch.softmax(pred_logits, dim=-1)
    scores   = probs[:, :2].max(dim=1).values
    best_idx = scores.argmax()

    cls   = probs[best_idx, :2].argmax().item()
    score = scores[best_idx].item()

    x_min, y_min, x_max, y_max = pred_boxes[best_idx].cpu().tolist()
    px1, py1 = int(x_min * w), int(y_min * h)
    px2, py2 = int(x_max * w), int(y_max * h)

    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
    label = cfg.CLASS_NAMES[cls] if cls < len(cfg.CLASS_NAMES) else f"cls{cls}"

    cv2.rectangle(canvas, (px1, py1), (px2, py2), color, 2)
    cv2.putText(canvas, f"{label} {score:.2f}", (px1, max(py1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ── ground-truth box (yellow) ─────────────────────────────────────────────
    if gt_bbox is not None:
        gx1, gy1, gx2, gy2 = [c.item() for c in gt_bbox.cpu()]
        gx1, gy1 = int(gx1 * w), int(gy1 * h)
        gx2, gy2 = int(gx2 * w), int(gy2 * h)
        cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
        cv2.putText(canvas, "GT", (gx1, max(gy1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas


def visualize_segmentation(image, pred_seg, gt_mask, gt_label):
    """
    Visualize segmentation prediction and ground truth.
    
    image      : (2, H, W) tensor — grayscale + edge
    pred_seg   : (num_classes, H, W) tensor — segmentation logits
    gt_mask    : (H, W) tensor — ground truth mask (class indices)
    gt_label   : int — ground truth class
    """
    base = image[0].cpu().numpy()
    base = (base * 255).astype(np.uint8)
    canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    
    # Get predicted class map
    pred_classes = torch.argmax(pred_seg, dim=0).cpu().numpy()  # (H, W)
    
    # Color map for classes: 0=benign=green, 1=malignant=red, 2=background=gray
    colors = {
        0: (0, 255, 0),      # benign - green
        1: (0, 0, 255),      # malignant - red  
        2: (128, 128, 128)   # background - gray
    }
    
    # Create colored mask overlay
    pred_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    gt_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls in [0, 1]:  # Only show foreground classes
        pred_overlay[pred_classes == cls] = colors[cls]
        gt_overlay[gt_mask == cls] = colors[cls]
    
    # Blend with original image
    alpha = 0.5
    pred_result = cv2.addWeighted(canvas, 1 - alpha, pred_overlay, alpha, 0)
    gt_result = cv2.addWeighted(canvas, 1 - alpha, gt_overlay, alpha, 0)
    
    # Add labels
    pred_label = cfg.CLASS_NAMES[gt_label] if gt_label < len(cfg.CLASS_NAMES) else f"cls{gt_label}"
    cv2.putText(pred_result, f"Pred: {pred_label}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(pred_result, "Prediction", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(gt_result, f"GT: {pred_label}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(gt_result, "Ground Truth", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Combine side by side
    combined = np.hstack([pred_result, gt_result])
    
    return combined

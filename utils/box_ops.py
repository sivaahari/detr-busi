import torch


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute IoU between two boxes in [x_min, y_min, x_max, y_max] normalized coords.
    Accepts 1-D tensors or plain lists/arrays.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return float(intersection / union)


def box_xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    """Convert [x_min, y_min, x_max, y_max] → [cx, cy, w, h]."""
    x_min, y_min, x_max, y_max = box.unbind(-1)
    return torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2,
                        x_max - x_min, y_max - y_min], dim=-1)


def box_cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] → [x_min, y_min, x_max, y_max]."""
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

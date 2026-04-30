import torch
import torch.nn.functional as F
from utils.matcher import HungarianMatcher
import configs.config as cfg


class DETRLoss:
    def __init__(self):
        self.matcher = HungarianMatcher(
            class_weight=cfg.COST_CLASS,
            bbox_weight=cfg.COST_BBOX,
        )

    def loss(self, pred_logits, pred_boxes, targets):
        batch_size = pred_logits.shape[0]
        device = pred_logits.device

        total_cls_loss   = 0.0
        total_bbox_loss  = 0.0
        total_prior_loss = 0.0

        for i in range(batch_size):
            logits = pred_logits[i]
            boxes  = pred_boxes[i]

            num_classes = logits.shape[-1]
            no_obj_idx  = num_classes - 1   # last class is always no-object

            # down-weight no-object class to fix 99:1 query imbalance
            cls_weight = torch.ones(num_classes, device=device)
            cls_weight[no_obj_idx] = cfg.NO_OBJ_WEIGHT

            target = {
                "labels": targets[i][1].unsqueeze(0),
                "boxes":  targets[i][0].unsqueeze(0),
            }

            idx_pred, idx_tgt = self.matcher.match(logits, boxes, target)

            tgt_labels = target["labels"][idx_tgt]
            tgt_boxes  = target["boxes"][idx_tgt]

            # ── classification ────────────────────────────────────────────────
            target_classes = torch.full(
                (logits.shape[0],), no_obj_idx, dtype=torch.long, device=device
            )
            target_classes[idx_pred] = tgt_labels
            cls_loss = F.cross_entropy(logits, target_classes, weight=cls_weight)

            # ── bounding box (L1) ─────────────────────────────────────────────
            matched_boxes = boxes[idx_pred]
            bbox_loss = F.l1_loss(matched_boxes, tgt_boxes)

            # ── geometric prior (aspect ratio + width) ────────────────────────
            pred_w = matched_boxes[:, 2] - matched_boxes[:, 0]
            pred_h = matched_boxes[:, 3] - matched_boxes[:, 1]
            tgt_w  = tgt_boxes[:, 2] - tgt_boxes[:, 0]
            tgt_h  = tgt_boxes[:, 3] - tgt_boxes[:, 1]

            pred_ratio = pred_h / (pred_w + 1e-6)
            tgt_ratio  = tgt_h  / (tgt_w  + 1e-6)

            prior_loss = F.l1_loss(pred_ratio, tgt_ratio) + F.l1_loss(pred_w, tgt_w)

            total_cls_loss   += cls_loss
            total_bbox_loss  += bbox_loss
            total_prior_loss += prior_loss

        total_loss = (
            total_cls_loss
            + total_bbox_loss
            + cfg.PRIOR_LOSS_WEIGHT * total_prior_loss
        )
        return total_loss

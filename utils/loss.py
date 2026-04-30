import torch
import torch.nn.functional as F
from utils.matcher import HungarianMatcher


class DETRLoss:
    def __init__(self):
        self.matcher = HungarianMatcher()

    def loss(self, pred_logits, pred_boxes, targets):
        batch_size = pred_logits.shape[0]

        total_cls_loss = 0
        total_bbox_loss = 0

        for i in range(batch_size):
            logits = pred_logits[i]   # (num_queries, num_classes)
            boxes = pred_boxes[i]     # (num_queries, 4)

            target = {
                "labels": targets[i][1].unsqueeze(0),
                "boxes": targets[i][0].unsqueeze(0),
            }

            idx_pred, idx_tgt = self.matcher.match(logits, boxes, target)

            tgt_labels = target["labels"][idx_tgt]
            tgt_boxes = target["boxes"][idx_tgt]

            # -------- CLASSIFICATION LOSS (WITH NO-OBJECT) --------
            target_classes = torch.full(
                (logits.shape[0],), 2, dtype=torch.long, device=logits.device
            )  # 2 = no-object

            target_classes[idx_pred] = tgt_labels

            cls_loss = F.cross_entropy(logits, target_classes)

            # -------- BBOX LOSS (ONLY MATCHED) --------
            matched_boxes = boxes[idx_pred]
            bbox_loss = F.l1_loss(matched_boxes, tgt_boxes)

            total_cls_loss += cls_loss
            total_bbox_loss += bbox_loss

        total_loss = total_cls_loss + total_bbox_loss

        return total_loss
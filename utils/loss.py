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
            logits = pred_logits[i]
            boxes = pred_boxes[i]

            target = {
                "labels": targets[i][1].unsqueeze(0),
                "boxes": targets[i][0].unsqueeze(0),
            }

            idx_pred, idx_tgt = self.matcher.match(logits, boxes, target)

            # matched predictions
            matched_logits = logits[idx_pred]
            matched_boxes = boxes[idx_pred]

            tgt_labels = target["labels"][idx_tgt]
            tgt_boxes = target["boxes"][idx_tgt]

            # classification loss
            cls_loss = F.cross_entropy(matched_logits, tgt_labels)

            # bbox loss (L1)
            bbox_loss = F.l1_loss(matched_boxes, tgt_boxes)

            total_cls_loss += cls_loss
            total_bbox_loss += bbox_loss

        total_loss = total_cls_loss + total_bbox_loss

        return total_loss
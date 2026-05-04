import torch
import torch.nn.functional as F
from utils.matcher import HungarianMatcher
import configs.config as cfg


def dice_loss(pred, target, smooth=1e-6):
    """
    Compute Dice loss for multi-class segmentation.
    pred: (B, C, H, W) logits
    target: (B, H, W) class indices
    """
    num_classes = pred.shape[1]
    device = pred.device
    
    # Convert target to one-hot
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    
    # Softmax predictions
    pred_soft = torch.softmax(pred, dim=1)
    
    # Compute Dice per class
    dice_scores = []
    for c in range(num_classes):
        pred_c = pred_soft[:, c]
        target_c = target_one_hot[:, c]
        
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.mean())
    
    return 1 - torch.stack(dice_scores).mean()


class DETRLoss:
    def __init__(self, use_segmentation=False):
        self.matcher = HungarianMatcher(
            class_weight=cfg.COST_CLASS,
            bbox_weight=cfg.COST_BBOX,
        )
        self.use_segmentation = use_segmentation

    def loss(self, pred_logits, pred_boxes, targets, pred_seg=None):
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

            # targets: (image, mask, bbox, label)
            target = {
                "labels": targets[i][3].unsqueeze(0),
                "boxes":  targets[i][2].unsqueeze(0),
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

        detection_loss = (
            total_cls_loss
            + total_bbox_loss
            + cfg.PRIOR_LOSS_WEIGHT * total_prior_loss
        )
        
        # ── segmentation loss ───────────────────────────────────────────────
        seg_loss = 0.0
        if self.use_segmentation and pred_seg is not None:
            for i in range(batch_size):
                # targets[i][1] is the mask (H, W) with class indices
                seg_pred = pred_seg[i].unsqueeze(0)  # (1, C, H, W)
                seg_target = targets[i][1].long().unsqueeze(0)  # (1, H, W)
                
                # Resize prediction to match target size
                seg_pred = F.interpolate(seg_pred, size=(seg_target.shape[1], seg_target.shape[2]), 
                                        mode='bilinear', align_corners=False)
                
                # BCE loss (without weight, handle class imbalance via Dice)
                bce_loss = F.cross_entropy(seg_pred, seg_target)
                
                # Dice loss
                d_loss = dice_loss(seg_pred, seg_target)
                
                seg_loss += (bce_loss + d_loss)
            
            seg_loss = seg_loss / batch_size
        
        # Combine losses
        if self.use_segmentation and pred_seg is not None:
            total_loss = detection_loss + cfg.SEG_LOSS_WEIGHT * seg_loss
            return total_loss, detection_loss, seg_loss
        
        return detection_loss

import torch
from scipy.optimize import linear_sum_assignment


class HungarianMatcher:
    def __init__(self, class_weight=1.0, bbox_weight=5.0):
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight

    def match(self, pred_logits, pred_boxes, targets):
        """
        pred_logits: (num_queries, num_classes)
        pred_boxes: (num_queries, 4)
        targets: dict with 'labels' and 'boxes'
        """

        # classification cost (negative log prob)
        prob = pred_logits.softmax(-1)
        tgt_ids = targets["labels"]

        cost_class = -prob[:, tgt_ids]

        # bbox L1 cost
        tgt_boxes = targets["boxes"]
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)

        # total cost
        C = self.class_weight * cost_class + self.bbox_weight * cost_bbox
        C = C.detach().cpu().numpy()

        indices = linear_sum_assignment(C)

        return indices
import torch
from models.detr import DETR
from utils.loss import DETRLoss
import configs.config as cfg

model     = DETR(num_classes=cfg.NUM_CLASSES, num_queries=cfg.NUM_QUERIES, use_segmentation=True)
criterion = DETRLoss(use_segmentation=True)

x = torch.randn(2, 2, cfg.IMG_SIZE, cfg.IMG_SIZE)
logits, boxes, seg_mask = model(x)

# Create dummy targets: (image, mask, bbox, label)
targets = [
    (x[0], torch.randint(0, 3, (256, 256)).float(), torch.tensor([0.2, 0.3, 0.5, 0.6]), torch.tensor(1)),
    (x[1], torch.randint(0, 3, (256, 256)).float(), torch.tensor([0.1, 0.2, 0.4, 0.5]), torch.tensor(0)),
]

loss, det_loss, seg_loss = criterion.loss(logits, boxes, targets, seg_mask)

print("Total loss:", loss.item())
print("Detection loss:", det_loss.item())
print("Segmentation loss:", seg_loss.item())
assert loss.item() > 0
print("OK")

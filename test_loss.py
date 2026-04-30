import torch
from models.detr import DETR
from utils.loss import DETRLoss
import configs.config as cfg

model     = DETR(num_classes=cfg.NUM_CLASSES, num_queries=cfg.NUM_QUERIES)
criterion = DETRLoss()

x = torch.randn(2, 2, cfg.IMG_SIZE, cfg.IMG_SIZE)
logits, boxes = model(x)

targets = [
    (torch.tensor([0.2, 0.3, 0.5, 0.6]), torch.tensor(1)),
    (torch.tensor([0.1, 0.2, 0.4, 0.5]), torch.tensor(0)),
]

loss = criterion.loss(logits, boxes, targets)

print("Loss:", loss.item())
assert loss.item() > 0
print("OK")

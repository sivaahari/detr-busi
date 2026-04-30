import torch
from models.detr import DETR
import configs.config as cfg

model = DETR(num_classes=cfg.NUM_CLASSES, num_queries=cfg.NUM_QUERIES)

x = torch.randn(2, 2, cfg.IMG_SIZE, cfg.IMG_SIZE)

class_logits, bbox = model(x)

print("Class logits shape:", class_logits.shape)  # (2, 100, 3)
print("BBox shape:        ", bbox.shape)           # (2, 100, 4)
assert class_logits.shape == (2, cfg.NUM_QUERIES, cfg.NUM_CLASSES)
assert bbox.shape          == (2, cfg.NUM_QUERIES, 4)
print("OK")

import torch
from models.detr import DETR
import configs.config as cfg

# Test with segmentation enabled
model = DETR(num_classes=cfg.NUM_CLASSES, num_queries=cfg.NUM_QUERIES, use_segmentation=True)

x = torch.randn(2, 2, cfg.IMG_SIZE, cfg.IMG_SIZE)

class_logits, bbox, seg_mask = model(x)

print("Class logits shape:", class_logits.shape)  # (2, 100, 3)
print("BBox shape:        ", bbox.shape)           # (2, 100, 4)
print("Seg mask shape:    ", seg_mask.shape)        # (2, 3, 256, 256)
assert class_logits.shape == (2, cfg.NUM_QUERIES, cfg.NUM_CLASSES)
assert bbox.shape          == (2, cfg.NUM_QUERIES, 4)
assert seg_mask.shape      == (2, cfg.NUM_CLASSES, cfg.IMG_SIZE, cfg.IMG_SIZE)
print("OK")

# Test without segmentation
model_no_seg = DETR(num_classes=cfg.NUM_CLASSES, num_queries=cfg.NUM_QUERIES, use_segmentation=False)
class_logits, bbox = model_no_seg(x)
print("Without segmentation: OK")

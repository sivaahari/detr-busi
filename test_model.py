import torch
from models.detr import DETR

model = DETR(num_classes=2, num_queries=100)

x = torch.randn(2, 2, 256, 256)  # batch size 2

class_logits, bbox = model(x)

print("Class logits shape:", class_logits.shape)
print("BBox shape:", bbox.shape)
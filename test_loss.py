import torch
from models.detr import DETR
from utils.loss import DETRLoss

model = DETR(num_classes=2)

criterion = DETRLoss()

# dummy batch
x = torch.randn(2, 2, 256, 256)

logits, boxes = model(x)

# fake targets (1 object per image)
targets = [
    (torch.tensor([0.2, 0.3, 0.5, 0.6]), torch.tensor(1)),
    (torch.tensor([0.1, 0.2, 0.4, 0.5]), torch.tensor(0)),
]

loss = criterion.loss(logits, boxes, targets)

print("Loss:", loss.item())
import torch
from datasets.busi import BUSIDataset
from models.detr import DETR


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BUSIDataset("data/BUSI")

    model = DETR(num_classes=3).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    ious = []

    for i in range(len(dataset)):
        image, gt_bbox, label = dataset[i]

        with torch.no_grad():
            logits, boxes = model(image.unsqueeze(0).to(device))

        probs = torch.softmax(logits[0], dim=-1)
        scores, _ = probs[:, :2].max(dim=1)

        best_idx = torch.argmax(scores)

        pred_box = boxes[0][best_idx].cpu().numpy()
        gt_box = gt_bbox.numpy()

        iou = compute_iou(pred_box, gt_box)
        ious.append(iou)

    avg_iou = sum(ious) / len(ious)

    print(f"Average IoU: {avg_iou:.4f}")


if __name__ == "__main__":
    evaluate()
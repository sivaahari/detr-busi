import os
import torch
import cv2
from datasets.busi import BUSIDataset
from models.detr import DETR
from utils.visualize import visualize_prediction
import configs.config as cfg


def run_inference(split: str = "test", num_samples: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset     = BUSIDataset(split=split)
    num_samples = min(num_samples, len(dataset))

    model = DETR(num_classes=cfg.NUM_CLASSES, num_queries=cfg.NUM_QUERIES).to(device)
    model.load_state_dict(
        torch.load(cfg.SAVE_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    os.makedirs("outputs", exist_ok=True)

    for i in range(num_samples):
        image, gt_bbox, label = dataset[i]

        with torch.no_grad():
            logits, boxes = model(image.unsqueeze(0).to(device))

        output    = visualize_prediction(image, logits[0], boxes[0], gt_bbox)
        save_path = f"outputs/result_{i}.png"
        cv2.imwrite(save_path, output)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    run_inference(split="test", num_samples=20)

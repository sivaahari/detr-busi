import torch
import cv2
import os
from datasets.busi import BUSIDataset
from models.detr import DETR
from utils.visualize import visualize_prediction


def run_inference(num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BUSIDataset("data/BUSI")

    model = DETR(num_classes=3).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    os.makedirs("outputs", exist_ok=True)

    for i in range(num_samples):
        image, bbox, label = dataset[i]

        with torch.no_grad():
            logits, boxes = model(image.unsqueeze(0).to(device))

        output = visualize_prediction(image, logits[0], boxes[0])

        save_path = f"outputs/result_{i}.png"
        cv2.imwrite(save_path, output)

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    run_inference(20)
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from datasets.busi import BUSIDataset
from models.detr import DETR
from utils.visualize import visualize_prediction, visualize_segmentation
import configs.config as cfg


def run_inference(split: str = "test", num_samples: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset     = BUSIDataset(split=split)
    num_samples = min(num_samples, len(dataset))

    model = DETR(num_classes=cfg.NUM_CLASSES, num_queries=cfg.NUM_QUERIES, use_segmentation=True).to(device)
    model.load_state_dict(
        torch.load(cfg.SAVE_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    os.makedirs("outputs", exist_ok=True)

    for i in range(num_samples):
        # Dataset returns: image, mask, bbox, label
        image, mask, gt_bbox, label = dataset[i]

        with torch.no_grad():
            logits, boxes, seg_mask = model(image.unsqueeze(0).to(device))
        
        # Resize segmentation to match image size
        seg_mask = F.interpolate(seg_mask, size=(image.shape[1], image.shape[2]), 
                                 mode='bilinear', align_corners=False)

        # Visualize detection
        det_output = visualize_prediction(image, logits[0], boxes[0], gt_bbox)
        
        # Visualize segmentation
        seg_output = visualize_segmentation(image, seg_mask[0], mask, label)
        
        # Combine both visualizations
        combined = np.hstack([det_output, seg_output])
        
        save_path = f"outputs/result_{i}.png"
        cv2.imwrite(save_path, combined)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    run_inference(split="test", num_samples=20)

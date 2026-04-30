import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class BUSIDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        
        for label, cls in enumerate(["benign", "malignant"]):
            folder = os.path.join(root_dir, cls)
            
            for file in os.listdir(folder):
                if file.endswith(".png") and "_mask" not in file:
                    img_path = os.path.join(folder, file)
                    mask_path = img_path.replace(".png", "_mask.png")
                    
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        bbox = self.mask_to_bbox(mask)
        
        return image, bbox, label

    def mask_to_bbox(self, mask):
        coords = np.where(mask > 0)
        
        if len(coords[0]) == 0:
            return [0, 0, 0, 0]
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return [x_min, y_min, x_max, y_max]
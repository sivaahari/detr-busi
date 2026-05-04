import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import configs.config as cfg


class BUSIDataset(Dataset):
    """
    BUSI dataset with stratified train/val/test splits and train-time augmentation.

    split: "train" | "val" | "test"
    """

    def __init__(self, root_dir: str = None, split: str = "train"):
        if root_dir is None:
            root_dir = cfg.DATA_ROOT

        self.split    = split
        self.img_size = cfg.IMG_SIZE

        # ── collect all samples ──────────────────────────────────────────────
        all_samples = []
        for label, cls in enumerate(["benign", "malignant"]):
            folder = os.path.join(root_dir, cls)
            for file in sorted(os.listdir(folder)):
                if file.endswith(".png") and "_mask" not in file:
                    img_path  = os.path.join(folder, file)
                    mask_path = img_path.replace(".png", "_mask.png")
                    if os.path.exists(mask_path):
                        all_samples.append((img_path, mask_path, label))

        # ── stratified split ─────────────────────────────────────────────────
        labels = [s[2] for s in all_samples]
        val_ratio_of_trainval = cfg.VAL_SPLIT / (cfg.TRAIN_SPLIT + cfg.VAL_SPLIT)

        train_val, test = train_test_split(
            all_samples,
            test_size=cfg.TEST_SPLIT,
            stratify=labels,
            random_state=cfg.SEED,
        )
        labels_tv = [s[2] for s in train_val]
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_of_trainval,
            stratify=labels_tv,
            random_state=cfg.SEED,
        )

        self.samples = {"train": train, "val": val, "test": test}[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        image = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask  = cv2.resize(mask,  (self.img_size, self.img_size))

        # ── augmentation (train only) ────────────────────────────────────────
        if self.split == "train":
            image, mask = self._augment(image, mask)

        # ── bbox from mask ───────────────────────────────────────────────────
        bbox = self._mask_to_bbox_normalized(mask)

        # ── image → float, normalize ─────────────────────────────────────────
        image = image.astype(np.float32) / 255.0

        # ── Sobel edge channel ────────────────────────────────────────────────
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edges  = np.sqrt(sobelx ** 2 + sobely ** 2)
        edges  = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)

        image = np.stack([image, edges], axis=0)

        # Normalize mask to [0, 1] binary
        mask_binary = (mask > 0).astype(np.float32)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask_binary, dtype=torch.float32),
            torch.tensor(bbox,  dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _mask_to_bbox_normalized(self, mask):
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        h, w = mask.shape
        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())
        return [x_min / w, y_min / h, x_max / w, y_max / h]

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        """Apply consistent geometric and photometric augmentations."""
        # horizontal flip
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask  = cv2.flip(mask,  1)

        # vertical flip
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask  = cv2.flip(mask,  0)

        # brightness / contrast shift
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta  = random.uniform(-20, 20)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(
                np.uint8
            )

        return image, mask

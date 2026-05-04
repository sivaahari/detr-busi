from datasets.busi import BUSIDataset

for split in ("train", "val", "test"):
    ds = BUSIDataset(split=split)
    print(f"{split:6}: {len(ds)} samples")

image, mask, bbox, label = BUSIDataset(split="train")[0]
print("\nImage shape:", image.shape)   # (2, 256, 256)
print("Mask shape: ", mask.shape)     # (256, 256)
print("BBox:       ", bbox)
print("Label:      ", label)
assert image.shape == (2, 256, 256)
assert mask.shape == (256, 256)
print("OK")

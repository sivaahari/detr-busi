from datasets.busi import BUSIDataset

dataset = BUSIDataset("data/BUSI")

image, bbox, label = dataset[0]

print("Image shape:", image.shape)  # should be (2, 256, 256)
print("BBox:", bbox)
print("Label:", label)
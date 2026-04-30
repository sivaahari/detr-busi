from datasets.busi import BUSIDataset

dataset = BUSIDataset("data/BUSI")

print("Total samples:", len(dataset))

image, bbox, label = dataset[0]

print("Image shape:", image.shape)
print("BBox:", bbox)
print("Label:", label)
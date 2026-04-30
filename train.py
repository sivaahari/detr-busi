import torch
from torch.utils.data import DataLoader
from datasets.busi import BUSIDataset
from models.detr import DETR
from utils.loss import DETRLoss
from tqdm import tqdm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = BUSIDataset("data/BUSI")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model
    model = DETR(num_classes=3).to(device)

    # Loss
    criterion = DETRLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 15

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(dataloader)

        for images, bboxes, labels in loop:
            images = images.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)

            # Forward pass
            logits, boxes = model(images)

            # Prepare targets
            targets = []
            for i in range(len(images)):
                targets.append((bboxes[i], labels[i]))

            # Compute loss
            loss = criterion.loss(logits, boxes, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")

        # Save model after each epoch
        torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()
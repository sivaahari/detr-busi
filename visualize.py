import torch
import cv2
from datasets.busi import BUSIDataset
from models.detr import DETR
from utils.visualize import visualize_prediction


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BUSIDataset("data/BUSI")

    model = DETR(num_classes=3).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    image, bbox, label = dataset[0]

    with torch.no_grad():
        logits, boxes = model(image.unsqueeze(0).to(device))

    output = visualize_prediction(image, logits[0], boxes[0])

    cv2.imshow("Prediction", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
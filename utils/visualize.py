import torch
import cv2
import numpy as np


def visualize_prediction(image, pred_logits, pred_boxes, threshold=0.7):
    """
    image: (2, H, W)
    pred_logits: (num_queries, num_classes)
    pred_boxes: (num_queries, 4)
    """

    image = image[0].cpu().numpy()  # use original channel
    image = (image * 255).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    probs = torch.softmax(pred_logits, dim=-1)

    for i in range(pred_boxes.shape[0]):
        cls = torch.argmax(probs[i]).item()
        score = torch.max(probs[i]).item()

        # ignore no-object (class 2)
        if cls == 2 or score < threshold:
            continue

        h, w = image.shape[:2]

        x_min, y_min, x_max, y_max = pred_boxes[i]

        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)

        color = (0, 255, 0) if cls == 0 else (0, 0, 255)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    return image
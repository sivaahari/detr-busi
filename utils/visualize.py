import torch
import cv2
import numpy as np


def visualize_prediction(image, pred_logits, pred_boxes, top_k=5):
    image = image[0].cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    probs = torch.softmax(pred_logits, dim=-1)

    # get top-k predictions (ignoring no-object)
    scores, classes = probs[:, :2].max(dim=1)

    topk_scores, topk_indices = torch.topk(scores, top_k)

    h, w = image.shape[:2]

    for idx in topk_indices:
        cls = classes[idx].item()
        score = scores[idx].item()

        x_min, y_min, x_max, y_max = pred_boxes[idx]

        x_min = int(x_min * w)
        y_min = int(y_min * h)
        x_max = int(x_max * w)
        y_max = int(y_max * h)

        color = (0, 255, 0) if cls == 0 else (0, 0, 255)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        cv2.putText(
            image,
            f"{cls}:{score:.2f}",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    return image
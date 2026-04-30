\# DETR for Breast Ultrasound Nodule Detection (BUSI)



\## Overview



This project implements a lightweight Detection Transformer (DETR) model for detecting breast nodules in ultrasound images using the BUSI dataset.



The goal is to adapt transformer-based object detection to \*\*medical imaging challenges\*\*, including:



\* Speckle noise

\* Low contrast boundaries

\* Irregular lesion shapes



\---



\## Key Contributions



\* Lightweight DETR (ResNet18 backbone)

\* Edge-based structural prior (Sobel filtering)

\* Geometric prior (aspect ratio + width constraints)

\* Hungarian matching for set-based detection

\* Custom loss design for medical object detection



\---



\## Architecture



```

Input (2-channel: Image + Edge)

&#x20;       ↓

ResNet18 Backbone

&#x20;       ↓

Transformer Encoder-Decoder

&#x20;       ↓

Object Queries (100)

&#x20;       ↓

Prediction Heads (Class + BBox)

```



\---



\## Results



\### Example Predictions



!\[Result 1](samples/result\_0.png)

!\[Result 2](samples/result\_1.png)

!\[Result 3](samples/result\_2.png)



\---



\## Observations



\* Model successfully localizes nodules in most cases

\* Bounding boxes are reasonably tight

\* Confidence scores remain low due to:



&#x20; \* Class imbalance (1 object vs many no-object queries)

&#x20; \* Limited dataset size



\---



\## Limitations



\* Low confidence calibration

\* Single-object assumption

\* No multi-scale feature handling (yet)



\---



\## Future Work



\* Add Deformable DETR (multi-scale attention)

\* Improve confidence calibration

\* Introduce IoU-based evaluation metrics

\* Expand dataset for better generalization



\---



\## Tech Stack



\* PyTorch

\* OpenCV

\* NumPy



\---



\## Author



Sivaa




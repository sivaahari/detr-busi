# Project Components — Plain English Guide

This document explains every part of the project from the ground up. After reading this, you should be able to explain the whole system to someone without any deep learning background.

---

## The Big Picture

We are building a system that looks at a breast ultrasound image and draws a box around any nodule it finds, then labels that nodule as either **benign** (harmless) or **malignant** (potentially cancerous).

The model is called **DETR** — Detection Transformer. The key idea is that instead of scanning the image in many different ways (like older detectors do), DETR reads the whole image at once using a Transformer (the same technology behind ChatGPT) and directly predicts where the nodule is.

---

## 1. The Dataset — `datasets/busi.py`

**What it is:** The BUSI (Breast Ultrasound Images) dataset. It contains grayscale ultrasound photos of breasts, paired with segmentation masks — images where the pixels of the nodule are highlighted in white.

**What we do with it:**
- We load each image and its mask
- From the mask, we find the bounding box (the smallest rectangle that fits around the nodule)
- We split all images into three groups: **70% for training**, **15% for validation**, **15% for testing**
- The split is *stratified* — meaning each group has a proportional mix of benign and malignant cases, not accidental imbalance

**Augmentation (training only):**
- We randomly flip images left-right and up-down
- We randomly brighten or darken them
- We do this to make the model more robust — it should detect nodules regardless of orientation or lighting. The mask gets flipped too so the bounding box stays accurate.

**The two-channel input:**
Instead of feeding just the grayscale image, we also compute a **Sobel edge map** — an image that highlights sharp boundaries and contours. Ultrasound images are blurry and noisy, so edges help the model find nodule boundaries.
- Channel 0: normalized grayscale image
- Channel 1: Sobel edge map

Both channels together form a `(2, 256, 256)` tensor (2 images stacked, each 256×256 pixels).

---

## 2. The Config — `configs/config.py`

**What it is:** A single file that holds every setting for the entire project — image size, number of epochs, learning rates, file paths, everything.

**Why it matters:** Instead of hardcoded numbers scattered across files, every parameter is defined in one place. To change the batch size, you change one line here.

Key settings:
- `NUM_CLASSES = 3` — benign, malignant, and "no object" (background)
- `NUM_QUERIES = 100` — the model makes 100 simultaneous predictions, of which usually only 1 is real
- `EPOCHS = 50` — how many full passes over the training data
- `LR = 1e-4` — learning rate for the transformer and prediction heads
- `LR_BACKBONE = 1e-5` — learning rate for the backbone (10× lower, because it's pre-trained)
- `NO_OBJ_WEIGHT = 0.1` — how much we penalize the model for getting the 99 "no object" queries wrong

---

## 3. The Model — `models/detr.py`

This is the brain of the system. It takes in an image and outputs predicted boxes and class labels. It has four main stages:

### Stage 1: The Backbone (ResNet-18)

**What it does:** Extracts features from the image — essentially learning what visual patterns look like.

Think of it like this: when you look at an image, your brain first detects simple things (edges, textures) and then builds up to complex things (shapes, objects). ResNet-18 does the same thing across 4 "layers" (called layer1 through layer4).

- layer1 → low-level features (edges, textures) — spatial size: 64×64
- layer2 → mid-level features — spatial size: 32×32
- layer3 → higher-level features — spatial size: 16×16
- layer4 → high-level semantic features — spatial size: 8×8

The backbone is **pre-trained** on ImageNet (a huge image dataset), so it already understands visual structure before we even train it on medical images. We modify only the first layer to accept 2 channels instead of 3 (RGB → grayscale+edges).

### Stage 2: Multi-Scale Fusion (`fusion_conv`)

**What it does:** Combines the 4 layers of features into a single rich representation.

The 4 feature maps are at different spatial sizes (64×64 down to 8×8). We:
1. Project all 4 to the same number of channels (256) using 1×1 convolutions
2. Resize all to the same spatial size (64×64)
3. Concatenate them all together
4. Pass through a learned 1×1 convolution to mix them into one `(256, 64, 64)` map

**Why not just sum them?** Earlier we were summing, but summing loses information — if two feature maps disagree, they cancel out. Concatenation + learned mixing lets the model decide how to weight each scale.

### Stage 3: The Transformer

**What it does:** This is where the magic happens. It lets every part of the image "talk to" every other part simultaneously.

Think of it like a room full of people where everyone can hear everyone else at once, rather than passing notes one by one (like older RNN models did).

It has two parts:
- **Encoder:** Reads the entire feature map and lets every spatial location attend to every other. After this, each location knows the global context of the whole image.
- **Decoder:** Takes 100 learned "object queries" (think of them as 100 detectives, each assigned to look for an object) and has each one attend to the encoder's output to find what it's looking for.

The result is 100 output vectors, one per query, each representing a potential detection.

### Stage 4: Prediction Heads

**What they do:** Turn the 100 query vectors into actual predictions.

- **Class head (`class_embed`):** A linear layer that outputs 3 scores per query — probability of being benign, malignant, or no-object
- **Box head (`bbox_embed`):** A small MLP that outputs 4 numbers per query — `[x_min, y_min, x_max, y_max]` all normalized between 0 and 1. Sigmoid activation ensures the values stay in range.

Final output:
- `class_logits`: shape `(batch, 100, 3)` — 100 class predictions per image
- `bbox`: shape `(batch, 100, 4)` — 100 box predictions per image

---

## 4. The Hungarian Matcher — `utils/matcher.py`

**What it does:** Figures out which of the 100 predicted boxes corresponds to the real nodule in the image.

This is the core of DETR's loss computation. We have 100 predictions and 1 ground truth — we need to assign the "best" prediction to that ground truth, and label the other 99 as "no object."

**How it works:**
1. Compute a cost for every possible pairing: how wrong would it be to say "prediction X corresponds to ground truth Y"?
   - Cost includes: how wrong the class prediction is + how far off the box location is
2. Use the **Hungarian Algorithm** (an optimal assignment algorithm) to find the minimum-cost pairing

Think of it like a job assignment problem: you have 100 candidates and 1 job, and you want to find which candidate is the best fit.

---

## 5. The Loss Function — `utils/loss.py`

**What it does:** Measures how wrong the model is, so the optimizer can improve it.

Three components, added together:

### Classification Loss
Cross-entropy loss over all 100 queries. For 99 of them, the target is class 2 (no-object). For the 1 matched query, the target is the true class (benign or malignant).

**The class imbalance fix:** Without weighting, the model learns to always predict "no-object" for everything (99% accuracy by doing nothing useful). We down-weight the no-object class by `0.1` so the model is forced to learn to find the real nodule.

### Bounding Box Loss (L1)
Simple absolute difference between the predicted box coordinates and the ground-truth box, computed only on the one matched query.

### Geometric Prior Loss
This is our domain knowledge injection. Real breast nodules have predictable shapes — they're roughly oval, not wildly elongated. We penalize predictions that have unrealistic aspect ratios (height/width ratio) or unrealistic widths compared to the ground truth.

**Total loss = classification + bbox + 0.5 × geometric prior**

---

## 6. The Training Pipeline — `train.py`

**What it does:** Runs the training loop that improves the model over 50 epochs.

Key design decisions:

### Differential Learning Rates
The backbone (ResNet-18) was pre-trained on ImageNet and already understands visual features. We train it slowly (`LR_BACKBONE = 1e-5`) to fine-tune gently.

The transformer and prediction heads are brand new and need to learn from scratch. We train them faster (`LR = 1e-4`).

### Gradient Clipping
After computing gradients, we clip their magnitude to a maximum of 0.1 before updating weights. This prevents "exploding gradients" — a common instability problem in transformer training where a single bad batch can derail weeks of training.

### Cosine Annealing LR Scheduler
The learning rate starts at its full value and gradually decreases following a cosine curve down to near-zero by epoch 50. This helps the model settle into a good solution without oscillating at the end.

### Validation Loop
After every training epoch, we run the model on the validation set (data it has never trained on) and compute:
- **Val Loss:** Is the model still improving on unseen data?
- **Val mIoU:** How well do predicted boxes overlap with ground truth?

We save the model only when validation loss improves — this is called **best model checkpointing** and prevents saving an overfit model.

### CSV Logging
Every epoch's metrics are saved to `logs/training_log.csv` so you can plot training curves later.

---

## 7. Box Utilities — `utils/box_ops.py`

Small utility functions used across the project:

- **`compute_iou`:** Given two bounding boxes, computes their Intersection over Union (IoU). A score of 1.0 means perfect overlap; 0.0 means no overlap at all.
- **`box_xyxy_to_cxcywh`:** Converts box format from corner coordinates `[x_min, y_min, x_max, y_max]` to center format `[cx, cy, width, height]`
- **`box_cxcywh_to_xyxy`:** The reverse conversion

---

## 8. Evaluation — `evaluate.py`

**What it does:** After training, measures how good the model actually is on the held-out test set.

**For each image:**
1. Run the model → get 100 predictions
2. Pick the query with the highest foreground confidence (highest probability of being benign OR malignant)
3. Compute IoU between the predicted box and the ground-truth box
4. If IoU ≥ 0.5 AND the class matches → **True Positive (TP)**
5. If IoU ≥ 0.5 but wrong class → **False Positive for predicted class, False Negative for real class**
6. If IoU < 0.5 → **False Negative** (missed the nodule)

**Output metrics per class:**
- **IoU:** How well the box overlaps (location quality)
- **Precision:** Of all detections made, what fraction were correct?
- **Recall:** Of all real nodules, what fraction did we find?
- **F1:** Harmonic mean of precision and recall — the overall balance score

---

## 9. Inference & Visualization — `inference.py`, `visualize.py`, `utils/visualize.py`

**What they do:** Run the trained model on images and draw the results.

`utils/visualize.py` draws two boxes on the image:
- **Colored box (green = benign, red = malignant):** The model's prediction, with the class name and confidence score
- **Yellow box:** The ground-truth annotation

This lets you visually inspect whether the model is finding nodules in the right place.

`inference.py` runs this on 20 test images and saves them to the `outputs/` folder.

`visualize.py` runs it on a single image and shows it on screen.

---

## How Everything Connects

```
Raw ultrasound image (PNG)
        ↓
BUSIDataset
  - Resize to 256×256
  - Extract bounding box from mask
  - Add Sobel edge channel
  - Apply augmentation (train only)
        ↓
DETR Model
  - ResNet-18 backbone → 4 feature scales
  - Learned fusion → single feature map
  - Positional encoding added
  - Transformer encoder → global context
  - 100 object queries → transformer decoder
  - Class head → 100 class scores
  - Box head → 100 bounding boxes
        ↓
Training:
  - Hungarian matcher assigns 1 query to real nodule
  - Loss = classification + L1 bbox + geometric prior
  - Backprop → update weights
  - Validate every epoch → save best model
        ↓
Evaluation:
  - Load best model
  - Pick highest-confidence foreground query
  - Compute IoU, Precision, Recall, F1
```

---

## Key Terms Glossary

| Term | Plain English |
|---|---|
| **Bounding Box** | A rectangle drawn around the nodule |
| **Anchor-free detection** | Predicting boxes directly without pre-defined reference boxes |
| **Hungarian Matching** | An algorithm that finds the optimal one-to-one assignment between predictions and ground truths |
| **IoU (Intersection over Union)** | How much two boxes overlap, as a fraction (0 = no overlap, 1 = perfect match) |
| **Epoch** | One complete pass through the entire training dataset |
| **Gradient Clipping** | Capping gradient values to prevent training instability |
| **Class Imbalance** | When one class (no-object) is much more common than others (benign, malignant), causing the model to ignore the rare classes |
| **Precision** | Of the boxes the model drew, how many were actually nodules? |
| **Recall** | Of all real nodules in the images, how many did the model find? |
| **Validation Set** | A separate set of images used to check model performance during training — never used for weight updates |
| **Checkpointing** | Saving the model only when it improves, so you keep the best version |
| **Stratified Split** | Dividing data so each split has a proportional representation of every class |

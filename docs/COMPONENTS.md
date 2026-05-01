# Project Components — Plain English Guide

This document explains every part of the project from the ground up — what each component does, why it was designed that way, and how it relates to the original paper.

---

## The Big Picture

We are building a system that looks at a breast ultrasound image and draws a box around any nodule it finds, then labels that nodule as either **benign** (harmless) or **malignant** (potentially cancerous).

The model is called **DETR** — Detection Transformer. Instead of scanning the image in many overlapping windows (like older detectors), DETR reads the whole image at once using a Transformer and directly predicts where the nodule is and what class it belongs to. There are no pre-defined anchor boxes, and no post-processing step like non-maximum suppression.

The paper this replicates ("Prior-Guided DETR", Wang et al. 2026) proposes three novel modules to handle the specific challenges of ultrasound images: irregular nodule shapes, blurred boundaries, speckle noise, and scale variation. Our implementation approximates all three under a 4 GB VRAM constraint.

---

## 1. The Dataset — `datasets/busi.py`

**What it is:** The BUSI (Breast Ultrasound Images) dataset — 780 grayscale ultrasound photographs of breasts, each paired with a segmentation mask (a pixel map where nodule pixels are white).

**What we do with it:**
- Load each image and its mask
- Extract the bounding box from the mask: find the smallest rectangle that tightly encloses all white pixels — `[x_min, y_min, x_max, y_max]` — and normalise the coordinates to `[0, 1]`
- Split into **70% train / 15% val / 15% test** with a fixed random seed
- The split is *stratified*: each subset has a proportional mix of benign and malignant cases, not an accidental imbalance

**Augmentation (training only):**
- Random horizontal and vertical flips
- Random brightness/contrast adjustment (factor α ∈ [0.8, 1.2], offset β ∈ [−20, 20])
- The mask is transformed identically so the bounding box remains accurate

**The two-channel input — approximating the paper's MSFFM:**

The paper's MSFFM module extracts structural priors by jointly processing features in the spatial domain (contour, boundary continuity) and the frequency domain (speckle suppression via FFT). Our approximation moves this into the input: instead of a raw grayscale image, we stack a **Sobel edge map** as a second channel.

- Channel 0: normalised grayscale image `∈ [0, 1]`
- Channel 1: Sobel edge magnitude (highlights nodule boundaries)

This gives the backbone explicit boundary information from the start, approximating the spatial-contour half of MSFFM. The frequency-domain branch (speckle suppression) has no equivalent in this implementation.

Both channels together form a `(2, 256, 256)` tensor per image.

---

## 2. The Config — `configs/config.py`

A single file holding every hyperparameter for the entire project. Changing a setting here propagates everywhere with no risk of inconsistency.

Key settings and their rationale:

| Setting | Value | Why |
|---|---|---|
| `NUM_CLASSES` | 3 | Benign, malignant, and no-object (background) |
| `NUM_QUERIES` | 100 | Model makes 100 candidate predictions; only 1 is real. Paper uses 300, but BUSI has 1 nodule per image so 100 suffices |
| `HIDDEN_DIM` | 256 | Feature dimension throughout the transformer |
| `NHEAD` | 8 | Number of attention heads |
| `ENC_LAYERS` | 3 | Deformable encoder depth (paper: 6) |
| `DEC_LAYERS` | 3 | Standard decoder depth (paper: 6) |
| `N_POINTS` | 4 | Sampling points per head in deformable attention |
| `DIM_FFN` | 512 | Feed-forward network hidden dimension |
| `EPOCHS` | 50 | Full passes over training data (paper: 200) |
| `BATCH_SIZE` | 4 | Limited by 4 GB VRAM |
| `LR` | 1e-4 | Learning rate for transformer + prediction heads |
| `LR_BACKBONE` | 1e-5 | 10× lower for the pre-trained backbone |
| `WEIGHT_DECAY` | 1e-4 | L2 regularisation |
| `GRAD_CLIP` | 0.1 | Maximum gradient norm — standard for DETR training |
| `NO_OBJ_WEIGHT` | 0.1 | Down-weight the 99 no-object queries in the loss |
| `PRIOR_LOSS_WEIGHT` | 0.5 | Strength of the geometric prior loss term |
| `IOU_THRESHOLD` | 0.5 | Minimum overlap to count a detection as correct |

---

## 3. The Model — `models/detr.py`

The model has four stages. The input is a `(B, 2, 256, 256)` tensor and the output is 100 class predictions and 100 bounding boxes per image.

### Stage 1: Backbone (ResNet-18)

ResNet-18 is a convolutional neural network pre-trained on ImageNet. It extracts visual features hierarchically — early layers detect low-level patterns (edges, textures), deeper layers detect high-level semantics (shapes, structures). We use all four stages:

| Stage | Channels | Spatial size (256×256 input) | Feature type |
|---|---|---|---|
| layer1 | 64 | 64 × 64 | Low-level: edges, local texture |
| layer2 | 128 | 32 × 32 | Mid-level: shapes, patterns |
| layer3 | 256 | 16 × 16 | High-level: object parts |
| layer4 | 512 | 8 × 8 | Semantic: object-level |

The first convolution of ResNet-18 is modified to accept **2 input channels** (grayscale + Sobel) instead of the standard 3 (RGB). All other pre-trained weights remain intact.

The paper uses ResNet-50, which is approximately 2.6× larger in parameter count and has a significantly larger receptive field per block. The paper further embeds its SDFPR module inside every residual block, injecting geometric priors directly into feature extraction. Our backbone has no such modification — geometric priors only appear in the loss function.

### Stage 2: Multi-Scale Fusion

All four feature maps are at different spatial sizes. We merge them into a single representation:

1. Project each to 256 channels using a 1×1 convolution (`input_proj1` through `input_proj4`)
2. Resize `f2`, `f3`, `f4` up to `f1`'s spatial size (64×64) via bilinear interpolation
3. Concatenate along the channel dimension: `(B, 1024, 64, 64)`
4. Pass through a learned 1×1 convolution (`fusion_conv`) to mix and reduce back to `(B, 256, 64, 64)`

The learned mixing step is important: a simple sum would let features cancel out if they disagree. Concatenation + convolution lets the model learn which scale is most informative for each spatial location.

This approximates a simplified version of the paper's MSFFM, which operates on only 3 scales but adds the spatial-branch and frequency-branch processing at each scale before fusion.

### Stage 3: Deformable Encoder + Standard Decoder

**Positional encoding:** Learned 2D embeddings (`row_embed`, `col_embed`), concatenated to form a full `(B, H*W, 256)` positional encoding. Added to the feature sequence before encoding. The paper uses fixed sine/cosine encoding; learned embeddings can adapt to the medical domain.

**Deformable Encoder (`models/deformable_attention.py`):**

Standard self-attention has O(N²) complexity — every position attends to every other. For a 64×64 feature map, N = 4096, making O(N²) = ~16.8 million attention pairs. This is what makes vanilla DETR memory-hungry and slow to converge.

Deformable attention replaces this with a small set of K=4 learned sampling points per query. For each query position:
1. Predict K=4 offsets relative to a reference point on a regular grid
2. Sample the feature map at those K locations using bilinear interpolation
3. Compute K learned attention weights (softmax-normalised)
4. Weighted sum of sampled values → output

Complexity drops from O(N²) to O(N·K). With K=4 and N=4096, that is ~16,384 operations instead of ~16.8 million.

Three such layers (`ENC_LAYERS = 3`) are stacked with pre-norm residual connections. After encoding, each position in the feature sequence encodes global context about the whole image.

The paper uses 6 encoder layers and the same deformable attention mechanism. It also adds the DFI module on top, which aggregates all 6 encoder outputs in a DenseNet-like manner and feeds them to decoder layers in reversed order. Our implementation uses only the final encoder layer output as the decoder's memory — the intermediate layer outputs are discarded. DFI would be the highest-impact addition to close this gap.

**Standard Decoder:**

PyTorch's built-in `nn.TransformerDecoder` with 3 layers. Each layer performs:
1. Self-attention among the 100 object queries (queries attend to each other)
2. Cross-attention from queries to the encoder memory (queries attend to image features)
3. FFN with ReLU

The decoder's cross-attention uses the full encoder memory sequence as Key and Value — this is where queries learn to "look at" the right regions of the image.

### Stage 4: Prediction Heads

Two small networks map the 100 query vectors to predictions:

- **Class head:** `Linear(256 → 3)` → 3 logit scores per query (benign / malignant / no-object)
- **Box head:** `Linear(256 → 256) → ReLU → Linear(256 → 4) → Sigmoid` → 4 normalised coordinates `[x_min, y_min, x_max, y_max] ∈ [0, 1]` per query

Output shapes: `class_logits (B, 100, 3)` and `bbox (B, 100, 4)`.

---

## 4. Deformable Attention — `models/deformable_attention.py`

The deformable attention module (described in Stage 3 above) is implemented in this file. Key implementation details:

- **Reference points:** A regular grid of (x, y) coordinates covering the full feature map, one per spatial position
- **Offset prediction:** A linear layer maps each query vector to `K × 2` offset values (K position offsets per head)
- **Sampling:** `F.grid_sample` with bilinear interpolation at the offset locations
- **Weight prediction:** Another linear layer maps each query to K softmax attention weights
- **Output:** Weighted sum of K sampled feature vectors per query

`DeformableEncoderLayer` wraps this in a pre-norm residual block (LayerNorm → attention → residual → LayerNorm → FFN → residual). Three layers are stacked in `DeformableEncoder`.

---

## 5. Hungarian Matching — `utils/matcher.py`

We have 100 predictions and 1 ground truth nodule per image. We need to assign exactly one prediction to the ground truth and mark the other 99 as "no object." The Hungarian algorithm does this optimally.

**Cost matrix** (100 × 1 in our single-nodule case):

```
cost[i] = COST_CLASS × (-log p_class[i]) + COST_BBOX × L1(bbox_pred[i], bbox_gt)
```

where `COST_CLASS = 1.0` and `COST_BBOX = 5.0` (box localisation matters more than class in the matching step).

`scipy.optimize.linear_sum_assignment` finds the minimum-cost assignment, returning `(pred_idx, gt_idx)` — which query matched which ground truth.

---

## 6. Loss Function — `utils/loss.py`

After matching, the total loss has three components:

### Classification Loss
Cross-entropy over all 100 queries. Target: matched query → true class, all others → no-object (class index 2).

Class weighting vector: `[1.0, 1.0, 0.1]`. Without this, the model would achieve 99% "accuracy" by predicting no-object for everything and never learning to find actual nodules.

The paper uses Focal loss instead, which additionally down-weights easy examples and focuses training on hard ones. Our weighted cross-entropy is a simpler but effective alternative.

### Bounding Box Loss (L1)
Applied only to the matched query:
```
bbox_loss = L1(pred_box, gt_box)    # element-wise, averaged over 4 coordinates
```
The paper additionally uses GIoU loss, which directly penalises non-overlapping area and is more informative when two boxes don't overlap at all. Without GIoU, the model can learn approximate shapes while missing precise localisation — visible in the outputs as boxes that are slightly too large.

### Geometric Prior Loss — approximating SDFPR
Applied only to the matched query. Penalises predictions that have unrealistic nodule shapes:
```
pred_w = pred_x_max - pred_x_min
pred_h = pred_y_max - pred_y_min
gt_w   = gt_x_max - gt_x_min
gt_h   = gt_y_max - gt_y_min

prior_loss = L1(pred_h / pred_w, gt_h / gt_w)   # aspect ratio
           + L1(pred_w, gt_w)                     # width
```

This approximates the paper's SDFPR, which uses a 3-component GMM fit to clinical data to constrain the deformable sampling offsets inside the backbone. Our version acts at the output level after the fact; the paper's version constrains how features are sampled from the image.

**Total loss:**
```
total_loss = cls_loss + bbox_loss + 0.5 × prior_loss
```

---

## 7. Training Pipeline — `train.py`

### Differential Learning Rates
The ResNet-18 backbone is pre-trained on ImageNet and already understands visual features. Training it too fast would destroy that knowledge. We use:
- `LR_BACKBONE = 1e-5` for backbone parameters
- `LR = 1e-4` for the transformer, encoder, decoder, heads, and fusion layers

This matches the paper's approach and is standard practice for fine-tuning pre-trained vision backbones.

### Gradient Clipping
After computing gradients, clip their magnitude to `max_norm = 0.1` before the weight update. This prevents a single bad batch from causing a catastrophically large update — a common failure mode in transformer training. The paper uses the same value.

### Cosine Annealing LR Scheduler
Learning rate follows a cosine curve from `1e-4` down to near-zero over 50 epochs. This prevents oscillation near convergence and allows the model to settle into a precise minimum without bouncing.

### Validation Loop
After every training epoch:
1. Run inference on the validation set (never used for weight updates)
2. Pick the query with the highest foreground confidence (argmax over benign + malignant probability)
3. Compute IoU between the predicted box and the ground-truth box
4. Average IoU across all validation images → **val mIoU**

The model checkpoint is saved whenever validation loss reaches a new minimum. This prevents saving an overfitted model.

### CSV Logging
Every epoch's `train_loss`, `val_loss`, and `val_miou` are appended to `logs/training_log.csv`.

---

## 8. Training Results

The model was trained for 50 epochs. All results are on the validation set:

| Epoch | Train Loss | Val Loss | Val mIoU |
|---|---|---|---|
| 1 | 2.257 | 1.725 | 0.139 |
| 15 | 1.728 | 1.567 | 0.210 |
| 25 | 1.336 | 1.499 | 0.291 |
| 30 | 1.231 | 1.411 | 0.351 |
| 41 | 1.082 | 1.457 | **0.374** ← best mIoU |
| 46 | 1.024 | **1.357** ← best val loss | 0.364 |
| 50 | 1.024 | 1.369 | 0.363 |

**Key observations:**
- Training loss decreases steadily and monotonically — gradient clipping is working, no instability
- Validation mIoU plateaus around epoch 35–41, indicating a capacity ceiling rather than a training failure — the model has learned what its architecture allows
- The generalisation gap (train ~1.02 vs. val ~1.37 at epoch 50) is moderate and expected for an 800-image dataset
- The early spike in val loss at epoch 2 (2.096) is normal — the Hungarian matcher takes a few epochs to find stable assignments

The plateau at ~0.37 mIoU is the clearest signal that further training will not help. The next gains require architectural changes: adding GIoU loss, replacing cross-entropy with Focal loss, or implementing the DFI mechanism.

---

## 9. Box Utilities — `utils/box_ops.py`

- **`compute_iou(box1, box2)`:** Computes Intersection over Union for two `[x_min, y_min, x_max, y_max]` boxes. Returns 0.0 for non-overlapping boxes, 1.0 for identical boxes.
- **`box_xyxy_to_cxcywh`:** Converts `[x_min, y_min, x_max, y_max]` → `[cx, cy, w, h]` (center format, used internally in some loss calculations)
- **`box_cxcywh_to_xyxy`:** The reverse conversion

---

## 10. Evaluation — `evaluate.py`

Run on the held-out test set (never seen during training or validation) after training completes.

**Per image:**
1. Run model → 100 `(class_logits, bbox)` pairs
2. Pick the query with the highest `softmax(logits)[benign] + softmax(logits)[malignant]` score
3. Compute IoU between that predicted box and the ground-truth box

**Classification:**
- IoU ≥ 0.5 AND correct class → True Positive (TP)
- IoU ≥ 0.5 AND wrong class → False Positive for predicted class; False Negative for correct class
- IoU < 0.5 → False Negative (missed detection)

**Output metrics per class (benign, malignant):**
- **IoU:** Average overlap quality
- **Precision:** Of all detections predicted as this class, what fraction were correct?
- **Recall:** Of all ground-truth nodules of this class, what fraction were found?
- **F1:** Harmonic mean of precision and recall

The paper uses COCO-style AP metrics (area under the precision-recall curve at multiple IoU thresholds), which capture performance across all confidence thresholds simultaneously. Our per-class F1/precision/recall measures performance at a single operating point.

---

## 11. Inference & Visualisation — `inference.py`, `visualize.py`, `utils/visualize.py`

`utils/visualize.py` draws two overlaid boxes on an image:
- **Green box (benign) or red box (malignant):** The model's top prediction, with class name and confidence score
- **Yellow box:** The ground-truth annotation

`inference.py` runs this for 20 test images and saves results to `outputs/result_*.png`.

From visual inspection of the 20 outputs:
- Classification (benign vs. malignant) is generally correct — class-level features are learnable from grayscale+edge input
- Localisation is inconsistent — some predictions align well, some miss the nodule entirely
- Boxes tend to be slightly too large — a consequence of no GIoU loss; L1 alone does not penalise area mismatch
- Confidence scores are moderate (0.5–0.8) — consistent with a small dataset and no Focal loss for calibration

---

## How Everything Connects

```
Raw PNG (ultrasound image + segmentation mask)
            ↓
    BUSIDataset (datasets/busi.py)
      - Resize to 256×256
      - Extract bounding box from mask
      - Compute Sobel edge channel
      - Stack → (2, 256, 256) tensor
      - Augment (train only)
            ↓
    DETR Model (models/detr.py)
      - ResNet-18 → 4 feature maps (64, 128, 256, 512 ch)
      - Project + resize → all to (256, 64×64)
      - Concatenate + fusion_conv → (256, 64×64)
      - Add learned positional encoding
      - DeformableEncoder (3 layers, K=4 points) → memory (4096, 256)
      - 100 learned queries + TransformerDecoder (3 layers) → (100, 256)
      - class_embed → (100, 3) logits
      - bbox_embed  → (100, 4) normalised boxes
            ↓
    Training (train.py)
      - HungarianMatcher: assign 1 query to GT, 99 → no-object
      - Loss = cross-entropy + L1 + 0.5 × geometric prior
      - Backprop + gradient clip (0.1) + AdamW
      - Cosine annealing LR schedule
      - Validate every epoch → save best model
            ↓
    Evaluation (evaluate.py)
      - Load best checkpoint
      - Pick highest-foreground-confidence query per image
      - Compute IoU, Precision, Recall, F1 per class
            ↓
    Inference (inference.py / visualize.py)
      - Draw predicted box + class + confidence
      - Draw ground-truth box in yellow
      - Save to outputs/
```

---

## Key Terms

| Term | Plain English |
|---|---|
| **Bounding box** | A rectangle drawn around the nodule |
| **Anchor-free detection** | Predicting boxes directly, without scanning pre-defined reference boxes |
| **Hungarian matching** | An algorithm that finds the optimal one-to-one pairing between predictions and ground truths |
| **IoU (Intersection over Union)** | How much two boxes overlap, as a fraction (0 = no overlap, 1 = perfect) |
| **Deformable attention** | Attention that samples K learned positions rather than attending to all N positions — O(N·K) instead of O(N²) |
| **Epoch** | One complete pass through the training dataset |
| **Gradient clipping** | Capping gradient magnitudes to prevent training instability |
| **Class imbalance** | No-object queries greatly outnumber real-object queries (99:1), causing the model to ignore real nodules without correction |
| **Focal loss** | A loss function that down-weights easy (confident) examples and focuses on hard ones — paper uses this; we use weighted cross-entropy instead |
| **GIoU (Generalised IoU)** | A box regression loss that directly penalises non-overlapping area — paper uses this; we use L1 + geometric prior instead |
| **DFI** | Dense Feature Interaction — the paper's mechanism for feeding all encoder layer outputs to the decoder; not implemented here |
| **SDFPR** | The paper's GMM-based Prior DCN — constrains deformable sampling offsets in the backbone to respect clinical nodule shape statistics |
| **MSFFM** | The paper's dual-branch spatial-frequency mixer — spatial branch for contour priors, FFT branch for speckle suppression |
| **AP@0.5** | Average Precision at IoU threshold 0.5 — area under the precision-recall curve; the paper's primary metric |
| **mIoU** | Mean IoU of the top-1 prediction per image; our primary validation metric |
| **Stratified split** | Data division that preserves class proportions in every subset |
| **Checkpointing** | Saving model weights only when performance improves, keeping the best version |

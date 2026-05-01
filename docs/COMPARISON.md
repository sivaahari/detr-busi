# Paper vs. Implementation: Side-by-Side Comparison

## Paper

**Title:** Prior-Guided DETR for Ultrasound Nodule Detection  
**Authors:** Wang et al., Beihang University + Cancer Hospital CAMS  
**Preprint:** arXiv, January 2026  
**Code:** https://github.com/wjj1wjj/Ultrasound-DETR

**Goal:** Detect thyroid and breast nodules in ultrasound images by progressively injecting three forms of prior knowledge — geometric, structural, and interaction-level — into a Deformable-DETR pipeline.

## Our Implementation

A hardware-constrained replication targeting the same task on the BUSI dataset. Trained on a single consumer GPU (RTX 3050, 4 GB VRAM). The three novel modules from the paper are approximated at varying fidelity; the core DETR paradigm is faithfully preserved.

---

## Architecture Comparison

| Component | Paper | Our Implementation |
|---|---|---|
| **Base architecture** | Deformable DETR | Deformable DETR (deformable encoder + standard decoder) |
| **Backbone** | ResNet-50 | ResNet-18 |
| **Input channels** | Standard grayscale ultrasound | 2-channel: grayscale + Sobel edge map |
| **Image resolution** | Not specified (standard ~800px) | 256 × 256 |
| **Encoder layers** | 6 | 3 |
| **Decoder layers** | 6 | 3 |
| **Object queries** | 300 | 100 |
| **Positional encoding** | Sine/cosine fixed | Learned 2D embeddings |
| **Classification loss** | Focal loss | Cross-entropy with no-object weighting |
| **Box regression loss** | L1 + GIoU | L1 only |
| **Domain prior in loss** | — (priors injected into backbone and encoder) | Geometric prior loss (aspect ratio + width L1) |
| **Training epochs** | 200 | 50 |
| **Batch size** | 2 (single RTX 3090, 24 GB) | 4 (RTX 3050, 4 GB) |
| **Learning rate** | 1e-4 | 1e-4 (transformer), 1e-5 (backbone) |
| **Optimizer** | AdamW | AdamW |
| **Gradient clipping** | max\_norm = 0.1 | max\_norm = 0.1 |
| **LR scheduler** | Not specified | Cosine annealing |
| **Augmentation** | Not detailed | Horizontal/vertical flip, brightness/contrast |
| **Dataset** | BUSI + 3 thyroid datasets (13,308 total images) | BUSI only (780 images) |
| **BUSI split** | 397 train / 136 val / 132 test (normal class excluded) | 546 train / 117 val / 117 test (all classes, stratified) |
| **Evaluation metrics** | AP, AP@0.5, AP@0.75, APs, APm, APl, AP@0.5-BN, AP@0.5-MN | mIoU, Precision, Recall, F1 per class |
| **Parameters (approx.)** | ~41 M (ResNet-50 backbone) | ~15.8 M (ResNet-18 backbone) |

---

## Module-by-Module Comparison

### Module 1: Geometric Prior — SDFPR vs. Geometric Prior Loss

**Paper — SDFPR (Spatially-adaptive Deformable FFN with Prior Regularization):**

The paper fits a 3-component Gaussian Mixture Model (GMM) to the distribution of nodule aspect ratios `r = h/w` and log-widths `log(w)` from clinical data. At every forward pass, a prior sample `(r_prior, w_prior)` is drawn from the GMM and used to modulate and clamp the deformable offset predictions inside a custom DCNv4 block — called "Prior DCN" — which replaces the standard 3×3 convolution in every residual block of the ResNet-50 backbone. This constrains *where the backbone samples features from*, forcing the receptive field itself to respect clinically realistic nodule geometries. A second sub-module (Mix FFN with DropPath) handles global semantic context. Together they form SDFPR, embedded after every residual block.

**Our approximation — Geometric Prior Loss:**

A loss penalty added to the training objective:
```
prior_loss = L1(pred_h / pred_w,  gt_h / gt_w)   # aspect ratio
           + L1(pred_w,           gt_w)            # width
total_loss = cls_loss + bbox_loss + 0.5 × prior_loss
```

**Where the approximation falls short:**

The paper's prior constrains feature *extraction* — it shapes what the model attends to in the image. Our prior only penalises the *output* — if the predicted box has an unrealistic shape, we push it back after the fact. The backbone still extracts features without any geometric constraint, so it will struggle with irregular and blurred nodule boundaries in the same way an unconstrained backbone does. The paper's ablation shows SDFPR alone raises AP by +0.030 (0.612 → 0.642) on Thyroid I.

---

### Module 2: Structural Prior — MSFFM vs. Sobel Edge Channel

**Paper — MSFFM (Multi-scale Spatial-Frequency Feature Mixer):**

For each of three backbone feature scales, a Dual-Branch Feature Fusion Module (DBFFM) runs two branches in parallel:

- **Spatial branch (PAConv):** A perception-aggregation convolution with a large-kernel depth-wise separable convolution for a wide receptive field, followed by SE attention and a FFN. Captures contour continuity, margin subtlety, and local homogeneity.
- **Frequency branch:** A 2D FFT transforms features to the spectral domain. Learnable PWConv + BN + ReLU filters reweight individual frequency components — suppressing speckle-dominated high-frequency noise, enhancing low-frequency morphology components. An inverse FFT reconstructs the refined spatial map.

The two branches are combined with a learnable scalar α (initialised to 0.5):
```
F(x) = α * F_spatial(x) + (1 − α) * F_frequency(x)
```
The paper's ablation shows MSFFM alone raises APs by +0.140 (+14 pts on small nodules) versus the baseline, validating the importance of frequency-domain speckle suppression.

**Our approximation — Sobel edge channel:**

A Sobel operator is applied to the grayscale image before it enters the model, producing an edge map that is stacked as a second input channel. The backbone then processes both channels together from layer 1 onward.

**Where the approximation falls short:**

This approximates only the spatial/contour half of the spatial branch, and only at the input level. There is no frequency-domain processing, so speckle noise and acoustic shadowing are not suppressed at any feature scale. The Sobel edge map is also fixed — it does not adapt during training the way the learned spatial and frequency filters do.

---

### Module 3: Feature Interaction — DFI vs. Standard Decoder Cross-Attention

**Paper — DFI (Dense Feature Interaction):**

After the encoder produces outputs `E1, ..., E6` from each of its 6 layers, DFI aggregates them top-down in a DenseNet-like manner. Each level `i` concatenates its own output with projections from all higher layers and compresses back to the original dimension, producing enhanced multi-level features `M1, ..., M6`. These are fed to the 6 decoder layers in *reversed order*: `D1` attends to `M6` (strongest global semantics), `D6` attends to `M1` (finest spatial detail). This ensures every decoder layer benefits from prior-modulated features from across the full encoder depth, rather than only the final encoder representation.

The paper's ablation shows DFI raises AP by +0.015 over the baseline (0.612 → 0.627) and significantly improves AP@0.75 (tight localisation), suggesting it aids refinement quality rather than coarse detection.

**Our implementation — Standard DETR cross-attention:**

Only the final encoder layer output (`memory`) is passed as Key and Value to the decoder. All intermediate encoder representations are discarded after their layer completes.

**Where the approximation falls short:**

This is the largest structural gap. The 3 deformable encoder layers in our model do compute intermediate representations, but they are all discarded. Implementing DFI here would be straightforward — aggregate `E1, E2, E3` top-down and feed them to `D3, D2, D1` respectively — and would likely produce the most immediate improvement of any single change.

---

## What We Preserved

| Idea | How |
|---|---|
| DETR set-prediction paradigm | Hungarian matching, no NMS, direct box regression |
| Deformable attention in encoder | `DeformableEncoder` with K=4 sampling points per head, O(N·K) complexity |
| Multi-scale feature extraction | All 4 ResNet-18 stages extracted and fused |
| Learned multi-scale fusion | 1×1 convolution over concatenated 4-scale features (not simple summation) |
| Geometric prior concept | Enforced via loss penalty on aspect ratio and width |
| Structural prior concept | Sobel edge map approximates spatial contour awareness |
| Differential learning rate | Backbone at 1e-5, transformer/heads at 1e-4 |
| No-object class imbalance handling | `NO_OBJ_WEIGHT = 0.1` in cross-entropy |
| Best-model checkpointing | Save on validation loss improvement |
| Stratified data splits | Equal benign/malignant ratio across train/val/test |

## What We Simplified

| Paper | Our simplification | Reason |
|---|---|---|
| ResNet-50 backbone | ResNet-18 | VRAM: ResNet-50 requires ~2× the memory at the same batch size |
| 6 + 6 transformer layers | 3 + 3 | VRAM and training time |
| 300 object queries | 100 | Memory; acceptable since BUSI has 1 nodule per image |
| GMM Prior DCN inside every residual block | Geometric prior in loss function | Implementing DCNv4 with GMM sampling requires significant custom CUDA/autograd work |
| MSFFM with FFT frequency branch | Sobel edge input channel | FFT branch requires significant custom module; Sobel is a reasonable zero-cost proxy |
| DFI across all encoder layers | Final encoder layer only | Not implemented; highest-priority addition for future work |
| Focal loss | Cross-entropy with no-object weighting | Focal loss requires careful γ tuning; weighted cross-entropy is a simpler stable alternative |
| L1 + GIoU box loss | L1 + geometric prior loss | GIoU requires computing overlapping areas; our geometric prior adds domain knowledge instead |
| 200 epochs | 50 epochs | Training time: 200 epochs on this hardware would take ~4× longer |
| Multiple datasets (13,308 images) | BUSI only (780 images) | Thyroid datasets are private and not publicly available |

---

## Results Comparison

### Paper results on BUSI (Table 5)

| Metric | Paper (Proposed) | Faster-RCNN | Deformable-DETR | DN-Deformable-DETR |
|---|---|---|---|---|
| AP | **0.472** | 0.413 | 0.388 | 0.441 |
| AP@0.5 | **0.706** | 0.691 | 0.573 | 0.666 |
| AP@0.75 | **0.585** | 0.464 | 0.454 | 0.463 |
| APs | **0.600** | 0.200 | 0.476 | 0.364 |
| APm | 0.389 | 0.229 | 0.308 | 0.302 |
| APl | **0.470** | 0.438 | 0.379 | 0.437 |
| AP@0.5-BN | **0.668** | 0.689 | 0.491 | 0.618 |
| AP@0.5-MN | **0.585** | 0.556 | 0.505 | 0.591 |

### Our results on BUSI validation set

| Metric | Our Value | Notes |
|---|---|---|
| Best val mIoU | **0.374** (epoch 41) | Mean IoU of top-1 prediction per image |
| Best val loss | **1.357** (epoch 46) | Model checkpoint saved here |
| Final train loss | 1.024 | Epoch 50 |
| Final val loss | 1.369 | Epoch 50 |

> Note: mIoU and AP@0.5 are different metrics and cannot be directly compared. AP@0.5 is the area under the full precision-recall curve at IoU threshold 0.5; mIoU is the mean IoU of the single highest-confidence prediction. A mIoU of ~0.37 roughly corresponds to a model that correctly localises nodules above IoU=0.5 for a minority of images, placing the estimated equivalent AP@0.5 in the **0.35–0.45** range — consistent with a Deformable-DETR baseline before the paper's three prior modules are applied.

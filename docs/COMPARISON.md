# Paper vs Our Implementation: Side-by-Side Comparison

## Paper Overview

**Title:** Prior-Guided DETR for Ultrasound Nodule Detection  
**Goal:** Detect thyroid and breast nodules in ultrasound images using a DETR-based model that injects clinical and physical knowledge (priors) into three stages of the network.

## Our Implementation Overview

A lightweight replication of the paper's core ideas, scaled down to run on a single consumer GPU (RTX 3050, 4GB VRAM). We preserve the spirit of the three key innovations while simplifying each one.

---

## Component-by-Component Comparison

| Component | Paper | Our Implementation | Notes |
|---|---|---|---|
| **Base Architecture** | Deformable DETR | Standard DETR (`nn.Transformer`) | Deformable DETR has multi-scale adaptive attention; standard DETR is simpler but still valid |
| **Backbone** | ResNet-50 or ResNet-101 | ResNet-18 | Lighter model, fewer parameters, fits in 4GB VRAM |
| **Input Channels** | Single grayscale channel | 2-channel (grayscale + Sobel edges) | We add edges explicitly; paper's MSFFM learns to extract structure internally |
| **Geometric Prior — Where** | SDFPR module inside the backbone (learnable) | Geometric prior loss term (at training time) | Paper injects priors into feature extraction; we apply them as a loss penalty instead |
| **Geometric Prior — What** | Statistical distribution of nodule aspect ratios and widths | L1 loss on predicted vs GT aspect ratio and width | Same goal: make predictions respect realistic nodule shapes |
| **Structural Prior** | MSFFM — processes image in both spatial domain and frequency domain (FFT) | Sobel edge map as a second input channel | Paper uses FFT to suppress speckle noise globally; we use Sobel edges to highlight boundaries locally |
| **Multi-scale Feature Fusion** | MSFFM with learned spatial-frequency mixing | Learned 1×1 conv over concatenated 4-scale features | Paper's fusion is more expressive; ours is simpler but still learned (not just summation) |
| **Dense Feature Interaction (DFI)** | Prior-enhanced features shared across ALL encoder layers | Not implemented | Paper ensures every encoder layer sees geometric/structural priors; we only apply priors at the loss level |
| **Object Queries** | 300 queries | 100 queries | Fewer queries = less memory; acceptable since BUSI has only 1 nodule per image |
| **Positional Encoding** | Sine/cosine fixed encoding | Learned 2D embedding | Both approaches work; learned encoding can adapt to the domain |
| **Class Imbalance Handling** | Likely focal loss or custom weighting | `NO_OBJ_WEIGHT = 0.1` in cross-entropy | Down-weights the 99 no-object queries so the model focuses on the 1 real nodule |
| **Bounding Box Loss** | L1 + GIoU | L1 + geometric prior | GIoU is theoretically better for overlap quality; our geometric prior adds domain knowledge instead |
| **Training Epochs** | Likely 300–500 epochs | 50 epochs | Transformers need long training; 50 is a practical compromise for local hardware |
| **Batch Size** | 8–16 (multi-GPU) | 4 | Limited by 4GB VRAM |
| **Learning Rate Strategy** | Backbone: 1e-5, transformer: 1e-4, warmup | Backbone: 1e-5, transformer/heads: 1e-4 | We match the differential LR approach from the paper |
| **Optimizer** | AdamW | AdamW | Identical |
| **Gradient Clipping** | max_norm = 0.1 | max_norm = 0.1 | Identical — standard for DETR |
| **Data Augmentation** | Rich augmentations (flips, elastic, intensity) | Horizontal flip, vertical flip, brightness/contrast | We implement the most impactful ones; elastic deformation omitted (compute cost) |
| **Dataset Splits** | Proper train/val/test with cross-validation | 70% train / 15% val / 15% test (stratified) | Stratified split ensures equal class balance across splits |
| **Datasets Tested** | Multiple clinical + public datasets (thyroid + breast) | BUSI (breast ultrasound only) | Paper tests generalization; we focus on a single dataset |
| **Evaluation Metrics** | mAP, AP50, AP75 | IoU, Precision, Recall, F1 per class | Paper uses COCO-style metrics; we use detection-specific metrics |
| **Total Parameters** | ~41M (ResNet-50 backbone) | ~15.8M (ResNet-18 backbone) | 2.6× fewer parameters |

---

## What We Preserved (Core Ideas)

1. **DETR as the base architecture** — set-based detection with Hungarian matching, no NMS needed
2. **Domain priors injected into training** — geometric prior loss enforces realistic nodule shapes
3. **Structural boundary awareness** — Sobel edges help the model find low-contrast boundaries
4. **Multi-scale feature extraction** — 4 backbone stages fused into one representation
5. **Differential learning rate** — pre-trained backbone trained slowly, new transformer trained faster

## What We Simplified (Hardware Constraints)

1. Deformable attention → standard attention (biggest architectural simplification)
2. Frequency-domain branch (FFT) → Sobel edge channel
3. DFI across all encoder layers → not implemented
4. Large dataset cross-validation → single stratified split
5. 500 epochs → 50 epochs

# DETR-BUSI: Multi-Task Breast Ultrasound Nodule Detection and Segmentation

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia)
![Dataset](https://img.shields.io/badge/Dataset-BUSI-informational)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end DETR-based framework for simultaneous breast nodule **detection** (bounding box + classification) and **segmentation** (pixel-level mask) on breast ultrasound images.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project implements a unified detection and segmentation pipeline for breast cancer screening in ultrasound images. Built upon the Detection Transformer (DETR) paradigm, the model performs:

1. **Object Detection**: Predicts bounding boxes and classifies nodules as benign or malignant
2. **Semantic Segmentation**: Generates pixel-level masks delineating nodule boundaries

The architecture combines the transformer-based detection framework with a U-Net style segmentation decoder, enabling both tasks to share backbone features while maintaining specialized output heads.

---

## Architecture

```
Input Ultrasound Image (256 × 256)
            │
            ▼
┌──────────────────────────────────────┐
│  Preprocessing  (datasets/busi.py)   │
│  • Grayscale normalisation           │
│  • Sobel edge map  ──────────────┐   │
│  • Stack → (2, 256, 256) tensor  │   │
└─────────────────────────────────┼───┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
┌─────────────────────────────────────────┐   ┌─────────────────────┐
│  ResNet-18 Backbone (modified conv1)    │   │ Segmentation Branch │
│  • layer1 → f1  (64 ch,  64×64)         │   │ (MkUNet-style)      │
│  • layer2 → f2  (128 ch, 32×32)         │   │                     │
│  • layer3 → f3  (256 ch, 16×16)         │   │ Skip connections    │
│  • layer4 → f4  (512 ch,  8×8)          │   │ from backbone       │
└────────────┬────────────────────────────┘   └─────────┬───────────┘
             │                                        │
             ▼                                        ▼
┌──────────────────────────────────────────────────────────────┐
│  Multi-Scale Fusion                                         │
│  • 1×1 conv: each fᵢ → 256 ch                              │
│  • Bilinear upsample: f₂,f₃,f₄ → 64×64                     │
│  • Concatenate → (1024, 64×64)                              │
│  • Learned 1×1 conv → (256, 64×64)                          │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Positional Encoding (learned 2D)                            │
│  row_embed (256, 128) + col_embed (256, 128)                │
│  → (B, 4096, 256) added to feature sequence                 │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Deformable Encoder  (3 layers)                              │
│  • K=4 learned sampling offsets per head                     │
│  • Bilinear interpolation at offset locs                     │
│  • Softmax attention weights over K points                  │
│  • Pre-norm residual + FFN (dim=512)                        │
└────────────────────────────┬─────────────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│  Detection Branch      │         │  Segmentation Decoder   │
│  (Transformer Decoder) │         │  (U-Net style)          │
│  • 100 object queries  │         │  • Attention gates      │
│  • 3 decoder layers    │         │  • MKIR blocks           │
│  • Self + Cross attention│       │  • Skip connections     │
└───────────┬─────────────┘         └─────────┬───────────────┘
            │                                 │
            ▼                                 ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  Class Head         │   │  Box Head           │   │  Segmentation Head   │
│  Linear(256 → 3)    │   │  Linear → ReLU      │   │  Conv2d(256 → 3)     │
│  (B, 100, 3) logits │   │  → Linear → Sigmoid│   │  (B, 3, 256, 256)    │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Backbone** | ResNet-18 with modified first conv for 2-channel input (grayscale + Sobel edge) |
| **Deformable Attention** | Efficient attention mechanism with K=4 sampling points per query |
| **Detection Decoder** | Standard transformer decoder with 3 layers, 100 object queries |
| **Segmentation Decoder** | U-Net style decoder with attention gates and skip connections from backbone |
| **Multi-Task Loss** | Combined loss: Detection (cls + bbox) + Segmentation (Dice + BCE) |

---

## Dataset

**BUSI — Breast Ultrasound Images** ([Al-Dhabyani et al., 2020](https://doi.org/10.1016/dib.2019.104863))

| Split | Images | Benign | Malignant |
|---|---|---|---|
| Train (70%) | 452 | ~305 | ~147 |
| Val  (15%) | 97 | ~66  | ~32 |
| Test (15%) | 98 | ~66  | ~32 |
| **Total** | **647** | **437** | **210** |

**Data Format:**
- Input: Grayscale ultrasound (256×256) + Sobel edge channel
- Segmentation mask: Pixel-level labels (0=benign, 1=malignant, 2=background)
- Detection: Bounding boxes derived from masks via tight-enclosure

---

## Results

### Detection Performance (Test Set)

| Class | IoU | Precision | Recall | F1-Score |
|-------|-----|-----------|--------|----------|
| Benign | 0.713 | 0.808 | 0.318 | 0.457 |
| Malignant | 0.958 | 0.813 | 0.406 | 0.542 |
| **Mean** | **0.807** | **0.810** | **0.347** | **0.486** |

### Segmentation Performance (Test Set)

| Class | Dice Score |
|-------|------------|
| Benign | **0.963** |
| Malignant | 0.709 |
| Background | 1.000 |
| **Foreground (avg)** | **0.836** |

### Training Dynamics

| Epoch | Train Loss | Det Loss | Seg Loss | Val mIoU |
|-------|------------|----------|----------|----------|
| 1 | 3.55 | 2.44 | 1.11 | 0.12 |
| 25 | 1.82 | 1.32 | 0.50 | 0.32 |
| 50 | 1.40 | 0.95 | 0.45 | 0.42 |

---

## Installation

```bash
git clone https://github.com/sivaahari/detr-busi.git
cd detr-busi
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10
- PyTorch 2.5.1
- OpenCV, NumPy, SciPy, scikit-learn, matplotlib, albumentations

---

## Usage

### Training

```bash
python train.py
```

Logs are saved to `logs/training_log.csv`. Best checkpoint saved to `checkpoints/best_model.pth`.

### Evaluation

```bash
python evaluate.py
```

Computes detection (IoU, Precision, Recall, F1) and segmentation (Dice) metrics on test set.

### Inference

```bash
python inference.py
```

Runs inference on 20 test images and saves visualizations to `outputs/`.

### Visualization (Single Image)

```bash
python visualize.py
```

Interactive single-image visualization.

---

## Repository Structure

```
detr-busi/
├── configs/
│   └── config.py              # All hyperparameters and paths
├── data/
│   └── BUSI/                  # Dataset (not tracked in git)
├── datasets/
│   └── busi.py                # Dataset loader with augmentation
├── docs/
│   ├── COMPARISON.md          # Paper vs. implementation analysis
│   └── COMPONENTS.md          # Architecture guide
├── models/
│   ├── detr.py                # Full DETR model
│   ├── segmentation_head.py   # MkUNet-style segmentation decoder
│   └── deformable_attention.py # Deformable attention module
├── utils/
│   ├── box_ops.py             # IoU, box format conversions
│   ├── loss.py                # DETRLoss + segmentation loss
│   ├── matcher.py             # Hungarian matching
│   └── visualize.py           # Prediction visualization
├── logs/
│   └── training_log.csv       # Training metrics
├── checkpoints/
│   └── best_model.pth         # Best checkpoint
├── outputs/                   # Inference visualizations
├── train.py                   # Training loop
├── evaluate.py                # Evaluation script
├── inference.py               # Batch inference
├── visualize.py               # Single-image visualization
└── README.md
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{aldhabyani2020busi,
  title   = {Dataset of breast ultrasound images},
  author  = {Al-Dhabyani, Walid and Gomaa, Mohammed and Khaled, Hussien and Fahmy, Aly},
  journal = {Data in Brief},
  volume  = 28,
  pages   = 104863,
  year    = 2020,
  doi     = {10.1016/j.dib.2019.104863}
}
```

---

## Acknowledgements

- [Al-Dhabyani et al. (2020)](https://doi.org/10.1016/j.dib.2019.104863) for the BUSI dataset
- [Carion et al. (2020)](https://arxiv.org/abs/2005.12872) for the original DETR
- [Zhu et al. (2020)](https://arxiv.org/abs/2010.04159) for Deformable DETR
- [Rahman & Marculescu (2025)](https://github.com/SLDGroup/MK-UNet) for MkUNet architecture insights
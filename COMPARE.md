# Summary of Key Trade-offs: Paper vs. Implementation

| Gap Type | Paper Approach | Implementation Gap | Impact |
|----------|---------------|-------------------|--------|
| **Geometric prior** | GMM → DCNv4 → backbone feature sampling | L1 penalty on output boxes | Prior only constrains predictions, not feature extraction |
| **Structural prior** | Dual-branch MSFFM (spatial + FFT at every scale) | Single Sobel edge channel at input | No learned frequency-domain speckle suppression |
| **Feature interaction** | DFI aggregates all 6 encoder outputs to decoder | Only final encoder layer used | Decoder lacks multi-level semantic guidance |
| **Box loss** | L1 + GIoU | L1 only | No penalty for area mismatch → boxes too large |
| **Classification** | Focal loss | Weighted cross-entropy | Weaker calibration on hard examples |
| **Model capacity** | ResNet-50 + 6+6 transformer | ResNet-18 + 3+3 transformer | 2.6× fewer parameters, 2× fewer layers |

---

## Detailed Gap Explanations

### 1. Geometric Prior (SDFPR)

**What the paper does:** The paper fits a 3-component Gaussian Mixture Model (GMM) to the clinical distribution of nodule aspect ratios (height/width) and log-widths from the training data. This GMM is then used to modulate deformable sampling offsets inside a custom DCNv4 convolution layer that replaces every standard 3×3 convolution in the ResNet-50 backbone. The deformable attention mechanism literally shifts its sampling points to respect clinically realistic nodule shapes before features are even extracted.

**What the implementation does:** The geometric prior is applied only as a loss penalty — after the model makes a prediction, we penalise boxes whose aspect ratio or width deviates from the ground truth. The backbone has no prior knowledge of what shape nodules should have.

**Why this matters:** The paper's approach constrains *feature extraction* — the model literally cannot attend to unrealistic geometries because the prior shapes the deformable offsets. The implementation only catches unrealistic predictions *after* the fact. This means the backbone still extracts features without any geometric guidance, making it harder to handle irregular nodule boundaries — one of the key challenges in ultrasound imaging.

---

### 2. Structural Prior (MSFFM)

**What the paper does:** For each of three feature scales, the paper runs two parallel branches: a spatial branch using Perception-Aggregation Convolution (PAConv) with a large receptive field and SE attention to capture contour continuity and boundary details; and a frequency branch that applies 2D FFT to transform features to the spectral domain, applies learnable filters to suppress speckle-dominated high frequencies (noise) while enhancing low-frequency morphology, then uses inverse FFT to reconstruct spatial features. The two branches are fused with a learnable scalar α. This operates at *every feature scale*, not just the input.

**What the implementation does:** A Sobel edge detector is applied to the raw grayscale image once before feeding it to the model, stacking the edge map as a second input channel. This is a fixed preprocessing step — it does not adapt during training.

**Why this matters:** Ultrasound images suffer from speckle noise (a grainy texture) that degrades boundary visibility. The paper's frequency branch directly addresses this by learning which frequency components to suppress. The Sobel edge map approximates only the spatial contour half of the spatial branch, and only at the input level. There is no frequency-domain processing to suppress speckle noise at the feature level, making boundary detection less robust.

---

### 3. Feature Interaction (DFI)

**What the paper does:** After the encoder produces outputs from all 6 layers (E₁, E₂, ..., E₆), the Dense Feature Interaction module aggregates them in a DenseNet-like top-down manner. Each level concatenates its output with projections from all higher layers and compresses back to the original dimension, producing enhanced multi-level features (M₁, ..., M₆). These are then fed to the 6 decoder layers in *reverse order*: early decoder layers (which refine coarse queries) attend to high-level semantics from later encoder layers, while later decoder layers attend to low-level spatial details from earlier encoder layers. This ensures every decoder layer benefits from prior-modulated features across the full encoder depth.

**What the implementation does:** Only the final encoder layer output (the "memory") is passed to the decoder as Key and Value. The outputs of intermediate encoder layers are computed but then discarded.

**Why this matters:** This is the largest structural gap. The decoder receives only one representation of the image — the final one. High-level semantic information from deep encoder layers and fine spatial details from early encoder layers are not available to the decoder simultaneously. The paper's ablation shows DFI alone raises AP by +0.015, with particularly strong improvements in AP@0.75 (tight localisation), indicating the decoder refines queries more effectively when it has access to multi-level features.

---

### 4. Box Regression Loss (GIoU)

**What the paper does:** The box regression loss combines L1 loss (smooth L1 on coordinate differences) with GIoU (Generalised Intersection over Union). GIoU directly measures how well two boxes overlap and handles the case where they don't overlap at all — something L1 cannot do. When boxes are far apart or one is contained within the other, GIoU still provides a meaningful gradient that pushes them toward overlap.

**What the implementation does:** The box loss uses only L1 (mean absolute error on the four coordinates).

**Why this matters:** L1 loss only cares about the distance between coordinate values — it does not directly penalise area mismatch. A box that is slightly shifted from the ground truth can have the same L1 loss as a box that is the correct size but shifted by the same amount. In practice, this causes predicted boxes to be systematically too large because the model learns that a larger box covers more area (more likely to have some overlap) without being heavily penalised for the excess area. Visual inspection of the implementation's outputs confirms this pattern — boxes tend to overestimate nodule boundaries.

---

### 5. Classification Loss (Focal Loss)

**What the paper does:** Focal loss is specifically designed for extreme class imbalance. It down-weights confident (easy) examples and focuses training on hard examples by adding a modulating factor (1 - p)^γ to the cross-entropy loss. In detection, this means the model is forced to keep learning from nodules that are difficult to classify rather than becoming overconfident on easy ones.

**What the implementation does:** Standard cross-entropy loss with manual class weighting: `[1.0, 1.0, 0.1]` where 0.1 is the weight for the "no-object" class. This down-weights the 99 queries that should predict "no object," but does not differentiate between easy and hard positive predictions.

**Why this matters:** In this dataset, there is a 99:1 imbalance between "no object" queries and actual nodule queries. Without Focal loss, easy negatives (correctly predicting "no object") dominate the gradient, while hard positive examples (difficult-to-classify nodules) receive less attention. The confidence scores in the implementation's outputs cluster in the 0.5–0.8 range — moderate but not well-calibrated. Focal loss would sharpen the distinction between confident correct predictions and uncertain ones.

---

### 6. Model Capacity (Architecture Scale)

**What the paper does:** Uses a ResNet-50 backbone (~41M parameters) with 6 deformable encoder layers and 6 standard decoder layers, 300 object queries, operating on ~800px images (standard resolution).

**What the implementation does:** Uses a ResNet-18 backbone (~15.8M parameters) with 3 deformable encoder layers and 3 standard decoder layers, 100 object queries, operating on 256×256 images.

**Why this matters:** The implementation has roughly 2.6× fewer parameters and 2× fewer transformer layers. Combined with the 4× reduction in image resolution, the model has significantly less representational capacity. This manifests as a validation mIoU plateau around epoch 35–41 — the model has learned what its architecture allows, and further training provides diminishing returns. The paper's deeper model (6+6 layers) and larger resolution allow for more sophisticated query refinement and finer spatial localisation, which directly contributes to the gap in AP@0.5 (0.706 vs. estimated 0.35–0.45).

---

## Conclusion

The implementation preserves the core DETR paradigm — set-based prediction, Hungarian matching, and deformable attention — while approximating the three novel prior modules from the paper under strict hardware constraints. The performance gap (~0.70 AP@0.5 vs. ~0.35–0.45) is consistent with ablating the three prior modules, as noted in the paper's ablation results. The highest-impact improvements for future work would be implementing DFI (most straightforward), adding GIoU loss (direct drop-in), and transitioning to Focal loss (minimal code change).
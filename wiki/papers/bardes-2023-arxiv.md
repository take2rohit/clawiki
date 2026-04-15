---
title: "MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features"
type: paper
paper_id: P036
authors:
  - "Bardes, Adrien"
  - "Ponce, Jean"
  - "LeCun, Yann"
year: 2023
venue: "arXiv (Meta AI)"
arxiv_id: "2307.12698"
url: "https://arxiv.org/abs/2307.12698"
pdf: "../../raw/bardes-2023-arxiv.pdf"
tags: [JEPA, optical-flow, motion-features, content-features, multi-task-learning, self-supervised-learning, VICReg, video-understanding]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
cited_by:
  - bardes-2024-tmlr
---

# MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features

> **MC-JEPA** (Motion-Content JEPA) jointly learns optical flow estimation and content features within a single shared encoder via multi-task self-supervised learning, combining a JEPA-based optical flow estimator (M-JEPA) with VICReg-based content learning, and demonstrating that the two objectives mutually benefit each other -- flow estimation improves content features for segmentation while content learning improves flow quality.

**Authors:** Adrien Bardes (Meta AI, FAIR; Inria, Ecole normale superieure, CNRS, PSL Research University), Jean Ponce (Courant Institute & Center for Data Science, NYU; Inria, ENS, CNRS, PSL), Yann LeCun (Meta AI, FAIR; Courant Institute & Center for Data Science, NYU) | **Venue:** arXiv preprint (Meta AI) | **arXiv:** [2307.12698](https://arxiv.org/abs/2307.12698)

---

## Problem & Motivation

Self-supervised visual representation learning has been dominated by methods that learn content features -- representations that identify and differentiate objects in images. These methods (contrastive, non-contrastive, masked modeling) focus on *what* is in the scene but cannot capture information at the pixel level, such as motion or fine-grained textures. On the other hand, self-supervised optical flow estimation learns dense pixel correspondences between consecutive video frames but does not understand the semantic content of what is moving.

These two capabilities -- understanding *what* objects are (content) and *how* they move (motion) -- have been learned separately by different methods optimized for different tasks. The authors propose to **unify** them in a single encoder trained with a joint-embedding predictive architecture ([[lecun-2022-openreview]] ([LeCun 2022](../papers/lecun-2022-openreview.md))), combining self-supervised optical flow estimation from videos with content feature learning from images. The key question is whether these two objectives benefit each other when trained jointly.

---

## Core Idea

MC-JEPA introduces a multi-task self-supervised learning approach with a **single shared ConvNeXt-T encoder** that simultaneously learns:

1. **Motion features** via self-supervised optical flow estimation (M-JEPA component): Given consecutive video frames, the encoder produces pyramidal features that feed into a PWC-Net-style flow estimator with coarse-to-fine refinement, cycle consistency, and VICReg regularization at every feature layer.
2. **Content features** via VICReg-based joint embedding (content learning component): Given two augmented views of the same image (from ImageNet), the encoder produces representations that are made invariant via VICReg loss.

Both tasks share the same encoder weights and are trained simultaneously by alternating batches from video datasets (for flow) and ImageNet (for content). The resulting features capture both motion and content information and transfer to a wide range of downstream tasks.

---

## How It Works

### Architecture

**Shared backbone**: Modified ConvNeXt-T (21M parameters) with 6 pyramidal feature levels. The stem layer is split into two smaller convolutional layers (3x3 and 4x4 kernels) to support fine-grained flow estimation at the lowest level.

**Flow estimator** (M-JEPA): Based on PWC-Net (Sun et al., 2018), adapted for the ConvNeXt features:
- Produces pyramidal features {X^(l)_t} for consecutive frames I_t, I_{t+1}.
- Estimates flow coarse-to-fine: initial flow at lowest resolution, then iteratively refined by predicting residual flow at each higher level.
- Uses 4D correlation volumes V = X_hat_{t+1} * X^T_{t+1} for matching.
- LayerNorm after each convolutional layer in the estimator (critical for stability).

**Content learning**: VICReg expander network (768-8192-8192-8192 MLP) on top of the shared encoder.

### Training Losses

**Flow losses** (applied on video data D1):
1. **Regression loss** (Eq. 2): Multi-scale L2 between warped and actual features: L_reg = sum_l ||X^(l)_{t+1} - X_hat^(l)_{t+1}||^2
2. **Reconstruction loss** (Eq. 3): Image-level photometric loss between I_{t+1} and the warped I_hat_{t+1}, using combined L2, L1, and SSIM.
3. **Smoothness loss** (Eq. 4): Edge-aware smoothness regularizer for the predicted flow.
4. **Cycle consistency loss** (Eq. 5): Forward-backward flow consistency: L_cycle = sum_l ||X^(l)_t - f_{t+1,t}(f_{t,t+1}(X^(l)_t))||^2
5. **Variance-covariance regularization** (Eq. 6): VICReg terms applied at *every feature layer* to prevent collapse during multi-task training (critical for stability).

**Content loss** (applied on ImageNet data D2):
- VICReg loss L_ssl: variance + covariance + invariance terms on embeddings of two augmented views.

**Total loss** (Eq. 7):
```
L = sum_{D1} (L_rec + L_reg + L_smooth + L_cycle + L_vc) + sum_{D2} L_ssl
```

Losses are balanced with carefully tuned coefficients at each feature layer (Table 8 in appendix). Higher layers need less regularization.

### Multi-task Training Strategy

At each iteration:
1. Sample a batch of video sequences from flow datasets (KITTI, Sintel, FlyingThings, FlyingChairs, HD1K).
2. Compute all flow losses and backpropagate.
3. Sample a batch of images from ImageNet.
4. Compute VICReg SSL loss and backpropagate.
5. Gradients from both tasks update the shared encoder.

The flow estimation objective starts after 10 epochs of ImageNet-only pretraining (Figure 5(1)), as features change too rapidly in early training. Batch alternation (not epoch alternation) produces the best results (Table 5).

### Inference

The trained encoder directly produces features for downstream tasks:
- **Optical flow**: Feed two frames through encoder + flow estimator.
- **Segmentation/classification**: Feed single image through encoder, extract features for linear probing or fine-tuning.
- **Video segmentation**: Use frozen features for nearest-neighbor-based tracking (DAVIS protocol).

---

## Results

### Optical Flow Estimation (Table 1)

| Method | Backbone | Sintel Clean (train EPE) | Sintel Final (train EPE) | KITTI 2015 (train EPE / test F1) |
|---|---|---|---|---|
| UFlow | PWC | 2.50 / 5.21 | 3.39 / 6.50 | 2.71 / 11.13 |
| ARFlow | PWC | 2.79 / 4.78 | 3.73 / 5.89 | 2.85 / 11.80 |
| UPFlow | PWC | 2.33 / 4.68 | 2.67 / 5.32 | 2.45 / 9.38 |
| SMURF | RAFT | 1.71 / 3.15 | 2.58 / 4.18 | 2.00 / 6.83 |
| M-JEPA | CNX-T | 2.98 / -- | 3.82 / -- | 3.01 / -- |
| **MC-JEPA** | **CNX-T** | **2.81 / 5.01** | **3.51 / 6.12** | **2.67 / 11.33** |

MC-JEPA (with content learning) improves over M-JEPA (flow only) on all benchmarks. Results are on par with dedicated flow methods (UFlow, ARFlow) while simultaneously learning strong content features.

### Image and Video Segmentation (Table 1, continued)

| Method | Backbone | Pascal VOC (Frozen / FT mIoU) | CityScapes (Frozen / FT mIoU) | ADE20K (Frozen / FT mIoU) | DAVIS 2017 (J&F)_m |
|---|---|---|---|---|---|
| VICReg | CNX-T | 60.1 / 77.8 | 59.8 / 76.3 | 28.6 / 41.1 | 58.1 |
| VICRegL | CNX-T | 66.8 / 79.7 | 64.9 / 78.3 | 30.6 / 44.1 | 66.7 |
| MoCo v3 | ViT-S | 57.1 / 75.9 | 56.5 / 74.0 | 23.7 / 39.8 | -- |
| DINO | ViT-S | 65.2 / 79.5 | 64.8 / 78.1 | 30.5 / 43.5 | 69.9 |
| **MC-JEPA** | **CNX-T** | **67.1 / 79.9** | **65.5 / 78.4** | **30.8 / 44.2** | **70.5** |

MC-JEPA outperforms VICReg (the content method it builds upon) by large margins on all segmentation tasks, demonstrating that the flow estimation pretext task significantly improves content features. MC-JEPA achieves results on par with VICRegL and DINO, which have among the best self-supervised features available.

On **DAVIS 2017 video segmentation** (J&F)_m = 70.5, MC-JEPA outperforms all compared methods including DINO (69.9) and VICRegL (66.7), demonstrating the value of motion-aware features for video tasks.

### Ablation: Flow Datasets (Table 2)

Adding more diverse flow datasets (KITTI + Sintel + FlyingThings + FlyingChairs + HD1K) consistently improves flow quality. The benefit on segmentation is independent of which flow dataset is used -- any flow training helps content features.

### Ablation: Multi-task Balancing (Figure 5(3))

Increasing the flow loss coefficient improves both flow and segmentation up to a threshold (~0.1), after which segmentation degrades. The multi-task coefficient must be tuned carefully.

### Ablation: VICReg on Feature Layers (Table 11, Appendix E)

Applying variance-covariance regularization at every feature layer is critical for stability in the multi-task setup. Without it, training crashes. Using it only at the last layer is sufficient to prevent collapse but applying it at all layers gives the best combined performance.

---

## Comparison to Prior Work

| | **MC-JEPA** | **VICReg** | **DINO** | **SMURF** |
|---|---|---|---|---|
| Task | Motion + Content (multi-task) | Content only | Content only | Flow only |
| Encoder | ConvNeXt-T (shared) | ConvNeXt-T | ViT-S | RAFT |
| Training data | ImageNet + video datasets | ImageNet | ImageNet | Video datasets |
| Flow estimation | Yes (competitive) | No | No | Yes (state-of-the-art) |
| Segmentation (ADE20K FT) | 44.2 | 41.1 | 43.5 | N/A |
| Video segmentation (DAVIS) | **70.5** | 58.1 | 69.9 | N/A |

**vs [[lecun-2022-openreview]] ([JEPA position paper](../papers/lecun-2022-openreview.md)):** LeCun 2022 advocated for joint-embedding architectures that learn to predict in representation space. MC-JEPA instantiates this vision for motion: the flow estimator works on encoder features (not pixels), and the VICReg content learning operates in embedding space. The paper's multi-task approach aligns with the position paper's emphasis on learning multiple aspects of the world.

**vs [[bardes-2024-tmlr]] ([V-JEPA](../papers/bardes-2024-tmlr.md)):** V-JEPA learns video representations by predicting masked spatiotemporal features. MC-JEPA takes a different approach: rather than masking video, it explicitly learns optical flow as a pretext task alongside content features. V-JEPA focuses on global video understanding while MC-JEPA produces dense, pixel-level motion features.

**vs dedicated flow methods (SMURF, RAFT):** MC-JEPA's flow is competitive but not state-of-the-art compared to methods optimized solely for flow. The goal is not to produce the best flow but to use flow as a pretext task that enriches the shared encoder's features.

---

## Strengths

- **Novel multi-task combination**: First work to jointly learn self-supervised optical flow and content features in a single shared encoder, demonstrating mutual benefits between the two objectives.
- **Strong empirical validation of mutual benefit**: Adding flow estimation improves content features (ADE20K mIoU: 41.1 -> 44.2; DAVIS: 58.1 -> 70.5 vs VICReg alone). Adding content learning improves flow quality (M-JEPA -> MC-JEPA improvements on all flow benchmarks).
- **Single model for diverse tasks**: One encoder performs well on optical flow, image segmentation, and video segmentation -- tasks that previously required separate specialized models.
- **Practical training recipes**: Detailed ablations on training stability (LayerNorm in estimator, VICReg at every layer, flow start epoch, batch alternation strategy) provide actionable guidance.
- **Thorough ablation study**: Systematic evaluation of flow datasets, estimator architecture, backbone choice, data sampling strategy, and loss coefficients.

---

## Weaknesses & Limitations

- **Small-scale experiments only**: All experiments use ConvNeXt-T (21M params) trained on 8 V100 GPUs for 100 epochs. No ViT-Base/Large results or scaling analysis.
- **Training instability**: The multi-task setup is "particularly challenging" -- requires LayerNorm in the flow estimator, VICReg at every feature layer, clipped flow values, and careful learning rate tuning to avoid NaN gradients. This fragility limits practical adoption.
- **ConvNeXt-T backbone only**: The paper does not evaluate on Vision Transformers, which dominate modern SSL. The modified ConvNeXt-T with custom stem limits direct comparison with ViT-based methods.
- **Not a JEPA in the standard sense**: The content learning uses VICReg (a non-contrastive method), not masked prediction in embedding space. The "JEPA" terminology is somewhat loose -- the flow component is JEPA-like (predicting in feature space) but the overall system is a multi-task VICReg + flow setup.
- **Careful coefficient tuning required**: Loss coefficients differ at each encoder layer (Table 8) and the multi-task balancing coefficient has a narrow optimal range (~0.1). This makes the method sensitive to hyperparameters.

---

## Key Takeaways

- **Motion and content learning are mutually beneficial**: Learning optical flow as a pretext task significantly improves content features (+12.4 points on DAVIS video segmentation vs VICReg alone), and content learning improves flow estimation quality.
- **VICReg regularization at every feature layer is critical for multi-task stability**: Without it, joint training of flow and content objectives causes training collapse (NaN gradients).
- **A single shared encoder can perform well on diverse tasks**: MC-JEPA's ConvNeXt-T produces competitive results on optical flow, image segmentation, and video segmentation simultaneously.
- **Flow estimation as a pretext task enriches features with pixel-level information**: Content features learned with flow estimation capture motion, spatial correspondences, and fine-grained details that pure content methods miss.
- **This work is a precursor to V-JEPA**: The same authors (Bardes, Ponce, LeCun) later developed [[bardes-2024-tmlr]] ([V-JEPA](../papers/bardes-2024-tmlr.md)), which takes a different approach to video representation learning via masked prediction rather than explicit flow estimation.

---

## BibTeX

{% raw %}
```bibtex
@article{bardes2023mcjepa,
  title={{MC-JEPA}: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features},
  author={Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  journal={arXiv preprint arXiv:2307.12698},
  year={2023}
}
```
{% endraw %}

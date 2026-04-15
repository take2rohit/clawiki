---
title: "Self-Supervised JEPA-based World Models for LiDAR Occupancy Completion and Forecasting"
type: paper
paper_id: P041
authors:
  - "Zhu, Haoran"
  - "Choromanska, Anna"
year: 2026
venue: "arXiv"
arxiv_id: "2602.12540"
url: "https://arxiv.org/abs/2602.12540"
pdf: "../../raw/zhu-2026-arxiv.pdf"
tags: [JEPA, world-model, autonomous-driving, LiDAR, occupancy-forecasting, self-supervised-learning, spatiotemporal, BEV]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
  - balestriero-2025-iclr
  - zhu-2025-aaai
cited_by: []
---

# Self-Supervised JEPA-based World Models for LiDAR Occupancy Completion and Forecasting

> **AD-LiST-JEPA** (Autonomous Driving with LiDAR data in a SpatioTemporal setting via JEPA) extends AD-L-JEPA from single-frame representation learning to a self-supervised spatiotemporal world model that predicts future multi-frame LiDAR BEV representations, demonstrating improved occupancy completion and forecasting (OCF) performance over training from scratch on the Waymo dataset (39.41% vs 38.56% IoU_full with SIGReg regularization).

**Authors:** Haoran Zhu, Anna Choromanska (NYU) | **Venue:** arXiv 2026 | **arXiv:** [2602.12540](https://arxiv.org/abs/2602.12540)

---

## Problem & Motivation

World models that predict how an environment evolves spatiotemporally are fundamental for autonomous driving planning. Existing driving world models fall into two categories, each with limitations:

1. **Generative world models** (GAIA-1, Genie) explicitly generate pixel-level future frames. They are computationally expensive and can produce physically implausible hallucinations.

2. **Latent world models** (LAW, World4Drive) predict future states in latent space, which is more efficient but suffers from **representation collapse** -- all encoder outputs collapse to constant vectors. LAW mitigates this with supervised waypoint regression (noisy, suboptimal), while World4Drive uses auxiliary pretrained encoders (require labeled data).

The JEPA framework from [[lecun-2022-openreview]] offers a principled solution: learn world models entirely in latent space without representation collapse, using either EMA-based target encoders or explicit regularization. The predecessor [[zhu-2025-aaai]] (AD-L-JEPA) validated JEPA for single-frame LiDAR representation learning. AD-LiST-JEPA extends this to the spatiotemporal setting, learning a world model that predicts future multi-frame representations from past LiDAR data.

---

## Core Idea

AD-LiST-JEPA learns to predict the embeddings of masked multi-frame LiDAR point cloud regions given unmasked regions, capturing the spatiotemporal evolution of the driving environment. Two key design choices distinguish it from its single-frame predecessor:

1. **Group BEV-guided masking**: A novel masking strategy that handles the challenges of multi-frame LiDAR data -- ego motion compensation and varying non-empty grid patterns across time.
2. **Two regularization variants**: The paper compares variance regularization (from AD-L-JEPA) with SIGReg from [[balestriero-2025-iclr]] (LeJEPA), finding that the latter significantly outperforms for this architecture.

The quality of the learned world model is evaluated via a downstream **occupancy completion and forecasting (OCF)** task, which jointly tests perception and prediction capabilities.

---

## How It Works

### Phase 1: Self-Supervised World Modeling

#### Group BEV-Guided Masking

The masking strategy addresses two challenges specific to multi-frame LiDAR:

1. **Ego motion compensation**: Points from time -T to T are transformed into a common coordinate system at time 0 using known ego-motion poses (p'_j <- R^T * p_j + c). Without this, propagating masks across ego-centric frames would leak information about ego motion.

2. **Temporal consistency of masking**: Non-empty grids vary across frames due to occlusion -- an object visible at time t may be occluded at time t+1. The **group masking** strategy first aggregates all points across all frames to determine which grids are ever non-empty, then propagates this group-level mask to individual frames. This ensures consistent masking decisions across time while respecting per-frame occupancy.

#### Network Architecture

The architecture inherits from AD-L-JEPA with multi-frame extensions:

- **Context and target encoders**: Same single-frame sparse 3D convolutional network as AD-L-JEPA, applied independently to each frame (following the standard AD architecture pattern of single-frame encoding + feature aggregation).
- **BEV representation**: Per-frame 3D voxel outputs are reshaped into per-frame BEV representations and concatenated along the height dimension to form multi-frame BEV representations.
- **Predictor**: A simple 3D convolutional predictor predicts the target multi-frame BEV representations from the context multi-frame BEV representations.

#### Two Regularization Variants

**Variant 1 -- Variance regularization**: Uses cosine-similarity embedding prediction loss with variance regularization at non-empty grids, plus EMA target encoder updates (from AD-L-JEPA).

**Variant 2 -- SIGReg**: Uses L2 distance embedding prediction loss with SIGReg regularization from [[balestriero-2025-iclr]]. In this setting, the context and target encoders share identical weights (no EMA needed), and SIGReg is applied to outputs of both encoders.

The overall loss is: **L = L_jepa + lambda_reg * L_reg**

### Phase 2: OCF Fine-tuning

The pretrained sparse 3D encoder from Phase 1 is frozen and used to extract per-frame BEV features from past 5 frames. A lightweight decoder (three convolutional layers + linear layer) predicts next 5 frames' completed occupancy via binary cross-entropy loss. The decoder is intentionally simple to isolate the encoder's representation quality.

### Training

- **Phase 1**: 30 epochs on Waymo dataset, 8x A100 GPUs, Adam optimizer, one-cycle LR scheduler (lr=0.0003, weight decay=0.01). Batch size 32 for variance regularization, 16 for SIGReg. Masking ratio: 50% non-empty, 50% empty grids.
- **Phase 2**: 3 epochs, single A100, batch size 4, constant lr=0.0005. Predicts next 5 frames' OCF given past 5 frames.

### Inference

The pretrained encoder processes each input frame independently, producing BEV features. These are concatenated and fed through the lightweight decoder to predict future completed occupancy grids.

---

## Results

### Occupancy Completion and Forecasting on Waymo (Table 1)

| Method | IoU_full (%) | IoU_close (%) |
|---|---|---|
| Scratch, linear | 38.56 +/- 0.19 | 42.87 +/- 0.17 |
| AD-LiST-JEPA, small | 39.09 +/- 0.36 | 43.43 +/- 0.39 |
| AD-LiST-JEPA, small, SIGReg | 39.35 +/- 0.24 | 43.70 +/- 0.24 |
| AD-LiST-JEPA, full | 39.01 +/- 0.47 | 43.46 +/- 0.44 |
| **AD-LiST-JEPA, full, SIGReg** | **39.41 +/- 0.31** | **43.86 +/- 0.30** |

"Small" and "full" denote Phase 1 pretraining with 190 and 950 Waymo sequences, respectively. Results are mean +/- std across 3 random seeds.

Key findings:
- All pretrained variants outperform training from scratch.
- **SIGReg consistently outperforms variance regularization** for both data scales, indicating that purely regularization-based methods (without EMA) are a promising direction for avoiding representation collapse.
- Scaling from small to full dataset improves performance, with SIGReg benefiting more from additional data.

---

## Comparison to Prior Work

| | **AD-LiST-JEPA** | AD-L-JEPA | LAW | World4Drive |
|---|---|---|---|---|
| **Setting** | Multi-frame world model | Single-frame SSL | Multi-frame latent WM | Multi-frame latent WM |
| **Modality** | LiDAR point clouds | LiDAR point clouds | Camera images | Camera images |
| **Supervision** | Self-supervised (JEPA) | Self-supervised (JEPA) | Semi-supervised (waypoints) | Pretrained encoders |
| **Collapse prevention** | Variance reg. or SIGReg | Variance reg. + EMA | Waypoint supervision | Auxiliary encoders |
| **Downstream task** | Occupancy completion & forecasting | 3D object detection | End-to-end driving | End-to-end driving |

**vs [[zhu-2025-aaai]] (AD-L-JEPA):** AD-LiST-JEPA extends AD-L-JEPA from spatial (single-frame) to spatiotemporal (multi-frame) world modeling. The key additions are: (1) group BEV-guided masking for temporal consistency, (2) ego motion compensation, (3) 3D convolutional predictor for multi-frame prediction, and (4) evaluation on OCF rather than 3D detection.

**vs [[balestriero-2025-iclr]] (LeJEPA):** The paper's SIGReg variant directly uses LeJEPA's SIGReg regularization, which eliminates the need for EMA and provides provable collapse prevention. The positive results validate SIGReg's applicability beyond vision transformers to sparse 3D convolutional networks.

**vs [[bardes-2024-tmlr]] (V-JEPA):** While V-JEPA applies JEPA to video prediction using vision transformers on camera data, AD-LiST-JEPA targets LiDAR point cloud sequences in autonomous driving with fundamentally different architectures (sparse 3D convolutions) and masking strategies (BEV-guided).

**vs [[hafner-2023-arxiv]] (DreamerV3):** DreamerV3 learns a world model for RL control using a generative approach (RSSM). AD-LiST-JEPA learns a non-generative world model using JEPA, focusing on LiDAR scene representation quality rather than control policies.

---

## Strengths

- **Natural extension to world models**: Cleanly extends the validated single-frame AD-L-JEPA to the spatiotemporal setting, directly addressing the original paper's noted limitation of missing temporal dynamics.
- **SIGReg validation in a new domain**: Demonstrates that LeJEPA's theoretically-grounded SIGReg regularization works for sparse 3D convolutional architectures on LiDAR data, not just vision transformers on images.
- **Principled masking strategy**: The group BEV-guided masking thoughtfully handles ego motion and temporal occlusion consistency, which are real challenges specific to multi-frame LiDAR data.
- **Appropriate evaluation task**: OCF jointly assesses perception and prediction, making it a meaningful proxy for world model quality compared to detection-only evaluation.

---

## Weaknesses & Limitations

- **Proof of concept only**: The paper is explicitly framed as a proof of concept. Results are on a limited number of Waymo sequences (190 and 950) with modest improvements (+0.85 IoU_full over scratch).
- **Simple downstream decoder**: The lightweight 3-layer decoder may be too simple to fully exploit the pretrained representations, potentially understating the benefits of pre-training.
- **No comparison to other world models**: The only baseline is training from scratch. No comparison to LAW, World4Drive, or other latent world models is provided.
- **Limited analysis**: No ablation studies on masking strategy, no visualization of learned representations, and no analysis of what the world model captures spatiotemporally.
- **Small network architecture**: Only uses the relatively small VoxelBackBone8x. The paper notes that experiments with larger architectures are future work.

---

## Key Takeaways

- **JEPA-based world models work for LiDAR**: Self-supervised pre-training via JEPA improves downstream occupancy completion and forecasting, validating that prediction in representation space captures useful spatiotemporal structure.
- **SIGReg is the better regularization choice**: For this architecture and task, SIGReg (no EMA, provable collapse prevention) consistently outperforms variance regularization (with EMA), aligning with theoretical predictions from [[balestriero-2025-iclr]].
- **Group masking handles multi-frame LiDAR challenges**: The proposed ego-motion-compensated group masking is a necessary adaptation for applying JEPA to temporally consistent LiDAR sequences.
- **Promising but early**: This is a proof-of-concept establishing feasibility. Larger-scale experiments with bigger architectures and more comprehensive comparisons are needed to determine whether JEPA-based world models can compete with generative alternatives for autonomous driving.

---

## BibTeX

{% raw %}
```bibtex
@article{zhu2026self,
  title={Self-Supervised JEPA-based World Models for LiDAR Occupancy Completion and Forecasting},
  author={Zhu, Haoran and Choromanska, Anna},
  journal={arXiv preprint arXiv:2602.12540},
  year={2026}
}
```
{% endraw %}

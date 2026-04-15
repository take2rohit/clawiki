---
title: "AD-L-JEPA: Self-Supervised Spatial World Models with Joint Embedding Predictive Architecture for Autonomous Driving with LiDAR Data"
type: paper
paper_id: P040
authors:
  - "Zhu, Haoran"
  - "Dong, Zhenyuan"
  - "Topollai, Kristi"
  - "Sha, Beiyao"
  - "Choromanska, Anna"
year: 2025
venue: "AAAI 2026"
arxiv_id: "2501.04969"
url: "https://arxiv.org/abs/2501.04969"
pdf: "../../raw/zhu-2025-aaai.pdf"
tags: [JEPA, autonomous-driving, LiDAR, self-supervised-learning, point-cloud, 3D-object-detection, BEV, representation-learning]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
cited_by:
  - zhu-2026-arxiv
---

# AD-L-JEPA: Self-Supervised Spatial World Models with Joint Embedding Predictive Architecture for Autonomous Driving with LiDAR Data

> **AD-L-JEPA** is the first JEPA-based self-supervised pre-training framework for automotive LiDAR 3D object detection, predicting Bird's-Eye-View (BEV) embeddings of masked regions rather than reconstructing point clouds, achieving consistent improvements over Occupancy-MAE and other baselines across KITTI3D, Waymo, and ONCE datasets while reducing GPU hours by 1.9--2.7x and GPU memory by 2.8--4x.

**Authors:** Haoran Zhu, Zhenyuan Dong, Kristi Topollai, Beiyao Sha, Anna Choromanska (NYU) | **Venue:** AAAI 2026 | **arXiv:** [2501.04969](https://arxiv.org/abs/2501.04969)

---

## Problem & Motivation

Current autonomous driving systems require large amounts of labeled data for training, which is expensive and limits scalability. Self-supervised learning (SSL) offers a way to leverage vast unlabeled data for pre-training. However, directly applying popular contrastive or generative methods to LiDAR point clouds has been challenging:

1. **Contrastive methods** (PointContrast, DepthContrast, BYOL) struggle because defining meaningful positive/negative pairs via data augmentation in driving scenarios is difficult -- scenes contain multiple objects and augmentations designed for natural images do not straightforwardly transfer.

2. **Generative methods** (Occupancy-MAE, ALSO) reconstruct masked point clouds or scene surfaces, but explicit pixel/point-level generation is computationally expensive and may be insufficient for capturing the diverse, high-uncertainty nature of driving scenes where multiple plausible futures exist (e.g., different car rears can share the same semantics).

The core insight from [[lecun-2022-openreview]] is that Joint Embedding Predictive Architectures (JEPAs) can avoid both issues by predicting in embedding space rather than input space. AD-L-JEPA applies this principle to autonomous driving LiDAR data for the first time.

---

## Core Idea

AD-L-JEPA predicts BEV embeddings of masked LiDAR regions in representation space rather than reconstructing the actual point clouds. This approach offers three key advantages over existing methods:

1. **No manual augmentation engineering**: Unlike contrastive methods, no positive/negative pair construction is needed.
2. **Semantic abstraction**: Unlike generative methods, the model predicts in embedding space and can ignore irrelevant low-level details, handling the inherent uncertainty and multi-modality of driving scenes.
3. **Efficiency**: By operating at the BEV embedding level and omitting dense 3D convolutional decoders, AD-L-JEPA uses significantly less GPU memory and training time than methods like Occupancy-MAE.

A critical design choice is the **modified BEV-guided masking** strategy that creates masks for both empty and non-empty BEV grids, forcing the network to learn representations for all spatial regions including free space.

---

## How It Works

### Architecture Overview

AD-L-JEPA consists of three components within the sparse 3D convolution encoder framework:

1. **Context encoder** (f_theta): A sparse 3D convolutional network (VoxelBackBone8x) that processes the unmasked (context) point cloud. It extracts 3D voxel representations which are reshaped into BEV representations.

2. **Target encoder** (f_theta_bar): An identical architecture whose weights are an exponential moving average (EMA) of the context encoder. It processes the masked (target) point cloud and produces target BEV embeddings. The target encoder always processes a different masking pattern than the context encoder.

3. **Spatial predictor** (g_phi): A lightweight three-layer 2D convolutional network that predicts target BEV embeddings from the context BEV embeddings.

### Modified BEV-Guided Masking

The masking strategy extends BEV-MAE's approach with two key modifications:

1. **Masks are created in BEV embedding space** and recursively upsampled to identify which input points to mask -- this is computationally efficient.
2. **Both empty and non-empty BEV grids are masked** (50% of each). The original BEV-MAE only masks non-empty grids, assuming the network knows which grids are empty. AD-L-JEPA forces the network to predict embeddings for all invisible regions, enhancing representation quality.

### Learnable Empty and Mask Tokens

- A **learnable empty token** replaces all unmasked empty grid embeddings in both context and target BEV representations. This is critical because sparse 3D convolutions produce near-zero vectors for empty regions, and direct L2 normalization of these near-zero vectors would yield high-variance, meaningless embeddings.
- A **learnable mask token** replaces all masked grid embeddings in the context BEV representation.

### Training

The overall loss combines two objectives:

**L = lambda_jepa * L_jepa + lambda_reg * L_reg**

- **Embedding prediction loss** (L_jepa): Cosine-similarity-based loss applied only at masked BEV grids, separately weighted for empty grids (alpha_0 = 0.25) and non-empty grids (alpha_1 = 0.75).
- **Variance regularization loss** (L_reg): Applied to non-empty grid embeddings at both context encoder and predictor outputs. Ensures average variance across embedding dimensions stays above threshold gamma, preventing representation collapse.

The target encoder is updated via EMA: theta_bar <- eta * theta_bar + (1 - eta) * theta, where eta starts at 0.996 and increases linearly to 1.0.

**Pre-training setup**: VoxelBackBone8x encoder with 12 sparse convolution layers, trained for 30 epochs using Adam optimizer with one-cycle learning rate scheduler (lr=0.0003, weight decay=0.01). Batch size 16 on KITTI3D (4x 1080 Ti), 16 on Waymo (1x A100), up to 128 on ONCE 1M (8x A100).

### Inference

After pre-training, the context encoder weights are used to initialize the backbone of a downstream 3D object detector (SECOND, PV-RCNN, or CenterPoint). The entire network is fine-tuned end-to-end for 80 epochs using default OpenPCDet hyperparameters.

---

## Results

### KITTI3D 3D Object Detection (Table 1)

Pre-trained and fine-tuned on KITTI3D using SECOND:

| Method | Cars | Ped. | Cycl. | Overall | Diff. vs scratch |
|---|---|---|---|---|---|
| No pre-training | 81.99 | 52.02 | 65.07 | 66.36 | -- |
| Occupancy-MAE | 81.15 | 50.36 | **69.74** | 67.08 | +0.72 |
| ALSO | 81.48 | 52.50 | 65.95 | 66.64 | +0.28 |
| **AD-L-JEPA** | 81.68 | **54.15** | 67.93 | **67.92** | **+1.56** |

AD-L-JEPA achieves the highest overall mAP, outperforming both Occupancy-MAE and ALSO across R_40 and R_11 metrics.

### Waymo 3D Object Detection (Table 2)

Pre-trained on Waymo 20% or 100%, fine-tuned on Waymo 20% using CenterPoint:

| Method | Veh. | Ped. | Cycl. | Overall AP | Diff. |
|---|---|---|---|---|---|
| No pre-training | 63.28 | 63.95 | 66.77 | 64.67 | -- |
| Occupancy-MAE, 20% | 63.20 | 64.20 | 67.20 | 64.87 | +0.20 |
| AD-L-JEPA, 20% | 63.18 | 64.35 | 67.68 | 65.07 | +0.40 |
| **AD-L-JEPA, 100%** | **63.58** | **64.58** | **68.07** | **65.41** | **+0.74** |

### ONCE Dataset (Table 3)

Pre-trained with increasing data scale (100k, 500k, 1M frames), fine-tuned using SECOND:

| Method | Veh. | Ped. | Cycl. | Overall | Diff. |
|---|---|---|---|---|---|
| No pre-training | 71.19 | 26.44 | 58.04 | 51.89 | -- |
| **AD-L-JEPA, 100k** | **73.18** | **29.19** | 58.14 | **53.50** | **+1.61** |
| **AD-L-JEPA, 500k** | **73.25** | **31.91** | **59.47** | **54.87** | **+2.98** |
| AD-L-JEPA, 1M | 73.01 | 31.94 | 59.16 | 54.70 | +2.81 |

Performance scales well with data but saturates around 500k frames, likely due to redundancy in similar driving scenarios.

### Transfer Learning (Table 4)

Pre-training on Waymo 20%, fine-tuning on KITTI3D with varying label fractions:

| Method | 20% labels | 50% labels | 100% labels |
|---|---|---|---|
| No pre-training | 62.01 | 64.21 | 66.36 |
| Occupancy-MAE | 62.12 | 64.49 | 66.63 |
| **AD-L-JEPA** | **63.30** | **65.27** | **66.99** |
| AD-L-JEPA (100% Waymo) | 63.33 | 65.27 | 67.71 |

AD-L-JEPA consistently outperforms baselines across all label efficiencies, with the strongest gains at 100% labels using the full Waymo pre-training.

### Pre-training Efficiency

| Metric | Occupancy-MAE | AD-L-JEPA | Reduction |
|---|---|---|---|
| GPU memory (Waymo, bs=8) | 34.6 GB | 10.7 GB | **3.2x less** |
| GPU hours (Waymo 20%) | 56 hrs | 19.25 hrs | **2.9x less** |
| GPU hours (Waymo 100%) | 210 hrs | 77 hrs | **2.7x less** |
| GPU hours (ONCE 100k) | 56 hrs | 30 hrs | **1.9x less** |

---

## Comparison to Prior Work

| | **AD-L-JEPA** | Occupancy-MAE | ALSO | PointContrast |
|---|---|---|---|---|
| **Paradigm** | JEPA (embedding prediction) | Generative (occupancy reconstruction) | Generative (surface reconstruction) | Contrastive |
| **Prediction space** | BEV embedding | 3D voxel occupancy | Scene surface | Embedding (pairs) |
| **Masking** | Modified BEV-guided (empty + non-empty) | BEV-guided (non-empty only) | N/A | N/A |
| **Collapse prevention** | EMA + variance regularization | N/A (generative) | N/A (generative) | Negative pairs |
| **GPU memory** | Low (10.7 GB) | High (34.6 GB) | Moderate | Moderate |
| **KITTI3D mAP** | 67.92 | 67.08 | 66.64 | N/A |

**vs Occupancy-MAE (Min et al., 2023):** Both mask BEV regions and predict from context. The critical difference is that Occupancy-MAE reconstructs occupancy in input space using dense 3D convolutions, while AD-L-JEPA predicts in embedding space. This gives AD-L-JEPA 2.8--3.4x less GPU memory usage and 1.9--2.7x less training time while achieving consistently better downstream detection performance.

**vs [[assran-2023-cvpr]] (I-JEPA):** AD-L-JEPA adapts the I-JEPA framework from images to LiDAR point clouds. Key adaptations include: operating on sparse 3D voxelized point clouds rather than image patches, using BEV-guided masking rather than image-block masking, introducing learnable empty tokens for sparse LiDAR regions, and using variance regularization instead of only relying on EMA for collapse prevention.

**vs [[lecun-2022-openreview]]:** AD-L-JEPA is the first concrete instantiation of LeCun's JEPA framework for autonomous driving, validating that predicting in representation space works for LiDAR data. The follow-up work AD-LiST-JEPA ([[zhu-2026-arxiv]]) extends this to spatiotemporal world modeling.

---

## Strengths

- **First JEPA for LiDAR**: Introduces the JEPA paradigm to autonomous driving LiDAR pre-training, opening a new family of methods beyond contrastive and generative approaches.
- **Dramatic efficiency gains**: 2.8--4x less GPU memory and 1.9--2.7x less training time than the SOTA Occupancy-MAE, making large-scale LiDAR pre-training more accessible.
- **Consistent improvements across datasets and detectors**: Gains are observed on KITTI3D, Waymo, and ONCE across SECOND, PV-RCNN, and CenterPoint architectures, demonstrating generalizability.
- **Thoughtful masking design**: The modified BEV-guided masking for both empty and non-empty grids is well-motivated by the sparse nature of LiDAR data and ablation-validated.
- **Comprehensive evaluation**: Transfer learning, label efficiency, scalability, and extensive ablation studies are provided. The singular value decomposition analysis (Figure 5) provides additional insight into representation quality.

---

## Weaknesses & Limitations

- **Single-frame only**: AD-L-JEPA operates on single LiDAR frames, missing temporal dynamics crucial for driving. This limitation is addressed by the follow-up AD-LiST-JEPA ([[zhu-2026-arxiv]]).
- **Moderate absolute improvements**: While consistent, the gains over Occupancy-MAE are modest (e.g., +0.84 mAP on KITTI3D R_40). The efficiency gains are arguably more significant than accuracy gains.
- **No planning or end-to-end driving evaluation**: Pre-trained features are only evaluated on 3D object detection. It remains unclear whether the learned representations benefit downstream planning or other AD tasks.
- **Limited architecture exploration**: Only the VoxelBackBone8x encoder is used. The paper does not explore ViT-based or larger architectures that might better exploit self-supervised pre-training.
- **Saturation at scale**: On ONCE, performance saturates after 500k frames and slightly drops at 1M, suggesting diminishing returns from simply scaling data without architectural improvements.

---

## Key Takeaways

- **JEPA translates effectively to LiDAR**: Predicting in BEV embedding space rather than reconstructing occupancy produces better representations for 3D object detection while being significantly more compute-efficient.
- **Modified BEV-guided masking is essential**: Masking both empty and non-empty grids (ablation shows +1.10 mAP improvement) forces the encoder to reason about all spatial regions, not just occupied ones.
- **Learnable empty tokens solve the sparse LiDAR problem**: Empty regions in sparse 3D convolutions produce near-zero vectors. Replacing them with learnable tokens prevents pathological variance issues and improves downstream performance by +1.55 mAP.
- **Variance regularization complements EMA**: Adding explicit variance regularization on non-empty grid embeddings prevents dimensional collapse and yields +0.72 mAP improvement over EMA alone.
- **Pre-training efficiency is as important as accuracy**: The 3x reduction in GPU memory makes large-scale LiDAR pre-training practical on single GPUs, which is a significant practical contribution.

---

## BibTeX

{% raw %}
```bibtex
@article{zhu2025ad,
  title={AD-L-JEPA: Self-Supervised Spatial World Models with Joint Embedding Predictive Architecture for Autonomous Driving with LiDAR Data},
  author={Zhu, Haoran and Dong, Zhenyuan and Topollai, Kristi and Sha, Beiyao and Choromanska, Anna},
  journal={arXiv preprint arXiv:2501.04969},
  year={2025}
}
```
{% endraw %}

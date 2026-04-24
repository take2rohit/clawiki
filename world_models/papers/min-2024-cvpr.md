---
title: "DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving"
type: paper
paper_id: P018
authors:
  - "Min, Chen"
  - "Xiao, Liang"
  - "Zhao, Dawei"
  - "Nie, Yiming"
  - "Li, Zheng"
  - "Ge, Zheng"
  - "Li, Yinhao"
  - "Li, Zeming"
  - "Sun, Jianjian"
  - "Tian, Tao"
  - "He, Shixiang"
  - "Ni, Bin"
  - "Dai, Bin"
year: 2024
venue: CVPR 2024
arxiv_id: "2405.04390"
url: "https://arxiv.org/abs/2405.04390"
pdf: "../../raw/min-2024-cvpr.pdf"
tags: [world-model, autonomous-driving, bev, planning, occupancy-grid]
created: 2026-04-10
updated: 2026-04-10
cites: []
cited_by:
  - feng-2025-arxiv
  - li-2025-arxiv
---

# DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving

> **DriveWorld** — a 4D world model pre-training framework using a Memory State-Space Model (MSSM) on nuScenes and OpenScene improves 3D object detection mAP by 6.4% and planning L2 error by 15.1% over supervised-only baselines at CVPR 2024.

**Authors:** Chen Min et al. (Peking University, 13 authors) | **Venue:** CVPR 2024 | **arXiv:** [2405.04390](https://arxiv.org/abs/2405.04390)

---

## Problem & Motivation

3D scene understanding for autonomous driving — 3D object detection, online mapping, motion forecasting, and planning — typically requires large amounts of expensive labeled 3D data. Self-supervised pre-training on driving video (without labels) can reduce annotation requirements, but prior approaches operate on 2D features or single-frame 3D representations, failing to capture the temporal dynamics that are critical for driving. Methods like [[hafner-2023-arxiv]] DreamerV3 and MILE use 1D compressed RSSM states that cannot reconstruct 3D spatial structures. Existing autonomous driving pre-training methods (e.g., BEV-MAE, GeoMAE) focus on static 3D representations without modeling scene dynamics. There is no pre-training approach that jointly models spatial 3D scene structure and temporal dynamics in a way that directly transfers to all major AV perception and planning tasks.

---

## Core Idea

DriveWorld introduces a 4D pre-training framework that treats autonomous driving scenes as evolving spatiotemporal entities. The key realization is that multi-camera images can be lifted to Bird's-Eye-View (BEV) feature representations, and these BEV features can be used to train a probabilistic generative world model that learns to predict both the current 3D occupancy and future 3D scenes — without any 3D labels during pre-training. The pre-trained model learns a rich 4D representation (3D space + time) through occupancy reconstruction and forecasting objectives, which then transfers to multiple downstream tasks via fine-tuning.

---

## How It Works

### Overview

Multi-camera images → BEV Representation Model → BEV feature map b_t → Memory State-Space Model (MSSM) → latent state s_t + static BEV b̂ → 3D Occupancy Decoder → predict current/future 3D occupancy → Action Decoder → predict ego-vehicle actions. At inference, BEV features + MSSM states are fine-tuned for downstream tasks using task prompts.

### BEV Representation Model

Multi-camera images (typically 6 cameras for nuScenes) are processed by an image backbone (e.g., ResNet, InternImage) and transformed to BEV space using deformable attention cross-view transformers. This produces a BEV feature map b_t ∈ R^(H×W×C) for each time step t. The BEV features capture the spatial layout of the driving scene in a canonical top-down view, providing a compact 2D representation of the 3D environment.

### Memory State-Space Model (MSSM)

The MSSM is the central innovation of DriveWorld. Unlike RSSM (used in DreamerV3 and MILE) which compresses features into a 1D tensor, MSSM:

1. Maintains a **Dynamics Memory Bank** — a set of BEV-structured memory slots that capture temporal dynamics of moving objects. The deterministic state h_t is updated as: h_t = f_θ(h_{t-1}, MLN(s_{t-1}))
2. Maintains a **Static BEV feature** b̂ — the persistent spatial context (road geometry, lane structure) that does not change over time. This separation of dynamic and static information is a key design choice.
3. Uses a **Stochastic State Model**: s_t ~ q_φ(s_t | h_t, a_{t-1}, o_t) — a Gaussian distribution conditioned on the deterministic state, previous action, and current observation.

The graphical model has: deterministic h_t (square nodes) dependent on h_{t-1} via Dynamics Memory Bank; stochastic s_t (circle nodes) dependent on h_t, actions a_t, and observations; and output y_t dependent on h_t, s_t, b̂.

**Key difference from RSSM:** RSSM compresses everything into 1D and uses RNN, which suffers from long-term memory loss. MSSM uses context BEV features to reconstruct 3D scenes and separates dynamic/static information.

### 3D Occupancy Decoder

From state (h_t, s_t, b̂), predicts:
- **Current 3D occupancy** ŷ_t: past occupancy reconstruction — supervised with semantic occupancy labels (from LiDAR fusion of multiple frames via nuScenes + OpenScene).
- **Future 3D occupancy** ŷ_{T+k}: future occupancy prediction — supervised with occupancy flow for motion-aware labels.

3D occupancy ground truth is derived from fusing multiple LiDAR frames, providing dense labels including occluded regions — a richer signal than single-frame point clouds.

### Action Decoder

Predicts ego-vehicle actions â_t ~ p_θ(â_t | h_t, s_t) as a Laplace distribution (L1 loss). This provides self-supervised action grounding during pre-training without requiring manual action labels beyond the ego-trajectory.

### Training Objective

The pre-training maximizes a variational lower bound (ELBO) on log p(y_{1:T+L}, a_{1:T+L}):

L = Σ E[log p(y_t | h_t, s_t, b̂) + log p(a_t | h_t, s_t)]
  + Σ E[log p(y_{T+k} | h_T, s_T, b̂) + log p(a_{T+k} | h_T, s_T)]
  - Σ D_KL(q(s_t | o_{≤t}, a_{<t}) || p(s_t | h_{t-1}, s_{t-1}))

This has five terms: past occupancy loss, past action loss, future occupancy loss, future action loss, and KL regularization of the posterior.

### Datasets

- **nuScenes:** 700/150/150 sequences for train/val/test. Boston and Singapore. 3D occupancy GT from multi-frame LiDAR fusion.
- **OpenScene:** Largest 3D occupancy dataset, 120+ hours, semantic labels + occupancy flow from Boston, Pittsburgh, Las Vegas, Singapore.

### Task Prompt (Inference/Fine-tuning)

Task-specific prompts are added to BEV maps before each downstream decoder. For 3D object detection: "The task is for 3D object detection of the current scene." For planning: "The task involves planning with consideration for both the current and future scenes." A shared encoder network processes task prompts that are transferred to downstream tasks during fine-tuning, enabling semantic disambiguation of features for different tasks.

### Inference

Pre-trained BEV features + MSSM states serve as general representations. For each downstream task, a task-specific decoder is fine-tuned. The task prompt encoder guides feature adaptation — BEV maps emphasize spatial accuracy for detection, and temporal context for planning.

---

## Results

### 3D Object Detection (nuScenes val)

| Method | mAP ↑ | NDS ↑ |
|--------|--------|--------|
| BEVFormer (supervised only) | 37.4 | 44.8 |
| BEVFormer + DriveWorld pre-train | **43.8** | **49.6** |
| BEV-MAE pre-train | 38.6 | 46.0 |
| GeoMAE pre-train | 40.0 | 47.2 |

DriveWorld pre-training improves mAP by +6.4pp (+17%) over supervised-only BEVFormer, and outperforms prior self-supervised pre-training methods by 3.8pp.

### Online Mapping (nuScenes val)

| Method | mIoU ↑ |
|--------|---------|
| HDMapNet (supervised) | 39.7 |
| **+ DriveWorld pre-train** | **45.2** |

+5.5pp improvement in online mapping mIoU.

### Motion Forecasting (nuScenes val)

| Method | minADE ↓ | minFDE ↓ |
|--------|---------|---------|
| FIERY (supervised) | 0.81 | 1.68 |
| **+ DriveWorld pre-train** | **0.72** | **1.45** |

Significant reductions in both minimum Average Displacement Error (minADE) and minimum Final Displacement Error (minFDE).

### Planning (nuScenes val, L2 error / collision rate)

| Method | L2 (m) ↓ | Collision (%) ↓ |
|--------|----------|----------------|
| ST-P3 (supervised) | 2.11 | 0.23 |
| **+ DriveWorld pre-train** | **1.79** | **0.17** |

15.1% reduction in L2 planning error; 26% reduction in collision rate.

### Ablations

Removing the future occupancy prediction objective (keeping only past reconstruction) drops 3D detection mAP by 2.3%, proving future dynamics modeling is critical. Removing the static BEV memory b̂ from MSSM reduces mAP by 1.8%, confirming that separating static scene context from dynamic states is necessary. Removing task prompts decreases performance by 1.2% mAP, showing that task-specific disambiguation of shared features is beneficial. Using RSSM instead of MSSM drops mAP by 4.1%, validating the architectural advantage of preserving spatial BEV structure in the state representation.

---

## Comparison to Prior Work

| Method | 3D Representation | Temporal Modeling | Pre-training Signal | Downstream Tasks |
|--------|-------------------|------------------|--------------------|--------------------|
| BEV-MAE | BEV (static) | None | Masked reconstruction | Detection, mapping |
| GeoMAE | BEV (static) | None | Masked reconstruction | Detection |
| MILE | BEV | RSSM (1D) | Reconstruction | Planning |
| [[hafner-2023-arxiv]] DreamerV3 | None (proprioceptive) | RSSM (1D) | ELBO | RL tasks |
| **DriveWorld** | BEV + 3D occupancy | MSSM (2D BEV-structured) | Occupancy + action | Detection, mapping, motion, planning |

**BEV-MAE / GeoMAE** are static masked autoencoders for BEV representations. They capture spatial structure but no temporal dynamics; DriveWorld outperforms them on detection by 3.8pp / 6.4pp respectively.

**MILE** uses an RSSM-based world model in BEV space for imitation learning. It uses a 1D compressed state which cannot reconstruct 3D geometry; DriveWorld's MSSM uses BEV-structured context for 3D reconstruction, achieving better generalization across all tasks.

**[[hafner-2023-arxiv]] DreamerV3** is a general-purpose world model for RL; its 1D RSSM state is designed for scalar reward prediction, not 3D spatial understanding.

---

## Strengths
- Covers all major AV perception and planning tasks in a single pre-training framework, providing consistent improvements across all of them.
- MSSM's separation of dynamic memory and static BEV is a principled architectural choice validated by ablations (+4.1pp over RSSM).
- Future occupancy prediction as a pre-training signal is more informative than past-only reconstruction — captures causal world dynamics.
- Task prompts provide an elegant mechanism to share a single encoder across semantically different downstream tasks.
- Uses nuScenes + OpenScene (combined 120+ hours of diverse urban driving) for pre-training — large enough for meaningful generalization.

## Weaknesses & Limitations
- Evaluation is limited to nuScenes; generalization to other datasets (Waymo, Argoverse) is not demonstrated.
- 3D occupancy ground truth requires dense LiDAR fusion — itself a labeling bottleneck that limits the scalability of this pre-training scheme.
- The task prompt design is currently simple and does not leverage LLM-scale semantic understanding; only a fixed set of AV-specific prompts are tested.
- Planning results use L2 open-loop metrics which are known to be weakly correlated with actual closed-loop performance.
- Unlike [[hu-2023-arxiv]] GAIA-1 or [[agarwal-2025-arxiv]] Cosmos, DriveWorld does not generate photorealistic video — it predicts 3D occupancy, not pixel-space observations.

## Key Takeaways
- DriveWorld pre-training improves 3D object detection mAP by +6.4pp (37.4→43.8), online mapping mIoU by +5.5pp, and planning L2 by 15.1% over supervised-only baselines.
- MSSM outperforms RSSM by 4.1pp mAP by preserving BEV spatial structure in the state representation rather than compressing to 1D.
- Future occupancy prediction is the most important pre-training signal: removing it drops 2.3pp mAP.
- The framework achieves 4D understanding (spatial 3D + temporal dynamics) from multi-camera video without requiring 3D labels during pre-training itself.
- Task prompts allow a single shared BEV encoder to be specialized for detection, mapping, forecasting, or planning at fine-tuning time.

---

## BibTeX
```bibtex
@inproceedings{min2024driveworld,
  title={DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving},
  author={Min, Chen and Xiao, Liang and Zhao, Dawei and Nie, Yiming and Li, Zheng and Ge, Zheng and Li, Yinhao and Li, Zeming and Sun, Jianjian and Tian, Tao and He, Shixiang and Ni, Bin and Dai, Bin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
  eprint={2405.04390},
  archivePrefix={arXiv}
}
```

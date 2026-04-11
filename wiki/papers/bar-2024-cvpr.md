---
title: "Navigation World Models"
type: paper
paper_id: P015
authors:
  - "Bar, Amir"
  - "Zhou, Gaoyue"
  - "Tran, Danny"
  - "Darrell, Trevor"
  - "LeCun, Yann"
year: 2024
venue: CVPR 2025 (Best Paper Honorable Mention)
arxiv_id: "2412.03572"
url: "https://arxiv.org/abs/2412.03572"
pdf: "../../raw/bar-2024-cvpr.pdf"
tags: [world-model, diffusion, planning, robotics]
created: 2026-04-10
updated: 2026-04-10
cites:
  - lecun-2022-openreview
  - alonso-2024-neurips
cited_by: []
---

# Navigation World Models

> **NWM** — a 1B-parameter Conditional Diffusion Transformer trained on egocentric robot video achieves ATE of 1.13m on RECON (vs. 1.87m for GNM and 1.95m for NoMaD) in standalone goal-conditioned visual navigation planning.

**Authors:** Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun | **Venue:** CVPR 2025 (Best Paper Honorable Mention) | **arXiv:** [2412.03572](https://arxiv.org/abs/2412.03572)

---

## Problem & Motivation

Visual navigation for robots requires planning multi-step trajectories in physical environments, but current state-of-the-art navigation policies (e.g., GNM, NoMaD) are "hard-coded" after training: they cannot dynamically incorporate new constraints such as "no left turns" or "avoid the cliff edge." They also cannot adaptively allocate more computation to harder navigation decisions. Prior world models for planning (e.g., DIAMOND) are designed for fixed game environments and do not generalize across diverse robot embodiments and outdoor/indoor environments. The core challenge is to learn a flexible, controllable world model for navigation that works across many robots and environments — trained purely from video and actions, without requiring 3D maps or explicit environment models.

---

## Core Idea

NWM learns to predict future visual observations given past frames and navigation actions, then uses this model to plan by simulating candidate trajectories and selecting those that best reach a goal image. The key insight is that video-based world modeling — trained on a diverse soup of egocentric robot data — can serve as a general navigation prior that supports both constraint-aware planning (by zeroing out disallowed actions) and trajectory ranking (by scoring externally proposed trajectories). In unknown environments, NWM imagines trajectories from a single image by leveraging its learned visual priors without any 3D reconstruction.

---

## How It Works

### Overview

Egocentric video frames + navigation actions (Δx, Δy, Δφ) → CDiT encodes past frames as context → diffusion-denoises next frame latent → pixel-space decoded image used to score/select trajectories → planning or ranking output.

### Conditional Diffusion Transformer (CDiT)

The core architecture is a novel CDiT block that efficiently conditions future frame generation on (1) a context sequence of past frames and (2) the current navigation action and time shift. Unlike a standard Diffusion Transformer (DiT), which uses self-attention over all tokens simultaneously (O(m²n²d) complexity), CDiT restricts self-attention only to tokens of the target denoised frame, and uses cross-attention to attend to the context frames. This makes complexity O(mn²d), linear in the number of context frames m. This allows using longer context sequences (up to 4 frames in the default setting) without prohibitive memory costs. For action conditioning, each scalar in the action tuple a = (u, φ, k) is mapped to R^(d/4) via sine-cosine features and a 2-layer MLP, then summed to a single conditioning vector ξ fed into AdaLN blocks. The time-shift k ∈ [T_min, T_max] allows the model to jump forward or backward in time, learning temporal dynamics without being anchored to fixed frame rates. CDiT scales favorably: CDiT-XL (1B parameters) achieves better LPIPS with 4× fewer TFLOPs than DiT-XL.

### World Model Formulation

NWM learns a stochastic mapping from past observations s_τ and action a_τ to next latent state s_{τ+1}. Observations are encoded using a pretrained Stable Diffusion VAE. Navigation actions are 3-DoF: (Δx, Δy, Δφ) representing forward/backward, left/right, and yaw rotation. The time-shift input k further specifies how many real-world seconds to jump forward, enabling the model to reason about temporal dynamics beyond frame-by-frame prediction. Action sequences can be composed: u_{τ→m} = Σ u_t and φ_{τ→m} = Σ φ_t mod 2π.

### Training

- **Loss:** Standard diffusion denoising MSE — L_simple = E[||s_{τ+1} - F_θ(s^(t)_{τ+1} | s_τ, a_τ, t)||²] — plus variational lower bound loss L_vlb for covariance prediction.
- **Datasets (known environments):** SCAND (8.7h, 138 trajectories), TartanDrive (5h off-road), RECON (40h, 9 open-world environments), HuRoN (75h indoor social robot).
- **Unlabeled data:** Ego4D (1619 videos, 908h) — for which only the time-shift action is used (no translation/rotation).
- **Model sizes:** CDiT-S (5M), CDiT-B (20M), CDiT-L (80M), CDiT-XL (1B parameters). Default experiments use CDiT-XL with 4 context frames and 4 navigation goals.
- **Compute:** 8 H100 machines × 8 GPUs = 64 H100s; AdamW optimizer, lr = 8e-5.

### Inference

**Standalone planning:** Uses the Cross-Entropy Method (CEM) — a gradient-free stochastic optimizer. Trajectories of length 8 with time shift k=0.25s are sampled from a Gaussian distribution. Each candidate is simulated through NWM, and the last predicted frame is scored by LPIPS similarity to the goal image. The distribution is updated toward top performers. Constraints (e.g., "no left turns") are enforced by zeroing out disallowed action components.

**Ranking:** Sample N ∈ {16, 32} trajectories from an external policy (NoMaD), simulate each through NWM, score by LPIPS to goal, select the best. This improves upon the base policy without any retraining.

---

## Results

### Goal-Conditioned Visual Navigation (RECON)

| Model | ATE ↓ | RPE ↓ |
|-------|--------|--------|
| GNM | 1.87 | 0.73 |
| NoMaD | 1.95 | 0.52 |
| NWM + NoMaD (×16) | 1.83 | 0.50 |
| NWM + NoMaD (×32) | 1.78 | 0.48 |
| **NWM (planning only)** | **1.13** | **0.35** |

NWM as a standalone planner outperforms all prior methods by a large margin — 40% lower ATE than GNM. This proves that video-based world model planning can exceed specialized navigation policies trained with labeled data.

### Video Synthesis Quality (FVD, 16s at 4FPS on RECON)

| Model | FVD ↓ |
|-------|--------|
| DIAMOND | 762.7 ± 3.4 |
| **NWM (ours)** | **201.0 ± 5.6** |

NWM produces significantly higher quality video predictions — 3.8× lower FVD — confirming that CDiT is a better video world model architecture than U-Net-based DIAMOND.

### Generalization to Unknown Environments (Go Stanford)

| Training Data | LPIPS ↓ | DreamSim ↓ | PSNR ↑ |
|---------------|---------|------------|--------|
| In-domain only | 0.658 | 0.478 | 11.031 |
| **+ Ego4D (unlabeled)** | **0.652** | **0.464** | **11.083** |

Adding 908 hours of unlabeled Ego4D video (using only time-shift action) improves prediction quality on completely unseen environments, demonstrating that large-scale unlabeled egocentric data transfers visual priors to novel settings.

### Ablations

Removing action conditioning and using time only causes catastrophic degradation (LPIPS 0.760 vs. 0.296), proving action grounding is essential. Removing time conditioning with action-only also hurts (0.318 LPIPS), confirming both inputs are complementary. Increasing context frames from 1 to 4 yields consistent improvement across all metrics. Increasing the number of planning goals from 1 to 4 improves prediction by 5.5% in PSNR. CDiT is 4× faster than DiT at equal parameter count, proving the architectural design is both better and more efficient.

---

## Comparison to Prior Work

| Method | Paradigm | Action-Conditioned | Multi-Environment | Open-World Planning |
|--------|----------|-------------------|-------------------|---------------------|
| DIAMOND | U-Net diffusion WM | Yes (game-specific) | No | No |
| NoMaD | Diffusion policy | Yes | Yes | No |
| GNM | Discriminative nav. | Yes | Yes | No |
| **NWM (ours)** | CDiT diffusion WM | Yes | Yes | Yes |

**[[alonso-2024-neurips]] ([DIAMOND](../papers/alonso-2024-neurips.md))** is a diffusion world model for Atari/game environments trained with offline RL. It uses a U-Net backbone and fixed action spaces; NWM outperforms it 3.8× on FVD and can generalize across diverse physical environments.

**NoMaD** is a strong goal-conditioned diffusion navigation policy trained on the same datasets. NWM ranks its trajectories to improve it by 5–10%, and NWM standalone planning beats NoMaD by 42% in ATE.

**GNM** is a general navigation model using a discriminative graph-based policy. NWM in standalone planning achieves 40% lower ATE; combining NWM ranking with NoMaD also outperforms GNM.

---

## Strengths
- CDiT architecture elegantly solves the context-scaling problem for video diffusion: linear rather than quadratic complexity in context length.
- Demonstrated planning under explicit constraints (action zeroing) without any retraining — a major practical advantage over hard-coded policies.
- Shows that unlabeled egocentric data (Ego4D) with minimal annotation (only time shift) improves generalization to novel environments.
- Single model generalizes across diverse robots (wheeled, legged, humanoid-POV) and environments (indoor, outdoor, off-road, urban).
- CVPR 2025 Best Paper Honorable Mention validates community recognition of the contribution.

## Weaknesses & Limitations
- Mode collapse: in unknown environments the model slowly generates frames that resemble training distribution rather than the actual scene ("hallucination" after ~8 seconds).
- Struggles with simulating non-rigid dynamic agents (pedestrians), though partial success is shown.
- Only 3-DoF navigation actions (flat surface assumption); 6-DoF extension (e.g., drone, robotic arm) left for future work.
- Inference speed is currently 30s per trajectory on RTX 6000 Ada; distillation reduces to 14.7s, quantization could reach <1s but not yet demonstrated.
- NWM does not explicitly learn a structured map; it remains unclear what geometric representation emerges in its latent space.

## Key Takeaways
- NWM achieves ATE of 1.13m (standalone) on RECON — 40% better than NoMaD (1.95m) and GNM (1.87m) — proving world model planning surpasses learned navigation policies.
- CDiT is 4× faster than DiT at equal parameter count and achieves better LPIPS, validating the cross-attention context design.
- Adding 908h of unlabeled Ego4D data (time-shift action only) improves generalization to novel environments, quantified by DreamSim improvement from 0.478 to 0.464.
- Constraint-aware planning (zeroing disallowed actions) adds only minor performance degradation (Table 3), enabling safe deployment constraints without retraining.
- FVD of 201 vs. 763 for DIAMOND demonstrates that CDiT is a substantially better video world model backbone for navigation.

---

## BibTeX
```bibtex
@inproceedings{bar2024navigation,
  title={Navigation World Models},
  author={Bar, Amir and Zhou, Gaoyue and Tran, Danny and Darrell, Trevor and LeCun, Yann},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
  note={Best Paper Honorable Mention},
  eprint={2412.03572},
  archivePrefix={arXiv}
}
```

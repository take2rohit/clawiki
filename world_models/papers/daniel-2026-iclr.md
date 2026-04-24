---
title: "Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling"
type: paper
paper_id: P048
authors:
  - "Daniel, Tal"
  - "Qi, Carl"
  - "Haramati, Dan"
  - "Zadeh, Amir"
  - "Li, Chuan"
  - "Tamar, Aviv"
  - "Pathak, Deepak"
  - "Held, David"
year: 2026
venue: ICLR 2026 (Oral)
arxiv_id: "2603.04553"
url: "https://arxiv.org/abs/2603.04553"
pdf: "../../raw/daniel-2026-iclr.pdf"
tags: [world-model, object-centric, self-supervised, stochastic-dynamics, keypoints, latent-actions, particles, vae, video-prediction, imitation-learning]
created: 2026-04-15
updated: 2026-04-15
cites: []
cited_by: []
---

# Latent Particle World Models (LPWM)

> **One sentence** — LPWM is a self-supervised object-centric world model that discovers keypoints, bounding boxes, and object masks from video without any supervision, models stochastic per-particle dynamics via a novel latent action module, and achieves state-of-the-art video prediction on diverse real-world and simulated multi-object datasets while also supporting goal-conditioned imitation learning.

**Authors:** Tal Daniel, Carl Qi, Dan Haramati, Amir Zadeh, Chuan Li, Aviv Tamar, Deepak Pathak, David Held | **Venue:** ICLR 2026 (Oral) | **arXiv:** [2603.04553](https://arxiv.org/abs/2603.04553)

---

## Problem & Motivation

General-purpose video generation models (diffusion-based, Transformer-based) achieve high visual fidelity but are computationally prohibitive — training requires thousands of GPU hours, inference is slow, and they do not decompose scenes into objects, making them unsuitable as world models for decision-making. Object-centric approaches (slot-based, patch-based, particle-based) offer structured, interpretable, and more compact representations, but have so far been confined to simple simulated environments or single-agent real-world settings with limited interactions. Scaling object-centric world models to complex, real-world, multi-object environments remains a substantial open challenge. Prior particle-based methods like DDLP (Daniel & Tamar, 2024) rely on explicit particle tracking and sequential encoding, which restricts parallelization and prevents stochastic dynamics modeling. Meanwhile, existing latent action approaches use a single global action vector for all scene changes, which cannot disentangle independent per-object dynamics in multi-entity environments. The paper asks: can we build a self-supervised, object-centric world model that scales to real-world complexity, models stochastic dynamics, and supports flexible conditioning for decision-making?

---

## Core Idea

LPWM combines Deep Latent Particle (DLP) representations — where each object is a compact latent "particle" encoding position, scale, depth, transparency, and appearance — with a novel per-particle latent action module that captures stochastic transitions between frames. Unlike prior work that uses a single global latent action for the entire scene, LPWM assigns each particle its own continuous latent action, enabling natural disentanglement of independent object dynamics (e.g., multiple enemies moving simultaneously in Mario, or multiple robotic arms). The latent actions are learned end-to-end via a latent inverse dynamics head (training) and a latent policy prior (inference), eliminating the need for explicit tracking or sequential encoding. This design enables parallel encoding of all frames, stochastic rollouts at inference time, and flexible conditioning on actions, language, images, or goal states — advancing particle-based representations into a full world model capable of decision-making in complex multi-object environments.

---

## How It Works

### Architecture Overview

LPWM is a temporal VAE with four jointly trained components:

1. **Encoder** E_phi — encodes each frame into a set of M foreground latent particles + 1 background particle
2. **Decoder** D_theta — reconstructs images from particle sets via learned RGBA glimpses composited onto the canvas
3. **Context** K_psi — the novel per-particle latent action module; models stochastic transitions
4. **Dynamics** F_xi — a causal spatio-temporal Transformer that autoregressively predicts next-timestep particle states

The pipeline: frames are encoded into particle sets by the Encoder, decoded back to images by the Decoder for reconstruction loss, processed by the Context module to sample per-particle latent actions, and combined with particles in the Dynamics module to predict next-step particle states via KL-divergence matching.

### Encoder E_phi

Takes an image frame and outputs a set of latent particles. Each frame I_t is represented by M foreground particles {z_fg^{m,t}}_{m=0}^{M-1} and one background particle z_bg^t. Each foreground particle z_fg^m is in R^{6+d_obj}, where the first 6 dimensions encode disentangled stochastic attributes:

- **Position** z_p ~ N(mu_p, sigma_p^2) in R^2 — 2D keypoint coordinates
- **Scale** z_s ~ N(mu_s, sigma_s^2) in R^2 — bounding-box height and width
- **Depth** z_d ~ N(mu_d, sigma_d^2) in R — compositing order (front-to-back)
- **Transparency** z_t ~ Beta(a, b) in [0, 1] — visibility
- **Appearance** z_f ~ N(mu_f, sigma_f^2) in R^{d_obj} — visual features of the local region

Particles originate from per-patch learned keypoints (no explicit tracking required, unlike DDLP). All frames are encoded in parallel. The background particle z_bg encodes visual features from a masked version of the image where visible foreground regions are removed.

### Decoder D_theta

Takes a set of L <= M foreground particles plus the background particle and reconstructs an image frame. Particle filtering (L < M) based on transparency or confidence reduces memory usage. Each particle is decoded independently into an RGBA glimpse in R^{S x S x 4} (S is the glimpse size). The alpha channel is modulated by the particle's transparency and depth attributes. Foreground glimpses are composited to produce x_hat_fg, the background is decoded via a standard upsampling network to produce x_hat_bg, and the final reconstruction is:

x_hat = alpha * x_hat_fg + (1 - alpha) * x_hat_bg

where alpha is the effective mask from the compositing process. This yields per-object masks, bounding boxes, and keypoints as emergent byproducts of reconstruction.

### Context Module K_psi (Novel Contribution)

The key architectural innovation. Designed to model per-particle stochastic dynamics in actionless videos. It is implemented as a causal spatio-temporal Transformer that processes particles across space and time with autoregressive temporal dependencies.

The Context module has two complementary heads:

1. **Latent inverse dynamics** p_psi^inv(z_c^t | z^{t+1}, z^t, ..., z^0, c_t) — predicts the latent action responsible for the transition between consecutive states. Used during training to ensure latent actions are consistent with observed transitions.

2. **Latent policy** p_psi^policy(z_c^t | z^t, ..., z^0, c_t) — models the distribution of latent actions conditioned on the current state only. Used at inference time to sample stochastic rollouts without access to future frames.

Latent actions are modeled as Gaussian distributions z_c ~ N(mu_c, sigma_c^2), one per particle per timestep. The latent policy serves as the prior that regularizes the inverse dynamics via a KL-divergence penalty in the VAE objective. This per-particle formulation (vs. global latent actions in prior work) enables:

- Representation of multiple simultaneous independent object interactions
- Capture of multimodality (e.g., an object may move left or right from the same state)
- Natural disentanglement of stochastic aspects from deterministic dynamics

**Conditioning:** The Context module optionally accepts external signals c_t (global actions, language instructions, goal images). It maps these global scene-level signals into per-particle latent actions. For instance, given a language instruction "move the blue cube diagonal to the red circle," K_psi translates this into per-particle latent actions that drive only the relevant particles.

### Dynamics Module F_xi

Implements the autoregressive dynamics prior p_xi(z^t | z^{t-1}, ..., z^0). It is a causal spatio-temporal Transformer conditioned on current particles and their corresponding latent actions via AdaLN (Adaptive Layer Normalization). Outputs distribution parameters for predicting next-timestep particle states, which serve as the prior in the KL-divergence between the encoder posterior and the dynamics prior.

Unlike DDLP, LPWM retains all M encoded particles with their identities (patch origins) across timesteps, operating in an implicit tracking regime where particles can move within a certain region around their origin. This eliminates explicit tracking, enabling parallel encoding.

### Training

LPWM is trained end-to-end by maximizing a temporal ELBO, decomposed into static and dynamic terms:

L_LPWM = -sum_{t=0}^{T-1} ELBO(x_t = I_t) = L_static + L_dynamic

- **Static term** (first frame): single-frame VAE setting — per-particle KL with respect to fixed priors, plus regularization on particle transparency
- **Dynamic term** (subsequent frames): includes KL losses for both latent actions and predicted future particles; reconstruction losses for all frames

Reconstruction losses:
- Pixel-wise MSE for simulated datasets
- MSE + LPIPS for real-world data

KL contributions are masked using the particle transparency attribute so only visible particles affect the KL loss (a key difference from DDLP). Latent action dimension d_ctx = 7 for all experiments. Optimized with Adam (lr = 8 x 10^{-5}). All datasets trained at 128 x 128 resolution. Implemented in PyTorch.

### Inference

At inference time, latent actions are sampled from the **latent policy** head (not the inverse dynamics head, which requires future frames). Given initial conditioning frames, LPWM autoregressively generates future particle states by:

1. Encoding initial frames into particle sets
2. Sampling per-particle latent actions from the latent policy
3. Predicting next-step particles via the Dynamics module
4. Decoding particles back to images via the Decoder

Stochastic sampling from the latent policy yields diverse plausible rollouts from the same initial conditions. When conditioning on external signals (actions, language, goal images), the Context module maps these into per-particle latent actions that replace or bias the policy sampling.

---

## Results

### Stochastic Video Prediction (Table 2 — Main Results)

Evaluated on unconditional (U), action-conditioned (A), and language-conditioned (L) video prediction with FVD (lower is better) for stochastic generation and LPIPS (lower is better) for perceptual quality.

| Dataset | Setting | Model | PSNR | SSIM | LPIPS | FVD |
|---------|---------|-------|------|------|-------|-----|
| Sketchy-U | t=20, c=6, p=44 | DVAE | 25.75 | 0.86 | 0.113 | 140.06 |
| | | PlaySlot | — | — | — | — |
| | | **LPWM** | **28.41** | **0.91** | **0.079** | **85.45** |
| BAIR-U | t=16, c=1, p=15 | DVAE | 26.3 | 0.90 | 0.063 | 164.41 |
| | | PlaySlot | 17.56 | 0.57 | 0.483 | — |
| | | **LPWM** | **25.66** | **0.89** | **0.062** | **163.91** |
| Mario-U | t=20, c=6, p=44 | DVAE | 24.35 | 0.93 | 0.087 | 277.41 |
| | | PlaySlot | 16.38 | 0.68 | 0.314 | — |
| | | **LPWM** | **27.50** | **0.95** | **0.045** | **195.95** |
| Sketchy-A | t=20, c=6, p=44 | DVAE | 25.33 | 0.85 | 0.111 | — |
| | | **LPWM** | **27.06** | **0.88** | **0.083** | — |
| LanguageTable-L | t=20, c=1, p=15 | DVAE | 29.29 | 0.94 | 0.038 | 26.78 |
| | | **LPWM** | **29.5** | **0.94** | **0.037** | **15.96** |

LPWM outperforms all baselines on LPIPS and FVD across stochastic dynamic datasets under all conditioning settings. It preserves object permanence throughout generation and models complex multi-object interactions without blurring or deformation. A compact LPWM model trained on BAIR-64 matches larger video generation models in FVD (89.4, Table 9 in appendix), demonstrating that object-centric inductive biases can compensate for raw scale.

### Imitation Learning (Table 3)

Goal-conditioned imitation learning on PandaPush (robotic cube pushing) and OGBench-Scene (long-horizon planning with drawers and buttons):

| Task (PandaPush) | EC Diffusion Policy | EC Diffuser | LPWM |
|------------------|---------------------|-------------|------|
| 1 Cube | 88.7 +/- 3 | 94.8 +/- 1.5 | 92.7 +/- 4.5 |
| 2 Cubes | 38.8 +/- 10.6 | 91.7 +/- 3 | 74 +/- 4 |
| 3 Cubes | 66.8 +/- 17 | 89.4 +/- 2.5 | 62.1 +/- 4.4 |

| Task (OGBench-Scene) | GCIVL | HIQL | LPWM |
|----------------------|-------|------|------|
| task1 | 84 +/- 4 | 80 +/- 6 | **100 +/- 0** |
| task2 | 24 +/- 8 | 81 +/- 7 | 6 +/- 9 |
| task3 | 16 +/- 8 | 61 +/- 11 | **89 +/- 9** |

On PandaPush, LPWM outperforms all baselines except EC Diffuser (which trains separate per-task policies, while LPWM trains a single model for all tasks). The multi-view LPWM variant is used here, demonstrating the framework's flexibility. On OGBench, LPWM achieves 100% success on task1 and 89% on task3, outperforming all baselines on tasks involving up to four atomic behaviors. Performance drops on task2/task4/task5 due to challenges from suboptimal unstructured "play" training data.

### Ablation Analysis

Key ablation findings (Appendix A.10.3):

- **Per-particle vs. global latent actions:** Per-particle latent actions are essential for strong performance — global latent actions cannot disentangle independent object dynamics in multi-entity scenes
- **Latent action dimension:** The model is robust to latent action dimension near the effective particle dimension (6 + d_obj); d_ctx = 7 used for all experiments
- **Positional embeddings:** AdaLN-based positional information outperforms standard additive positional embeddings

---

## Comparison to Prior Work

| Dimension | LPWM | DVAE (patch baseline) | PlaySlot | DDLP | DreamerV3 |
|-----------|------|-----------------------|----------|------|-----------|
| Representation | Particles (keypoints) | Patches | Slots | Particles | Categorical latents (RSSM) |
| Object-centric | Yes | No | Yes | Yes | No |
| Self-supervised | Yes | Yes | Partial | Yes | Needs rewards |
| Latent actions | Continuous, per-particle | — | Discrete, global | — | — (uses env actions) |
| End-to-end | Yes | Yes | No (2-stage) | Yes | Yes |
| Stochastic dynamics | Yes | — | — | No | Yes (via RSSM) |
| Text conditioning | Yes | — | — | No | No |
| Multi-view | Yes | — | — | No | No |
| Dynamics module | Transformer | Transformer | Transformer | Transformer | GRU (RSSM) |

**[[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)):** DreamerV3 is a powerful general-purpose world model for RL that uses categorical latent states in an RSSM without any object-centric decomposition. It requires environment actions and reward signals for training. LPWM occupies a complementary niche — it is self-supervised (trained purely from video), discovers object structure without supervision, and learns its own latent action space. DreamerV3 excels at single-agent RL across 150+ tasks; LPWM targets multi-object scene understanding, video prediction, and goal-conditioned imitation learning. LPWM's per-particle latent actions are conceptually analogous to DreamerV3's action-conditioned RSSM transitions but operate at the object level rather than the scene level.

**[[maes-2026-arxiv]] ([LeWorldModel](../papers/maes-2026-arxiv.md)):** LeWorldModel is a stable end-to-end JEPA that learns predictive representations from pixels without reconstruction. Like LPWM, it is self-supervised and operates from pixels, but it uses a patch-based representation without object-centric decomposition. LPWM's explicit object structure (keypoints, bounding boxes, masks) provides interpretability and compositional generalization that patch-based JEPAs lack, at the cost of requiring the object-centric inductive bias to be applicable to the domain.

**[[lecun-2022-openreview]] ([A Path Towards Autonomous Machine Intelligence](../papers/lecun-2022-openreview.md)):** LeCun's position paper advocates for world models that learn structured, hierarchical representations of the world for planning and reasoning. LPWM's object-centric decomposition with per-particle latent actions aligns with this vision — it learns a structured, compositional world model where each entity has its own dynamics, and planning can operate at the object level. The latent policy prior in LPWM can be seen as a learned "actor" in LeCun's framework, while the dynamics module serves as the "world model" predictor. However, LPWM does not implement the hierarchical multi-scale prediction or energy-based formulation that LeCun envisions.

**DDLP (Daniel & Tamar, 2024):** LPWM's direct predecessor. DDLP jointly trains a Transformer dynamics model with the DLP particle representation, but relies on explicit particle tracking and sequential frame encoding, which prevents parallelization and stochastic modeling. LPWM eliminates tracking by retaining particle identities via patch origins, encodes all frames in parallel, and introduces the per-particle latent action module for stochastic dynamics.

**PlaySlot (Villar-Corrales & Behnke, 2025):** A slot-based approach with discrete global latent actions. LPWM outperforms PlaySlot substantially on all shared benchmarks due to: (1) continuous per-particle latent actions vs. discrete global ones, (2) particle representations that scale to many objects vs. slots that suffer from inconsistent decompositions, and (3) end-to-end training vs. two-stage slot inference + dynamics.

---

## Strengths

- First self-supervised object-centric world model that scales to complex real-world multi-object datasets (BAIR, Bridge, Sketchy, LanguageTable), not just toy simulations
- Per-particle latent actions are a principled and effective solution for modeling independent stochastic dynamics of multiple objects simultaneously — a clear advance over global latent action approaches
- Discovers interpretable scene structure (keypoints, bounding boxes, per-object masks) purely from video reconstruction without any supervision or annotation
- Supports flexible conditioning on actions, language, images, and multi-view inputs within a single unified architecture
- End-to-end training (unlike two-stage slot methods) simplifies the pipeline and avoids error compounding between decomposition and dynamics stages
- Parallel frame encoding eliminates the sequential bottleneck of DDLP, improving training efficiency
- Practical applicability demonstrated beyond video prediction: goal-conditioned imitation learning on multi-object manipulation tasks achieves competitive or state-of-the-art results
- Compact model matches larger video generation models on FVD (89.4 on BAIR-64), validating the efficiency advantage of object-centric inductive biases

## Weaknesses

- Operates in an implicit tracking regime where particles move within a region around their patch origin — this limits the ability to track objects that traverse the entire canvas, and the paper acknowledges this tracking limitation (Appendix A.4.4)
- All experiments are at 128 x 128 resolution — scalability to higher-resolution real-world video (256+) is not demonstrated
- The number of particles M is a fixed hyperparameter that must be set per dataset; the model does not dynamically adjust the number of objects
- Imitation learning results on PandaPush with 2 and 3 cubes lag behind EC Diffuser, which trains per-task policies — the single-model advantage comes at a performance cost on harder manipulation tasks
- OGBench results are mixed (strong on task1/task3, weak on task2/task4/task5), suggesting sensitivity to suboptimal training data and task complexity
- No comparison to large-scale video generation models (e.g., diffusion-based world models) on shared benchmarks — the baselines are primarily other object-centric or small-scale models
- The latent policy prior may not capture complex multi-modal action distributions well, as it is parameterized as a unimodal Gaussian per particle
- Requires the object-centric inductive bias to be a good fit for the domain — may not generalize to scenes without well-defined discrete objects (e.g., fluid dynamics, deformable objects)

## Key Takeaways

- Per-particle latent actions are strictly superior to global latent actions for modeling stochastic dynamics in multi-entity scenes — ablations confirm this is essential, not just beneficial
- Object-centric inductive biases (keypoints, bounding boxes, masks) emerge naturally from a reconstruction objective on video data without any supervision, enabling interpretable world models at scale
- A compact object-centric model can match or exceed larger patch-based and diffusion-based video generation models on FVD metrics, demonstrating that structured representations provide a strong efficiency advantage
- Self-supervised pre-training on actionless video produces latent actions that effectively encode dynamics — these can be mapped to real actions with a simple attention-based policy head, enabling goal-conditioned imitation learning without action labels during world model training
- The representation-dynamics co-training principle ("the representation is trained to be predictable by the dynamics module") yields better decompositions than two-stage approaches where decomposition is learned independently of dynamics

---

## BibTeX
{% raw %}
```bibtex
@inproceedings{daniel2026lpwm,
  title     = {Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling},
  author    = {Daniel, Tal and Qi, Carl and Haramati, Dan and Zadeh, Amir and Li, Chuan and Tamar, Aviv and Pathak, Deepak and Held, David},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.04553},
  eprint    = {2603.04553},
  archivePrefix = {arXiv}
}
```
{% endraw %}

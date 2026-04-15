---
title: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
type: paper
paper_id: P031
authors:
  - "Maes, Lucas"
  - "Le Lidec, Quentin"
  - "Scieur, Damien"
  - "LeCun, Yann"
  - "Balestriero, Randall"
year: 2026
venue: arXiv
arxiv_id: "2603.19312"
url: "https://arxiv.org/abs/2603.19312"
pdf: "../../raw/maes-2026-arxiv.pdf"
tags: [world-model, JEPA, self-supervised-learning, latent-planning, robotics, model-predictive-control]
created: 2026-04-10
updated: 2026-04-10
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - balestriero-2025-iclr
  - ha-2018-neurips
  - micheli-2023-iclr
  - alonso-2024-neurips
  - hansen-2022-icml
  - hansen-2024-iclr
  - terver-2025-iclr
cited_by:
  - nam-2026-arxiv
  - zhang-2026-arxiv
---

# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

> **LeWorldModel (LeWM)** is the first JEPA world model trained stably end-to-end from raw pixels without stop-gradients or pretrained encoders, using only two loss terms — a next-embedding MSE prediction loss and SIGReg — achieving 48× faster planning than DINO-WM while matching or exceeding it on most control tasks with a 15M-parameter model trainable on a single GPU.

**Authors:** Lucas Maes\*, Quentin Le Lidec\* (equal contribution), Damien Scieur, Yann LeCun (NYU), Randall Balestriero (Brown Univ.) | **Venue:** arXiv (March 2026) | **arXiv:** [2603.19312](https://arxiv.org/abs/2603.19312)

---

## Problem & Motivation

JEPA is a natural framework for learning latent world models — it focuses on predicting abstract representations of future states rather than reconstructing pixels, keeping the model compact and focused on dynamically relevant information. But applying JEPA to world modeling hits the same collapse problem as in static SSL: without constraints, the encoder collapses all inputs to the same embedding, making the prediction trivially satisfied but the model useless.

Existing approaches resolve this in ways that each introduce serious costs:

- **Foundation-based methods (DINO-WM)**: Freeze a large pretrained DINOv2 encoder and only learn the dynamics predictor. Avoids collapse but inherits the frozen encoder's distribution, limiting expressivity, and requires a massive pretrained model (DINOv2 trained on 124M images).
- **End-to-end with VICReg-based loss (PLDM)**: Learns both encoder and predictor from scratch using a 7-term VICReg-style objective. Trains end-to-end but introduces 6 hyperparameters, unstable non-monotonic training curves, and O(n⁶) hyperparameter search.
- **Task-specific RL-based (Dreamer, TD-MPC)**: Require reward signals or privileged state access — not available in generic observation-only settings.

LeWM addresses all four limitations: it is end-to-end, task-agnostic, pixel-only, reconstruction-free, and has just one effective hyperparameter.

---

## Core Idea

Balestriero & LeCun (LeJEPA) proved that embeddings should follow an isotropic Gaussian distribution and provided SIGReg to enforce this. LeWM takes that regularizer and applies it directly to the sequential world model setting: train an encoder and an action-conditioned predictor jointly from raw pixel observations, using next-embedding MSE as the dynamics objective and SIGReg as the only anti-collapse mechanism. The entire anti-collapse apparatus — stop-gradients, EMA teachers, negative samples, feature whitening — disappears. What remains is a two-term loss that is theoretically grounded, produces smooth monotonic training curves, and runs on a single GPU in a few hours.

---

## How It Works

### Overview

LeWM consists of two components trained jointly from offline trajectory data: an **Encoder** that maps raw pixel observations to compact latent vectors, and an **action-conditioned Predictor** that models temporal dynamics by predicting the next latent state. At test time, the frozen world model serves as the simulator for latent planning via Model Predictive Control (MPC).

### Encoder

A Vision Transformer (ViT-tiny, ~5M parameters):
- Patch size: 14; 12 layers; 3 attention heads; hidden dim: 192
- Output: [CLS] token embedding from last layer
- Projection: 1-layer MLP with Batch Normalization maps CLS embedding to latent z_t

The Batch Normalization is critical — the ViT's final Layer Normalization would prevent the SIGReg objective from being optimized effectively, so BN replaces it in the projection head.

### Predictor

A Transformer (~10M parameters):
- 6 layers, 16 attention heads, 10% dropout
- Action conditioning: Adaptive Layer Normalization (AdaLN) at each layer, initialized to zero so action conditioning ramps up progressively
- Input: history of N frame embeddings with temporal causal masking
- Auto-regressive: predicts next embedding given current latent z_t and action a_t
- Followed by a projector network (same architecture as encoder's projector)

### Training Objective

The total training objective is:

```
L_LeWM = L_pred + λ · SIGReg(Z)
```

**Prediction loss** (teacher-forcing MSE):
```
L_pred = ||ẑ_{t+1} - z_{t+1}||²,   ẑ_{t+1} = pred(z_t, a_t)
```

**SIGReg** (anti-collapse, from [[balestriero-2025-iclr]]):
```
SIGReg(Z) = (1/M) ∑_{m=1}^M T(Zu^(m))
```
where u^(m) are M random unit-norm directions and T is the Epps-Pulley test comparing the 1D projected distribution to a standard Gaussian. Applied to embeddings Z collected over history length N.

**Default hyperparameters**: λ=0.1, M=1024 projections. M has negligible effect on downstream performance — λ is the only effective hyperparameter, searchable via binary bisection in O(log n) time (vs. PLDM's O(n⁶) polynomial search across 6 params).

All gradients flow through both encoder and predictor — no stop-gradient, no EMA.

```python
def LeWorldModel(obs, actions, lambd=0.1):
    emb = encoder(obs)              # (B, T, D)
    next_emb = predictor(emb, actions)  # (B, T, D)
    pred_loss = F.mse_loss(emb[:, 1:] - next_emb[:, :-1])
    sigreg_loss = mean(SIGReg(emb.transpose(0, 1)))
    return pred_loss + lambd * sigreg_loss
```

**Model scale**: ~15M total parameters — trainable on a single GPU in a few hours.

### Latent Planning (Inference)

Given initial observation o₁ and goal observation o_g:

1. Encode: z₁ = enc(o₁), z_g = enc(o_g)
2. Initialize random action sequence a_{1:H}
3. Roll out the predictor auto-regressively: ẑ_{t+1} = pred(ẑ_t, a_t) for H steps
4. Compute terminal cost: C(ẑ_H) = ||ẑ_H - z_g||²
5. Optimize a_{1:H} using Cross-Entropy Method (CEM) — iteratively update sampling distribution toward best-performing action sequences
6. Execute only first K actions (MPC receding horizon), then replan from new observation

The planning horizon H trades off long-horizon foresight against compounding prediction error. Full planning completes in **0.98 seconds** (vs. 47s for DINO-WM), making near-real-time control feasible.

### Data

Fully offline, reward-free trajectories: (o_{1:T}, a_{1:T}) collected from behavioral policies (no optimality requirements). No reward signals, no task labels, no goal annotations during training. This is the same setup as the JEPA line of work — generic world models from passive observation.

---

## Results

### Planning Performance (Figure 6, success rate %)

Four environments: Two-Room (2D navigation), Reacher (2D arm), Push-T (2D manipulation), OGBench-Cube (3D manipulation). Baselines: PLDM (end-to-end VICReg-based), DINO-WM (frozen DINOv2 encoder), GCBC/GCIQL (goal-conditioned behavioral cloning / offline RL).

| Environment | LeWM | PLDM | DINO-WM | GCBC | GCIQL |
|---|---|---|---|---|---|
| Two-Room | 87 | 100 | 100 | 100 | 100 |
| Reacher | 84 | 78 | 79 | — | — |
| Push-T | **90** | 13 | 75† | — | — |
| OGBench-Cube | 74 | ~50 | **86** | — | — |

†DINO-WM here used additional proprioceptive inputs; LeWM uses pixels only.

**Interpreting the results:**
- **Push-T** is the most striking: LeWM achieves 90% vs. PLDM's 13% — a 77 percentage-point gap showing that the 7-term VICReg objective genuinely fails on harder tasks while SIGReg alone succeeds. LeWM also beats DINO-WM (75%) even though DINO-WM has extra proprioception.
- **Reacher**: LeWM slightly outperforms both PLDM and DINO-WM (84 vs. 78/79).
- **OGBench-Cube**: DINO-WM wins (86 vs. 74), likely because DINOv2's pretraining on 124M diverse images gives it richer 3D visual features. This is the one setting where frozen pretraining pays off.
- **Two-Room**: PLDM and DINO-WM both reach 100%; LeWM reaches 87%. The low intrinsic dimensionality of this environment (agent position in a 2D room) makes matching an isotropic Gaussian in a high-dimensional latent space unnecessarily difficult — SIGReg spreads features across all dimensions when the signal truly only needs a few.

### Planning Speed (Figure 3)

| | LeWM | DINO-WM |
|---|---|---|
| Full planning time | **0.98s** | 47s |
| Speedup | **48×** faster | — |
| Push-T success (fixed FLOPs) | **90** | 13 |
| OGBench-Cube success (fixed FLOPs) | **74** | 48 |

At equal compute budget (fixed FLOPs), LeWM dramatically outperforms DINO-WM. The gap exists because DINO-WM encodes ~200× more tokens per observation (large pretrained ViT with all patch tokens vs. LeWM's single CLS token), making each rollout step much slower.

### Physical Understanding: Probing (Table 1, Push-T)

Physical quantities recoverable from latent representations via linear/MLP probes:

| Property | Model | Linear MSE↓ | Linear r↑ | MLP MSE↓ | MLP r↑ |
|---|---|---|---|---|---|
| Agent Location | DINO-WM | 1.888 | 0.977 | 0.003 | **0.999** |
| Agent Location | PLDM | 0.090 | 0.955 | 0.014 | 0.993 |
| Agent Location | **LeWM** | **0.052** | **0.974** | **0.004** | **0.998** |
| Block Location | DINO-WM | **0.006** | **0.997** | 0.002 | **0.999** |
| Block Location | PLDM | 0.122 | 0.938 | 0.011 | 0.994 |
| Block Location | **LeWM** | **0.029** | **0.986** | **0.001** | **0.999** |
| Block Angle | DINO-WM | **0.050** | **0.979** | 0.009 | **0.995** |
| Block Angle | PLDM | 0.446 | 0.745 | 0.056 | 0.972 |
| Block Angle | **LeWM** | 0.187 | 0.902 | 0.021 | 0.990 |

LeWM consistently outperforms PLDM on all three properties, and the MLP probes achieve near-DINOv2 quality — remarkable given that DINOv2 was trained on 100× more images. This confirms the latent space genuinely encodes physical structure, not just task-solving heuristics.

### Violation-of-Expectation (Surprise Detection)

LeWM assigns significantly higher surprise (prediction error) to physically implausible events (object teleportation) vs. normal trajectories, across all three environments (paired t-test, p < 0.01). Color changes (visual perturbation) produce a weaker, less consistent surprise signal — LeWM is more sensitive to **physical** violations than visual ones. This is a desirable property for a world model: it cares about the causal structure of the environment, not just pixels.

### Ablations

- **Number of SIGReg projections (M)**: Negligible effect on performance — M is not a meaningful hyperparameter.
- **Embedding dimension**: Must be sufficiently large; performance saturates quickly beyond a threshold.
- **Encoder architecture** (ViT vs. ResNet-18): Competitive performance — LeWM is architecture-agnostic.
- **Training stability**: LeWM's 2-term loss converges smoothly and monotonically. PLDM's 7-term loss shows noisy, non-monotonic behavior requiring careful gradient balancing.

---

## Comparison to Prior Work

| | **LeWM** | **DINO-WM** | **PLDM** | **Dreamer/TD-MPC** |
|---|---|---|---|---|
| Encoder training | End-to-end from pixels | Frozen (pretrained DINOv2) | End-to-end from pixels | Latent-based (reconstruction) |
| Anti-collapse | SIGReg (provable) | Frozen encoder | VICReg (7 terms) | Reconstruction loss |
| Hyperparameters | 1 (λ) | Multiple (predictor architecture) | 6 (VICReg weights) | Multiple + task-specific |
| Reward-free | Yes | Yes | Yes | No (requires reward) |
| Planning | CEM in latent space | CEM in latent space | CEM in latent space | RL policy or MPPI |
| Planning speed | **0.98s** | 47s | — | — |
| Model size | 15M params | DINOv2 + predictor | 15M params | 12M–400M |
| Hardware | Single GPU | Multiple GPUs | Single GPU | Multiple GPUs |
| Push-T success | **90%** | 75% | 13% | — |
| OGBench-Cube | 74% | **86%** | ~50% | — |

**vs [[balestriero-2025-iclr]] ([LeJEPA](balestriero-2025-iclr.md)):** LeJEPA is the parent paper that derives SIGReg and proves the isotropic Gaussian optimality result for static SSL. LeWM adapts this framework to sequential world modeling: the prediction task shifts from cross-view consistency (LeJEPA) to action-conditioned next-state prediction (LeWM), and the downstream use case shifts from image classification (frozen encoder + linear probe) to latent planning for robotic control. LeJEPA operates on images with no temporal structure; LeWM operates on action-observation sequences and produces a model for planning.

**vs DINO-WM (Zhou et al., ICML 2025):** DINO-WM builds dynamics predictors on top of a frozen DINOv2 encoder, which guarantees no collapse but tightly couples the world model to DINOv2's pretraining distribution and makes each planning step expensive (~200× more tokens per frame). LeWM learns its own encoder end-to-end, achieving 48× faster planning and matching DINO-WM on most tasks (losing only on the most visually complex 3D environment).

**vs PLDM (end-to-end VICReg baseline):** PLDM is LeWM's closest architectural cousin — both learn encoder + predictor end-to-end from pixels. But PLDM uses a 7-term VICReg-derived objective with 6 hyperparameters and O(n⁶) search complexity. The result is unstable training and poor performance on harder tasks (13% on Push-T vs. LeWM's 90%). Theorem 3 from LeJEPA explains why: moment-based tests like VICReg are provably insufficient to guarantee non-collapse.

**vs [[ha-2018-neurips]] ([World Models](ha-2018-neurips.md)), [[hafner-2023-arxiv]] ([DreamerV3](hafner-2023-arxiv.md)):** Dreamer-family models operate in latent space too, but require reward signals for the RL training loop and use reconstruction-based SSL (RSSM + decoder) rather than JEPA-style prediction. LeWM is reward-free and reconstruction-free by design.

**vs [[hansen-2024-iclr]] ([TD-MPC2](hansen-2024-iclr.md)):** TD-MPC2 also uses latent dynamics + MPPI planning, but requires state access and task-specific reward signals. LeWM works purely from raw pixels and reward-free data.

---

## Strengths

- **End-to-end with a single hyperparameter**: Combines the expressivity of end-to-end training with the simplicity of one tunable parameter (λ), searchable in O(log n) time.
- **48× faster planning**: Makes real-time control feasible; DINO-WM at 47s/plan is unusable in closed-loop settings.
- **Strong physical encoding**: Latent space recovers physical quantities (position, angle) at near-DINOv2 quality without any physical supervision.
- **Single GPU**: 15M parameters, training in hours — accessible to academic labs without large compute.
- **Emergent properties**: Temporal path straightening, physical violation detection, and visual reconstruction all emerge without explicit supervision.

---

## Weaknesses & Limitations

- **Short planning horizon**: Compounding prediction errors limit effective horizon. Hierarchical world modeling (noted as future work) is needed for long-horizon tasks.
- **Two-Room failure**: SIGReg's isotropic Gaussian constraint is counterproductive for very low-intrinsic-dimensionality environments — the regularizer pushes representations to be high-dimensional when the task only needs a few dimensions.
- **Requires action labels**: Action conditioning requires labeled actions in training data; inverse dynamics modeling could reduce this dependency.
- **3D complexity gap**: Trails DINO-WM on OGBench-Cube (74% vs. 86%) due to lack of diverse pretraining. Domain-specific pretraining on video data (as suggested in conclusion) could close this.
- **Offline data requirement**: Requires datasets with sufficient environment coverage; limited diversity can harm SIGReg.

---

## Key Takeaways

- **The two-term loss works**: SIGReg alone (applied from LeJEPA) replaces the entire VICReg 7-term machinery and dramatically outperforms it — 90% vs. 13% on Push-T — validating that the theoretically grounded regularizer is strictly superior to the heuristic one.
- **End-to-end beats frozen pretraining at planning speed**: LeWM's compact 15M-param model is 48× faster at planning than DINO-WM's frozen-DINOv2 approach, despite competitive task performance, because fewer tokens means faster rollouts.
- **Physical understanding emerges without explicit supervision**: Physical probing and violation-of-expectation results confirm the latent space captures causal structure, not just appearance.
- **One failure mode for SIGReg**: The isotropic Gaussian prior can hurt in low-intrinsic-dimensionality settings (Two-Room) by spreading representations across dimensions unnecessarily — a known limitation noted by the authors.
- **Practical recipe**: 15M params, single GPU, O(log n) hyperparameter search, hours of training — a genuine democratization of JEPA world model research.

---

## BibTeX

```bibtex
@article{maes2026leworldmodel,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2026}
}
```

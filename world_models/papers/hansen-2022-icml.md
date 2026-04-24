---
title: "Temporal Difference Learning for Model Predictive Control"
type: paper
paper_id: P012
authors:
  - "Hansen, Nicklas"
  - "Wang, Xiaolong"
  - "Su, Hao"
year: 2022
venue: ICML 2022
arxiv_id: "2203.04955"
url: "https://arxiv.org/abs/2203.04955"
pdf: "../../raw/hansen-2022-icml.pdf"
tags: [world-model, model-predictive-control, continuous-control, latent-dynamics, td-learning, dmcontrol, meta-world]
created: 2026-04-10
updated: 2026-04-10
cites: []
cited_by:
  - hansen-2024-iclr
  - maes-2026-arxiv
  - terver-2025-iclr
  - bagatella-2025-iclr
  - zhang-2026-arxiv
---

# Temporal Difference Learning for Model Predictive Control

> **TD-MPC** jointly learns a task-oriented latent dynamics model and terminal value function via temporal difference learning, achieving superior sample efficiency and asymptotic performance on 92 continuous control tasks from DMControl and Meta-World, including the first documented solution of complex Dog locomotion tasks within 3M environment steps.

**Authors:** Nicklas Hansen, Xiaolong Wang, Hao Su | **Venue:** ICML 2022 | **arXiv:** [2203.04955](https://arxiv.org/abs/2203.04955)

---

## Problem & Motivation

Model-based RL with planning has two fundamental tensions. Long-horizon planning is computationally expensive, and learned dynamics models compound errors over extended rollouts, often performing worse than model-free methods on continuous control benchmarks. Prior model-based approaches learn to predict future states or observations directly (PlaNet, Dreamer), which forces the model to capture task-irrelevant details like shading and background textures — the model must be accurate about everything, making learning harder. Simultaneously, MPC with a ground-truth simulator (MPC:sim) shows strong asymptotic performance but requires prohibitive wall-clock time and a known simulator. The key insight TD-MPC addresses is that a model only needs to be accurate about quantities relevant to reward and value, not the full observation.

---

## Core Idea

Instead of predicting future observations, TD-MPC learns a task-oriented latent dynamics model (TOLD) that is only required to predict reward and value estimates — quantities directly relevant to the RL objective. A short planning horizon (H=5) handles local trajectory optimization, while a jointly learned terminal value function extends estimates to infinite horizon. Both model and value function are trained together end-to-end using TD-learning, with gradients flowing through multiple rollout steps. This makes the model dramatically easier to learn and avoids compounding errors from predicting irrelevant perceptual details.

---

## How It Works

### Overview

At each decision step, TD-MPC encodes the current observation into a latent state, runs MPPI-based planning in latent space using the learned model over H=5 steps augmented by the value function for terminal return, and executes the first action from the best planned trajectory. Training continuously improves the five TOLD components using data from a replay buffer.

### TOLD: Task-Oriented Latent Dynamics (Five Components)

TOLD consists of five jointly learned networks parameterized by θ:

| Component | Symbol | Role |
|-----------|--------|------|
| Representation | h_θ(s_t) → z_t | Encodes raw observation to latent state |
| Latent dynamics | d_θ(z_t, a_t) → z_{t+1} | Predicts next latent state |
| Reward | R_θ(z_t, a_t) → r̂_t | Predicts single-step reward |
| Value (Q-function) | Q_θ(z_t, a_t) → q̂_t | State-action value for terminal estimates |
| Policy | π_θ(z_t) → â_t | Learned policy to guide planning |

All components are deterministic MLPs (no RNNs, no probabilistic models), with hidden dimension 512, ELU activations, latent dimension 100 (Humanoid/Dog) or 50 (otherwise). For image-based tasks, h_θ is a 4-layer CNN with kernel sizes (7,5,3,3), stride (2,2,2,2), 32 filters per layer.

### Training Objective

The model is trained with a temporally weighted sum of three loss terms over a trajectory of length H:

**J(θ; Γ) = Σ_{i=t}^{t+H} λ^{i−t} L(θ; Γ_i)**

where λ=0.5 weights near-term predictions more heavily, and the single-step loss L has three components:

1. **Reward loss** (c_1=0.5): ‖R_θ(z_i, a_i) − r_i‖²  
2. **Value loss** (c_2=0.1): ‖Q_θ(z_i, a_i) − (r_i + γQ_{θ⁻}(z_{i+1}, π_θ(z_{i+1})))‖² (TD target via fitted Q-iteration)  
3. **Latent state consistency loss** (c_3=2): ‖d_θ(z_i, a_i) − h_{θ⁻}(s_{i+1})‖² (predicted next latent vs. target-network encoding of next observation — no observation reconstruction required)

Crucially, gradients from all three terms are back-propagated through time across multiple rollout steps. The target network θ⁻ is a slow-moving average of θ with momentum coefficient ζ=0.99. Policy training minimizes J_π(θ; Γ) = −Σ λ^{i−t} Q_θ(z_i, π_θ(sg(z_i))), where sg denotes stop-gradient.

### Planning (Inference)

TD-MPC uses MPPI (Model Predictive Path Integral) for trajectory optimization at each decision step:

1. Encode state: z_t = h_θ(s_t)
2. Run J=6 iterations of CEM-style refinement (J=12 for Humanoid, 8 for Dog):
   - Sample N=512 trajectories of length H=5 from current distribution N(μ, σ²I), plus N_π=5% from learned policy π_θ
   - For each trajectory, estimate return: φ_Γ = γ^H Q_θ(z_H, a_H) + Σ γ^t R_θ(z_t, a_t)
   - Select top-k=64 trajectories; update μ, σ using importance-weighted mean
3. Execute first action from final distribution; warm-start next step with 1-step shifted μ

The planning cost can be reduced by up to 50% at inference (reducing iterations from 6 to 3) with negligible performance loss. Default inference runs at ~20ms per step (50Hz).

### Training Loop

- **Optimizer:** Adam (β₁=0.9, β₂=0.999), lr=1e-3 (3e-4 for Dog/pixels)
- **Replay buffer:** Unlimited size, prioritized experience replay (PER, α=0.6, β=0.4)
- **Batch size:** 512 (256 for pixels, 2048 for Dog)
- **Discount factor:** γ=0.99
- **Exploration:** ε annealed 0.5→0.05 over 25k steps; planning horizon annealed 1→5 over 25k steps
- **Compute:** Single RTX3090 GPU; most tasks solve within 1 hour

---

## Results

### DMControl State-Based (15 Tasks)

TD-MPC achieves the highest median, IQM, and mean aggregate return across 15 state-based DMControl tasks at 500k environment steps. Notably:

| Method | Aggregate Mean (500k steps) |
|--------|----------------------------|
| **TD-MPC** | **~820** |
| LOOP | ~700 |
| SAC | ~640 |

TD-MPC is the **first documented method to solve Dog Walk and Dog Run** tasks from DMControl (A ∈ ℝ³⁸), which SAC, LOOP, and MPC:sim all fail at within the evaluated budget.

TD-MPC solves Walker Walk **16× faster than LOOP** and matches time-to-solve of SAC on Walker Walk and Humanoid Stand while being significantly more sample efficient. Wall-clock: TD-MPC takes 5.60 h/500k steps vs. LOOP's 18.5 h/500k steps on Walker Walk.

### DMControl Image-Based (100k Benchmark, 6 Tasks)

| Method | Cartpole Swingup | Reacher Easy | Cup Catch | Finger Spin | Walker Walk | Cheetah Run |
|--------|-----------------|--------------|-----------|-------------|-------------|-------------|
| **TD-MPC** | 770±70 | 628±105 | 933±34 | 943±59 | **577±208** | 222±88 |
| DrQ | 759±92 | 601±213 | 913±53 | 938±103 | 612±164 | 344±67 |
| EfficientZero* | 813±19 | 493±145 | 952±17 | 1000±0 | — | — |
| Dreamer | 326±27 | 277±12 | 246±174 | 341±70 | 221±43 | 165±123 |

TD-MPC uses the **same hyperparameters for all image-based tasks**, whereas baselines are tuned task-specifically. EfficientZero requires discretizing the action space (incompatible with Walker Walk and Cheetah Run).

### Meta-World (50 Tasks)

TD-MPC achieves substantially higher success rates than SAC on complex manipulation tasks (Bin Picking, Box Close, Hammer) within 1M steps, while being competitive on simpler tasks. In multi-task learning (MT10), a single TD-MPC policy trained on 10 tasks simultaneously outperforms SAC.

### Ablations

**No latent representation:** Replacing h_θ with identity (operating in state space) drops performance significantly on complex tasks, proving the latent space is not merely for compression but provides a beneficial learning signal.

**No consistency regularization (c_3=0):** Performance degrades across most tasks; reconstruction and contrastive alternatives both help vs. no regularization but consistency loss yields the most consistent results across all 15 DMControl tasks.

**Planning horizon:** Longer horizons (up to H=9) generally help on complex-dynamics tasks like Quadruped Walk (A ∈ ℝ¹²) but provide marginal gains on simpler tasks. H=5 is a good default.

**CEM iterations:** More iterations help for hard tasks; planning can be halved to 3 iterations with ~50% time reduction and no performance loss on Quadruped Walk.

---

## Comparison to Prior Work

| Method | Model Objective | Value | Inference | Continuous | Compute |
|--------|----------------|-------|-----------|------------|---------|
| SAC | None | Yes | Policy | Yes | Low |
| LOOP | State prediction | Yes | Policy + CEM | Yes | Moderate |
| PlaNet | Image prediction | No | CEM | Yes | High |
| Dreamer | Image prediction | Yes | Policy | Yes | Moderate |
| MuZero | Reward/value pred. | Yes | MCTS + policy | No | Moderate |
| EfficientZero | Reward/value pred. + contrast | Yes | MCTS + policy | No | Moderate |
| **TD-MPC** | Reward/value pred. + latent pred. | Yes | CEM + policy | Yes | Low |

**SAC** is the strongest model-free baseline. TD-MPC matches SAC's time-to-solve on Walker Walk/Humanoid Stand while using 15× fewer environment steps. SAC is more sample-efficient on Finger Turn Hard, suggesting exploration limitations.

**LOOP** (Sikchi et al., 2022) is the most similar prior work: it extends SAC with planning but constrains planned trajectories to be close to SAC's policy. TD-MPC differs by replacing the parametric policy with planning as the primary inference mechanism and learning a reward-centric latent model. TD-MPC is 16× faster to solve Walker Walk.

**[[hafner-2019-icml]] ([PlaNet](../papers/hafner-2019-icml.md)) / Dreamer** learn via image reconstruction, requiring much larger models to capture task-irrelevant details. Dreamer uses a learned policy rather than planning at inference. TD-MPC outperforms both with simpler, reward-centric representations.

**MuZero/EfficientZero** are most similar conceptually (reward/value prediction objectives) but use MCTS requiring discretized action spaces, making them incompatible with high-dimensional continuous control tasks like Dog (A ∈ ℝ³⁸).

---

## Strengths
- First complete framework to jointly learn a latent world model, value function, and policy via TD-learning for continuous control, achieving SOTA on 92 tasks.
- Task-oriented (reward-centric) objective dramatically simplifies model learning vs. full observation reconstruction.
- Naturally handles multi-modal inputs (proprioception + egocentric camera) and multi-task learning with a single shared representation.
- Computationally efficient: default inference at 50Hz on a single GPU; planning cost can be halved without performance loss.
- First to document solution of high-dimensional Dog locomotion tasks (A ∈ ℝ³⁸) in DMControl.
- Same hyperparameters across all image-based tasks — no task-specific tuning needed.

## Weaknesses & Limitations
- Planning with CEM/MPPI is still more computationally intensive than policy-only inference; the learned policy π_θ alone performs ~6× faster but worse.
- Hard exploration tasks (Finger Turn Hard) where SAC and LOOP outperform TD-MPC, suggesting the MPPI-based exploration may be insufficient in some settings.
- Deterministic MLP components may struggle in highly stochastic environments where uncertainty modeling is critical.
- The TOLD model is task-specific: the representation h_θ transfers well across related tasks but d_θ encodes task-specific behavior, limiting generalization to unrelated domains.
- Evaluated on simulation benchmarks only; no real-robot results.

## Key Takeaways
- Reward-centric latent consistency (predicting next latent state from reward and value gradients, not pixels) is more effective than reconstruction or contrastive objectives for continuous control world models.
- A terminal value function is essential: it converts short-horizon (H=5) MPC into effective infinite-horizon optimization at much lower compute than long-horizon planning.
- Jointly training model, value function, and policy with TD-learning enables gradients to flow through model rollouts, dramatically improving model quality vs. decoupled training.
- TD-MPC solves DMControl Dog tasks (A ∈ ℝ³⁸) that all prior model-based and model-free methods fail on, demonstrating strong scaling to high-dimensional action spaces.
- Planning cost can be reduced 50% at inference with negligible performance loss, enabling ~50Hz deployment on a single GPU.

---

## BibTeX
```bibtex
@inproceedings{hansen2022tdmpc,
  title={Temporal Difference Learning for Model Predictive Control},
  author={Hansen, Nicklas and Wang, Xiaolong and Su, Hao},
  booktitle={Proceedings of the 39th International Conference on Machine Learning},
  year={2022},
  organization={PMLR}
}
```

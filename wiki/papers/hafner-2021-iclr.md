---
title: Mastering Atari with Discrete World Models
type: paper
paper_id: P003
authors:
- Hafner, Danijar
- Lillicrap, Timothy
- Norouzi, Mohammad
- Ba, Jimmy
year: 2021
venue: ICLR 2021
arxiv_id: '2010.02193'
url: https://arxiv.org/abs/2010.02193
pdf: ../../raw/hafner-2021-iclr.pdf
tags:
- world-model
- rssm
- categorical-latents
- atari
- actor-critic
- model-based-rl
created: 2026-04-10
updated: 2026-04-10
cites:
- ha-2018-neurips
- hafner-2019-icml
cited_by:
- alonso-2024-neurips
- ding-2024-csur
- hafner-2023-arxiv
- hauri-2026-iclrws
- li-2025-arxiv
- mazzaglia-2024-neurips
- micheli-2023-iclr
- robine-2023-iclr
- wang-2025-iclr

---

# Mastering Atari with Discrete World Models (DreamerV2)

> **One sentence** — DreamerV2 is the first agent to achieve human-level performance on the Atari 55-game benchmark by learning behaviors purely from latent-space predictions of a world model with discrete (categorical) representations, surpassing Rainbow and IQN at 200M steps on a single GPU in 10 days.

**Authors:** Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, Jimmy Ba | **Venue:** ICLR 2021 | **arXiv:** [2010.02193](https://arxiv.org/abs/2010.02193)

---

## Problem & Motivation

Despite years of model-free RL progress on Atari (DQN, Rainbow, IQN), world model approaches had consistently failed to match model-free algorithms on this benchmark. Atari presents unique challenges: discrete action spaces, high visual variability, non-smooth transitions (e.g., objects disappearing), and tasks requiring long-horizon credit assignment. Previous world model attempts (Oh et al. 2015, Chiappa et al. 2017, Kaiser et al. 2019/SimPLe) either learned in pixel space (too expensive), used insufficient latent representations, or only evaluated on a subset of easier games. The question was whether a latent-space world model could be made accurate enough on diverse Atari games to derive competitive policies purely from imagination — without ever directly training the policy on real environment transitions.

---

## Core Idea

DreamerV2 makes two key modifications to the Dreamer/DreamerV1 framework: (1) replacing continuous Gaussian latent variables with discrete categorical variables (32 categoricals × 32 classes each = 1024-dimensional one-hot), and (2) introducing KL balancing to separately control the learning rates of the prior and posterior in the KL divergence loss. These changes, combined with Reinforce gradients for the actor on discrete actions and increased model size, turn a capable continuous-control world model into the first to achieve human-level aggregate Atari performance — all within a single GPU training run of 10 days.

---

## How It Works

### Overview

Real observations → encoder → posterior state (h_t, z_t) → world model trained on reconstruction + reward + discount + KL losses. Separately, actor-critic trained purely on imagined trajectories in latent space for H=15 steps. At environment interaction time, the posterior is used to maintain the current belief state; the actor selects actions.

### World Model — RSSM with Categorical Latents

The world model extends the RSSM from PlaNet/DreamerV1 with discrete stochastic states:

- **Recurrent model**: h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1}) — GRU with 600 units, deterministic backbone
- **Representation model (posterior)**: z_t ~ q_φ(z_t | h_t, x_t) — CNN encoder + MLP, produces posterior categorical distribution over current state
- **Transition predictor (prior)**: ẑ_t ~ p_φ(ẑ_t | h_t) — MLP predicting the stochastic state without access to current image
- **Image predictor**: x̂_t ~ p_φ(x̂_t | h_t, z_t) — transposed CNN for image reconstruction
- **Reward predictor**: r̂_t ~ p_φ(r̂_t | h_t, z_t) — MLP with univariate Gaussian output (tanh-transformed)
- **Discount predictor**: γ̂_t ~ p_φ(γ̂_t | h_t, z_t) — MLP with Bernoulli output for terminal detection

The stochastic state z_t is a vector of 32 categorical variables each with 32 classes, giving a 1024-dimensional binary vector with 32 active bits. Straight-through gradients (Bengio et al. 2013) allow backpropagation through the discrete sampling. Total world model parameters: 22M. Image inputs are 64×64 grayscale (downscaled from 84×84).

### KL Balancing

The standard ELBO KL loss KL[q(z_t | h_t, x_t) || p(z_t | h_t)] serves two purposes: training the prior p toward the posterior, and regularizing the posterior q toward the prior. These goals conflict — aggressive regularization of the posterior toward a poorly trained prior slows down representation learning. DreamerV2 decouples these with different learning rates controlled by α = 0.8:

```
kl_loss = alpha * KL(stop_grad(posterior) || prior)
        + (1 - alpha) * KL(posterior || stop_grad(prior))
```

This means the prior is updated 4× faster (α/(1-α) = 4) than the posterior. The result is that the prior learns to track the informative posterior more quickly, while the posterior is free to represent rich information from the images without being aggressively pulled toward the prior. KL balancing outperforms the standard KL on 44 of 55 Atari tasks.

### Behavior Learning (Actor-Critic in Imagination)

The actor-critic is trained entirely on imagined trajectories of H=15 steps, starting from posterior states encountered during world model training:

- **Actor**: â_t ~ p_ψ(â_t | ẑ_t) — 4-layer MLP with ELU activations, 400 units per layer, outputs categorical distribution over actions (1M parameters). For Atari, trained purely with Reinforce gradients (ρ=1).
- **Critic**: v_ξ(ẑ_t) — 4-layer MLP approximating the λ-return. Trained with TD learning toward λ-targets (λ=0.95). Target network updated every 100 gradient steps.

**λ-return** (Equation 4): V_t^λ = r̂_t + γ̂_t[(1-λ)v_ξ(ẑ_{t+1}) + λV_{t+1}^λ] — a weighted average of multi-step returns, prioritizing longer-horizon returns.

**Actor loss** (Equation 6): Combines Reinforce (unbiased, high variance) and dynamics backpropagation gradients (biased, low variance) with mixing weight ρ, plus entropy regularization with scale η:

L(ψ) = E[-ρ log p_ψ(â_t|ẑ_t)(V_t^λ - v_ξ(ẑ_t)) - (1-ρ)V_t^λ - η H[â_t|ẑ_t]]

For Atari: ρ=1 (pure Reinforce); for continuous control: ρ=0 (pure dynamics backprop).

The world model is held fixed during behavior learning — no gradients flow from the actor-critic back into the world model. This allows simulating 2500 latent trajectories of length 15 in parallel on a single GPU, providing far more diverse training signal than real environment steps.

### Training

- **Dataset**: FIFO replay buffer of 2×10⁶ transitions; each gradient step processes B=50 sequences of length L=50
- **World model optimizer**: Adam, learning rate 2×10⁻⁴; KL scale β=0.1; KL balancing α=0.8; gradient clipping 100
- **Actor optimizer**: Adam, 4×10⁻⁵; Critic: Adam, 1×10⁻⁴
- **Policy steps per gradient step**: 4 (behavior learning runs faster than world model)
- **Atari evaluation**: 200M environment steps, action repeat 4, sticky actions, full 18-action space, 108K steps/episode limit
- **Compute**: Single NVIDIA V100 GPU, ~10 days to 200M steps
- **Reward**: tanh-transformed to stabilize learning across games with different reward scales

---

## Results

### Atari 55-Game Benchmark (200M steps, sticky actions)

| Agent | Gamer Median | Gamer Mean | Record Mean | Clipped Record Mean |
|-------|-------------|------------|-------------|---------------------|
| **DreamerV2** | **2.15** | **11.33** | **0.44** | **0.28** |
| DreamerV2 (schedules) | 2.64 | 10.45 | 0.43 | 0.28 |
| IQN | 1.29 | 8.85 | 0.21 | 0.21 |
| Rainbow | 1.47 | 9.12 | 0.17 | 0.17 |
| C51 | 1.09 | 7.70 | 0.15 | 0.15 |
| DQN | 0.65 | 2.84 | 0.12 | 0.12 |

The authors recommend the Clipped Record Mean as the most robust metric (normalizes by human world record, clips at 1.0 to prevent outliers dominating). DreamerV2 (0.28) outperforms the best single-GPU model-free agents Rainbow (0.17) and IQN (0.21) on this metric. DreamerV2 achieves or exceeds model-free performance on most games except Video Pinball (where image reconstruction fails because the ball is a single pixel) and a few others.

Notable individual game strengths: James Bond, Up N Down, Assault (largest DreamerV2 advantages). The world model learns general image representations useful for acting even without task-specific reward optimization of the encoder.

### DreamerV2 vs. MuZero (Conceptual)

| Property | DreamerV2 | MuZero |
|----------|-----------|--------|
| Image modeling | Yes | No |
| Latent transitions | Yes | Yes |
| Single GPU | Yes | No (requires ~2 months GPU equivalent) |
| Frames to train | 200M | 20B |
| Parameters | 22M | 40M |
| Planning | Actor-critic in latent space | MCTS |

MuZero achieves stronger absolute performance but requires orders of magnitude more compute and is not publicly available. DreamerV2 is reproducible by research groups with a single GPU.

### Humanoid from Pixels (Appendix A)

DreamerV2 also solves the DeepMind Control Suite Humanoid Walk task from pixel inputs only — a 21-dimensional continuous action space — representing the first published result of solving this environment from pixels. For continuous control, dynamics backpropagation (ρ=0) replaces Reinforce.

### Ablation Study

| Agent | Clipped Record Mean |
|-------|---------------------|
| **DreamerV2** | **0.25** |
| No Layer Norm | 0.25 |
| No Reward Gradients | 0.24 |
| No Discrete Latents | 0.19 |
| No KL Balancing | 0.16 |
| No Policy Reinforce | 0.15 |
| No Image Gradients | 0.01 |

Removing discrete latents (→ Gaussian) drops from 0.25 to 0.19 (24% decrease), confirming categorical representations are important. Removing KL balancing drops to 0.16 (36% decrease), confirming this is the second most critical change. Removing image gradients entirely (to 0.01) proves the world model's reconstruction signal is essential — reward-only training is catastrophically worse. Removing reward gradients from encoder training slightly helps on some tasks, suggesting task-agnostic representations may generalize better.

---

## Comparison to Prior Work

| Method | Latent type | Policy training | Image modeling | Atari competitive |
|--------|------------|----------------|----------------|-------------------|
| **DreamerV2** | Categorical (discrete) | Actor-critic in imagination | Yes | Yes (human-level) |
| DreamerV1 | Gaussian | Actor-critic in imagination | Yes | No |
| SimPLe | Discrete (codebook) | PPO on imagined frames | Yes (pixel space) | No (36 games only) |
| MuZero | Learned (task-specific) | MCTS + value gradients | No | Yes (superhuman, huge compute) |
| Rainbow/IQN | N/A (model-free) | Q-learning / distributional | N/A | Yes (below human on median) |

**[[hafner-2019-icml]] ([Hafner et al. 2019, PlaNet](../papers/hafner-2019-icml.md)):** Used Gaussian latents; worked well on continuous control (DMC) but the continuous prior cannot match a mixture of categorical posteriors, making it less effective for modeling Atari's discrete multi-modal transitions. DreamerV2's categorical change is a targeted fix for this failure mode.

**SimPLe:** Learns a video prediction model in pixel space and trains PPO on imagined frames. Pixel-space imagination is computationally expensive and visually noisy; evaluated only on 36 games for 4M steps. DreamerV2 trains in compact latent space, enabling thousands of parallel imagined trajectories and scaling to all 55 games.

**MuZero:** Does not learn image reconstruction — trains a task-specific representation purely for value estimation via MCTS. This avoids the noise of image prediction but requires enormous compute (20B frames, multi-GPU for 80 accelerator-days) and the MCTS component is challenging to parallelize. DreamerV2 learns a general image representation and uses it for both the world model and actor-critic.

---

## Strengths
- First world model agent to achieve human-level aggregate performance on the full Atari benchmark (all 55 games, sticky actions, full protocol)
- Single GPU training in 10 days — dramatically more accessible than MuZero
- Categorical latents outperform Gaussian on 42/55 games with no increase in parameter count
- KL balancing is a principled and general technique applicable to any probabilistic world model
- Actor-critic in imagination generates 10,000× more training examples than real environment steps (2500 parallel latent trajectories × H=15 steps vs. 50M real frames)
- Also generalizes to continuous control tasks including humanoid from pixels

## Weaknesses & Limitations
- Fails on Video Pinball (single-pixel ball not reconstructed) — reconstruction-based representation learning has inherent failure modes for tasks with small but critical visual features
- Still requires 200M environment frames — not data-efficient by model-based standards (PlaNet solved DMC in 1,000 episodes)
- No explicit exploration mechanism; relies on policy entropy regularization which is insufficient for hard-exploration games (Montezuma's Revenge: DreamerV2 matches ICM but doesn't exceed it)
- Atari evaluation uses 200M steps — compute-heavy even for a single GPU; StarCraft-scale environments are out of scope
- Policy trained in imagination may not transfer if the world model drifts — model exploitation remains a concern
- Ablation compute cost (60,000 GPU-hours per change) makes comprehensive ablation infeasible

## Key Takeaways
- Categorical/discrete latents outperform Gaussian latents for Atari by 24% (clipped record mean: 0.25 vs 0.19), likely because categorical priors can perfectly fit categorical mixtures and better represent discrete environmental transitions
- KL balancing (α=0.8, updating prior 4× faster than posterior) is the single most important training change, contributing a 36% improvement over standard KL regularization
- Image reconstruction is essential: removing image gradients from encoder training collapses performance by 24× (0.25 → 0.01), proving that task-reward alone is insufficient for learning good representations
- Reinforce-only actor gradients work best for discrete (Atari) while dynamics backpropagation works better for continuous control — the optimal gradient estimator depends on the action space
- DreamerV2 learns 468B compact model states during 200M environment steps — 10,000× more experience in imagination than in the real environment, demonstrating that world model accuracy is the key bottleneck

---

## BibTeX
```bibtex
@inproceedings{hafner2021dreamerv2,
  title     = {Mastering Atari with Discrete World Models},
  author    = {Hafner, Danijar and Lillicrap, Timothy and Norouzi, Mohammad and Ba, Jimmy},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  url       = {https://arxiv.org/abs/2010.02193},
  eprint    = {2010.02193},
  archivePrefix = {arXiv}
}
```

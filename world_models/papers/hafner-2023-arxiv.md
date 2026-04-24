---
title: Mastering Diverse Domains through World Models
type: paper
paper_id: P005
authors:
- Hafner, Danijar
- Pasukonis, Jurgis
- Ba, Jimmy
- Lillicrap, Timothy
year: 2023
venue: arXiv
arxiv_id: '2301.04104'
url: https://arxiv.org/abs/2301.04104
pdf: ../../raw/hafner-2023-arxiv.pdf
tags:
- world-model
- rssm
- atari
- categorical-latents
- model-based-rl
created: 2026-04-10
updated: 2026-04-10
cites:
- ha-2018-neurips
- hafner-2019-icml
- hafner-2021-iclr
cited_by:
- alonso-2024-neurips
- bagatella-2025-iclr
- bredis-2026-arxiv
- ding-2024-csur
- hansen-2024-iclr
- hauri-2026-iclrws
- jaber-2026-arxiv
- li-2025-arxiv
- mazzaglia-2024-neurips
- parthasarathy-2025-arxiv
- terver-2025-iclr
- wang-2025-iclr
- zhang-2026-arxiv

---

# Mastering Diverse Domains through World Models (DreamerV3)

> **One sentence** — DreamerV3 uses a single fixed set of hyperparameters across 150+ tasks spanning 8 diverse domains (Atari, Minecraft, DMLab, ProcGen, continuous control, and more), outperforming specialized expert algorithms on each, and is the first algorithm to collect diamonds in Minecraft from scratch without human data in 100M environment steps.

**Authors:** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap | **Venue:** arXiv | **arXiv:** [2301.04104](https://arxiv.org/abs/2301.04104)

---

## Problem & Motivation

Prior reinforcement learning algorithms — including earlier Dreamer versions — require substantial domain-specific hyperparameter tuning to transfer between environments. A world model agent that performs well on Atari requires different KL scales, reward normalization schemes, and model capacities than one targeting continuous control or sparse-reward 3D environments. This brittleness is not merely inconvenient — it means applying RL to a new domain requires significant expertise and experimentation, fundamentally limiting the technology's practicability. The core challenge is that quantities like reward magnitudes, observation complexity, and return ranges vary by orders of magnitude across domains, causing standard loss functions and normalization schemes to fail in ways that are hard to diagnose and fix without domain knowledge.

---

## Core Idea

DreamerV3's key insight is that the instability of world model training across domains stems from a few identifiable sources: unbounded reward and return scales, unstable KL loss dynamics, and inadequately scaled gradient signals. Rather than tuning hyperparameters per domain, the authors introduce a suite of robustness techniques — symlog transformations, percentile return normalization, KL balancing with free bits, and twohot regression losses — that make each component of the algorithm scale-invariant by construction. The result is a single agent that can be applied out of the box to radically different tasks, including the long-standing AI challenge of collecting diamonds in Minecraft from sparse rewards alone.

---

## How It Works

### Overview

Input observations (images or vectors) → RSSM encoder → categorical latent state (h_t, z_t) → world model trained end-to-end with prediction + dynamics + representation losses → actor-critic trained on imagined H=16-step trajectories in latent space. All components use robustness techniques that operate scale-independently.

### World Model — RSSM with Robustness Techniques

The RSSM structure from DreamerV2 is preserved with important additions:

- **Sequence model**: h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1}) — Block-diagonal GRU with 8 blocks, allowing large recurrent state (8d units) without quadratic parameter growth; uses RMSNorm normalization and SiLu activation
- **Encoder**: z_t ~ q_φ(z_t | h_t, x_t) — CNN for images (stride-2 convolutions to 6×6 or 4×4), 3-layer MLP for vectors; symlog-transforms vector inputs
- **Dynamics predictor**: ẑ_t ~ p_φ(ẑ_t | h_t) — 1-layer MLP predicting next latent from recurrent state alone
- **Decoder**: x̂_t ~ p_φ(x̂_t | h_t, z_t) — transposed CNN for images, 3-layer MLP for vectors
- **Reward predictor**: r̂_t — 1-layer MLP; outputs twohot distribution over exponentially spaced bins
- **Continue predictor**: ĉ_t — 1-layer MLP; binary logistic regression for episode termination

**Categorical distributions**: Encoder, dynamics predictor, and actor are parameterized as mixtures of 99% neural network softmax output and 1% uniform — a "1% unimix" that prevents zero probabilities and infinite log-probabilities, stabilizing KL losses.

**Loss function** (Equation 2):

L(φ) = E[Σ_t (β_pred L_pred(φ) + β_dyn L_dyn(φ) + β_rep L_rep(φ))]

Where β_pred = 1, β_dyn = 1, β_rep = 0.1. The dynamics loss trains the sequence model by minimizing KL(sg[posterior] || prior); the representation loss trains the encoder by minimizing KL(posterior || sg[prior]). Both are clipped with free bits at 1 nat to prevent trivial collapse.

### Robust Predictions — Symlog and Twohot

**Symlog transformation** (Equation 9): A bi-symmetric log for observations and targets that compresses large values while preserving sign:

symlog(x) = sign(x) · ln(|x| + 1)
symexp(x) = sign(x) · (exp(|x|) − 1)

Applied to: vector observation inputs (encoder and decoder), reward prediction targets. This allows the same squared-error loss to work across environments where rewards vary from ±0.001 to ±10,000.

**Twohot regression** (Equations 10–11): For stochastic targets (rewards and returns), predictions are parameterized as softmax distributions over exponentially spaced bins B = symexp([-20, ..., +20]). Targets are encoded as twohot vectors (two adjacent bins receive weights proportional to proximity). The loss is categorical cross-entropy:

L(θ) = -twohot(y)^T log softmax(f(x, θ))

This decouples gradient magnitudes from target magnitudes — large rewards produce the same gradient scale as small rewards. Unlike squared-error loss, there is no gradient explosion from rare high-reward events.

### Return Normalization for the Actor

The actor's entropy regularizer scale η = 3×10⁻⁴ must work across environments with different return ranges. DreamerV3 normalizes returns before computing actor gradients, but avoids the non-stationarity of running-statistics normalization:

S = EMA(Per(R_t^λ, 95) − Per(R_t^λ, 5), 0.99)

The range S is computed from the 5th to 95th return percentile over the current batch and smoothed with exponential moving average. Returns are divided by max(1, S) — small returns (S < 1) are left untouched to avoid amplifying noise in near-zero reward environments. This allows exploration under sparse rewards (η dominates when returns are near zero) while exploiting efficiently under dense rewards (normalized returns dominate).

### Critic Learning — Distributional with Twohot

The critic predicts the full distribution of bootstrapped λ-returns rather than their expectation (Equation 5):

R_t^λ = r_t + γc_t((1-λ)v_ψ(s_{t+1}) + λR_{t+1}^λ)

Trained with maximum likelihood against twohot-encoded targets. Reading out predictions as the expectation of the learned distribution provides robustness to multi-modal return distributions. Discount γ = 0.997, λ = 0.95, horizon T = 16. The critic is additionally trained on replay buffer samples (loss scale β_repval = 0.3) to stabilize early learning, and output weights initialized to zero.

### Training and Architecture

- **Optimizer**: LaProp (RMSProp + momentum) with Adaptive Gradient Clipping (AGC) clipping per-tensor if gradient norm exceeds 30% of weight L2 norm; ε = 10⁻²⁰
- **Model sizes**: 12M to 400M parameters parameterized by hidden dimension d (256 to 1536); number of latents = d/16 codes per latent, 8d recurrent units. Default: 200M (d=1024)
- **Replay buffer**: Uniform sampling with online queue; latent states stored and refreshed during collection for efficient imagination initialization
- **Compute (default 200M model)**: Single Nvidia A100 GPU; ~8.9 GPU-days for Minecraft, 7.7 for Atari, 2.9 for DMLab, 16.1 for ProcGen
- **Evaluation**: 5 seeds per benchmark (1 for ProcGen, 10 for BSuite/Minecraft)

### Inference

Actor samples actions from π_θ(a_t | s_t) where s_t = {h_t, z_t} — no lookahead planning. The actor operates directly on the latent state maintained by the RSSM. This is computationally efficient at test time and identical in structure across all domains.

---

## Results

### Benchmark Summary (fixed hyperparameters, single A100 GPU)

| Domain | Tasks | Budget | Dreamer result |
|--------|-------|--------|----------------|
| Atari | 57 | 200M steps | Outperforms MuZero (fraction of compute), Rainbow, IQN |
| Atari100k | 26 | 400K steps | Outperforms IRIS, TWM, SPR, SimPLe (excluding EfficientZero) |
| ProcGen | 16 | 50M steps | Matches tuned PPG, outperforms Rainbow |
| DMLab | 30 | 100M steps | Exceeds IMPALA/R2D2+ at 1B steps (10× data efficiency) |
| Minecraft Diamond | 1 | 100M steps | First algorithm to collect diamonds from scratch |
| BSuite | 23 | varied | New state-of-the-art, outperforms Boot DQN |
| Proprio Control | 18 | 500K steps | New SOTA, outperforms D4PG, DMPO, MPO |
| Visual Control | 20 | 1M steps | New SOTA, outperforms DrQ-v2 and CURL |

### Minecraft Diamond (Headline Result)

| Method | Diamond discovery | Human data required |
|--------|-------------------|---------------------|
| **DreamerV3** | **Yes (~100% of runs)** | **No** |
| IMPALA | No (stops at iron pickaxe) | No |
| Rainbow | No | No |
| PPO | No | No |
| VPT | Yes | Yes (720 GPU-days of video pretraining) |

Dreamer is the only algorithm among those compared that reliably discovers a diamond, doing so across all seeds within 100M steps using 64 parallel environment instances on 1 A100 GPU (8.9 GPU-days). Intermediate milestone items (iron ingot, iron pickaxe) are collected reliably by ~50M steps. VPT achieves similar depth in the technology tree but requires human expert demonstration data and 720 GPU-days of pretraining.

### Atari (200M Steps, Sticky Actions)

DreamerV3 outperforms MuZero while using a fraction of the computational resources. It also outperforms Rainbow and IQN. In the Atari100k (data-efficient) setting, DreamerV3 outperforms IRIS, TWM, SPR, and SimPLe — placing second only to EfficientZero, which uses tree search, prioritized replay, and episode resets that are not standard practice.

### Ablations (14-task mean across diverse tasks)

| Configuration | Relative performance |
|--------------|---------------------|
| **Full DreamerV3** | **100% (baseline)** |
| No observation symlog | Lower on most tasks |
| No return normalization (advnorm) | Lower |
| No symexp twohot (use Huber) | Lower |
| No KL balance & free bits | Lower |
| Remove all robustness techniques | ~40-50% of full performance |

All robustness techniques collectively contribute substantially. Each individual technique may only affect a subset of tasks, but removing any of them degrades at least some domains (Figure 6a). The KL objective contributes most broadly, followed by return normalization and symexp twohot.

**Learning signals ablation**: Removing reconstruction gradients from the encoder collapses performance to near zero, while removing reward and value gradients has modest impact. This confirms DreamerV3 relies primarily on the unsupervised reconstruction objective — unlike most RL algorithms that rely on task-specific reward signals.

### Scaling Properties

Performance increases monotonically with model size from 12M to 400M parameters with the same hyperparameters. Critically, larger models also require fewer environment steps to achieve the same performance — the scaling is both in final performance and sample efficiency. Replay ratio also scales predictably: higher replay ratios (more gradient steps per env step) improve performance at the cost of compute. This gives practitioners a reliable "performance knob" without requiring hyperparameter retuning.

---

## Comparison to Prior Work

| Method | Fixed hyperparams | Domains covered | Human data | Planning |
|--------|------------------|----------------|-----------|---------|
| **DreamerV3** | Yes | 8+ | No | No (actor) |
| DreamerV2 | No (per-domain) | 2 (Atari, DMC) | No | No (actor) |
| MuZero | No | Atari + board games | No | Yes (MCTS) |
| VPT | No | Minecraft | Yes (video) | No |
| Gato | No | Many (but needs demos) | Yes | No |
| PPO (tuned) | Partial | General but weaker | No | No |

**[[hafner-2021-iclr]] ([DreamerV2](../papers/hafner-2021-iclr.md)):** Domain-specific hyperparameters required (e.g., KL scale β=0.1 for Atari vs β=1 for DMC). Limited to two domain types. DreamerV3 introduces robustness techniques that eliminate this tuning requirement, enabling a true single configuration across 8 domains.

**MuZero:** Achieves stronger Atari results but requires enormously more compute, does not learn image reconstruction, and is not publicly available. Its MCTS planning component does not transfer to Minecraft's continuous and procedurally generated world.

**VPT (Video PreTraining):** Uses behavioral cloning on 720 GPU-days of human gameplay to initialize an Minecraft policy, then fine-tunes with RL. Requires carefully curated human data and domain-specific action space engineering. DreamerV3 uses 64 parallel environments for 8.9 GPU-days with no human data.

**Gato:** Learns a single model across many tasks by conditioning on demonstrations but requires expert data for each new task and does not generalize to tasks without demonstrations. DreamerV3 requires no demonstrations.

---

## Strengths
- Single fixed configuration generalizes across 8 fundamentally different domains — visual, non-visual, discrete, continuous, sparse, dense, 2D, 3D
- First RL algorithm to collect diamonds in Minecraft from scratch — a significant milestone requiring long-horizon planning, exploration, and multi-step crafting chains
- Robustness techniques (symlog, twohot, return normalization, free bits) are principled, interpretable, and individually motivated — not black-box regularizers
- Predictable scaling: larger models and higher replay ratios reliably improve performance without additional tuning
- Performance rests primarily on unsupervised reconstruction (not task-specific rewards), enabling potential pretraining on internet videos
- Open-source implementation on a single A100 GPU makes it accessible to research labs

## Weaknesses & Limitations
- Does not outperform EfficientZero on Atari100k — tree search with prioritized replay provides data efficiency gains that actor-critic imagination does not match
- 200M model requires significant GPU memory; the 12M model achieves competitive results on control tasks but weaker performance on complex visual domains
- Minecraft experiments use block-breaking action space (not standard keyboard/mouse), which simplifies the control problem compared to raw human-playable settings
- ProcGen evaluation uses only 1 seed due to computational constraints — statistical reliability is lower there
- No explicit exploration mechanism — relies on entropy regularization, which may be insufficient for environments requiring systematic exploration beyond Minecraft's naturally information-rich open world
- The agent still trains from scratch per task — no cross-task transfer or pretraining is demonstrated despite the discussion of it as future work

## Key Takeaways
- Fixed hyperparameters across 150+ tasks are achievable by replacing scale-sensitive components: symlog transforms tame unbounded observations and rewards, twohot regression decouples gradient scale from target scale, and percentile return normalization adapts exploration to the actual return range
- DreamerV3 is the first algorithm to collect diamonds in Minecraft from scratch without human data (100M steps, 1 A100 GPU, 8.9 GPU-days) — demonstrating world models can support long-horizon, sparse-reward, open-world exploration
- World model representations are primarily shaped by unsupervised reconstruction, not reward prediction — removing reconstruction gradients collapses performance while removing reward gradients has modest effect
- Performance scales monotonically and predictably with model size (12M–400M parameters) and replay ratio with fixed hyperparameters, providing practitioners a clear compute-performance tradeoff
- DreamerV3 outperforms the best domain-specific expert algorithms on DMLab using 10× fewer environment steps (100M vs 1B), demonstrating that a general world model can be more data-efficient than specialized algorithms when robustness is built into the architecture

---

## BibTeX
```bibtex
@article{hafner2023dreamerv3,
  title     = {Mastering Diverse Domains through World Models},
  author    = {Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal   = {arXiv preprint arXiv:2301.04104},
  year      = {2023},
  url       = {https://arxiv.org/abs/2301.04104},
  eprint    = {2301.04104},
  archivePrefix = {arXiv}
}
```

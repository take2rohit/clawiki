---
title: Learning Latent Dynamics for Planning from Pixels
type: paper
paper_id: P002
authors:
- Hafner, Danijar
- Lillicrap, Timothy
- Fischer, Ian
- Villegas, Ruben
- Ha, David
- Lee, Honglak
- Davidson, James
year: 2019
venue: ICML 2019
arxiv_id: '1811.04551'
url: https://arxiv.org/abs/1811.04551
pdf: ../../raw/hafner-2019-icml.pdf
tags:
- world-model
- rssm
- latent-planning
- continuous-control
- latent-dynamics
created: 2026-04-10
updated: 2026-04-10
cites:
- ha-2018-neurips
cited_by:
- alonso-2024-neurips
- destrade-2025-workshop
- ding-2024-csur
- garrido-2026-arxiv
- hafner-2021-iclr
- hafner-2023-arxiv
- li-2025-arxiv
- mazzaglia-2024-neurips
- micheli-2023-iclr
- parthasarathy-2025-arxiv
- robine-2023-iclr
- terver-2025-iclr
- zhang-2026-arxiv

---

# Learning Latent Dynamics for Planning from Pixels (PlaNet)

> **One sentence** — PlaNet's Recurrent State Space Model (RSSM) learns latent dynamics from image observations and plans with CEM in latent space, achieving performance comparable to D4PG on 6 DeepMind Control Suite tasks using 200× fewer environment episodes.

**Authors:** Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson | **Venue:** ICML 2019 | **arXiv:** [1811.04551](https://arxiv.org/abs/1811.04551)

---

## Problem & Motivation

Planning is powerful when environment dynamics are known, but in image-based domains the agent must learn those dynamics from interaction data — and learned dynamics models had failed to scale to the difficulty of pixel-based control tasks. Prior model-based approaches in image domains accumulated prediction errors over multi-step rollouts, struggled to represent multiple possible futures from partially observable states, and became overconfident outside the training distribution. At the same time, model-free RL methods like D4PG required 100,000 or more episodes to reach good performance. The gap was stark: could a learned world model enable a purely model-based agent to be competitive on continuous control while using drastically fewer samples?

---

## Core Idea

The authors realized that a recurrent state space model with both deterministic and stochastic latent components is necessary and sufficient for reliable multi-step prediction from pixels. The deterministic path (a GRU) carries information across many steps reliably; the stochastic path captures aleatoric uncertainty from partial observability. Critically, all planning happens in the compact latent space — no pixel generation needed during action selection — enabling fast evaluation of thousands of candidate action sequences. A novel training objective called "latent overshooting" encourages the model to make consistent predictions across all look-ahead distances simultaneously, acting as a regularizer for long-horizon accuracy.

---

## How It Works

### Overview

Observations (64×64 RGB frames) → encoder → current belief state (h_t, s_t) → CEM planner searches over action sequences in latent space → best action executed in environment → new observation collected → dataset updated → model retrained iteratively.

### Recurrent State Space Model (RSSM)

The core contribution is a latent dynamics model that splits state into two components:

- **Deterministic state**: h_t = f(h_{t-1}, s_{t-1}, a_{t-1}) — a GRU recurrence that carries information reliably across many time steps
- **Stochastic state**: s_t ~ p(s_t | h_t) — a diagonal Gaussian sampled from the deterministic context, capturing environment stochasticity and partial observability

Three additional models hang off the hidden state:
- **Observation model**: o_t ~ p(o_t | h_t, s_t) — deconvolutional network for image reconstruction (training only)
- **Reward model**: r_t ~ p(r_t | h_t, s_t) — feed-forward network predicting scalar reward
- **Encoder**: q(s_t | h_t, o_t) — convolutional network mapping current image + RNN context to approximate posterior

Latent dimensionality: 30-dimensional diagonal Gaussians. GRU deterministic size: 200 units. All feed-forward layers: two layers of 200 units with ReLU.

This hybrid design (Figure 2 in the paper) was validated against a purely deterministic GRU and a purely stochastic SSM — both performed substantially worse, confirming that both components are essential.

### Training Objective (ELBO + Latent Overshooting)

**Standard ELBO** (Equation 3): For each timestep t, minimize:
- Reconstruction loss: E[log p(o_t | h_t, s_t)] — image reconstruction
- KL complexity penalty: KL[q(s_t | h_{t-1}, a_{t-1}, o_t) || p(s_t | h_{t-1}, a_{t-1})] — regularize posterior toward prior

**Latent overshooting** (Equation 7): A novel generalization that trains the transition model to make accurate predictions at all distances d = 1..D simultaneously, not just one step ahead. For each distance d, compute multi-step predictions by chaining the prior transition model d times, then compute a KL divergence between this multi-step prior and the filtering posterior. Average across all distances with weighting factors β_d. This regularizes the transition model to be consistent at all planning horizons — crucial because the planner will use the model for multi-step rollouts where errors accumulate. The KL losses for d > 1 stop gradients from flowing back through the posterior, so only the prior (transition) model is trained for long-horizon consistency.

### Planning with CEM

At each timestep, CEM (Cross-Entropy Method) searches for the best action sequence:
1. Initialize diagonal Gaussian belief over action sequences: q(a_{t:t+H}) ~ Normal(0, I)
2. Sample J = 1000 candidate action sequences
3. Evaluate each by rolling out the RSSM prior in latent space and summing predicted rewards over H = 12 steps
4. Refit the belief to the top K = 100 sequences (update mean and variance)
5. Repeat for I = 10 iterations
6. Execute the mean of the final belief for the current step

No policy network or value function is used — the planner re-plans from scratch at every step using the current state belief.

### Training

- **Dataset**: Start with S = 5 seed episodes (random actions), grow by 1 episode every C = 100 gradient steps
- **Optimizer**: Adam, learning rate 10⁻³, ε = 10⁻⁴, gradient clipping norm 1000
- **Batch**: B = 50 sequence chunks of length L = 50
- **KL free bits**: 3 nats (clip KL below this to prevent posterior collapse)
- **Action repeat**: varies by task (cartpole: R=8, reacher/cheetah/cup: R=4, finger/walker: R=2)
- **Compute**: 10–20 hours on a single Nvidia V100 GPU
- **Preprocessing**: Images bit-depth reduced to 5 bits

### Inference

At test time only: encode current observation with convolutional encoder → update state belief via filtering posterior → run CEM in latent space → execute first action of best plan → repeat. No image generation occurs during planning — all computation is in the 30-D latent space, enabling fast batch evaluation of action sequences.

---

## Results

### DeepMind Control Suite (6 tasks, from pixels)

| Method | Modality | Episodes | Cartpole | Reacher | Cheetah | Finger | Cup | Walker |
|--------|----------|----------|----------|---------|---------|--------|-----|--------|
| **PlaNet (RSSM)** | pixels | **1,000** | **821** | **832** | **662** | **700** | **930** | **951** |
| D4PG | pixels | 100,000 | 862 | 967 | 524 | 985 | 980 | 968 |
| A3C | proprio | 100,000 | 558 | 285 | 214 | 129 | 105 | 311 |
| CEM + true simulator | state | 0 | 850 | 964 | 656 | 825 | 993 | 994 |

PlaNet achieves near-D4PG performance (matching or exceeding on cheetah: 662 vs 524, +26%) with 100× fewer episodes on pixels. The comparison to CEM with the true simulator shows that the learned model introduces modest overhead, with the main gap on finger spin being the hardest task. A3C from proprioceptive states at 100K episodes is comprehensively outperformed despite PlaNet using pixels at 1K episodes.

### Model Design Ablations

| Model | Key property |
|-------|-------------|
| RSSM (full) | Deterministic + stochastic |
| Deterministic GRU | No stochastic component |
| Stochastic SSM | No deterministic component |

Both the purely deterministic GRU and purely stochastic SSM perform substantially worse than the combined RSSM on all 6 tasks (Figure 4). The deterministic-only model cannot capture multiple possible futures; the stochastic-only model struggles to carry information reliably over long sequences. This is one of the paper's strongest ablations — both components are individually necessary.

### Ablations

Removing the stochastic component (GRU-only) sharply degrades performance across all tasks, especially those with contact dynamics (walker, finger). Removing the deterministic component (SSM-only) also degrades performance substantially, confirming the complementary roles. Using random data collection (no planning-guided exploration) instead of online collection with the planner hurts on cartpole, finger, and walker — tasks requiring specific initial conditions that random exploration cannot easily encounter. Using random shooting (best of 1000 independent sequences) instead of iterative CEM refinement consistently underperforms PlaNet, though by a smaller margin — CEM's iterative refinement provides important gains, especially on sparse-reward tasks.

Latent overshooting improves the DRNN baseline substantially across tasks but slightly hurts RSSM performance (Appendix D, Figure 8), suggesting the RSSM's architecture already provides multi-step consistency that overshooting redundantly regularizes.

---

## Comparison to Prior Work

| Method | Latent space | Planning | Stochastic | Pixel input |
|--------|-------------|---------|-----------|------------|
| **PlaNet** | Continuous (RSSM) | CEM online | Yes (RSSM) | Yes |
| Ha & Schmidhuber 2018 | VAE + LSTM | None (evolution) | Yes (MDN) | Yes |
| PILCO | GP state space | Analytical | Yes (GP) | No |
| E2C / RCE | Locally linear latent | LQR | No | Yes |
| Nagabandi et al. 2017 | Neural dynamics (state) | MPC | No | No |

**[[ha-2018-neurips]] ([Ha & Schmidhuber 2018](../papers/ha-2018-neurips.md)):** World Models uses a similar VAE + recurrent latent space but trains a controller with evolution strategies rather than planning. PlaNet replaces the controller with online CEM planning, enabling direct gradient-free optimization without learning a separate policy. World Models only demonstrated results on simpler tasks (CarRacing, Doom); PlaNet tackles harder contact dynamics from pixels.

**PILCO:** Uses Gaussian processes over known low-dimensional states; scales poorly to pixel observations due to GP computational complexity. PlaNet operates directly from pixels.

**E2C / RCE:** Learn locally-linear latent dynamics and plan with LQR; work on simple simulated environments but cannot scale to the contact dynamics and partial observability of DeepMind Control Suite.

**Nagabandi et al. 2017:** Neural dynamics model with model-free fine-tuning; requires low-dimensional state, not pixels.

---

## Strengths
- Purely model-based with no policy network or value function — action selection is fully explainable as a search procedure
- 200× sample efficiency gain over D4PG with comparable final performance across most tasks
- RSSM design insight (det + stoch hybrid) is rigorously ablated and clearly load-bearing
- Latent overshooting is a clean, general regularizer applicable to any latent sequence model
- Multi-task single agent experiment demonstrates that one RSSM can learn to distinguish and solve 6 visually different tasks
- Planning compute can be increased at test time to improve performance (more CEM iterations/candidates)

## Weaknesses & Limitations
- CEM planning is expensive at inference time: 1000 × 12-step latent rollouts per step
- No amortized policy — cannot reuse computation from previous planning steps
- Latent overshooting slightly hurts the full RSSM, suggesting the regularizer is not universally beneficial
- Only tested on simulated continuous control; unclear how the approach scales to high-visual-complexity or discrete action domains (Atari)
- Planning horizon H=12 with action repeat is quite short; very long-horizon tasks may require significant re-engineering
- State diagnostics (Appendix I) show the latent space contains rich physical information but at the cost of interpretability

## Key Takeaways
- The RSSM hybrid (deterministic GRU backbone + stochastic Gaussian samples) is strictly better than either component alone for latent space planning, with ablations confirming both are necessary
- PlaNet achieves D4PG-level performance on 6 DeepMind Control tasks using 100× fewer episodes (1,000 vs 100,000), demonstrating a landmark improvement in model-based sample efficiency
- Planning purely in a 30-D latent space (no pixel generation during inference) makes CEM practical: thousands of candidate sequences can be evaluated per step in 10–20 GPU-hours total training
- Latent overshooting (training on multi-step KL losses simultaneously) improves weaker baselines substantially, providing a general recipe for better long-horizon model accuracy
- A single PlaNet agent can learn to solve all 6 tasks jointly, demonstrating that RSSM dynamics representations transfer across visually diverse environments

---

## BibTeX
```bibtex
@inproceedings{hafner2019planet,
  title     = {Learning Latent Dynamics for Planning from Pixels},
  author    = {Hafner, Danijar and Lillicrap, Timothy and Fischer, Ian and Villegas, Ruben and Ha, David and Lee, Honglak and Davidson, James},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  year      = {2019},
  url       = {https://arxiv.org/abs/1811.04551},
  eprint    = {1811.04551},
  archivePrefix = {arXiv}
}
```

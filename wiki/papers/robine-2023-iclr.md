---
title: Transformer-based World Models Are Happy With 100k Interactions
type: paper
paper_id: P014
authors:
- Robine, Jan
- Höftmann, Marc
- Uelwer, Tobias
- Harmeling, Stefan
year: 2023
venue: ICLR 2023
arxiv_id: '2303.07109'
url: https://arxiv.org/abs/2303.07109
pdf: ../../raw/robine-2023-iclr.pdf
tags:
- world-model
- transformer
- atari
- reinforcement-learning
- model-based-rl
created: 2026-04-10
updated: 2026-04-10
cites:
- ha-2018-neurips
- hafner-2019-icml
- hafner-2021-iclr
- micheli-2023-iclr
cited_by:
- alonso-2024-neurips
- wang-2025-iclr

---

# Transformer-based World Models Are Happy With 100k Interactions

> **TWM** (Transformer-based World Model) replaces the RNN dynamics backbone of DreamerV2 with a Transformer-XL that receives latent states, actions, and rewards as multimodal input, achieving a mean human-normalized score of 0.956 on the Atari 100k benchmark — outperforming all prior model-free and model-based methods at the time of publication.

**Authors:** Jan Robine, Marc Höftmann, Tobias Uelwer, Stefan Harmeling | **Venue:** ICLR 2023 | **arXiv:** [2303.07109](https://arxiv.org/abs/2303.07109)

---

## Problem & Motivation

World models for model-based RL typically use recurrent neural networks (RSSMs/GRUs) as their dynamics backbone. RNNs access past information only through a compressed recurrent hidden state, which creates two problems: they cannot directly attend to relevant past events that may be many steps back, and their recurrent bottleneck can be a limiting factor for modeling complex long-range dependencies. In the sample-limited Atari 100k regime (only 100k environment interactions, roughly 2 hours of human gameplay), exploiting all available sequential structure from the small dataset is critical. Transformers — which have direct access to all past elements in their context window — are better positioned to capture such dependencies, but prior work using transformers in world models (TransDreamer) required the transformer at inference time, making policy execution expensive. TWM shows that the transformer can be restricted to training only, with a model-free policy running efficiently at inference.

---

## Core Idea

TWM feeds not only the discrete latent states and actions but also the predicted rewards back into a Transformer-XL dynamics model at each step. This reward-conditioned autoregressive prediction allows the model to be aware of what rewards it has already generated during imagined rollouts — important when rewards are sampled from a distribution and the model cannot observe the noise. The Transformer-XL's recurrence mechanism and relative positional encodings enable it to capture long-range temporal dependencies efficiently, while a key design choice keeps the transformer out of the inference path: the policy is conditioned only on latent states z (not on the transformer hidden state h), making real-environment interaction fast and cheap.

---

## How It Works

### Overview

TWM consists of two decoupled components: (1) an observation model (VAE) that maps raw pixel observations to discrete latent states, and (2) an autoregressive dynamics model built on Transformer-XL that takes sequences of latent states, actions, and rewards as input and predicts next latent states, rewards, and episode termination. The actor-critic policy is trained purely in imagination using the dynamics model, and executed in the real environment using only the current latent state (no transformer at inference).

### Observation Model

A variational autoencoder (VAE) following the DreamerV2 architecture:
- **Encoder:** CNN that maps a 4-frame-stacked observation o_t to a discrete latent z_t — a vector of 32 categorical variables, each with 32 categories (1024-dimensional one-hot).
- **Decoder:** Reconstructs observations from z_t by predicting pixel-level independent Gaussian means.
- The observation model captures only the current time step's non-temporal information; temporal dynamics are entirely modeled by the Transformer-XL dynamics model.

**Observation loss:**

L^Obs_φ = E [ Σ_t ( −log p_φ(o_t | z_t) − α₁ H(p_φ(z_t | o_t)) + α₂ H(p_φ(z_t | o_t), p_ψ(ẑ_t | h_{t-1})) ) ]

Three terms: decoder log-likelihood, entropy regularizer on the encoder (prevent one-hot collapse), and consistency cross-entropy keeping the encoder close to the dynamics model's predicted distribution.

### Autoregressive Dynamics Model (Transformer-XL)

The aggregation model f_ψ is a causally masked Transformer-XL that computes a deterministic hidden state h_t from the history of ℓ=16 previous latent states, actions, and rewards: h_t = f_ψ(z_{t-ℓ:t}, a_{t-ℓ:t}, r_{t-ℓ:t-1}).

**Key inputs to the transformer:**
- Latent states z_t (stochastic discrete variables from encoder/predictor)
- Actions a_t (linear embedding)
- Rewards r_t (linear embedding) — including the predicted rewards from the model itself during imagination

The number of input tokens per time step is 3ℓ − 1 (three modalities, one token each, with the last reward excluded from current input). Each modality has its own modality-specific linear embedding layer. The Transformer-XL uses relative positional encodings (removing dependence on absolute time steps) and a recurrence segment mechanism for computational efficiency.

**Conditioned on h_t, three MLP predictors compute:**
- Reward: r̂_t ~ p_ψ(r̂_t | h_t) — normal distribution
- Discount: γ̂_t ~ p_ψ(γ̂_t | h_t) — Bernoulli (0=episode end, γ=discount otherwise)
- Next latent state: ẑ_{t+1} ~ p_ψ(ẑ_{t+1} | h_t) — vector of 32 categorical distributions

**Dynamics loss:**

L^Dyn_ψ = E [ Σ_t ( H(p_φ(z_{t+1} | o_{t+1}), p_ψ(ẑ_{t+1} | h_t)) − β₁ log p_ψ(r_t | h_t) − β₂ log p_ψ(γ_t | h_t) ) ]

Cross-entropy for latent state prediction (consistency), NLL for reward and discount prediction.

### Policy (Actor-Critic)

- **Input:** Current latent state z_t only (no h_t at inference — no transformer required).
- **Architecture:** Separate actor π_θ(a_t | z_t) and critic v_ξ(z_t) MLPs with SiLU activations.
- **Training:** Generalized Advantage Estimation (GAE) on H-step imagined trajectories generated by the dynamics model; actor trained with standard advantage actor-critic objective.
- **Thresholded Entropy Loss:** A novel modification to the standard entropy regularization: L^Ent_θ = max(0, Γ − H(π_θ) / ln(m)), where Γ ∈ [0,1] is a threshold and m is the number of actions. The loss only activates when entropy falls below Γ·ln(m), preventing both entropy collapse and unnecessary entropy inflation. This eliminates the need for ε-greedy exploration schedules and simplifies hyperparameter selection across games.
- **At training time:** Policy uses predicted latent states ẑ_t (from the dynamics model).
- **At inference time:** Policy uses real encoded latent states z_t from the VAE encoder; transformer is not needed.

### Balanced Dataset Sampling

During training, the dataset grows incrementally, causing uniform sampling to over-represent early low-quality experience. TWM uses a temperature-based softmax sampling procedure:

(p_1, …, p_T) = softmax(−v_1/τ, …, −v_T/τ)

where v_i counts how many times entry i has been sampled and τ > 0 controls oversampling strength (default τ = 20). This shifts focus toward recent, higher-quality data without discarding old experience. Setting τ = ∞ recovers uniform sampling.

### Training

- **History length:** ℓ = 16 time steps context
- **Imagination horizon:** H steps (policy rollouts in latent space)
- **Pretraining:** World model is warm-started on pre-collected data (within the 100k interaction budget) to get good latent representations before policy training begins
- **Compute:** ~23.3 hours per run on a single NVIDIA A100 (10 hours budget per run); Transformer-XL version is ~2× faster than vanilla transformer; 5 runs per game
- **Throughput (A100):** World model training: 16,800 samples/s; Transformer-XL imagination: 39,000 samples/s; Policy training: 700,000 samples/s

---

## Results

### Atari 100k Benchmark (26 Games)

| Method | Normalized Mean (↑) | Normalized Median (↑) |
|--------|--------------------|-----------------------|
| **TWM (ours)** | **0.956** | **0.505** |
| SPR | 0.616 | 0.396 |
| DrQ(ε) | 0.465 | 0.313 |
| CURL | 0.261 | 0.092 |
| DER | 0.350 | 0.189 |
| SimPLe | 0.332 | 0.134 |

TWM achieves the highest mean and median human-normalized scores across all four aggregate metrics (mean, median, IQM, optimality gap) with 95% stratified bootstrap confidence intervals, showing improvements are statistically robust. TWM outperforms SPR by 0.956 vs. 0.616 mean HNS — a 55% relative improvement. TWM exceeds all model-free methods (DER, CURL, DrQ) and prior model-based methods (SimPLe).

**Sample efficiency:** After only 25K interactions, TWM's mean HNS already exceeds DER, CURL, and SimPLe's final scores. After 50K interactions it outperforms SPR. This demonstrates strong early learning from limited data.

**Runtime:** TWM takes 23.3 hours per run (A100), compared to SimPLe at 500 hours (20× faster) but slower than model-free methods (SPR: 4.6h, DER: 2.1h). At inference (real environment), TWM runs at 653 frames/s (policy on z alone), comparable to model-free methods.

### Ablations

**Reward feedback:** Removing predicted rewards from the transformer input drops performance on 8 of 9 tested games (BankHeist, Boxing, Breakout, Crazy Climber, Gopher, MsPacman, Pong, Qbert). The reward signal helps the model react to its own stochastic predictions during imagination and improves dynamics modeling. Games where rewards are fully determined by state are unaffected.

**Balanced sampling (τ=20 vs. uniform):** Balanced sampling consistently improves performance in Breakout, KungFuMaster, MsPacman, and Pong. The dynamics loss is lower at the end of training with balanced sampling, confirming that uniform sampling causes overfitting to early data.

**History length (ℓ=16 vs. ℓ=4):** Reducing context to ℓ=4 drops performance in Assault, Krull, MsPacman, and Hero — demonstrating that longer context is important for world model quality.

**Policy input (z vs. [z,h]):** Using [z, h] as policy input can improve learning curves mid-training but leads to lower or equal final scores, as the policy must adapt to changing transformer states throughout training. Using only z is more stable.

**Thresholded entropy loss:** Without the threshold (Γ=1.0, standard entropy penalty), policy entropy behaves unfavorably — either collapsing or diverging across games. With the threshold (Γ=0.1), entropy stabilizes and final scores improve on Breakout and Pong.

---

## Comparison to Prior Work

| Method | Dynamics Model | Latent Space | Policy Input | Transformer at Inference? |
|--------|---------------|-------------|-------------|--------------------------|
| DreamerV2 | RSSM (GRU) | Discrete categorical | z or h | No (policy only) |
| TransDreamer | Transformer | Discrete categorical | h | Yes (costly) |
| SimPLe | CNN video pred. | Pixel space | Pixel obs. | Yes (planning) |
| **TWM** | Transformer-XL | Discrete categorical | z only | No |
| IRIS | GPT autoregressive | Discrete VQ tokens | Tokens | Yes (GPT at each step) |

**[[hafner-2021-iclr]] ([DreamerV2, Hafner et al., 2021](../papers/hafner-2021-iclr.md))** is the direct predecessor. TWM replaces its GRU-based RSSM with a Transformer-XL, adds reward feedback into the dynamics model, disentangles the balanced KL loss, and introduces the thresholded entropy loss. TWM reuses DreamerV2's VAE architecture for observations.

**TransDreamer** (Chen et al., 2022) also replaces DreamerV2's RNN with a transformer for the dynamics model. However, TransDreamer's policy depends on the transformer hidden state h, requiring the full transformer during real-environment inference — making deployment costly. TWM avoids this by conditioning the policy on z alone.

**SPR** (Schwarzer et al., 2021) is the strongest model-free competitor on Atari 100k. It uses self-predictive representations (consistency loss across augmented views over multiple time steps) without imagining full trajectories. TWM outperforms SPR by 55% in mean HNS.

**SimPLe** (Kaiser et al., 2020) is the original world model baseline on Atari 100k using pixel-space video prediction. TWM is 20× faster to train and significantly outperforms it.

---

## Strengths
- Achieves SOTA on Atari 100k at publication time, outperforming all model-free and model-based methods with a clean, principled architecture.
- Transformer-XL dynamics model captures long-range temporal dependencies that RNNs cannot, demonstrated by meaningful attention patterns across all 16 history steps.
- Computational efficiency at inference: transformer not needed for policy execution, enabling fast real-environment interaction (653 frames/s on CPU).
- Thresholded entropy loss is a simple, principled improvement that stabilizes policy entropy across diverse games without per-game tuning.
- Reward feedback into the dynamics model is well-motivated and empirically validated — allows the model to react to its own stochastic predictions.
- Balanced dataset sampling is a simple but effective fix to the well-known overfitting-on-early-data problem in growing replay buffers.

## Weaknesses & Limitations
- Training is significantly slower than model-free baselines (23h vs. 2-5h) — limiting practical iteration speed.
- Evaluated only on discrete-action Atari; no experiments on continuous control or other domains.
- The observation model (VAE) and dynamics model (Transformer-XL) are trained separately with separate parameters, which may limit joint optimization.
- History length is fixed at ℓ=16 — the model cannot adaptively extend context for games requiring longer memory.
- Stochastic latent prediction during imagination introduces rollout variance; the reward feedback helps but cannot eliminate compounding of stochastic samples.
- No comparison to IRIS (concurrent work) in the main paper, which also achieves competitive results on the same benchmark using a similar discrete-token + autoregressive-transformer approach.

## Key Takeaways
- Transformer-XL as the dynamics backbone outperforms GRU-based RSSM (DreamerV2) on Atari 100k, with a 55% relative improvement over the best prior model-free method (SPR).
- Feeding predicted rewards back into the transformer input is a low-cost change that improves dynamics modeling in games with stochastic reward timing.
- The thresholded entropy loss (activating only when entropy < Γ·ln(m)) is more stable than standard entropy regularization for diverse discrete-action games, eliminating ε-greedy schedules.
- Transformer-XL's recurrence mechanism (~2× faster than vanilla transformer) is critical for practical training speed at this context length.
- TWM already outperforms all model-free baselines at 25K environment interactions (4× before the budget ends), confirming strong sample efficiency.

---

## BibTeX
```bibtex
@inproceedings{robine2023twm,
  title={Transformer-based World Models Are Happy With 100k Interactions},
  author={Robine, Jan and H{\"o}ftmann, Marc and Uelwer, Tobias and Harmeling, Stefan},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

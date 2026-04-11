---
title: "Diffusion for World Modeling: Visual Details Matter in Atari"
type: paper
paper_id: P011
authors:
  - "Alonso, Eloi"
  - "Jelley, Adam"
  - "Micheli, Vincent"
  - "Kanervisto, Anssi"
  - "Storkey, Amos"
  - "Pearce, Tim"
  - "Fleuret, François"
year: 2024
venue: NeurIPS 2024 (Spotlight)
arxiv_id: "2405.12399"
url: "https://arxiv.org/abs/2405.12399"
pdf: "../../raw/alonso-2024-neurips.pdf"
tags: [world-model, diffusion, atari, reinforcement-learning, model-based-rl]
created: 2026-04-10
updated: 2026-04-10
cites:
  - ha-2018-neurips
  - hafner-2019-icml
  - hafner-2021-iclr
  - hafner-2023-arxiv
  - micheli-2023-iclr
cited_by:
  - bar-2024-cvpr
---

# Diffusion for World Modeling: Visual Details Matter in Atari

> **DIAMOND** trains a reinforcement learning agent entirely within a diffusion-based world model, achieving a mean human-normalized score of 1.46 on the Atari 100k benchmark — a new best for agents trained exclusively within a world model.

**Authors:** Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, François Fleuret | **Venue:** NeurIPS 2024 (Spotlight) | **arXiv:** [2405.12399](https://arxiv.org/abs/2405.12399)

---

## Problem & Motivation

World models for RL traditionally compress observations into compact discrete latent representations (e.g., VQ-VAE tokens) to model environment dynamics. This discretization helps avoid compounding errors over multi-step rollouts but necessarily loses information — small visual details that a policy depends on (e.g., a traffic light, a distant enemy, a moving ball) may be discarded. Increasing the number of discrete tokens mitigates the loss but raises computational cost substantially. Concurrently, diffusion models have become dominant in image generation, outperforming discrete-token approaches at modeling fine-grained visual detail. The key question DIAMOND addresses is: can a diffusion model serve as a stable, efficient world model for RL, operating directly in pixel space without discrete compression?

---

## Core Idea

Rather than encoding observations into a discrete bottleneck, DIAMOND models environment dynamics as a conditional denoising process: given past observations and actions, predict the next frame by iteratively reversing a noise process. The authors discovered that the choice of diffusion framework critically determines long-horizon stability — the EDM formulation (Karras et al., 2022) stays stable even with a single denoising step, while DDPM-based variants suffer rapid compounding error over imagined trajectories. Better visual fidelity in the world model translates directly to better agent performance, particularly in games where small pixel-level details carry reward-relevant information.

---

## How It Works

### Overview

A replay dataset is maintained from real environment interactions. A conditional diffusion model D_θ generates next-frame predictions given a window of past frames and actions. A separate reward/termination model R_ψ predicts scalar rewards and episode ends. An actor-critic (π_φ, V_φ) is trained entirely on imagined rollouts inside the world model. All three components are retrained on collected data each epoch.

### Diffusion World Model (D_θ)

**Input:** Noisy next observation x^τ_{t+1}, past L=4 clean frames x^0_{≤t}, past actions a_{≤t}, and diffusion time τ.  
**Output:** Denoised prediction of x^0_{t+1}.  
**Architecture:** Standard 2D U-Net with residual blocks (channels [64,64,64,64], layers [2,2,2,2]). Past frames are concatenated channel-wise to the noisy input (frame stacking). Actions and diffusion time τ are injected via adaptive group normalization layers in the residual blocks.  
**Framework:** EDM (Elucidated Diffusion Models, Karras et al. 2022). The network uses preconditioning scalars c^τ_{in}, c^τ_{out}, c^τ_{noise}, c^τ_{skip} to normalize inputs and targets across noise levels. The training objective adaptively mixes signal and noise, causing the network to predict the clean image when noise dominates — this is the key property that prevents compounding error.  
**Training loss:** L(θ) = E[‖F_θ(c^τ_{in} x^τ_{t+1}, y^τ_t) − (1/c^τ_{out})(x^0_{t+1} − c^τ_{skip} x^τ_{t+1})‖²], where σ(τ) is sampled from a log-normal distribution centered at P_mean = −0.4, P_std = 1.2.  
**Sampling at inference:** Euler's method with n=3 denoising steps (NFE=3), which provides a good balance between visual quality and compute.

### Reward / Termination Model (R_ψ)

A CNN-LSTM that takes sequences of frames and actions and predicts scalar reward and binary episode termination at each step. Shared convolutional trunk (channels [32,32,32,32]) + LSTM (dim 512). Trained with cross-entropy loss on clipped rewards {−1,0,1} and termination flags.

### Actor-Critic (π_φ, V_φ)

A shared CNN-LSTM actor-critic with policy head π_φ and value head V_φ. Policy trained with REINFORCE + entropy regularization (weight η=0.001). Value network trained to predict λ-returns (λ=0.95) over imagined horizons of H=15 steps. Discount factor γ=0.985.

### Training

- **Dataset:** Atari 100k benchmark (26 games), 100k real environment steps total per game (~2 hours of gameplay).
- **Loop:** Collect 100 real steps → update D_θ (400 steps/epoch, 1000 epochs) → update R_ψ → update π_φ and V_φ on H=15 imagined trajectories.
- **Optimizer:** AdamW, lr=1e-4, weight decay 1e-2 for D_θ and R_ψ; no weight decay for actor-critic.
- **Compute:** ~2.9 days on a single NVIDIA RTX 4090 (12GB VRAM) per game, 5 seeds; 1.03 GPU-years total.
- **Image resolution:** 64×64×3.
- **CS:GO experiment:** 381M-parameter model (including 51M upsampler), trained 12 days on an RTX 4090 on 87 hours (5M frames) of static CS:GO gameplay.

### Inference

At test time, the policy π_φ interacts with the learned world model D_θ. Given the current observation, π_φ selects an action; D_θ runs 3 Euler denoising steps to generate the next frame; R_ψ predicts reward and termination. This loop continues for up to H=15 steps during training, and indefinitely during evaluation (the agent is evaluated in the real environment, not in imagination).

---

## Results

### Atari 100k Benchmark

| Method | Mean HNS (↑) | IQM (↑) | #Superhuman |
|--------|-------------|---------|-------------|
| **DIAMOND (ours)** | **1.459** | **0.641** | **11** |
| STORM | 1.266 | 0.497 | 9 |
| DreamerV3 | 1.097 | 0.501 | 10 |
| IRIS | 1.046 | 0.459 | 8 |
| TWM | 0.956 | 0.130 | 8 |
| SimPLe | 0.332 | 0.130 | 1 |

DIAMOND outperforms all agents trained entirely within a world model by a notable margin in mean HNS, and achieves IQM parity or better compared to all baselines. The improvement is most pronounced in visually detailed games: Asterix (+2,700 vs IRIS), Breakout (132.5 vs IRIS's 83.7), Road Runner (20,673 vs STORM's 17,564). DIAMOND matches the IQM of STORM while using only 3 NFE per frame compared to IRIS's 16 NFE, making it faster and simpler.

### CS:GO World Model

DIAMOND's diffusion model was trained on 87 hours of static CS:GO Dust II gameplay. The resulting model functions as an interactive neural game engine: it generates stable trajectories over hundreds of timesteps in response to keyboard/mouse input, correctly models 3D perspective, lighting, and recoil. Failure modes include out-of-distribution areas of the map and incorrect consecutive-jump physics, both attributable to memory limits and data sparsity.

### Ablations

The main analytical ablation compares EDM vs. DDPM as the diffusion backbone. With DDPM and n≤10 denoising steps, imagined trajectories degrade rapidly (visually broken within t=50 steps); with EDM, trajectories remain stable past t=1,000 even with n=1. This is attributed to EDM's adaptive signal/noise mixing in the training target: when noise dominates, the network predicts the clean image rather than noise residual, yielding better score function estimates that prevent accumulating error.

A secondary ablation justifies n=3 over n=1: single-step denoising produces blurry predictions in games with multimodal observation distributions (e.g., Boxing, where opponent position is uncertain), while 3-step sampling resolves modes sharply through iterative refinement.

---

## Comparison to Prior Work

| Method | Latent Space | Dynamics Model | NFE/frame | Mean HNS |
|--------|-------------|---------------|-----------|----------|
| DIAMOND | Pixel (continuous) | Diffusion U-Net (EDM) | 3 | 1.46 |
| IRIS | Discrete VQ tokens | Autoregressive Transformer | 16 | 1.05 |
| STORM | Discrete VQ tokens | Stochastic Transformer | ~16 | 1.27 |
| DreamerV3 | Continuous + categorical latents | RSSM (GRU) | 1 | 1.10 |
| TWM | Discrete VQ tokens | Transformer (RSSM adapted) | — | 0.96 |

**[[micheli-2023-iclr]] ([IRIS, Micheli et al., 2023](../papers/micheli-2023-iclr.md))** uses a VQ-VAE to tokenize images and an autoregressive GPT to model token sequences over time. Qualitative comparison reveals that IRIS generates temporal inconsistencies (enemies morphing to rewards between frames) that DIAMOND avoids. IRIS requires 16 NFE per frame vs. DIAMOND's 3.

**[[hafner-2023-arxiv]] ([DreamerV3, Hafner et al., 2023](../papers/hafner-2023-arxiv.md))** uses a recurrent state-space model with a continuous+categorical latent, achieving fixed hyperparameters across many domains. It has lower mean HNS on Atari 100k and does not operate in pixel space, limiting interpretability and applicability as a game engine.

**STORM** (Zhang et al., 2023) adapts DreamerV3 with a transformer for the dynamics model and achieves 1.27 HNS — the closest prior competitor. DIAMOND surpasses it by ~0.2 mean HNS.

**SimPLe** (Kaiser et al., 2019) was the first world-model agent on Atari 100k; it used a convolutional video prediction model and is now a weak baseline.

---

## Strengths
- First to successfully use a diffusion model as a world model for online RL, establishing a new SOTA on Atari 100k for world-model agents.
- EDM framework provides surprising long-horizon stability with as few as 1 denoising step — far cheaper than comparable generative approaches.
- Pixel-space operation enables interpretable, playable world models (demonstrated with CS:GO engine).
- Only 3 NFE per frame at inference — competitive with non-diffusion baselines in wall-clock cost.
- Thorough analysis of failure modes (DDPM compounding error, single-step blurriness) with clear mechanistic explanations.

## Weaknesses & Limitations
- Evaluated primarily on discrete-action Atari; generalization to continuous-control domains (e.g., DMControl) is unverified.
- Frame stacking as the memory mechanism is a minimal approach; the context window is fixed at L=4 frames, limiting long-range memory.
- Reward/termination model is separate from the diffusion model, adding complexity; integrating these objectives into the diffusion backbone is non-trivial and left to future work.
- CS:GO experiments are qualitative only; no quantitative evaluation of the 3D world model's fidelity or downstream RL performance.
- Training cost (~3 GPU-days per game, 5 seeds) is higher than some baselines.

## Key Takeaways
- Diffusion models can replace discrete-token world models and achieve higher visual fidelity without the compression bottleneck, yielding a new world-model SOTA of 1.46 mean HNS on Atari 100k.
- The EDM diffusion framework is dramatically more stable for autoregressive rollouts than DDPM: EDM is stable at t=1,000 with n=1 step; DDPM breaks down by t=50 even with n=10 steps.
- Visual quality of the world model matters for agent performance: games where small details carry reward signals (Asterix, Breakout, Road Runner) show the largest gains over discrete-latent baselines.
- 3 denoising steps (NFE=3) per frame is sufficient — DIAMOND uses 5× fewer function evaluations than IRIS while achieving better performance.
- Diffusion world models can serve as interactive neural game engines, demonstrated by a playable CS:GO simulator trained on 87 hours of static video.

---

## BibTeX
```bibtex
@inproceedings{alonso2024diamond,
  title={Diffusion for World Modeling: Visual Details Matter in {Atari}},
  author={Alonso, Eloi and Jelley, Adam and Micheli, Vincent and Kanervisto, Anssi and Storkey, Amos and Pearce, Tim and Fleuret, Fran\c{c}ois},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024},
  note={Spotlight}
}
```

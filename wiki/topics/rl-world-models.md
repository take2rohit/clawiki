---
title: "RL World Models"
type: topic
tags: [world-model, model-based-rl, latent-dynamics, dream-training, rssm, atari, continuous-control]
created: 2026-04-10
updated: 2026-04-10
papers: [ha-2018-neurips, hafner-2019-icml, hafner-2021-iclr, hafner-2023-arxiv, micheli-2023-iclr, alonso-2024-neurips, robine-2023-iclr, hansen-2022-icml, hansen-2024-iclr, mazzaglia-2024-neurips]
---

# RL World Models

> World models for reinforcement learning learn a compact internal model of environment dynamics — typically in a low-dimensional latent space — and use that model to train policies through imagined rollouts rather than direct environment interaction. This paradigm achieves dramatic improvements in sample efficiency by decoupling representation learning from policy optimization and allowing agents to plan or train inside a "dream" without consuming additional environment steps.

## Background

The foundational insight dates to Schmidhuber (1990), but [[ha-2018-neurips]] ([World Models](../papers/ha-2018-neurips.md)) established the modern deep learning template: a VAE (vision) + MDN-RNN (memory) world model supporting a tiny CMA-ES controller trained entirely inside the model's hallucinated "dream." The work proved that a policy trained purely in imagination can transfer to the real environment, and introduced the V/M/C decomposition (Vision, Memory, Controller) that nearly all subsequent work inherits.

The critical follow-up came from Hafner et al., who replaced the MDN-RNN with a Recurrent State-Space Model (RSSM) — combining deterministic and stochastic latent states — in [[hafner-2019-icml]] ([PlaNet](../papers/hafner-2019-icml.md)), then added an actor-critic trained entirely in latent-space imagination in [[hafner-2021-iclr]] ([DreamerV2](../papers/hafner-2021-iclr.md)), and finally scaled to a single fixed-hyperparameter agent across 150+ tasks in [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)). In parallel, the continuous-control community developed [[hansen-2022-icml]] ([TD-MPC](../papers/hansen-2022-icml.md)) and [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)), which pair latent dynamics with MPPI planning rather than actor-critic imagination.

A second line of work attacked the discrete Atari benchmark, where transformer-based world models proved superior to recurrent ones: [[micheli-2023-iclr]] ([IRIS](../papers/micheli-2023-iclr.md)) and [[robine-2023-iclr]] ([TWM](../papers/robine-2023-iclr.md)) replaced the RSSM with autoregressive GPT-style models over VQ-tokenized frames, and [[alonso-2024-neurips]] ([DIAMOND](../papers/alonso-2024-neurips.md)) replaced discrete tokenization entirely with a diffusion model operating in pixel space.

## Key Approaches

### Recurrent State-Space Models (RSSM / Dreamer family)

The RSSM maintains a pair of latent states: a deterministic recurrent state h_t computed by a GRU, and a stochastic categorical state z_t sampled from an encoder posterior. The world model is trained end-to-end with a combined prediction + dynamics + representation loss. An actor-critic is then trained purely on imagined H-step rollouts inside the model, never touching the real environment during policy updates.

Key papers: [[ha-2018-neurips]] ([World Models](../papers/ha-2018-neurips.md)), [[hafner-2019-icml]] ([PlaNet](../papers/hafner-2019-icml.md)), [[hafner-2021-iclr]] ([DreamerV2](../papers/hafner-2021-iclr.md)), [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md))

### Transformer World Models (Discrete Token Approaches)

Instead of a recurrent backbone, these approaches tokenize each frame into a small number of discrete symbols via a VQ-VAE or discrete autoencoder, then model environment dynamics as sequence prediction with a causal transformer. This makes the world model structurally identical to a language model and allows leveraging GPT-style training objectives.

Key papers: [[micheli-2023-iclr]] ([IRIS](../papers/micheli-2023-iclr.md)), [[robine-2023-iclr]] ([TWM](../papers/robine-2023-iclr.md))

### Diffusion World Models

Rather than compressing frames into a discrete bottleneck, DIAMOND models next-frame prediction as a conditional denoising process. The EDM framework (Karras et al. 2022) provides long-horizon stability with as few as 3 denoising steps, achieving higher visual fidelity than discrete-token approaches at lower computational cost.

Key papers: [[alonso-2024-neurips]] ([DIAMOND](../papers/alonso-2024-neurips.md))

### Latent MPC / TD-MPC Family

Instead of training an actor-critic in imagination, TD-MPC and TD-MPC2 frame policy optimization as model-predictive control in latent space. An MPPI planning loop samples action sequences, rolls them forward through the latent dynamics, evaluates them with a learned value function, and returns the best first action. This approach is particularly well-suited to continuous control and scales from 1M to 317M parameters with log-linear performance improvement.

Key papers: [[hansen-2022-icml]] ([TD-MPC](../papers/hansen-2022-icml.md)), [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md))

### Foundation-Model-Augmented World Models

[[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md)) extends the RL world model paradigm by connecting a pretrained vision-language model (InternVideo2) to the world model's latent space via a connector-aligner module trained on vision-only data. This enables multi-task and data-free policy learning from language, image, or video prompts without language annotations in the embodied domain.

Key papers: [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md))

## Open Questions

- **Dream exploitation**: World model imperfections can be adversarially exploited by the policy (demonstrated by Ha & Schmidhuber's temperature ablation). Preventing this while maintaining training efficiency is unsolved.
- **Double exploration problem**: The world model can only simulate environment content it has observed; rare transitions create a ceiling on imagination-based learning (demonstrated in IRIS's Frostbite failure).
- **Long-horizon memory**: RSSM and frame-stacking approaches have limited context windows; extending to hundreds of steps without quadratic attention cost remains a challenge.
- **Continuous control at scale**: TD-MPC2 shows log-linear scaling to 317M parameters but has not plateaued — optimal scale and architecture for continuous control foundation models is unknown.
- **Joint optimization**: Most approaches train the world model and policy sequentially; end-to-end co-optimization of both components remains largely unexplored.
- **Cross-task transfer**: DreamerV3 uses fixed hyperparameters but still trains from scratch per task; true cross-task transfer via pretrained world model weights is nascent.

## Timeline

- **2018**: VAE + MDN-RNN "dream training" — [[ha-2018-neurips]] ([World Models](../papers/ha-2018-neurips.md))
- **2019**: RSSM + latent planning (PlaNet) — [[hafner-2019-icml]] ([PlaNet](../papers/hafner-2019-icml.md))
- **2021**: Actor-critic in imagination (DreamerV2) — [[hafner-2021-iclr]] ([DreamerV2](../papers/hafner-2021-iclr.md))
- **2022**: Latent MPC for continuous control (TD-MPC) — [[hansen-2022-icml]] ([TD-MPC](../papers/hansen-2022-icml.md))
- **2023**: Transformer world model, Atari SOTA (IRIS) — [[micheli-2023-iclr]] ([IRIS](../papers/micheli-2023-iclr.md))
- **2023**: Transformer world model with RSSM (TWM) — [[robine-2023-iclr]] ([TWM](../papers/robine-2023-iclr.md))
- **2023**: Fixed-hyperparameter, 150+ tasks, Minecraft diamond (DreamerV3) — [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md))
- **2024**: Scalable latent MPC to 317M params (TD-MPC2) — [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md))
- **2024**: Diffusion world model, new Atari SOTA (DIAMOND) — [[alonso-2024-neurips]] ([DIAMOND](../papers/alonso-2024-neurips.md))
- **2024**: VLM-grounded world model for generalization (GenRL) — [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md))

## Papers

| Paper | Year | Key Idea |
|-------|------|----------|
| [[ha-2018-neurips]] ([World Models](../papers/ha-2018-neurips.md)) | 2018 | VAE + MDN-RNN + CMA-ES; first dream training; V/M/C decomposition |
| [[hafner-2019-icml]] ([PlaNet](../papers/hafner-2019-icml.md)) | 2019 | RSSM with deterministic + stochastic latents; latent-space MPC (RSSM) |
| [[hafner-2021-iclr]] ([DreamerV2](../papers/hafner-2021-iclr.md)) | 2021 | Categorical latents; actor-critic in imagination; KL balancing |
| [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)) | 2023 | Symlog, twohot, return normalization; 150+ tasks, fixed hyperparams |
| [[micheli-2023-iclr]] ([IRIS](../papers/micheli-2023-iclr.md)) | 2023 | Discrete AE + GPT transformer world model; HNS 1.046 on Atari 100k |
| [[robine-2023-iclr]] ([TWM](../papers/robine-2023-iclr.md)) | 2023 | Transformer + RSSM hybrid world model; HNS 0.956 on Atari 100k |
| [[hansen-2022-icml]] ([TD-MPC](../papers/hansen-2022-icml.md)) | 2022 | Task-oriented latent dynamics + MPPI planning for continuous control |
| [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)) | 2024 | SimNorm + discrete regression; 317M multi-task scaling; log-linear perf |
| [[alonso-2024-neurips]] ([DIAMOND](../papers/alonso-2024-neurips.md)) | 2024 | Diffusion (EDM) world model; HNS 1.46 on Atari 100k; pixel-space |
| [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md)) | 2024 | VLM connector-aligner; data-free policy learning; 0.80 on 35 tasks |

---
title: "Learning Latent Action World Models In The Wild"
type: paper
paper_id: P055
authors:
  - "Garrido, Quentin"
  - "Nagarajan, Tushar"
  - "Terver, Basile"
  - "Ballas, Nicolas"
  - "LeCun, Yann"
  - "Rabbat, Michael"
year: 2026
venue: arXiv
arxiv_id: "2601.05230"
url: "https://arxiv.org/abs/2601.05230"
pdf: "../../raw/garrido-2026-arxiv.pdf"
tags: [world-model, latent-actions, video, self-supervised, planning]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - hafner-2019-icml
  - bar-2024-cvpr
cited_by: []
---

# Learning Latent Action World Models In The Wild

> **LAM-Wild** — a latent action world model trained entirely on uncurated, in-the-wild natural videos (YoutubeTemporal-1B) learns transferable, camera-relative action representations without any action labels. Key finding: continuous constrained latent actions (sparse or noisy) capture the complexity of real-world actions far better than vector quantization, and a lightweight controller mapping real actions to the learned latent space enables planning performance comparable to models trained on domain-specific, action-labeled data.

**Authors:** Quentin Garrido, Tushar Nagarajan, Basile Terver, Nicolas Ballas, Yann LeCun, Michael Rabbat (FAIR at Meta, Inria, NYU) | **Venue:** arXiv preprint (Jan 2026, v2) | **arXiv:** [2601.05230](https://arxiv.org/abs/2601.05230)

---

## Problem & Motivation

World models that predict the consequences of actions are essential for planning and reasoning in autonomous agents. Existing world models overwhelmingly require explicit action labels at training time ([[lecun-2022-openreview]], [[hafner-2019-icml]](../papers/hafner-2019-icml.md), Dreamer, GAIA-1). However, the vast majority of video data available online is unlabeled and contains diverse embodiments, making action-labeled training unscalable. Latent Action Models (LAMs) address this gap by jointly learning an inverse dynamics model (IDM) that infers latent actions from consecutive frames and a forward model that predicts the future conditioned on these latent actions.

Prior LAM work has been limited to narrow, task-aligned domains: video games (Genie), tabletop manipulation (UniVLA), or curated manipulation datasets. Even works that use some natural videos (e.g., Ego4D) only use a small fraction (~5%) of such data. The action diversity in uncurated in-the-wild videos is far richer than in these controlled settings -- it includes not only agent movements (navigation, manipulation) but also exogenous events like people entering scenes, objects flying through frames, leaves oscillating, and people dancing. This introduces three critical challenges:

1. **Action complexity:** In-the-wild actions span a much broader distribution than navigation or manipulation alone (Figure 1 in the paper illustrates this long tail).
2. **Environmental noise:** Latent actions risk capturing exogenous noise (background motion, lighting changes) rather than meaningful actions.
3. **No common embodiment:** Videos feature diverse agents and camera perspectives, making it impossible for the model to latch onto a single embodiment.

This paper systematically studies whether LAMs can be trained at scale on fully uncurated natural videos, what architectural and regularization choices matter, and whether the resulting latent actions are useful for downstream planning.

---

## Core Idea

The paper argues that to build truly general world models, we must train on large-scale in-the-wild video rather than domain-specific data. The central insight is that **continuous, constrained latent actions** (using sparsity or noise regularization) dramatically outperform the commonly used **vector quantization** approach when modeling the complex, diverse actions found in natural videos. Vector quantization, though popular in prior LAM work (ILPO, LAPO, Genie, UniVLA), struggles to scale its capacity and adapt to the richness of in-the-wild action distributions.

A second key finding is that the **absence of a common embodiment is not an obstacle** -- it is actually a strength. Without a shared embodiment to anchor on, the model learns spatially-localized, camera-relative actions. These actions generalize across semantically different objects (e.g., transferring human walking motion to a flying ball), enabling a form of abstract action transfer.

Finally, by training a lightweight **controller** that maps known real-world actions to the learned latent action space, the world model trained purely on natural videos can be repurposed for downstream robotic manipulation and navigation planning tasks, achieving performance comparable to models trained with full action supervision.

---

## How It Works

### Architecture Overview

The architecture follows the standard LAM framework (Figure 2 of the paper):

1. **Encoder** (frozen): A frame-causal V-JEPA 2-L encoder (`f_theta`) encodes video frames into representations `s_{0:T-1}`. The encoder is kept frozen during world model training.
2. **Inverse Dynamics Model (IDM)** (`g_phi`): Takes consecutive encoded representations `s_t` and `s_{t+1}` and predicts the latent action `z_t = g_phi(s_t, s_{t+1})`. This introduces a causal leak (the IDM sees the future to infer the action), so the information content of `z_t` must be carefully regulated.
3. **Forward Model / World Model** (`p_psi`): A ViT-L with RoPE positional embeddings, conditioned on latent actions via frame-wise AdaLN-zero. Predicts the next representation: `s_{t+1} = p_psi(s_{0:t}, z_t)`. Trained with teacher forcing.
4. **Training Loss:** `L_t = ||s_{t+1} - p_psi(s_{0:t}, z_t)||_1 + L_z(z_t)`, where `L_z` is the latent action regularizer.

Latent actions are 128-dimensional continuous vectors by default.

### Information Regularization of Latent Actions

The key methodological contribution is a systematic comparison of three mechanisms to regulate the information content of latent actions, each preventing the IDM from "cheating" by encoding the entire future frame:

**1. Sparsity (energy-based):** Encourages latent actions to have low L1 norm while maintaining well-structured distributions. Uses a VICReg-inspired Variance-Covariance-Mean (VCM) regularization:

```
L(Z) = VCM(Z) + (1/N) * sum_i E(Z_i)
```

where `E(z) = lambda_l2 * max(sqrt(D) - ||z||_2^2, 0) + lambda_l1 * ||z||_1` enforces sparsity, and VCM ensures adequate spread across dimensions (variance term), decorrelation (covariance term), and centering (mean term). Hyperparameters: `lambda_l2=1, lambda_V=0.1, lambda_C=0.001, lambda_M=0.1`; `lambda_l1` is varied to control capacity.

**2. Noise addition (VAE-like):** Adds stochastic noise to latent actions via a KL divergence term against a standard normal prior:

```
L(z_t) = -beta * D_KL(q(z_t | s_t, s_{t+1}) || N(0, 1))
```

The `beta` coefficient controls the noise-to-signal ratio. Higher `beta` means noisier, lower-capacity latent actions.

**3. Discretization (vector quantization):** The standard approach used in Genie, UniVLA, LAPO. Uses classical VQ with codebook reset for unused codes. Serves as the baseline. Codebook sizes `|C| in {16, 1024, 4096, 32768}`.

### Controller for Real-to-Latent Action Mapping

To use the world model for planning with real actions, a lightweight controller `h_Phi` is trained to map real actions (and optionally past representations) to latent actions:
- **Actions-only:** Simple MLP mapping real actions to latent actions.
- **Actions + representations:** A cross-attention-based adapter with 2 self-attention blocks processing `s_{t-1}`, followed by cross-attention between embedded real actions and processed representations. A 3-layer MLP embeds actions to the encoder dimension (1024), and a final linear layer projects to the 128-dimensional latent action space.
- Trained for 3000 iterations with AdamW, L2 loss, batch size 256.

### Planning Protocol

Planning uses the Cross-Entropy Method (CEM):
- **DROID (manipulation):** Follows the protocol of Terver et al. (2025). Given start state `s_t` and goal `s_g`, plans over horizon H=3 steps. CEM samples N=300 candidate action sequences, evaluates costs via world model unrolling, refits to top K=10 elite samples, iterates I=15 times.
- **RECON (navigation):** Follows NWM ([[bar-2024-cvpr]](../papers/bar-2024-cvpr.md)) protocol. N=120 candidates, single iteration, horizon H=8 (2 seconds at 4fps). Evaluates via Relative Pose Error (RPE).

---

## Experimental Details

- **Encoder:** V-JEPA 2-L (frozen), frame-causal.
- **Training data:** YoutubeTemporal-1B (Zellers et al., 2022), 16-frame clips at 4fps, batch size 1024, 30,000 iterations.
- **Optimization:** Muon optimizer (lr=0.02) + AdamW (lr=6.25e-4), linear warmup over 10% of training, cosine annealing, weight decay 0.04.
- **Hardware:** ~12 hours on 64 H100 GPUs.
- **Decoder (for visualization only):** Frame-causal ViT-L trained with L1 + perceptual loss to decode representations back to pixels. Not part of core evaluation.
- **Evaluation datasets:** Kinetics (human activity), RECON (navigation), DROID (robotic manipulation).

---

## Results / Key Findings

### 1. Continuous Regularizations Outperform Discretization

On in-the-wild prediction error (Figure 4), sparse and noisy latent actions achieve a wide range of capacity-vs-quality tradeoffs by adjusting their regularization strength. **Vector quantization struggles to scale its capacity** and remains close to the deterministic (no conditioning) baseline. Even at maximum sparsity, sparse latent actions with `d=128` still retain meaningful structure, while at high `beta`, noisy latent actions effectively become noise (equivalent to no conditioning).

Qualitatively (Figure 3), sparse and noisy actions can capture complex in-the-wild events like a person entering a scene, while discrete actions show only a vague blob -- highlighting that quantization cannot represent the full complexity.

### 2. No Future Leakage in Practice

Despite the causal leak inherent in the IDM design, the paper shows (Table 1) that when scene changes are artificially introduced by stitching video ends together, prediction error more than doubles across all regularization types. This indicates no model is simply copying the next frame into the latent action. The complexity of in-the-wild data makes the cheating solution harder to find.

### 3. Latent Actions Transfer Across Videos

**Cycle consistency test** (Table 2, Figure 7): Infer actions on Video A, apply them to Video B, re-infer from the prediction, apply back to Video A. The small increase in prediction error (1.03x--1.34x on Kinetics, 1.03x--1.22x on RECON) shows that latent actions transfer reliably. More constrained latent actions transfer better but capture less fine-grained motion.

**Cross-object transfer:** Motion from a walking human can be transferred to a flying ball (Figure 7), demonstrating that actions encode abstract, camera-relative motion rather than object-specific dynamics.

### 4. Spatially-Localized, Camera-Relative Actions

Without a common embodiment, the model learns **spatially-localized, camera-relative transformations** (Figure 8). Applying a locomotion action to a video with two people only moves the person closest to the spatial location where the action was originally inferred. This is a natural consequence of diverse embodiments -- the camera is the only common reference frame.

### 5. Planning Performance Approaches Action-Supervised Baselines

**DROID manipulation:** The LAM achieves planning performance (delta-xyz error) comparable to V-JEPA 2-AC (trained with real actions). The best performance comes from models with **moderate** latent action capacity -- not the highest or lowest. Notably, noisy latent actions yield the best planning performance despite producing the worst-looking unrollings (Figure 11).

**RECON navigation:** The LAM beats policy baselines like NoMaD (Sridhar et al., 2024) on RPE, though it does not fully match NWM ([[bar-2024-cvpr]](../papers/bar-2024-cvpr.md)), which was specifically designed for navigation with action labels. Egocentric navigation is harder due to additional information entering the frame at every step.

**Key nuance:** Rollout quality (visual fidelity) does not perfectly correlate with planning performance -- a common challenge in world model literature (Zhang et al., 2025).

### 6. Positive Scaling Trends

Figure 12 shows that across all three axes -- model size (Large/Huge/Giant), training time (30k--120k iterations), and data quantity (5.4 days to 150 years of video) -- IDM prediction quality and planning performance improve. Training time shows the clearest gains. The default recipe sees each video on average twice; performance only degrades below ~1% of the total training set size.

---

## Comparison to Prior Work

| Method | Data Source | Action Type | Regularization | Embodiment | Planning |
|---|---|---|---|---|---|
| **PlaNet** [[hafner-2019-icml]](../papers/hafner-2019-icml.md) | Game/sim | Real (labeled) | N/A | Single | MPC (CEM) |
| **NWM** [[bar-2024-cvpr]](../papers/bar-2024-cvpr.md) | Robot video | Real (labeled) | N/A | Single (ego) | CEM on diffusion WM |
| **Genie** (Bruce et al., 2024) | Video games | Latent (discrete) | VQ | Game-specific | Interactive env |
| **UniVLA** (Bu et al., 2025) | Curated manip. | Latent (discrete) | VQ | Task-specific | VLA pipeline |
| **LAPO** (Schmidt & Jiang, 2024) | Diverse video | Latent (discrete) | VQ | Multi | Downstream control |
| **AdaWorld** (Gao et al., 2025) | Real manip. | Latent (continuous) | Regularization | Single | WM training |
| **CoMo** (Yang et al., 2025) | Internet video | Latent (continuous) | Continuous | Multi | Robot learning |
| **This work (LAM-Wild)** | **YT-1B (in-the-wild)** | **Latent (continuous)** | **Sparse / Noisy / VQ** | **No common** | **CEM planning** |

**vs. [[lecun-2022-openreview]](../papers/lecun-2022-openreview.md) (LeCun's JEPA position paper):** This work directly implements a key component of LeCun's vision -- a world model that learns from observation without action labels. The LAM's forward model operates in JEPA representation space (V-JEPA 2 embeddings), and the latent actions serve as the latent variable `z` that parameterizes possible futures. The information regularization methods studied here directly address the challenge LeCun identified of preventing representational collapse and controlling the information content of latent variables.

**vs. [[maes-2026-arxiv]](../papers/maes-2026-arxiv.md) (LeWorldModel):** LeWorldModel trains an end-to-end JEPA world model from pixels for planning, but uses real action labels and domain-specific data. This work removes the action label requirement entirely by learning latent actions from uncurated video, complementing LeWorldModel's approach.

**vs. V-JEPA / V-JEPA 2:** The frozen V-JEPA 2-L encoder provides the representation space in which predictions and latent actions operate. The paper notes that integrating latent action training into V-JEPA 2 pretraining could unlock single-stage encoder/world-model training -- an exciting future direction.

**vs. [[hafner-2019-icml]](../papers/hafner-2019-icml.md) (PlaNet):** PlaNet requires action labels and operates in narrow domains (DeepMind Control Suite). The LAM achieves comparable planning capability without any action labels, trained on vastly more diverse data, using a fundamentally different representation space (JEPA embeddings vs. pixel-level VAE latents).

---

## Strengths

- **First systematic study of LAMs on large-scale in-the-wild video** -- moves beyond the narrow domains (games, curated manipulation) that dominated prior work.
- **Thorough comparison of three regularization families** (sparsity, noise, discretization) with clear takeaway: continuous constrained actions are superior to VQ for complex real-world actions.
- **Comprehensive evaluation protocol** including future leakage tests (scene stitching), cycle consistency for transferability, spatial locality analysis, and downstream planning on two different domains (manipulation + navigation).
- **Practical utility demonstrated:** A simple controller bridges the gap between latent and real action spaces, enabling planning comparable to action-supervised baselines.
- **Positive scaling trends** across model size, training time, and data quantity suggest the approach will benefit from further scaling.
- **Clean ablation-driven experimental design** with well-defined metrics (in-the-wild prediction error as a proxy for latent action capacity, LPIPS for cycle consistency, delta-xyz and RPE for planning).

## Weaknesses & Limitations

- **Static information constraints:** The regularization coefficient is fixed for all videos, but different videos contain actions of varying complexity. Adaptive, per-video capacity control could improve results (acknowledged by the authors).
- **Camera-relative actions only:** Without a common embodiment, learned actions are localized relative to the camera frame. This limits direct transferability to embodied agents with different viewpoints without the controller bridging step.
- **Planning in latent action space unexplored:** All planning experiments use real actions mapped through the controller. Direct planning in the latent action space (sampling latent actions without a controller) remains an open problem, partly because sampling from the continuous latent action distribution is non-trivial (see Appendix B on SGLD sampling challenges).
- **Frozen encoder:** The V-JEPA 2 encoder is not trained jointly with the world model. The representation space was not designed with prediction in mind, which could hinder performance. Joint training is left for future work.
- **Two-stage pipeline:** IDM and forward model are trained jointly, but the controller requires a separate training stage on domain-specific, action-labeled data -- partially undercutting the "no action labels needed" narrative.
- **Rollout quality degrades over time:** While planning performance is reasonable, multi-step unrollings degrade in visual quality, and the disconnect between rollout quality and planning performance is not fully resolved.
- **Limited comparison with concurrent work:** CoMo (Yang et al., 2025) and CLAM (Liang et al., 2025) are mentioned but not experimentally compared.

## Key Takeaways

- **Continuous constrained latent actions (sparse or noisy) dramatically outperform vector quantization** for modeling the rich action distributions in natural videos. VQ, despite being the standard choice in prior LAM work, cannot adapt its capacity to the complexity of in-the-wild data.
- **Training on uncurated, in-the-wild video is viable for LAMs.** The resulting world model captures meaningful, transferable actions despite environmental noise and the absence of a common embodiment.
- **Latent actions are spatially localized and camera-relative** -- a natural and useful consequence of training on diverse embodiments. Actions can transfer between semantically different objects (human to ball).
- **Future leakage is not a practical concern** in the in-the-wild setting, likely because dataset complexity prevents the IDM from finding the cheating solution.
- **A lightweight controller can bridge latent and real action spaces**, enabling planning on manipulation (DROID) and navigation (RECON) tasks with performance approaching action-supervised baselines.
- **Moderate latent action capacity yields the best planning** -- neither too constrained (insufficient information) nor too free (noise capture, poor transferability). This capacity-identifiability tradeoff is a key design consideration.
- **Scaling helps:** Larger models, longer training, and more data all improve both IDM prediction and downstream planning performance.

---

## BibTeX

{% raw %}
```bibtex
@article{garrido2026learning,
  title   = {Learning Latent Action World Models In The Wild},
  author  = {Garrido, Quentin and Nagarajan, Tushar and Terver, Basile and Ballas, Nicolas and LeCun, Yann and Rabbat, Michael},
  journal = {arXiv preprint arXiv:2601.05230},
  year    = {2026}
}
```
{% endraw %}

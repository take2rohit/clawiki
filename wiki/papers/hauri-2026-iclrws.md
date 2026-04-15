---
title: "Dreamer-CDP: Improving Reconstruction-free World Models Via Continuous Deterministic Representation Prediction"
type: paper
paper_id: P045
authors:
  - "Hauri, Michael"
  - "Zenke, Friedemann"
year: 2026
venue: ICLR 2026 Workshop
arxiv_id: "2603.07083"
url: "https://arxiv.org/abs/2603.07083"
pdf: "../../raw/hauri-2026-iclrws.pdf"
tags: [world-model, JEPA, Dreamer, reconstruction-free, model-based-rl]
created: 2026-04-15
updated: 2026-04-15
cites:
  - hafner-2021-iclr
  - hafner-2023-arxiv
  - lecun-2022-openreview
cited_by: []
---

# Dreamer-CDP: Improving Reconstruction-free World Models Via Continuous Deterministic Representation Prediction

> **One sentence** — Dreamer-CDP removes the pixel reconstruction decoder from DreamerV3 and replaces it with a JEPA-style predictor that performs continuous deterministic prediction (CDP) of the next observation embedding, matching DreamerV3's Crafter score (16.2 vs 14.5%) and outperforming all prior reconstruction-free Dreamer variants.

**Authors:** Michael Hauri, Friedemann Zenke (Friedrich Miescher Institute for Biomedical Research, Basel) | **Venue:** ICLR 2026 Workshop | **arXiv:** [2603.07083](https://arxiv.org/abs/2603.07083) | **Code:** [github.com/fmi-basel/Dreamer-CDP](https://github.com/fmi-basel/Dreamer-CDP)

---

## Problem & Motivation

[[hafner-2023-arxiv|DreamerV3]] is the dominant model-based RL framework, learning world models via a Recurrent State-Space Model (RSSM) trained with pixel reconstruction. But pixel reconstruction has a known weakness: it biases representations toward task-irrelevant visual details (texture, lighting, background clutter), wasting model capacity on information that does not help planning or control. This insight, articulated by [[lecun-2022-openreview|LeCun's JEPA framework]], motivates reconstruction-free world models.

Several prior attempts to remove reconstruction from Dreamer have been made:

- **DreamerPro** (Deng et al., 2022): Replaces reconstruction with prototypical representations using augmented views. Requires view augmentation.
- **MuDreamer** (Burchi & Timofte, 2024): Removes reconstruction but trains the sequence model via action prediction from the latent state. Requires auxiliary action prediction.

However, **none of these reconstruction-free Dreamer variants match the original DreamerV3 on Crafter**, the most challenging benchmark for Dreamer-family models. MuDreamer scores 7.3% vs DreamerV3's 14.5%, and DreamerPro scores 4.7%. The performance gap persists despite reconstructed Dreamer itself using a discrete stochastic latent that is inherently limited.

The root cause, Hauri & Zenke argue, is that prior reconstruction-free methods still train both the representation model and the transition model to predict Dreamer's **discrete probabilistic** latent states. These discrete targets are too coarse for a purely predictive objective — the internal KL-based prediction mechanism in the RSSM is necessary but not sufficient to learn good representations without reconstruction.

---

## Core Idea

Dreamer-CDP introduces a **continuous deterministic prediction (CDP)** objective into DreamerV3: a JEPA-style predictor that maps the recurrent hidden state h_{t+1} to an estimate of the next observation's continuous embedding u_{t+1}, trained with negative cosine similarity. This gives the sequence model a rich, continuous supervision signal that replaces the pixel decoder entirely. The key design choices are:

1. **Separate the encoder** into a deterministic feature extractor (observations to continuous embeddings u_t) followed by a stochastic encoder (embeddings to discrete latents z_t).
2. **Train a predictor** g_phi(h_{t+1}) to approximate u_{t+1} using cosine similarity — a JEPA-style "predict the next embedding" objective.
3. **No EMA target network** — unlike BYOL-Explore or other SSL-RL methods, the predictor targets are the raw encoder outputs, and convergence is ensured by training the sequence model with a higher learning rate.
4. **No view augmentation** — unlike DreamerPro, the method works on unaugmented observations.

The result: reconstruction loss is removed, the decoder is removed, and the world model trains purely from internal prediction objectives — exactly the setup LeCun's JEPA framework advocates.

---

## How It Works

### Overview

Observations x_t are encoded into continuous deterministic embeddings u_t via a CNN feature extractor. A stochastic encoder samples discrete latent states z_t from u_t and the recurrent hidden state h_t. The RSSM sequence model rolls forward as in standard Dreamer. A CDP predictor maps h_{t+1} to an estimate of the next continuous embedding u_{t+1}. The model is trained with a cosine similarity loss on these predictions, replacing the reconstruction loss entirely.

### Architecture Modifications to DreamerV3

The modifications to [[hafner-2023-arxiv|DreamerV3]] are minimal:

1. **Encoder decomposition**: The original encoder q_phi(z_t | h_t, x_t) is split into:
   - A **feature extractor**: x_t -> u_t (CNN producing continuous deterministic embeddings)
   - A **stochastic encoder**: z_t ~ q_phi(z_t | h_t, u_t) (maps features + hidden state to discrete categorical latent)

2. **CDP predictor**: An MLP that takes the hidden state and predicts the next continuous embedding:
   ```
   û_{t+1} = g_phi(h_{t+1})
   ```

3. **Decoder removed**: No pixel reconstruction decoder. No L_recon loss.

### Model Equations

The full model is:

```
Sequence model:     h_t    = f_phi(h_{t-1}, z_{t-1}, a_{t-1})
Feature extractor:  u_t    = CNN(x_t)
Encoder:            z_t    ~ q_phi(z_t | h_t, u_t)
Dynamics predictor: ẑ_t    ~ p_phi(ẑ_t | h_t)
Reward predictor:   r̂_t   ~ p_phi(r̂_t | h_t, z_t)
Continue predictor: ĉ_t   ~ p_phi(ĉ_t | h_t, z_t)
CDP predictor:      û_{t+1} = g_phi(h_{t+1})
```

### Training Objective

The loss replaces L_recon with L_CDP:

```
L(phi) = E[ sum_t (beta_CDP * L_CDP(phi) + beta_aux * L_aux(phi) + beta_dyn * L_dyn(phi) + beta_rep * L_rep(phi)) ]
```

where:

- **L_CDP** = negative cosine similarity = -sum_t cos(SG(u_{t+1}), û_{t+1})
  - SG = stop-gradient on the target embeddings u_{t+1}
  - The predictor learns to match the direction (not magnitude) of the next embedding
- **L_aux** = reward + continuation prediction losses (same as DreamerV3)
- **L_dyn** = max(1, D_KL[SG(q_phi(z_t|h_t,x_t)) || p_phi(z_t|h_t)]) — dynamics loss with free bits
- **L_rep** = max(1, D_KL[q_phi(z_t|h_t,x_t) || SG(p_phi(z_t|h_t))]) — representation loss with free bits

### Avoiding Collapse Without EMA

A critical design choice: Dreamer-CDP does **not** use an EMA target network (unlike BYOL-Explore and SimSiam-based methods). Instead, it relies on two mechanisms:

1. **The sequence model must converge to a fixed point** of its own dynamics when the representation network parameters change. Tang et al. (2023) and Khetarpal et al. (2025) established that this constraint forces the RNN's internal predictions to track the encoder's outputs.
2. **Higher learning rate for the sequence model** ensures the RSSM adapts quickly enough to track the moving encoder targets, preventing representational drift from causing divergence.

### Comparison of Dreamer Variants (Table 1)

| Method | Recon-free | Non-contrastive | No action pred. | No view augment. |
|---|---|---|---|---|
| Dreamer (V3) | No | No | Yes | Yes |
| DreamerPro | Yes | Yes | Yes | No |
| MuDreamer | Yes | No | No | Yes |
| **Dreamer-CDP** | **Yes** | **Yes** | **Yes** | **Yes** |

Dreamer-CDP is the only reconstruction-free variant that requires **none** of the auxiliary signals (action prediction, view augmentation, contrastive losses).

---

## Results

### Crafter Benchmark (Main Result)

All models trained for 1M environment interactions on a single Nvidia V100 GPU. Performance measured by Crafter score (n=7 seeds):

| Method | Crafter Score | Cum. Reward |
|---|---|---|
| Dreamer (V3) | 14.5 +/- 1.6% | 11.7 +/- 1.9 |
| DreamerPro | 4.7 +/- 0.5% | — |
| MuDreamer | 7.3 +/- 2.6% | 5.6 +/- 1.6 |
| **Dreamer-CDP** | **16.2 +/- 2.1%** | **9.8 +/- 0.4** |

Dreamer-CDP achieves **16.2%** vs DreamerV3's **14.5%** (t-test p=0.10, i.e., on par). It is the first reconstruction-free Dreamer variant to match the reconstruction-based original on Crafter. The gap to prior reconstruction-free methods is large: +8.9 points over MuDreamer, +11.5 points over DreamerPro.

The only method that outperforms Dreamer-CDP on Crafter is prioritized experience replay (Kauvar et al., 2023), which achieves 19.4% — but this is an orthogonal technique that could also be combined with CDP.

### Ablation Studies

**Removing L_CDP** (equivalent to Dreamer without reconstruction):
- Crafter score drops to **3.2 +/- 1.2%** — catastrophic collapse, confirming that the RSSM's internal KL-based prediction alone is insufficient for reconstruction-free learning.

**Removing reward predictor gradients** (no gradient flow from reward head):
- Crafter score drops to **12.7 +/- 1.6%** — a moderate drop, showing reward prediction contributes meaningfully but is not the primary learning signal.

**Removing L_dyn/L_rep** (no KL alignment objectives):
- Crafter score drops to **6.3 +/- 1.9%** — substantial degradation, indicating CDP alone is necessary but not sufficient. The KL alignment objectives remain essential for good performance.

### Visual Reconstruction Quality (Figure 1c)

The authors train decoders independently (with detached gradients) for visualization purposes. Dreamer-CDP's latent representations produce reconstructions that are notably sharper and more detailed than those from a model trained without any prediction objective, and comparable to DreamerV3's reconstruction-supervised representations. This confirms the CDP objective learns visually informative representations even though it never sees pixels during training.

---

## Relationship to JEPA and LeWorldModel

This paper provides **direct evidence for [[lecun-2022-openreview|LeCun's JEPA]] thesis** within the Dreamer framework: pixel reconstruction is unnecessary when you have a predictor that operates in a learned embedding space.

The connection to [[maes-2026-arxiv|LeWorldModel (LeWM)]] is especially instructive. Both papers arrive at the same conclusion — reconstruction-free world models via next-embedding prediction — from different starting points:

| | **Dreamer-CDP** | **LeWorldModel** |
|---|---|---|
| Base framework | DreamerV3 (RSSM, RL) | Custom ViT (planning, no RL) |
| Prediction target | Continuous CNN embeddings | Continuous ViT CLS embeddings |
| Loss | Cosine similarity | MSE |
| Anti-collapse | Stop-gradient + higher LR | SIGReg (isotropic Gaussian) |
| Stochastic latents | Yes (categorical) | No (purely deterministic) |
| Evaluation | RL (Crafter) | Goal-conditioned planning (Push-T, Reacher) |
| Task type | Reward-based RL | Reward-free offline planning |
| Key finding | Matches DreamerV3 on Crafter | 48x faster planning than DINO-WM |

The convergent finding across these independent works is striking: replacing reconstruction with embedding prediction works across fundamentally different architectures (RSSM vs ViT), training paradigms (online RL vs offline planning), and evaluation settings.

---

## Comparison to Prior Work

**vs [[hafner-2023-arxiv|DreamerV3]]:** Dreamer-CDP removes the reconstruction decoder and loss entirely, replacing them with a cosine-similarity predictor on continuous embeddings. Performance on Crafter is statistically equivalent (16.2% vs 14.5%, p=0.10). The decoder removal should yield computational savings (not quantified in this paper), and the representations are freed from pixel-level detail reconstruction.

**vs [[hafner-2021-iclr|DreamerV2]]:** DreamerV2 introduced the categorical latent representation and KL-balanced training that Dreamer-CDP inherits. The key departure is that DreamerV2's representations are shaped primarily by the reconstruction objective — removing reconstruction collapses DreamerV2/V3 to 3.2% Crafter score, while adding CDP restores performance to 16.2%.

**vs DreamerPro (Deng et al., 2022):** DreamerPro uses prototypical representations with augmented views but achieves only 4.7% on Crafter. Dreamer-CDP's continuous deterministic targets (vs DreamerPro's discrete prototypes) appear to be a strictly better supervisory signal for reconstruction-free learning.

**vs MuDreamer (Burchi & Timofte, 2024):** MuDreamer replaces reconstruction with action prediction (predicting which action led to a transition), scoring 7.3% on Crafter. The weak action signal in Crafter (limited discrete actions) likely explains the gap. Dreamer-CDP's embedding prediction is richer because it captures the full state transition, not just the action.

**vs [[maes-2026-arxiv|LeWorldModel]]:** Both are reconstruction-free JEPA-style world models, but LeWM operates in a completely different regime — offline, reward-free, goal-conditioned planning with a ViT encoder and SIGReg regularization. Dreamer-CDP demonstrates that the same core idea (predict embeddings, not pixels) works within the online RL setting with recurrent latent-state models. Together, these papers bracket the reconstruction-free world model thesis across both major paradigms.

**vs [[micheli-2023-iclr|IRIS]] and [[alonso-2024-neurips|DIAMOND]]:** These transformer-based and diffusion-based world models move in the opposite direction — toward richer, more expressive reconstruction (autoregressive token prediction for IRIS, diffusion for DIAMOND). Dreamer-CDP's result suggests this reconstruction investment may be unnecessary, at least in the Crafter domain.

**vs BYOL-Explore (Guo et al., 2022):** BYOL-Explore uses a BYOL-style predictor with EMA target networks for exploration bonuses in Atari. Dreamer-CDP's contribution is showing that the same non-contrastive prediction idea works for the world model's primary representation learning — not just exploration — and does so without EMA targets.

---

## Strengths

- **Minimal modification, strong result**: Only two changes to DreamerV3 (encoder split + CDP predictor), yet matches the original on the hardest benchmark (Crafter) where all prior reconstruction-free methods fail badly.
- **Clean ablation**: The 3.2% vs 16.2% ablation (no CDP vs CDP) is a crisp demonstration that continuous deterministic prediction is the key ingredient, not any other auxiliary loss.
- **No auxiliary signals**: Unlike DreamerPro (needs augmented views) and MuDreamer (needs action prediction), Dreamer-CDP relies solely on internal embedding prediction — the purest realization of the JEPA idea within Dreamer.
- **Architecture-preserving**: Retains the RSSM, categorical latents, KL-balanced training, and actor-critic — all of DreamerV3's proven infrastructure. Only the reconstruction loss and decoder are removed.
- **Open-source code**: Available at github.com/fmi-basel/Dreamer-CDP.

---

## Weaknesses & Limitations

- **Single benchmark**: Evaluated only on Crafter. DreamerV3's strength is its single-hyperparameter generalization across 150+ tasks in 8 domains. Whether CDP maintains parity with reconstruction on Atari, DMC, Minecraft, and other domains is unknown.
- **Statistical significance**: The Crafter score difference (16.2 vs 14.5%) has p=0.10, which is suggestive but not definitive. With only n=7 seeds and high variance, the claim is "matches" rather than "exceeds."
- **No computational savings quantified**: Removing the decoder should reduce FLOPs and memory, but the paper does not measure wall-clock time, GPU memory, or parameter count comparisons.
- **Higher learning rate sensitivity**: The reliance on a higher sequence model learning rate (instead of EMA) for collapse avoidance may introduce its own tuning sensitivity, which is not ablated.
- **Cumulative reward gap**: While Crafter score is higher (16.2 vs 14.5), cumulative reward is lower (9.8 vs 11.7). The Crafter score weights novel achievements more, so Dreamer-CDP may discover more achievements but exploit them less reliably.
- **No scaling study**: DreamerV3 showed predictable scaling from 12M to 400M parameters. Whether CDP's advantages persist at different scales is untested.
- **Workshop paper scope**: At 4 pages of main content, the paper leaves many questions unanswered — deeper ablations, multi-domain evaluation, and theoretical analysis of the collapse-avoidance mechanism are all deferred.

---

## Key Takeaways

- **Continuous deterministic prediction closes the reconstruction gap**: Prior reconstruction-free Dreamer variants scored 4.7-7.3% on Crafter; adding CDP to DreamerV3 without reconstruction scores 16.2%, matching the 14.5% reconstruction baseline. The secret ingredient is predicting continuous embeddings rather than discrete latents.
- **The RSSM's internal KL prediction is necessary but not sufficient**: Without CDP, removing reconstruction collapses Dreamer to 3.2%. With CDP but without KL alignment, performance drops to 6.3%. Both objectives work together — CDP provides rich representation learning signal; KL alignment maintains the structured latent space needed for imagination.
- **Convergent evidence with LeWorldModel**: Two independent groups — Hauri & Zenke (Dreamer/RL) and Maes et al. (JEPA/planning) — simultaneously demonstrate that next-embedding prediction replaces pixel reconstruction in world models. This convergence across architectures and training paradigms strengthens the case that reconstruction-free is the future.
- **The decoder was the bottleneck, not the latent structure**: DreamerV3's categorical latents and KL-balanced training remain intact in Dreamer-CDP. The reconstruction decoder was the problematic component — forcing representations to preserve pixel-level detail. Removing it and adding embedding prediction frees representations to focus on dynamically relevant structure.
- **Practical implication**: For practitioners building on Dreamer, this suggests removing the decoder and adding a simple cosine-similarity predictor on CNN embeddings is a drop-in improvement — simpler architecture, no view augmentation needed, no action prediction needed, with equivalent or better performance on sparse-reward environments.

---

## BibTeX

{% raw %}
```bibtex
@article{hauri2026dreamercdp,
  title={Dreamer-CDP: Improving Reconstruction-free World Models Via Continuous Deterministic Representation Prediction},
  author={Hauri, Michael and Zenke, Friedemann},
  journal={arXiv preprint arXiv:2603.07083},
  year={2026}
}
```
{% endraw %}

---
title: "JEPA for RL: Investigating Joint-Embedding Predictive Architectures for Reinforcement Learning"
type: paper
paper_id: P056
authors:
  - "Kenneweg, Tristan"
  - "Kenneweg, Philip"
  - "Hammer, Barbara"
year: 2025
venue: "ESANN 2025"
arxiv_id: "2504.16591"
url: "https://arxiv.org/abs/2504.16591"
pdf: "../../raw/kenneweg-2025-esann.pdf"
tags: [JEPA, reinforcement-learning, representation-learning, vision-transformer, model-collapse, self-supervised-learning, CartPole]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
cited_by: []
---

# JEPA for RL: Investigating Joint-Embedding Predictive Architectures for Reinforcement Learning

> **JEPA for RL** adapts the Joint-Embedding Predictive Architecture to reinforcement learning from images by using a vision transformer as the encoder with a temporal frame-stacking scheme, a two-layer MLP predictor conditioned on actions, and a classification token for compact state embeddings, demonstrating on Cart Pole that combining JEPA loss with RL gradient backpropagation through the encoder yields the best performance while JEPA-only training (without RL gradients) collapses unless explicit variance regularization is added.

**Authors:** Tristan Kenneweg, Philip Kenneweg, Barbara Hammer (University of Bielefeld) | **Venue:** ESANN 2025 | **arXiv:** [2504.16591](https://arxiv.org/abs/2504.16591)

---

## Problem & Motivation

Reinforcement learning from images is slow and compute-intensive because images are extremely high-dimensional state descriptions. In Cart Pole, the image state has 720,000 dimensions (400x600x3), while the true state is only 4-dimensional (cart position, angle, velocity, angular velocity). Learning a compact, informative representation from images is essential.

**Autoencoders** (including VAEs used in World Models by Ha & Schmidhuber) are the standard approach for learning such representations, but they have a fundamental limitation: they assign equal importance to every pixel. In Cart Pole, a white background pixel receives the same reconstruction weight as a pixel of the pole, even though the pole carries far more task-relevant information. This leads to reconstructions where backgrounds are crisp but task-relevant moving parts are blurry.

**JEPA** ([[lecun-2022-openreview]]) offers a solution: by predicting in embedding space rather than pixel space, the model can choose to create embeddings that ignore irrelevant details. However, JEPA is much more prone to **representation collapse** -- constant encoder outputs with an identity predictor achieve perfect loss.

The paper investigates how to adapt JEPA to RL from images, focusing on collapse prevention and the interaction between JEPA and RL training signals.

---

## Core Idea

The core adaptation is to use JEPA as a **representation learning module** for RL agents operating from pixel observations. The x-encoder (context encoder) produces a compact embedding from stacked image frames, which is fed to both:
1. A **predictor** that predicts the next-frame embedding (JEPA objective).
2. An **actor-critic** network that produces actions and value estimates (RL objective).

The key question explored is: **what combination of JEPA loss, RL gradient backpropagation, and regularization produces the best RL performance?**

---

## How It Works

### Input and Encoder (Figure 2)

- **Input x**: Patch embeddings from the last 3 frames (f_{t-2}, f_{t-1}, f_t) of the environment.
- **x-encoder V(theta, x)**: A Vision Transformer (ViT) that processes all patch embeddings from all 3 frames simultaneously. Positional encodings encode both spatial (i, j) and temporal (t) coordinates.
- **Embedding**: A learnable classification token is prepended to the ViT. Only the last-layer embedding of this CLS token (dimension d_emb = 64) serves as the state representation s_x. This is deliberately compact -- sufficient for tasks describable with a short state vector.

### Target Encoder

- **Target y**: Frames f_{t-1}, f_t, f_{t+1} (shifted by one timestep into the future).
- **y-encoder V(theta_bar, y)**: Same ViT architecture. Weights are an EMA of x-encoder weights: theta_bar_{t+1} = 0.99 * theta_bar_t + 0.01 * theta_{t+1}. No gradients flow through the y-encoder.

### Predictor

- A shallow **two-layer MLP** that predicts the target embedding s^p_y from the context embedding s_x.
- **Action conditioning**: The one-hot encoded action taken by the actor (from state s_t to s_{t+1}) is projected to the hidden dimension via a linear layer and added to the embedding after the first MLP layer.
- The predictor is intentionally small to force the embedding stage (not the predictor) to solve the state prediction task.

### Loss Functions

**JEPA loss**: L2 distance between predicted and target embeddings:

L_JEPA = ||s^p_y - s_y||^2_2

**Variance regularization** (collapse prevention):

L_reg = -min(1, (1/d_emb) * sum_i Var(s_x)_i)

This encourages batch-wise variance in embeddings, clamped at 1 since variance is unbounded. Inspired by VICReg.

**Total loss**: L = L_JEPA + L_actor + L_critic + L_reg

where L_reg is optional and L_actor + L_critic are PPO losses that can optionally be backpropagated through the encoder.

### Four Experimental Configurations

| Config | JEPA Loss | RL Gradients to Encoder | Regularization |
|---|---|---|---|
| J-hat, grad, R-hat | No | Yes | No |
| J, grad, R-hat | Yes | Yes | No |
| J, grad-hat, R-hat | Yes | No | No |
| J, grad-hat, R | Yes | No | Yes |

### Inference

The x-encoder produces s_x, which is fed to the actor for action selection and the critic for value estimation. The predictor and y-encoder are not used at inference.

---

## Results

### Cart Pole Performance (Figure 3)

Results over 100k environment steps, 5 runs each, using PPO actor-critic:

**J-hat, grad, R-hat (Baseline -- no JEPA, RL gradients only):**
- The agent learns some advantageous behavior but in a limited fashion. Episodic return increases slowly and plateaus around 40-50.
- Embedding variance stays in a reasonable range (0-1).

**J, grad, R-hat (JEPA + RL gradients -- best configuration):**
- **Best results.** Reward increases much faster and does not plateau, reaching ~100 by 100k steps.
- Embedding variances remain reasonable.
- The combination of JEPA and RL signals produces the most informative representations.

**J, grad-hat, R-hat (JEPA only, no RL gradients):**
- **Model collapse occurs.** The encoder maps all inputs to the same embedding. Batch-wise variance drops below 10^-7. JEPA loss becomes trivially low. The actor cannot learn anything since embeddings carry no information.
- In some runs, embeddings eventually recover from collapse with extended training.

**J, grad-hat, R (JEPA + regularization, no RL gradients):**
- **Collapse is prevented** by the variance regularization. Batch-wise variance remains healthy.
- The actor can learn from informative embeddings, but learning is much slower than with RL gradients.
- Demonstrates that JEPA can produce useful state representations purely through self-supervised learning (no task-specific reward signal), although combining with RL gradients is much more effective.

---

## Comparison to Prior Work

| | **JEPA for RL** | World Models (Ha & Schmidhuber) | CURL (Srinivas et al.) | DreamerV3 |
|---|---|---|---|---|
| **Representation** | JEPA (embedding prediction) | VAE (pixel reconstruction) | Contrastive | RSSM (generative) |
| **Encoder** | Vision Transformer | CNN | CNN | CNN |
| **Prediction space** | Embedding | Pixel | Embedding (contrastive) | Latent + pixel |
| **Action conditioning** | In predictor MLP | In dynamics model | N/A | In transition model |
| **Collapse prevention** | EMA + variance reg + RL grads | N/A (generative) | Negative pairs | N/A (generative) |
| **Scale** | Cart Pole (proof of concept) | VizDoom, car racing | Atari, DMControl | Atari, DMControl, Minecraft |

**vs [[lecun-2022-openreview]]:** This paper directly implements LeCun's JEPA proposal for RL from images. The key finding is that JEPA alone (without RL gradients) tends to collapse in the RL setting, requiring either variance regularization or task-specific gradient signals. This is a practically important observation for future JEPA-based RL systems.

**vs [[assran-2023-cvpr]] (I-JEPA):** I-JEPA uses multi-block spatial masking for image SSL. JEPA for RL instead uses temporal frame prediction (predict embedding of future frames given past frames), which is more natural for sequential decision-making. The CLS token approach (compact single-vector embedding) differs from I-JEPA's patch-level predictions.

**vs World Models (Ha & Schmidhuber, 2018):** The VAE-based World Models reconstruct pixels, spending capacity on irrelevant details. JEPA for RL predicts in embedding space, theoretically allowing more task-relevant representations. However, the paper only evaluates on Cart Pole, making direct performance comparison with World Models on complex environments impossible.

---

## Strengths

- **Clear problem formulation**: Provides a well-structured investigation of how to adapt JEPA to RL, with clearly defined experimental conditions that isolate the contributions of each component.
- **Thorough collapse analysis**: The four configurations systematically reveal when and why collapse occurs, and how to prevent it -- actionable guidance for practitioners.
- **Practical architecture decisions**: Using a CLS token for compact embeddings, temporal frame stacking with positional encodings, and action-conditioned prediction are sensible adaptations that could transfer to more complex environments.
- **Important negative result**: Showing that JEPA alone (without RL gradients) collapses in the RL setting is a valuable finding that tempers expectations and motivates further research on collapse prevention.

---

## Weaknesses & Limitations

- **Only Cart Pole**: Evaluation on a single, simple environment severely limits the conclusions that can be drawn. Cart Pole is trivially solvable with many methods; the value of JEPA would be better demonstrated on environments where standard approaches struggle.
- **No comparison to baselines**: No comparison to VAEs, CURL, DrQ, or other standard image-RL representation learning methods. It is unclear whether JEPA offers any advantage over existing approaches.
- **Preliminary results**: The paper is a 6-page workshop-style publication with proof-of-concept experiments. There are no ablation studies on architecture choices (ViT depth, embedding dimension, EMA momentum, etc.).
- **Limited scale**: The ViT and MLP are small. Whether the findings transfer to larger architectures and more complex visual environments is unknown.
- **No pre-training experiment**: The paper only explores online learning where JEPA and RL are trained simultaneously. Pre-training JEPA representations on environment data before RL fine-tuning (as done in vision JEPA papers) is not explored.
- **Collapse with JEPA-only is concerning**: The fact that JEPA alone collapses (even with EMA) and requires either RL gradients or explicit regularization suggests that temporal prediction in RL may be harder than spatial prediction in vision for collapse prevention.

---

## Key Takeaways

- **JEPA + RL gradients is the winning combination**: Combining JEPA prediction loss with RL gradient backpropagation through the encoder produces the best representations for RL from images, outperforming either signal alone.
- **JEPA alone collapses without additional measures**: Unlike in vision (where I-JEPA works with just EMA), temporal JEPA for RL collapses without either RL gradients or explicit variance regularization. This is a critical practical finding.
- **Variance regularization enables JEPA-only representation learning**: With the VICReg-inspired regularization, JEPA can learn informative representations without any task-specific signal, validating the self-supervised JEPA principle for RL -- albeit with slower learning.
- **CLS token gives compact embeddings**: Using a classification token to produce a single d_emb-dimensional vector (d_emb=64) is an effective strategy for RL where a compact state representation is needed, rather than predicting all patch embeddings.
- **More work needed on complex environments**: Cart Pole is a proof of concept. The real test for JEPA in RL will be on Atari, DMControl, or other visually complex environments where pixel reconstruction struggles.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{kenneweg2025jepa,
  title={JEPA for RL: Investigating Joint-Embedding Predictive Architectures for Reinforcement Learning},
  author={Kenneweg, Tristan and Kenneweg, Philip and Hammer, Barbara},
  booktitle={Proceedings of the European Symposium on Artificial Neural Networks (ESANN)},
  year={2025},
  note={arXiv:2504.16591}
}
```
{% endraw %}

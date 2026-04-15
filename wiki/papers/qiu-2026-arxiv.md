---
title: "Self-Improving World Modelling with Latent Actions (SWIRL)"
type: paper
paper_id: P061
authors:
  - "Qiu, Yifu"
  - "Zhao, Zheng"
  - "Li, Weixian Waylon"
  - "Ziser, Yiftah"
  - "Korhonen, Anna"
  - "Cohen, Shay B."
  - "Ponti, Edoardo M."
year: 2026
venue: arXiv
arxiv_id: "2602.06130"
url: "https://arxiv.org/abs/2602.06130"
pdf: "../../raw/qiu-2026-arxiv.pdf"
tags: [world-model, self-improving, latent-actions, inverse-dynamics, forward-dynamics, grpo, vlm, llm]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
cited_by: []
---

# Self-Improving World Modelling with Latent Actions (SWIRL)

> **One sentence.** SWIRL enables LLMs and VLMs to self-improve their intrinsic world modelling ability from unlabelled state-only sequences by alternating between a Forward World Model and an Inverse Dynamics Model trained reciprocally via GRPO, achieving +16% on AURORA-BENCH, +28% on ByteMorph, +16% on WorldPredictionBench, and +14% on StableToolBench over supervised baselines.

**Authors:** Yifu Qiu, Zheng Zhao, Weixian Waylon Li, Yiftah Ziser, Anna Korhonen, Shay B. Cohen, Edoardo M. Ponti | **Venue:** arXiv 2026 | **arXiv:** [2602.06130](https://arxiv.org/abs/2602.06130)

---

## Problem & Motivation

Large foundation models (LLMs and VLMs) have been shown to internalize world models to some extent during pretraining, enabling them to reason about states and actions. However, building robust internal world models faces two bottlenecks: (1) current approaches require densely action-annotated trajectories, which are expensive or infeasible to collect for open-world tasks; and (2) the inherent ambiguity of inverse dynamics -- a transition between two states may be explained by multiple valid actions -- makes purely supervised learning brittle. Existing self-improving methods focus on reasoning or code generation, not on enhancing the model's ability to predict future states and infer actions from state transitions.

---

## Core Idea

SWIRL formalises world modelling as two complementary components -- a Forward World Model (FWM) that predicts next states given current state and latent action, and an Inverse Dynamics Model (IDM) that infers latent actions from state transitions -- and trains them reciprocally using reinforcement learning without ground-truth action annotations. The key insight is that FWM and IDM can serve as critic and policy for each other: in Phase I, the IDM rewards the FWM for generating identifiable futures (maximising a variational lower bound on Conditional Mutual Information); in Phase II, the FWM rewards the IDM for inferring plausible actions (maximising the Evidence Lower Bound). Both phases use GRPO, with the frozen counterpart providing the reward signal.

---

## How It Works

### Overview

Input: unlabelled dataset of state pairs D = {(x_i, y_i)} (no action annotations). SWIRL alternates two RL phases until convergence, with each phase using GRPO and the other model as a frozen reward function.

### Phase I: FWM Optimisation (Variational Information Maximisation)

1. Freeze IDM Q_phi, optimise FWM P_theta
2. For each state x, sample latent action z ~ Q_phi(z | x, y) from the IDM
3. Generate G rollouts y_hat_1, ..., y_hat_G ~ P_theta(. | x, z)
4. Compute reward: r_k = log Q_phi(z | x, y_hat_k) -- reward is high when the IDM can recover the action from the generated future
5. Update theta via GRPO to maximise advantage-weighted log P_theta(y_hat_k | x, z)

**Theorem 3.1:** This objective maximises a variational lower bound on the Conditional Mutual Information I(Z; Y_hat | X), encouraging the FWM to produce futures that are identifiable -- distinct outcomes for different latent actions.

### Phase II: IDM Optimisation (ELBO Maximisation)

1. Freeze FWM P_theta, optimise IDM Q_phi
2. For each state pair (x, y), sample G action candidates z_1, ..., z_G ~ Q_phi(. | x, y)
3. Compute reward: r_k = log P_theta(y | x, z_k) -- reward is high when the FWM can reproduce the observed transition given the inferred action
4. Update phi via GRPO to maximise advantage-weighted log Q_phi(z_k | x, y)

**Theorem 3.2:** This objective maximises the Evidence Lower Bound (ELBO) of log P_theta(y|x), encouraging the IDM to infer actions that are plausible under the current forward dynamics.

### Algorithm (SWIRL)

```
Repeat:
  Phase I: Freeze IDM. For each batch x ~ D:
    z ~ Q_phi(z|x,y), generate rollouts y_hat ~ P_theta(.|x,z)
    R_k = log Q_phi(z|x,y_hat_k)
    Update theta via GRPO
  Phase II: Freeze FWM. For each batch (x,y) ~ D:
    z_1,...,z_G ~ Q_phi(.|x,y)
    R_k = log P_theta(y|x,z_k)
    Update phi via GRPO
Until convergence
```

### Experimental Setup

**VLM experiments:** Liquid 7B (unified VLM) as base model. SFT warm-up on PICO-BANANA-400K + AURORA data. SWIRL RL on 30K unlabelled videos per iteration from VidGen-1M. Trained on 32 NVIDIA H200 140GB GPUs (FWM) + 8 H200 (IDM).

**LLM experiments:** Qwen-2.5-3B-Instruct. Fine-tuned with GRPO using DeepSpeed on 8 NVIDIA Grace-Hopper GH100 GPUs.

**Benchmarks:**
- **Visual (single-turn):** AURORA-BENCH (5 image editing datasets), ByteMorph (action-conditioned image editing)
- **Visual (multi-turn):** WorldPredictionBench (up to T=6 autoregressive rollouts)
- **Textual:** ScienceWorld, Mind2Web, StableToolBench

### Separate vs Shared Weights

The authors find that separate weights for FWM (theta) and IDM (phi) yield stable optimisation with monotonic improvement across iterations. Shared weights (theta = phi) reduce memory but introduce instability, with performance dropping at Iteration 3.

---

## Results

### AURORA-BENCH (Visual World Modelling, Single-Turn)

| Method | GPT-4o Avg | DE Avg | CLIP Avg |
|---|---|---|---|
| Liquid-SFT | 4.36 | 0.36 | 0.84 |
| SWIRL (IDM -> FWM) | 4.83 | 0.36 | 0.84 |
| SWIRL (Iterative) | 5.06 | 0.38 | 0.84 |
| **SWIRL (Iter. + Share)** | **5.00** | **0.39** | **0.84** |
| BAGEL-14B | 6.44 | 0.22 | 0.91 |
| OmniGen2 | 6.05 | 0.25 | 0.89 |

SWIRL raises GPT-4o score from 4.36 (SFT) to 5.06 (+16%), outperforming the bootstrapping baseline and test-time verification (N=8). Competitive with much larger unified VLMs (BAGEL-14B, OmniGen2).

### ByteMorph (Action-Conditioned Editing)

| Method | Average |
|---|---|
| Liquid-SFT | 43.38 |
| SWIRL (Iterative) | **53.77** |
| SWIRL (Iter. + Share) | 55.72 |
| OmniGen2 | 60.14 |

+26.4% improvement over Liquid-SFT baseline. Competitive with OmniGen2 (a much larger model).

### WorldPredictionBench (Multi-Turn, T=1 to T=6)

| Method | T=1 | T=4 | T=6 |
|---|---|---|---|
| Liquid-SFT | 3.09 | 1.17 | 0.97 |
| SWIRL (Iterative) | 3.23 | 1.32 | 1.11 |
| SWIRL (Best) | **3.16** | **1.59** | **1.11** |
| BAGEL | 4.29 | 3.22 | -- |

SFT degrades rapidly from T=1 to T=6 (3.09 -> 0.97). SWIRL maintains significantly higher fidelity (+14.4% at T=6), demonstrating that the reciprocal cycle internalises more robust physical dynamics.

### StableToolBench (LLM, Tool Calling)

| Method | Average |
|---|---|
| Qwen-2.5-3B-SFT | 12.85 |
| **Qwen-2.5-3B-SWIRL** | **14.61** |
| Qwen-2.5-32B-Instruct | 7.31 |

SWIRL on 3B model surpasses 32B instruction-tuned models, demonstrating that self-improving world modelling can compensate for model size in structured tool-calling environments.

### Data Efficiency (RL vs SFT)

Across all 5 benchmarks and all training budgets (3.2K to 12.8K samples), SWIRL (RL) consistently outperforms both SFT-Continue and SFT-Merge. The gap widens at larger data scales (average 4.27 -> 4.73), indicating RL captures latent structure more effectively than supervised imitation.

---

## Comparison to Prior Work

| | SWIRL | SFT (supervised) | Test-Time Verification | Bootstrapping |
|---|---|---|---|---|
| Requires action annotations | No | Yes (for actions) | No | Partially |
| Self-improving | Yes (iterative) | No | No | Partially |
| Theoretical guarantees | Yes (CMI + ELBO) | No | No | No |
| Multi-turn consistency | Strong (T=6 robustness) | Weak (degrades rapidly) | Moderate | Moderate |

**vs [[lecun-2022-openreview]] ([H-JEPA](../papers/lecun-2022-openreview.md)):** LeCun envisions world models that predict future states conditioned on actions in latent space. SWIRL implements this vision within LLMs/VLMs by decomposing world modelling into FWM and IDM, and training them without explicit action annotations through a reciprocal RL loop. While LeCun focuses on learned latent representations, SWIRL operates in the text/image generation space of foundation models.

**vs [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)):** DreamerV3 learns a world model from environment interactions with explicit action labels and uses it for RL policy training. SWIRL instead enhances the intrinsic world modelling capabilities already present in pretrained LLMs/VLMs, without requiring action labels or environment interaction, and validates on a much broader range of tasks (visual editing, web, tool calling).

**vs [[maes-2026-arxiv]] ([LeWorldModel](../papers/maes-2026-arxiv.md)):** LeWorldModel trains a JEPA-based world model from pixels with explicit action conditioning for robotic planning. SWIRL addresses a different setting: improving the internal world model of large pretrained models from unlabelled data. The two approaches are complementary -- LeWorldModel for embodied control from scratch, SWIRL for enhancing foundation model capabilities.

---

## Strengths

- **No action annotations required:** SWIRL learns from unlabelled state-only sequences, dramatically reducing data requirements compared to supervised approaches
- **Principled theoretical framework:** Each phase has a formal justification -- Phase I maximises a variational CMI lower bound, Phase II maximises the ELBO -- grounding the reciprocal training in information theory
- **Broad applicability:** Validated across 4 distinct environment types (visual dynamics, textual physics, web, tool calling) with both VLMs and LLMs, demonstrating generality
- **Self-improvement across iterations:** Clear evidence of cumulative gains (GPT-4o scores 4.36 -> 4.56 -> 4.98 -> 5.06 across SFT + 3 iterations), with training rewards and evaluation scores improving monotonically in the separate-weights configuration
- **Multi-turn robustness:** SWIRL shows dramatically less degradation than SFT on long-horizon rollouts (WorldPredictionBench T=6), suggesting it learns actual dynamics rather than memorising single-step transitions
- **Latent actions remain meaningful:** Analysis shows generated actions maintain high diversity (>94% unique), increasing descriptive complexity, and stable naturalness (PPL ratio) across iterations -- no reward collapse

---

## Weaknesses & Limitations

- **Weak base model:** Liquid 7B is a mid-size VLM whose generation quality caps SWIRL's absolute performance. Much larger models (BAGEL-14B, OmniGen2) still outperform on absolute metrics, though SWIRL closes the gap substantially
- **Compute-intensive:** Requires 32+8 H200 GPUs for VLM experiments and multiple RL iterations; the overhead over SFT is significant
- **Shared weights unstable:** The more parameter-efficient shared-weight variant (theta = phi) introduces instability by Iteration 3, suggesting gradient interference between FWM and IDM objectives
- **SFT warm-up required:** Without initial SFT, random policy outputs make GRPO exploration impossible. The approach bootstraps from supervised initialisation, not fully from scratch
- **Limited physical grounding:** Camera zoom/motion improvements are limited since unlabelled video data (VidGen-1M) is predominantly static; self-improvement is strongest for object/human motion and interactions
- **No comparison to DreamerV3-style world models:** The paper benchmarks against SFT and VLM baselines but not against dedicated world model architectures used for planning

---

## Key Takeaways

- SWIRL decomposes world modelling into Forward World Modelling (FWM) and Inverse Dynamics Modelling (IDM), training them reciprocally via GRPO without action annotations -- FWM optimisation maximises variational Conditional Mutual Information (identifiability), IDM optimisation maximises the ELBO (data fidelity)
- Self-improving world modelling from unlabelled data is effective: SWIRL achieves +16% on AURORA-BENCH, +26% on ByteMorph, +14% on StableToolBench over supervised baselines, with cumulative gains across RL iterations
- Multi-turn consistency is a key differentiator: SFT degrades rapidly on long-horizon WorldPredictionBench (3.09 -> 0.97 at T=6), while SWIRL maintains significantly higher fidelity (3.23 -> 1.11), indicating it internalises actual dynamics rather than memorising single transitions
- Separate FWM/IDM weights yield stable monotonic improvement, while shared weights reduce memory but cause instability after multiple iterations -- a practical consideration for deployment
- A 3B LLM with SWIRL outperforms 32B instruction-tuned models on StableToolBench, demonstrating that self-improving world modelling can compensate for raw model scale in structured reasoning tasks

---

## BibTeX

{% raw %}
```bibtex
@article{qiu2026swirl,
  title     = {Self-Improving World Modelling with Latent Actions},
  author    = {Qiu, Yifu and Zhao, Zheng and Li, Weixian Waylon and Ziser, Yiftah and Korhonen, Anna and Cohen, Shay B. and Ponti, Edoardo M.},
  journal   = {arXiv preprint arXiv:2602.06130},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.06130}
}
```
{% endraw %}

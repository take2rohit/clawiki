---
title: "TD-MPC2: Scalable, Robust World Models for Continuous Control"
type: paper
paper_id: P013
authors:
  - "Hansen, Nicklas"
  - "Su, Hao"
  - "Wang, Xiaolong"
year: 2024
venue: ICLR 2024
arxiv_id: "2310.16828"
url: "https://arxiv.org/abs/2310.16828"
pdf: "../../raw/hansen-2024-iclr.pdf"
tags: [world-model, model-predictive-control, continuous-control, multi-task, latent-dynamics, td-learning, dmcontrol, meta-world]
created: 2026-04-10
updated: 2026-04-10
cites:
  - hansen-2022-icml
  - hafner-2023-arxiv
cited_by:
  - maes-2026-arxiv
  - terver-2025-iclr
  - bagatella-2025-iclr
---

# TD-MPC2: Scalable, Robust World Models for Continuous Control

> **TD-MPC2** scales the TD-MPC framework to a single 317M parameter agent trained on 80 tasks across multiple domains, achieving consistently strong results across 104 continuous control tasks with a single set of hyperparameters — performance that scales log-linearly with model size.

**Authors:** Nicklas Hansen, Hao Su, Xiaolong Wang | **Venue:** ICLR 2024 | **arXiv:** [2310.16828](https://arxiv.org/abs/2310.16828)

---

## Problem & Motivation

The original TD-MPC required task-specific hyperparameter tuning and did not scale: naively increasing model or data size of TD-MPC leads to a net decrease in performance, as is commonly observed in RL. Current approaches to generalist embodied agents assume near-expert demonstrations for behavior cloning (severely limiting data availability) or rely on discrete action tokenization (incompatible with high-dimensional continuous control). A scalable continuous control algorithm that can consume large, mixed-quality datasets spanning multiple embodiments, action spaces, and task domains — without hyperparameter tuning — is needed. TD-MPC2 directly addresses both the robustness and scalability gaps in TD-MPC.

---

## Core Idea

TD-MPC2 makes two key sets of changes to TD-MPC: (1) it redesigns the architecture and training objective for robustness, using SimNorm latent normalization, LayerNorm+Mish activations, an ensemble of Q-functions, and discrete regression for rewards and values; and (2) it introduces learnable task embeddings and action masking to handle multiple heterogeneous tasks, embodiments, and action spaces without domain knowledge. These changes allow the same hyperparameters to work across 104 diverse tasks and allow performance to improve as model parameters scale from 1M to 317M.

---

## How It Works

### Overview

TD-MPC2 shares the same five-component TOLD architecture as TD-MPC (encoder h, latent dynamics d, reward R, value Q, policy prior p), but replaces or improves each component to achieve robustness and scalability. At decision time, MPPI planning in latent space generates action sequences evaluated using the learned model and value function; the first action from the best trajectory is executed. Training alternates between collecting environment data and updating all model components jointly.

### Architecture Changes vs. TD-MPC

| Design Choice | TD-MPC | TD-MPC2 |
|---------------|--------|---------|
| Activations | ELU | LayerNorm + Mish |
| Latent normalization | None (can explode) | SimNorm (simplicial embedding) |
| Q-functions | 2, continuous regression | 5, ensemble + 1% Dropout |
| TD targets | Minimum of 2 Q-functions | Minimum of 2 randomly subsampled from 5 |
| Reward/value regression | Continuous (MSE) | Discrete (soft cross-entropy in log-space) |
| Policy prior | Deterministic + Gaussian noise | Maximum entropy (SAC-style) |
| Replay sampling | PER | Uniform |
| Default model size | ~1M params | 5M (single-task), up to 317M (multi-task) |

**SimNorm** is the key representation stabilization technique. The latent z is partitioned into L groups, and each group is projected onto an L-dimensional simplex via softmax with temperature τ. This biases z toward sparsity, maintains a small ℓ₂-norm, and prevents the representation collapse that caused TD-MPC to diverge on some tasks.

**Discrete regression** for rewards and values uses soft cross-entropy with scalar targets treated as soft label distributions in a log-transformed bucket space (following DreamerV3/C51). This makes the training objective invariant to the magnitude of task rewards — critical for multi-task learning where rewards may differ by orders of magnitude across tasks.

### Task Embeddings and Action Masking (Multi-task)

Each task is associated with a learnable fixed-dimensional embedding e, jointly trained with all other components. All five TOLD components are conditioned on e. The ℓ₂-norm of e is constrained to ≤ 1 for training stability and to encourage semantically coherent embedding geometry. For new tasks, e can be initialized from a semantically similar task embedding or random vector and finetuned via online RL.

Action masking zero-pads all model inputs and outputs to the largest action dimension across tasks, and masks out invalid dimensions during planning and in policy prior predictions. This allows a single model to handle variable action spaces (e.g., A ∈ ℝ¹ for Cartpole to A ∈ ℝ³⁹ for MyoSuite) without domain knowledge.

### Training Objective

The joint model objective is:

**L(θ) = E_{(s,a,r,s')~B} [ Σ_{t=0}^{H} λᵗ ( ‖z'_t − sg(h(s'_t))‖² + CE(r̂_t, r_t) + CE(q̂_t, q_t) ) ]**

where the first term is joint-embedding prediction (consistency loss, continuous regression), the second is discrete reward prediction (soft cross-entropy), and the third is discrete value prediction (soft cross-entropy against TD targets). λ=0.5 weights near-term steps more heavily. The TD target q_t = r_t + γQ(z', p(z', e)) uses an exponential moving average of Q as the target network.

The policy prior objective trains p with maximum entropy RL:

**L_p(θ) = E [ Σ λᵗ (αQ(z_t, p(z_t)) − βH(p(·|z_t))) ]**

where α is tuned via moving statistics of Q values to prevent entropy collapse across varying task difficulties.

### Planning

MPPI-based planning identical to TD-MPC, with two simplifications for efficiency: (1) momentum between iterations is removed (found comparable results without it), (2) code-level optimizations including Q-ensemble vectorization improve throughput by ~2×, keeping wall-time comparable to TD-MPC despite 5× more parameters.

### Training Configuration

- **Single-task:** 5M parameter model; batch size 256; uniform replay; same hyperparameters across all 104 tasks
- **Multi-task:** 1M–317M parameter models; batch size 1024 (for 317M); data from replay buffers of 240 single-task TD-MPC2 agents (545M transitions total); 33 GPU days (RTX 3090) for 317M model
- **Few-shot:** 19M model pretrained on 70 tasks, finetuned for 20k steps on 10 held-out tasks with empty replay buffer

---

## Results

### Single-Task Performance (104 Tasks)

TD-MPC2 consistently outperforms all baselines (SAC, DreamerV3, TD-MPC, CURL, DrQ-v2) across all four task domains using a single set of hyperparameters:

| Domain | Tasks | TD-MPC2 vs. SAC | TD-MPC2 vs. DreamerV3 | TD-MPC2 vs. TD-MPC |
|--------|-------|-----------------|-----------------------|--------------------|
| DMControl | 39 | Better | Better | Better (stable) |
| Meta-World | 50 | Better | Better | Better |
| ManiSkill2 | 5 | Much better | Much better | Better |
| MyoSuite | 10 | Better | Better | Better (no prior results) |

On high-dimensional locomotion (Dog A ∈ ℝ³⁸, Humanoid A ∈ ℝ²¹): SAC and DreamerV3 show numerical instabilities on Dog tasks; TD-MPC2 reliably solves all Dog and Humanoid variants. On hard manipulation (ManiSkill2 Pick YCB — 74 object categories): TD-MPC2 achieves ~50% success vs. near-0% for SAC and DreamerV3.

### Visual RL (10 Image-Based DMControl Tasks)

With encoder replaced by a shallow CNN (no other changes), TD-MPC2 is comparable to DrQ-v2 and DreamerV3 on 10 visual DMControl tasks — without any hyperparameter modification.

### Multi-Task Scaling (80 Tasks)

| Model Params | GPU Days | Normalized Score |
|-------------|---------|-----------------|
| 1M | 3.7 | 16.0 |
| 5M | 4.2 | 49.5 |
| 19M | 5.3 | 57.1 |
| 48M | 12 | 68.0 |
| **317M** | **33** | **70.6** |

Performance scales approximately log-linearly with model parameters. The 317M model performs 80 tasks spanning DMControl, Meta-World, and multiple embodiments from a single set of weights. Scaling has not saturated at 317M — further improvements are expected with larger models.

### Few-Shot Learning

A 19M parameter model pretrained on 70 tasks and finetuned for 20k steps (online RL, empty replay) on 10 held-out tasks achieves ~2× the normalized score of the same model trained from scratch (47.0 vs. 24.0). Task embedding similarity analysis shows that semantically similar tasks (Door Open / Door Close) cluster in embedding space, with similarity correlating more with dynamics than task objective.

### Ablations

All ablations conducted on three hardest single-task and 80-task multi-task settings:

- **Planning vs. policy-only:** Planning + policy achieves ~54 normalized score (multitask); policy alone achieves ~42 — planning is essential and contributes ~30% improvement.
- **SimNorm vs. LayerNorm vs. no norm:** SimNorm achieves 54.2 (multitask); LayerNorm+SimNorm also 54.2; no normalization 46.8. SimNorm is essential for training stability.
- **Discrete vs. continuous regression:** Discrete achieves 54.2 (multitask); continuous 49.6. Discrete regression provides ~10% improvement by decoupling loss magnitude from reward scale.
- **Q-function ensemble size:** 5 Q-functions (54.2 multitask) > 2 (default TD-MPC level) > 10 (marginal benefit over 5). 5 is the sweet spot.
- **Task embedding normalization:** Normalized embeddings (ℓ₂ ≤ 1) lead to more semantically coherent task relations and marginally better multi-task performance.

---

## Comparison to Prior Work

| Method | Single-task hyperparams | Multi-task | Scalable | Latent space | Action spaces |
|--------|------------------------|-----------|---------|-------------|---------------|
| SAC | Per-task tuning | No | No | State | Continuous |
| DreamerV3 | Fixed (claimed) | No at this scale | Partially | Continuous latent | Continuous |
| TD-MPC | Per-task tuning | Limited (same domain) | No (degrades) | Task-oriented latent | Continuous |
| **TD-MPC2** | Single set, all tasks | Yes (80 tasks, 4 domains) | Yes (log-linear) | Task-oriented latent + SimNorm | Continuous (variable) |

**[[hafner-2023-arxiv]] ([DreamerV3, Hafner et al., 2023](../papers/hafner-2023-arxiv.md))** is the closest model-based competitor and also claims fixed hyperparameters. However, DreamerV3 experiences numerical instabilities on Dog tasks, struggles with fine-grained manipulation (ManiSkill2), and does not consider multi-task scaling across domains in the same paper. DreamerV3 uses ~20M parameters vs. TD-MPC2's 5M default.

**[[hansen-2022-icml]] ([TD-MPC, Hansen et al., 2022](../papers/hansen-2022-icml.md))** uses per-task hyperparameters and degrades when naively scaled — the opposite of TD-MPC2. The architectural differences (SimNorm, discrete regression, ensemble Q-functions) are what enable scaling.

**Gato / RT-1 / behavior cloning approaches** require near-expert demonstration data, limiting scalable data acquisition. TD-MPC2 learns from unstructured online RL data with no expert demonstrations required.

---

## Strengths
- Demonstrates that implicit world models can scale to 317M parameters with log-linear performance improvement — the clearest scaling result for model-based RL to date.
- Single hyperparameter set across 104 tasks spanning 4 domains, 4 embodiment types, and action spaces from A ∈ ℝ¹ to A ∈ ℝ³⁹.
- SimNorm is a simple, principled normalization scheme that prevents representation collapse and training instability without requiring careful tuning.
- Releases 300+ model checkpoints, datasets, and training code at tdmpc2.com — enabling the community to build on large pretrained world models.
- Few-shot adaptation of pretrained world models achieves 2× better performance than learning from scratch, opening the door to foundation world models.

## Weaknesses & Limitations
- Still requires reward signals for training; extending to reward-free or goal-conditioned settings is mentioned as future work.
- Evaluated exclusively on simulation; real-robot transfer is addressed in concurrent work (MoDem-v2) but not in this paper.
- Discrete action spaces are not natively supported (Appendix I discusses an extension, but not evaluated in main results).
- The 317M model requires 33 GPU days and 545M prerecorded transitions — resource requirements are non-trivial.
- Risk of reward misspecification at scale: unintended behaviors from faulty reward functions become harder to detect and correct in larger agents.
- Multi-task generalization to truly unrelated tasks (not in the training distribution) remains limited; the encoder h_θ generalizes better than the dynamics d_θ.

## Key Takeaways
- SimNorm (simplicial normalization) and discrete regression are the two most impactful architectural changes enabling stable scaling: together they prevent representation collapse and decouple training from reward magnitude.
- TD-MPC2 achieves log-linear performance scaling from 1M to 317M parameters across 80 diverse tasks — the first clear scaling result for model-based RL in continuous control.
- A single 317M parameter world model can perform 80 tasks spanning DMControl, Meta-World, multiple embodiments, and action spaces up to A ∈ ℝ³⁹ with no domain knowledge.
- Pretrained world models enable 2× faster few-shot adaptation (20k steps) to held-out tasks, suggesting a path to foundation models for continuous control.
- Planning (MPPI) contributes ~30% performance improvement over policy-only inference even for large models, making it essential rather than optional.

---

## BibTeX
{% raw %}
```bibtex
@inproceedings{hansen2024tdmpc2,
  title={{TD-MPC2}: Scalable, Robust World Models for Continuous Control},
  author={Hansen, Nicklas and Su, Hao and Wang, Xiaolong},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
{% endraw %}

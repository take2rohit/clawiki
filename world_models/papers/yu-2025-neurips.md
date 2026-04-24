---
title: "Why and How Auxiliary Tasks Improve JEPA Representations"
type: paper
paper_id: P026
authors:
  - "Yu, Jiacan"
  - "Chen, Siyi"
  - "Liu, Mingrui"
  - "Horiuchi, Nono"
  - "Braverman, Vladimir"
  - "Xu, Zicheng"
  - "Haramati, Dan"
  - "Balestriero, Randall"
year: 2025
venue: "NeurIPS 2025"
arxiv_id: "2509.12249"
url: "https://arxiv.org/abs/2509.12249"
pdf: "../../raw/yu-2025-neurips.pdf"
tags: [JEPA, auxiliary-tasks, representation-learning, reinforcement-learning, bisimulation, collapse-prevention]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
  - balestriero-2025-iclr
cited_by: []
---

# Why and How Auxiliary Tasks Improve JEPA Representations

> **P-JEPA** adds an auxiliary regression head to a JEPA encoder trained jointly with latent dynamics, and proves a *No Unhealthy Representation Collapse* theorem: in deterministic MDPs, non-equivalent observations (differing in transition dynamics or auxiliary value) must map to distinct latent representations, theoretically grounding how auxiliary tasks control which information a JEPA encoder preserves.

**Authors:** Jiacan Yu, Siyi Chen (Johns Hopkins University), Mingrui Liu (Northwestern University), Nono Horiuchi (University of Rochester), Vladimir Braverman, Zicheng Xu (Johns Hopkins University), Dan Haramati (Brown University), Randall Balestriero (Brown University) | **Venue:** NeurIPS 2025 | **arXiv:** [2509.12249](https://arxiv.org/abs/2509.12249)

---

## Problem & Motivation

Joint-Embedding Predictive Architecture (JEPA) has become a go-to approach for image/video representation learning ([[assran-2023-cvpr]] ([I-JEPA](../papers/assran-2023-cvpr.md)), [[bardes-2024-tmlr]] ([V-JEPA](../papers/bardes-2024-tmlr.md))) and is increasingly used in model-based reinforcement learning. Yet its success is not "out-of-the-box": practitioners report brittleness and representation collapse unless carefully tuned. Critically, there is no existing theory explaining *which* design knobs matter and *why*. Previous SSL theories either only connect methods to each other or provide guarantees in unrealistic infinite/nonparametric regimes. The authors ask: can we provide actionable theoretical guarantees for a practical JEPA variant that explain how auxiliary tasks shape the learned representations?

---

## Core Idea

The paper introduces **P-JEPA** (Practical JEPA), a minimal JEPA variant augmented with an auxiliary regression head that is trained jointly with the latent-transition dynamics loss. The core theoretical contribution is the **No Unhealthy Representation Collapse** theorem (Theorem 1): if training drives both the latent-transition consistency loss and the auxiliary regression loss to zero, then any pair of *non-equivalent* observations -- those not in the largest bisimulation over the MDP and the auxiliary function -- must map to distinct latent representations. In other words, the auxiliary task *anchors* which distinctions the encoder must preserve. The choice of auxiliary function determines the equivalence relation and thus controls what information is encoded vs. discarded.

---

## How It Works

### Architecture (P-JEPA)

P-JEPA consists of three components:

1. **Encoder** E_phi: maps observations o to latent representations z = E_phi(o).
2. **Latent transition model** T_psi: predicts the next latent state given current state and action: T_psi(z_t, a_t) -> z_hat_{t+1}.
3. **Auxiliary head** P_theta: a neural network on top of the encoder that regresses to an auxiliary function p of observations: P_theta(z_t) -> p(o_t).

There is no stop-gradient in JEPA's latent dynamics loss; no target/EMA encoder is used. E_phi is updated by both the dynamics loss and the auxiliary loss.

### Training

The total loss combines two terms:

```
L(theta, phi, psi) = L_dyn + c_p * L_p
```

- **Latent transition loss**: L_dyn = E[||T_psi(E_phi(o_t), a_t) - E_phi(o_{t+1})||^2] -- measures consistency between predicted and actual next-state embeddings.
- **Auxiliary loss**: L_p measures the difference between P_theta(E_phi(o)) and p(o) -- how well the auxiliary head can predict the auxiliary function from the encoder's representation.
- c_p is a hyperparameter weighting the auxiliary loss.

The auxiliary function p can be the reward function, a Q-function, or even a randomly initialized neural network.

### Theory

**Definition 1 (Largest Bisimulation):** Given an MDP M and auxiliary function p, the largest bisimulation B* is defined via a monotone operator F that iteratively collects pairs of observations distinguishable by either (a) different p values or (b) different transition dynamics after the same action sequence. B* is the complement of the fixed point R*.

**Theorem 1 (No Unhealthy Representation Collapse):** If P-JEPA is well-trained (T_psi(E_phi(o), a) = E_phi(f(o,a)) and P_theta(E_phi(o)) = p(o) for all o, a), then any pair of observations not in the largest bisimulation must map to distinct representations: o_i not equivalent to o_j implies E_phi(o_i) != E_phi(o_j).

This means: the auxiliary task and transition dynamics jointly define an equivalence relation; the encoder can collapse within equivalence classes but must preserve distinctions across them.

### Inference

Standard representation evaluation. The encoder produces representations that are used for downstream tasks or planning via the learned latent dynamics model.

---

## Results

### Counting Environment (64x64 RGB, k in {0,...,8} objects)

The authors design a controlled counting environment where observations contain varying numbers of objects (0-8). Actions increase or decrease count by one. Reward is 1 iff count equals n=4, else 0.

**P-JEPA with reward auxiliary (top row of Figure 2):**
- PCA visualization shows **9 distinct clusters**, one per object count, matching the theory's prediction of 9 non-bisimilar sets.
- Pairwise L2 distances within clusters are smaller than across clusters.
- A decoder trained *without* backpropagating into the encoder cannot recover shape, color, or position -- confirming the encoder abstracts away these redundant factors while preserving count.

**P-JEPA with 256-D random auxiliary (middle row):**
- No count-based clustering in PCA; embeddings are separated but not organized by count.
- Decoder *can* recover position and partial shape/color information, since the random auxiliary makes almost all observation pairs non-bisimilar, preventing most collapse.

**Ablation -- training losses separately (bottom row):**
- Reward loss alone: only coarse separation (3 rough groups: 0-2, 3-5, 6-8).
- Latent transition loss alone: complete collapse into a single cluster.
- **Combined (P-JEPA): 9 separated clusters**, demonstrating that joint training produces richer representations than either loss alone.

---

## Comparison to Prior Work

| | **P-JEPA (this paper)** | **JEPA** ([[lecun-2022-openreview]] ([LeCun 2022](../papers/lecun-2022-openreview.md))) | **LeJEPA** ([[balestriero-2025-iclr]] ([Balestriero 2025](../papers/balestriero-2025-iclr.md))) |
|---|---|---|---|
| Focus | RL / model-based control | General vision SSL | General vision SSL |
| Anti-collapse mechanism | Auxiliary regression head | Conceptual (no recipe) | SIGReg (isotropic Gaussian) |
| Theoretical guarantee | Non-equivalent observations cannot collapse (bisimulation-based) | None (position paper) | Provably collapse-free via Cramer-Wold |
| Controls *what* is preserved | Yes (via choice of auxiliary function) | No | No (enforces isotropy globally) |
| Target/EMA encoder | Not required | Proposed conceptually | Not required |
| Setting | Deterministic MDPs | General | Image SSL |

**vs TD-MPC2 (Hansen et al., 2024):** TD-MPC2 uses reward and Q-function prediction as auxiliary tasks alongside a JEPA-based latent dynamics model. P-JEPA's theory provides the first formal explanation of *why* this design works: the reward/Q auxiliary defines the equivalence relation that the encoder preserves.

**vs [[lecun-2022-openreview]] ([JEPA position paper](../papers/lecun-2022-openreview.md)):** LeCun 2022 proposed JEPA conceptually but noted the collapse problem without solving it. P-JEPA demonstrates that auxiliary tasks are a principled lever to prevent collapse and control the encoder's abstraction.

**vs [[balestriero-2025-iclr]] ([LeJEPA](../papers/balestriero-2025-iclr.md)):** LeJEPA prevents collapse by enforcing isotropic Gaussian embeddings globally. P-JEPA takes a complementary approach: rather than constraining the *distribution* of embeddings, it constrains *which observations must be distinguishable*, offering task-specific control over what information is preserved. Notably, Randall Balestriero is an author on both papers.

---

## Strengths

- **First theoretical characterization of JEPA with auxiliary tasks**: Proves exactly which observations must be distinguished, grounding heuristic design choices in bisimulation theory.
- **Actionable design principle**: The auxiliary function directly controls the equivalence relation, enabling practitioners to choose what information the encoder preserves by selecting the appropriate auxiliary.
- **Clean experimental validation**: The counting environment directly tests the theory's predictions (9 clusters for 9 non-bisimilar sets) and confirms them.
- **Knowledge discovery interpretation**: Frames the JEPA + auxiliary system as discovering knowledge triples (encoder, dynamics model, auxiliary predictor) that explain a user-specified phenomenon.

---

## Weaknesses & Limitations

- **Restricted to deterministic MDPs**: Theorem 1 requires deterministic transition dynamics. Stochastic environments (most real-world settings) are not covered.
- **Only a counting environment**: Experiments are limited to a simple 64x64 counting task. No standard RL benchmarks (Atari, DMControl) or large-scale vision experiments are provided.
- **Finite-data version deferred to appendix**: The main theorem assumes perfect training (zero loss); the finite-sample version is in Appendix B.1.
- **No quantitative downstream evaluation**: The paper does not evaluate on standard metrics like reward obtained by a downstream policy, linear probing accuracy, or any task beyond cluster visualization.
- **One-sided guarantee**: The theory proves non-equivalent observations *cannot* collapse, but does *not* require bisimilar observations to merge. Cluster compactness within equivalence classes is not guaranteed.

---

## Key Takeaways

- **Auxiliary tasks are not heuristics -- they are a principled lever for controlling JEPA representations**: The choice of auxiliary function defines the equivalence relation the encoder must respect.
- **The No Unhealthy Representation Collapse theorem provides the first formal guarantee for JEPA with auxiliary tasks**: Non-bisimilar observations must map to distinct representations under perfect training.
- **Joint training of dynamics + auxiliary produces richer representations than either alone**: Latent dynamics loss alone causes complete collapse; reward loss alone gives only coarse separation; their combination yields the full set of theoretically predicted clusters.
- **This paper explains why TD-MPC2-style designs work**: Using reward or Q-functions as auxiliary tasks in JEPA-based world models has theoretical grounding through bisimulation theory.
- **The theory is complementary to distributional approaches like LeJEPA**: P-JEPA controls *which* distinctions are preserved (task-specific), while [[balestriero-2025-iclr]] ([LeJEPA](../papers/balestriero-2025-iclr.md)) controls *how* embeddings are distributed (task-agnostic).

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{yu2025auxiliary,
  title={Why and How Auxiliary Tasks Improve {JEPA} Representations},
  author={Yu, Jiacan and Chen, Siyi and Liu, Mingrui and Horiuchi, Nono and Braverman, Vladimir and Xu, Zicheng and Haramati, Dan and Balestriero, Randall},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={arXiv:2509.12249}
}
```
{% endraw %}

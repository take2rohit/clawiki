---
title: "Hierarchical Planning with Latent World Models"
type: paper
paper_id: P049
authors:
  - "Zhang, Wancong"
  - "Terver, Basile"
  - "Zholus, Artem"
  - "Chitnis, Soham"
  - "Sutaria, Harsh"
  - "Assran, Mido"
  - "Balestriero, Randall"
  - "Bar, Amir"
  - "Bardes, Adrien"
  - "LeCun, Yann"
  - "Ballas, Nicolas"
year: 2026
venue: arXiv
arxiv_id: "2604.03208"
url: "https://arxiv.org/abs/2604.03208"
pdf: "../../raw/zhang-2026-arxiv.pdf"
tags: [world-model, hierarchical-planning, latent-dynamics, robotics, JEPA]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - hafner-2019-icml
  - hafner-2023-arxiv
  - hansen-2022-icml
  - bar-2024-cvpr
  - balestriero-2025-iclr
  - maes-2026-arxiv
cited_by: []
---

# Hierarchical Planning with Latent World Models

> **HWM** is a model-agnostic hierarchical MPC framework that couples two latent world models operating at different temporal resolutions via a shared latent space, enabling zero-shot non-greedy control on real robotic manipulation (70% pick-and-place success on Franka vs. 0% for flat planning) while reducing inference-time planning cost by up to 4x across three diverse world model architectures.

**Authors:** Wancong Zhang\* (FAIR at Meta / NYU), Basile Terver (FAIR), Artem Zholus (Mila), Soham Chitnis (NYU), Harsh Sutaria (NYU), Mido Assran (FAIR), Randall Balestriero (Brown / FAIR), Amir Bar (FAIR), Adrien Bardes (FAIR), Yann LeCun (NYU / FAIR), Nicolas Ballas (FAIR) | **Venue:** arXiv (April 2026) | **arXiv:** [2604.03208](https://arxiv.org/abs/2604.03208)

---

## Problem & Motivation

Planning with learned world models via Model Predictive Control (MPC) has become a powerful paradigm for zero-shot, goal-conditioned embodied control — world models trained on reward-free, task-agnostic offline data can generalize to novel tasks at inference time. However, flat MPC with a single world model struggles with long-horizon tasks for two compounding reasons:

1. **Error accumulation**: Small one-step prediction errors compound during autoregressive rollouts, making long-horizon latent predictions increasingly inaccurate.
2. **Exponential search space**: The space of candidate action sequences grows exponentially with the planning horizon, making optimization intractable for long tasks.

These limitations are especially damaging for **non-greedy tasks** — tasks where the agent must temporarily move away from the goal (e.g., picking up an object before placing it) — because such tasks require genuinely long-horizon reasoning that cannot be approximated by short-horizon planning toward the goal.

A natural solution is hierarchical planning, which has been extensively studied in reinforcement learning (options, skills, hierarchical policies) and hierarchical model-based RL (Director, Puppeteer, THICK). However, existing hierarchical approaches typically require task-specific reward supervision, learned hierarchical policies, hand-engineered state spaces, or known dynamics — limiting their applicability to zero-shot settings with learned world models operating from high-dimensional pixel observations.

---

## Core Idea

HWM introduces temporal hierarchy directly at inference time on pretrained latent world models, without requiring any changes to how the world models are trained. The key insight is that latent world models operating at different temporal resolutions can share the same latent space, enabling a simple coupling mechanism: high-level predictions serve as subgoals for low-level planning via latent-state matching.

Concretely, a **high-level world model** predicts long-horizon transitions conditioned on learned "macro-actions" (compressed sequences of primitive actions), while a **low-level world model** predicts short-horizon transitions conditioned on primitive actions. Both operate in the same encoder's representation space. At planning time, the high-level planner optimizes macro-actions to reach the final goal, producing intermediate latent subgoals. The low-level planner then optimizes primitive actions to reach the first subgoal. This top-down decomposition decouples long-horizon reasoning from fine-grained control, reducing both error accumulation (fewer autoregressive steps at each level) and search complexity (smaller action spaces at each level).

The framework is **model-agnostic** — it applies as a plug-in on top of any latent world model architecture. The paper demonstrates this across three architectures: VJEPA2-AC (JEPA-based robot manipulation), DINO-WM (DINO-based push manipulation), and PLDM (VICReg-based maze navigation).

---

## How It Works

### Overview

Given an initial observation s_1 and a goal observation s_g, HWM proceeds in two stages:

1. **High-level planning**: Encode both into a shared latent space (z_1 = E(s_1), z_g = E(s_g)). Optimize a sequence of H latent macro-actions to minimize the distance between the final predicted latent state and z_g, using the high-level world model. The intermediate predicted latent states become subgoals.
2. **Low-level planning**: Starting from z_1, optimize a sequence of h primitive actions to reach the first latent subgoal, using the low-level world model. Execute the first k actions, then replan in a receding-horizon MPC fashion.

### Low-Level World Model P^(1)

The low-level model predicts short-horizon transitions in latent space conditioned on primitive actions:

```
P^(1)(z_{t+1} | z_t, a_t)
```

This is the standard single-level world model. HWM reuses existing pretrained world models (VJEPA2-AC, DINO-WM, PLDM) without modification, preserving their training recipes and architectures. The model is trained with teacher-forcing and multi-step autoregressive rollout losses following JEPA-style predictive objectives.

### High-Level World Model P^(2)

The high-level model captures long-horizon dynamics by predicting transitions between temporally distant waypoint states, conditioned on latent macro-actions:

```
P^(2)(z_{t+h} | z_t, l_t)
```

where l_t is a latent macro-action encoding the intervening sequence of primitive actions between waypoints. Crucially, P^(2) operates in the **same latent space** as P^(1), sharing the encoder E. This enables direct coupling: high-level predictions are compatible as subgoals for low-level planning without any inverse model or goal-conditioned policy.

**Training**: Given a trajectory (s_1, a_1, ..., a_{T-1}, s_T), N waypoint indices are chosen such that 1 = t_1 < t_2 < ... < t_N. Each high-level transition consists of (s_{t_k}, actions a_{t_k:t_{k+1}}, s_{t_{k+1}}). The action encoder A_psi compresses each action subsequence into a latent macro-action l_{t_k}. The world model P^(2) is causal and trained via teacher-forcing with a latent prediction loss:

```
L_tf = (1/N) * sum_k ||z_hat_{t_{k+1}} - z_{t_{k+1}}||_1
```

Unlike fixed-stride temporal abstraction methods, HWM does not assume a fixed high-level horizon h — each high-level transition can correspond to a variable-length segment of low-level execution. In the Franka experiments, N=3 waypoints are sampled from trajectory segments spanning up to 4 seconds, with the middle waypoint chosen uniformly at random.

### Action Encoder

A transformer-based encoder that maps variable-length sequences of low-level actions between waypoints into fixed-dimensional latent macro-actions. The CLS token output is projected to the latent action space. The latent action dimension is a critical design parameter — too low and the high-level model cannot express non-greedy trajectories; too high and the subgoals become unreachable for the low-level planner. Empirically, dimension 4 strikes the best balance for Franka tasks.

### Top-Down Hierarchical Planning

**High-level energy function** (goal-conditioned):

```
E_2(l_hat_{1:H}; z_1, z_g) = ||z_g - P^(2)(l_hat_{1:H}; z_1)||_1
```

Optimized via CEM (Cross-Entropy Method) with 3000 samples over H latent macro-actions of dimension 4, prediction horizon 2. This yields intermediate latent subgoals z_tilde_i = P^(2)(l*_{1:i}; z_1) for i = 1,...,H.

**Low-level energy function** (subgoal-conditioned):

```
E_1(a_hat_{1:h}; z_1, z_tilde_1) = ||z_tilde_1 - P^(1)(a_hat_{1:h}; z_1)||_1
```

Optimized via CEM with 800 samples over primitive actions, prediction horizon 2. The agent replans every k=1 step in standard MPC fashion. Planning rollouts are parallelized across GPUs.

### Data

All world models are trained on **reward-free, task-agnostic offline data**:
- **Franka (VJEPA2-AC)**: ~130 hours of unlabeled real-robot manipulation from DROID and RoboSet. Observations include RGB images and end-effector proprioception; actions are end-effector delta poses.
- **Push-T (DINO-WM)**: Offline Push-T dataset from DINO-WM.
- **Diverse Maze (PLDM)**: MuJoCo PointMaze with 10x10 grid layouts, trained on 25 layouts and evaluated on 20 held-out layouts.

---

## Results

### Franka Arm: Real-World Robotic Manipulation (Table 1)

Pick-and-place with cup/box objects and drawer opening/closing on a 7-DoF Franka Emika Panda arm with a Robotiq gripper. These are **non-greedy tasks** — pick-and-place requires the arm to first move to the object (away from the goal location), grasp it, then transport it.

| Method | P&P Cup (oracle sub) | P&P Box (oracle sub) | P&P Cup | P&P Box | Drawer |
|---|---|---|---|---|---|
| Octo (VLA) | 20% | 10% | 0% | 0% | 43% |
| pi_0-FAST-DROID (VLA) | -- | -- | 52% | 18% | -- |
| pi_0.5-DROID (VLA) | -- | -- | 68% | 36% | -- |
| VJEPA2-AC (flat) | 80% | 80% | **0%** | **0%** | 30% |
| **VJEPA2-AC + HWM** | **80%** | **80%** | **70%** | **60%** | **70%** |

The flat VJEPA2-AC planner achieves 0% on pick-and-place without oracle subgoals — it greedily moves the gripper toward the goal location without first picking up the object. HWM recovers 70% success on cups and 60% on boxes by automatically generating the "grasp first" subgoal via high-level planning. Notably, HWM outperforms strong vision-language-action models (Octo, pi_0-FAST-DROID, pi_0.5-DROID) trained on ~77x more robotic interaction data. Failure cases primarily arise from perceptual imprecision (depth estimation errors) and occasional near-miss executions due to lack of joint optimization between hierarchical levels.

### Push-T: Long-Horizon Manipulation (Table 2)

Pushing a T-shaped object to match a goal configuration, with start-goal distances d from 25 to 75 timesteps — exceeding the d=25 used in original DINO-WM evaluation.

| Method | d=25 | d=50 | d=75 |
|---|---|---|---|
| GCIQL | 40% | 25% | 7.5% |
| HIQL | 55% | 30% | 20% |
| HILP | 25% | 13% | 0% |
| DINO-WM (flat) | 84% | 55% | 17% |
| **DINO-WM + HWM** | **89%** | **78%** | **61%** |

Hierarchical planning consistently outperforms flat DINO-WM as task difficulty increases. The improvement is most dramatic at the longest horizon (d=75): 61% vs. 17%, a +44 percentage point gain. Policy-based baselines (GCIQL, HIQL, HILP) degrade sharply at longer horizons, suggesting limited robustness to long-horizon generalization. Hierarchical planning also achieves 3x less compute per planning step compared to the flat planner at equivalent success rates (Figure 5).

### Diverse Maze: OOD Navigation (Table 3)

MuJoCo PointMaze with top-down RGB renderings, evaluated on 20 held-out maze layouts unseen during training. Start-goal pairs at varying grid distances D.

| Method | D in [5,8] | D in [9,12] | D in [13,16] |
|---|---|---|---|
| GCIQL | 85% | 40% | 33% |
| HIQL | 88% | 73% | 48% |
| HILP | 48% | 20% | 10% |
| PLDM (flat) | 100% | 63% | 44% |
| **PLDM + HWM** | **100%** | **95%** | **83%** |

The performance gap between HWM and flat PLDM widens with task horizon: from 0% at short distances to +39% at the longest distances. HWM achieves higher success with 4x less compute than the flat planner. Both PLDM variants outperform all goal-conditioned and zero-shot RL baselines on OOD mazes, highlighting the robustness of model-based planning to distributional shift in environment geometry.

### Compute Efficiency (Figure 5)

Across both Push-T and Diverse Maze, hierarchical planning matches or exceeds flat planner success rates while requiring approximately 3x less planning time per step. The high-level planner operates over a much smaller search space (low-dimensional macro-actions, short horizon), and the low-level planner only needs to plan to a nearby subgoal rather than the distant final goal.

---

## Analysis of Hierarchical Planning (Section 4)

### Latent Actions vs. Delta-Pose Actions (Table 4)

Learned latent macro-actions outperform raw delta-pose aggregation for high-level planning (cosine similarity to expert behavior: 0.88 vs. 0.80). Delta-pose representations collapse non-greedy action sequences (e.g., "move up then down") into a single displacement, discarding essential information about the intermediate trajectory. Learned latent actions can compactly encode the full action structure.

### High-Level vs. Low-Level Prediction Accuracy (Figure 6)

For prediction horizons up to ~1 second, the low-level model is more accurate (fewer autoregressive steps accumulate less error). Beyond ~1.5 seconds, the high-level model's single-step predictions achieve lower L1 error than the low-level model's multi-step autoregressive rollouts. This validates the hierarchical strategy: high-level guidance for long horizons, low-level precision for short horizons.

### Latent Action Dimensionality (Figure 7)

A critical design trade-off: latent action dimension must be high enough for the high-level planner to produce valid plans (>= 4 dimensions), but low enough that the resulting subgoals are reachable by the low-level planner via greedy execution. Higher dimensions increase plan expressivity but reduce action cosine similarity to expert behavior. Dimension 4 is optimal for the Franka tasks. Notably, reconstruction fidelity is not tightly correlated with planning performance — lower-dimensional latent spaces yield noisier predictions but still preserve coarse semantic structure (contact events, motion direction) sufficient for hierarchical planning.

---

## Comparison to Prior Work

| | **HWM** | Director (Hafner 2022) | THICK (Gumbsch 2023) | IQL-TD-MPC (Chitnis 2024) | CAVIN (Fang 2019) |
|---|---|---|---|---|---|
| Zero-shot | Yes | No | Partial | No | Yes |
| Observation | Pixels -> learned latent | Pixels -> learned latent | Pixels -> learned latent | State -> learned latent | Structured state |
| Training | Learned models (reward-free) | Model-based RL | Learned models + RL | Offline RL + MPC | Learned models |
| Hierarchical interface | **Subgoal matching (latent)** | Goal-conditioned policy | Subgoal (latent) + reward | Goal-conditioned policy | Subgoal matching (state) |

**vs [[lecun-2022-openreview]] ([JEPA / H-JEPA](lecun-2022-openreview.md)):** LeCun's 2022 position paper proposed hierarchical JEPA (H-JEPA) as a conceptual architecture for intelligent agents, with world models operating at multiple levels of abstraction and temporal scales. HWM is a concrete realization of that vision: it instantiates a two-level hierarchy of latent world models operating in a shared JEPA-derived representation space, with high-level planning generating subgoals for low-level execution. The key difference from the H-JEPA proposal is that HWM introduces hierarchy at inference time on separately trained models, rather than training a monolithic hierarchical architecture end-to-end.

**vs [[maes-2026-arxiv]] ([LeWorldModel](maes-2026-arxiv.md)):** LeWM and HWM come from the same LeCun/Balestriero research group at Meta/NYU. LeWM solves the end-to-end JEPA world model training problem (via SIGReg), producing a single-level model that achieves 48x faster planning than DINO-WM. HWM addresses the complementary problem of long-horizon planning — LeWM's own paper explicitly identifies "hierarchical world modeling" as future work needed to overcome compounding prediction errors. HWM can be applied on top of any world model, including LeWM.

**vs [[balestriero-2025-iclr]] ([LeJEPA](balestriero-2025-iclr.md)):** LeJEPA provides the theoretical foundation (SIGReg, isotropic Gaussian optimality) that underlies VJEPA2-AC, one of the three backbone world models used in HWM. The connection is indirect: LeJEPA provides the representation learning machinery; HWM provides the hierarchical planning framework that operates on top of those representations.

**vs [[hafner-2019-icml]] ([PlaNet](hafner-2019-icml.md)):** PlaNet pioneered latent dynamics planning with CEM, using a single RSSM model with flat MPC. HWM extends this paradigm by introducing a two-level hierarchy over learned latent models, addressing PlaNet's fundamental limitation of compounding errors over long horizons. Both use CEM as the planner, but HWM decomposes the optimization into two smaller, more tractable problems. PlaNet requires online data collection with reward signals; HWM operates zero-shot from offline data.

**vs [[hafner-2023-arxiv]] ([DreamerV3](hafner-2023-arxiv.md)):** DreamerV3 learns a single-level RSSM world model and trains an actor-critic policy via imagination. Director (Hafner et al., 2022, cited in HWM) extends this to hierarchical planning with a learned "manager" that proposes latent goals for a worker policy. HWM differs in that it does not learn hierarchical policies at all — hierarchy is introduced purely at planning time via CEM optimization at two temporal scales, enabling zero-shot generalization without task-specific reward training.

**vs [[hansen-2022-icml]] ([TD-MPC](hansen-2022-icml.md)):** TD-MPC combines learned latent dynamics with temporal-difference learning for model-predictive control. It operates at a single temporal scale and requires reward supervision. TD-MPC2 (Hansen et al., 2023) scales this to 80+ tasks but remains single-level and reward-dependent. HWM introduces temporal hierarchy and operates reward-free, enabling zero-shot transfer.

**vs [[bar-2024-cvpr]] ([Navigation World Models](bar-2024-cvpr.md)):** NWM (by co-author Amir Bar) learns a diffusion-based world model for visual navigation planning with CEM. NWM operates in pixel space and uses a single temporal scale. HWM operates in latent space at multiple temporal scales. The approaches are complementary — NWM could potentially serve as a backbone for HWM in navigation domains, though this is not explored in the paper.

---

## Strengths

- **Model-agnostic plug-in**: Demonstrated on three diverse world model architectures (VJEPA2-AC, DINO-WM, PLDM) across three different domains, consistently improving performance without modifying the underlying models.
- **Unlocks non-greedy zero-shot control**: The 0% to 70% jump on real-robot pick-and-place is a qualitative capability gain, not just an incremental improvement — flat world model planners fundamentally cannot solve these tasks.
- **Compute-efficient**: Higher success rates at 3-4x lower planning cost, because each level of the hierarchy operates over a smaller search space.
- **Outperforms VLAs with less data**: HWM with VJEPA2-AC outperforms vision-language-action models (Octo, pi_0 variants) trained on ~77x more robotic interaction data, demonstrating the power of zero-shot world model planning over learned policies.
- **Careful analysis**: Ablations on latent action dimensionality, prediction horizon crossover, and subgoal quality provide genuine insight into when and why hierarchy helps.

---

## Weaknesses & Limitations

- **No joint optimization across levels**: High-level and low-level planning are strictly sequential (top-down); there is no feedback from the low-level planner to the high-level planner. If a high-level subgoal is unreachable by low-level execution, the system fails silently.
- **Strictly top-down decomposition**: The authors note that improved hierarchical planning algorithms allowing feedback and interaction across levels could address this limitation. The current design cannot recover from poor subgoal proposals.
- **Performance still degrades with horizon**: Despite substantial gains, all methods including HWM see declining success at the longest horizons (61% at d=75 on Push-T, 83% at D in [13,16] on maze). Long-horizon manipulation remains an open problem.
- **Fixed two-level hierarchy**: The paper only explores a two-level hierarchy. Extending to deeper hierarchies (as in the original H-JEPA proposal) is left for future work.
- **Latent action dimension is task-specific**: The optimal dimension (4 for Franka, different for other domains) must be tuned per task domain, adding a meaningful hyperparameter.
- **Real-robot failures from perception**: Failure cases on Franka arise from depth estimation errors and perceptual imprecision rather than planning failures, suggesting the bottleneck may shift to perception at longer horizons.

---

## Key Takeaways

- **Hierarchy is the key to non-greedy zero-shot control**: A flat world model planner gets 0% on real-robot pick-and-place; adding hierarchical planning over the same model achieves 70% — validating that temporal abstraction, not model capacity, is the missing ingredient for long-horizon tasks.
- **Inference-time hierarchy suffices**: Rather than training a monolithic hierarchical architecture (as proposed in H-JEPA / [[lecun-2022-openreview]]), HWM shows that hierarchy can be introduced purely at planning time on separately pretrained models, making it a modular upgrade path for any existing world model.
- **Latent macro-actions beat raw action aggregation**: Learned action encoders that compress primitive action sequences into latent macro-actions capture non-greedy trajectory structure that delta-pose aggregation discards, improving high-level planning quality (cosine similarity 0.88 vs. 0.80).
- **High-level models become more accurate than low-level models at longer horizons**: Beyond ~1.5 seconds, a single high-level prediction has lower error than multi-step low-level autoregressive rollouts (Figure 6), empirically validating the crossover point where hierarchy pays off.
- **Model-agnostic gains across architectures**: Consistent improvements on VJEPA2-AC (+70 pp on pick-and-place), DINO-WM (+44 pp on Push-T at d=75), and PLDM (+39 pp on maze at D in [13,16]) demonstrate that the framework generalizes beyond any single world model architecture.

---

## BibTeX

{% raw %}
```bibtex
@article{zhang2026hierarchical,
  title={Hierarchical Planning with Latent World Models},
  author={Zhang, Wancong and Terver, Basile and Zholus, Artem and Chitnis, Soham and Sutaria, Harsh and Assran, Mido and Balestriero, Randall and Bar, Amir and Bardes, Adrien and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2604.03208},
  year={2026}
}
```
{% endraw %}

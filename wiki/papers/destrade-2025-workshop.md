---
title: "Value-guided action planning with JEPA world models"
type: paper
paper_id: P044
authors:
  - "Destrade, Matthieu"
  - "Bounou, Oumayma"
  - "Le Lidec, Quentin"
  - "Ponce, Jean"
  - "LeCun, Yann"
year: 2025
venue: World Modeling Workshop 2026
arxiv_id: "2601.00844"
url: "https://arxiv.org/abs/2601.00844"
pdf: "../../raw/destrade-2025-workshop.pdf"
tags: [JEPA, world-model, planning, value-function, goal-conditioned, iql, mpc, quasi-distance]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - hafner-2019-icml
cited_by: []
---

# Value-guided action planning with JEPA world models

> **Value-guided JEPA** shapes the representation space of a JEPA world model so that the negative goal-conditioned value function is approximated by the Euclidean distance (or quasi-distance) between state embeddings, enabling significantly improved planning with MPC compared to standard JEPA training on goal-reaching control tasks.

**Authors:** Matthieu Destrade (Ecole Polytechnique / ENS Paris)\*, Oumayma Bounou (NYU), Quentin Le Lidec (NYU), Jean Ponce (ENS Paris / NYU), Yann LeCun (NYU) | **Venue:** World Modeling Workshop 2026 (poster) | **arXiv:** [2601.00844](https://arxiv.org/abs/2601.00844)

---

## Problem & Motivation

JEPA world models ([[lecun-2022-openreview]]) learn predictive representations by training an encoder and an action-conditioned predictor to minimize prediction error in latent space. Recent works (Sobal et al. 2025; Zhou et al. 2025) have applied JEPA models to action planning using Model Predictive Control (MPC), where the planning cost is the distance between the predicted latent state and the goal's latent representation. However, this planning approach suffers from a critical limitation: the MPC cost landscape defined by Euclidean distance in the standard JEPA representation space has numerous **local minima**, making optimization difficult and planning unreliable.

The core issue is that standard JEPA representations are learned purely from a prediction objective -- they are shaped to make next-state prediction accurate, not to provide a smooth cost landscape for planning. There is no guarantee that minimizing distance in representation space corresponds to moving toward the goal in any dynamically meaningful sense.

---

## Core Idea

The authors propose to shape the JEPA encoder's representation space so that the Euclidean distance (or a quasi-distance) between embedded states approximates the negative goal-conditioned value function associated with a reaching cost. Specifically, for states s and goal g, they learn an encoder E_theta such that:

```
V_theta(s, g) = -||E_theta(s) - E_theta(g)||_2
```

approximates V*, the optimal goal-conditioned value function for the cost C(s, a, g) = 1_{s != g} (a step-penalty reaching cost). This is achieved by training the state encoder using an Implicit Q-Learning (IQL) loss from offline trajectory data, without requiring reward labels or online interaction.

The key insight is that if the distance in representation space reflects the true reaching cost, then minimizing MPC planning cost naturally drives the system toward the goal -- the cost landscape becomes inherently meaningful for planning, potentially eliminating the local minima problem that plagues standard JEPA representations.

---

## How It Works

### Overview

The method modifies JEPA world model training by adding a value-function loss to the state encoder, shaping its representations for planning. The system retains the standard JEPA components (state encoder, action encoder, predictor) but supplements them with an IQL-inspired training objective.

### JEPA Model Architecture

The standard JEPA model consists of:
- **State encoder** E_theta: maps observations to latent representations
- **Action encoder**: maps actions to latent action representations
- **Predictor**: maps (state embedding, action embedding) to predicted next-state embedding

The model is trained on sequences of (state, action) pairs with a prediction loss L_pred that minimizes the error between predicted and actual next-state embeddings. Standard approaches use VCReg loss or an EMA scheme to prevent collapse.

### Value Function Loss (IQL for JEPA)

For all state-goal pairs (s, g), the value function is defined as:

```
V_theta(s, g) = -||E_theta(s) - E_theta(g)||_2
```

The encoder parameters theta are trained to make V_theta approximate V*, the optimal goal-conditioned value function for the reaching cost C : (s, a, g) -> 1_{s != g}. This is done by minimizing the mean IQL loss:

```
L_VF((s_t), (g_n)) = sum_{n=0}^{N} sum_{t=0}^{T-1} L^2_tau(-1_{s_t != g_n} + gamma * V_bar_theta(s_{t+1}, g_n) - V_theta(s_t, g_n))
```

where:
- The overbar denotes a stop-gradient
- tau, gamma are in (0, 1) and close to 1
- L^2_tau(x) = |tau - 1_{x<0}| * x^2 performs expectile regression
- gamma is the discount factor of the value function
- Goals are sampled from both the last state of training trajectories and random goals from training batches

### Quasi-distance Extension

Because the goal-conditioned value function is not symmetric in general (reaching from A to B may differ from B to A), the authors also explore replacing the Euclidean distance with a **quasi-distance** following Wang et al. (2023) and Wang & Isola (2022). The quasi-distance formulation uses interval quasimetric embeddings that can represent asymmetric reachability costs, enhancing the expressiveness of the learned representation.

### Two Training Approaches

1. **Separate ("Sep")**: Train the state encoder first with L_VF alone, then freeze it and train the action encoder and predictor with L_pred. This keeps the value-shaped representations untouched by the prediction objective.

2. **Joint**: Train all networks together using L_VF + L_pred as the combined objective. This allows the prediction task to also influence the representation space.

### Training Approaches Evaluated (Table 1)

| Name | State encoder loss | Sep | | Name | State encoder loss | Sep |
|---|---|---|---|---|---|---|
| Contrastive | L_contrastive | Yes | | VF_pred | L_VF | No |
| Regressive | L_regressive + L_VCReg | Yes | | VF_quasi | L_VF + quasi-distance | Yes |
| pred_VCReg | L_VCReg | No | | VF_quasi_pred | L_VF + quasi-distance | No |
| pred_EMA | EMA procedure | No | | VF_VCReg | L_VF + L_VCReg | Yes |
| VF | L_VF | Yes | | VF_VCReg_pred | L_VF + L_VCReg | No |

Baselines include contrastive learning (successive states as positives, random pairs as negatives), regressive approaches (enforcing unit distance between successive states), and standard JEPA training with VCReg or EMA.

### Planning (Inference)

Planning uses MPC with MPPI (Model Predictive Path Integral) optimization:
1. Encode initial observation and goal observation into latent space
2. Sample candidate action sequences
3. Roll out the predictor autoregressively for each candidate
4. Evaluate planning cost as distance between final predicted state and goal in representation space
5. Iteratively refine the action distribution toward lower-cost sequences
6. Execute the first action and replan (receding horizon)

Because the representation space is shaped by the value function, the planning cost ||z_predicted - z_goal||_2 now approximates the negative value function, providing a meaningful gradient signal that naturally drives the model toward the goal.

### Experimental Setup

- **Offline, reward-free**: Models are trained on random trajectories with no reward labels
- **Two environments**: Wall (2D navigation with a wall and door) and Maze (random maze navigation based on MuJoCo PointMaze)
- **Observations**: 64x64 images (2-channel for Wall, 3-channel for Maze)
- **Model architecture**: Flat representations of size 512; CNN state encoder (2.2M params) with convolutions and residual connections; MLP predictor (1.3M params); identity action encoder
- **Training**: Adam optimizer, lr=0.0028, cosine schedule, trajectory segments of length 16
- **Planning**: MPPI with 2000 perturbations (wall) or 500 (maze), planning horizons of 96-200 steps (wall) or 100 steps (maze)

### Hyperparameters

For VF-based approaches: gamma = 0.98, tau = 0.80. For VF_quasi-based approaches: gamma = 0.93, tau = 0.60. These were optimized on a held-out Wall dataset.

---

## Results

### Planning Accuracy (Table 2, success rate on goal-reaching)

| Method | WS (Wall Small) | WB (Wall Big) | Maze |
|---|---|---|---|
| Contrastive | 0.49 | 0.59 | 0.50 |
| Regressive | 0.54 | 0.57 | 0.46 |
| pred_VCReg | 0.55 | 0.89 | 0.54 |
| pred_EMA | 0.46 | 0.43 | 0.04 |
| VF | 0.63 | 0.94 | 0.49 |
| VF_pred | 0.55 | 0.75 | 0.49 |
| **VF_quasi** | **0.71** | **0.96** | **0.63** |
| VF_quasi_pred | 0.61 | 0.85 | 0.43 |
| VF_VCReg | 0.49 | 0.75 | 0.39 |
| VF_VCReg_pred | 0.47 | 0.67 | 0.39 |

**Key findings:**

1. **VF_quasi is the best method across all three environments** (0.71/0.96/0.63), substantially outperforming all baselines. The quasi-distance formulation consistently improves over the symmetric Euclidean distance version (VF).

2. **IQL-inspired approaches outperform intuitive and prediction-based baselines**: The value function loss provides better guidance for planning than contrastive or regressive shaping of the representation space.

3. **Separate training ("Sep") outperforms joint training**: VF_quasi (Sep) achieves 0.71/0.96/0.63 vs. VF_quasi_pred (joint) at 0.61/0.85/0.43. Training the state encoder with the value function loss alone, then training the predictor separately, consistently works better than combining both losses.

4. **VCReg hurts when combined with IQL loss**: VF_VCReg (0.49/0.75/0.39) performs substantially worse than VF alone (0.63/0.94/0.49), indicating that VCReg's diversity regularization interferes with the value-shaped geometry.

5. **pred_EMA collapses on Maze**: The EMA-based training approach achieves only 0.04 success on the maze environment, demonstrating the fragility of standard JEPA training for planning.

6. **WB results are better than WS**: Larger action norms (WB dataset) produce trajectories that explore more of the environment, leading to better coverage for value function learning.

---

## Discussion

### Locality of Training

The value functions learned are imperfect, particularly for distant state pairs. Two factors contribute:
- **Sparse sampling**: Distant triplets (start, following state, goal) are rarely sampled together during training, making long-range value estimation unreliable.
- **Vanishing gradients**: The gradient of the discounted value function becomes small when the state is far from the goal, producing low signal-to-noise ratios.

The authors suggest that a **hierarchy of representation spaces** -- where higher levels model longer-range transitions using coarser trajectories -- could better capture distant relationships. This connects directly to LeCun's H-JEPA proposal in [[lecun-2022-openreview]].

### Influence of the Dataset

The IQL loss theoretically depends only on the support of the training policy (states visited), not the policy's quality, when tau approaches 1. In practice, highly suboptimal trajectories can create misleading training signals where nearby states in observation space are far apart in trajectory distance. Expert trajectories could help but sacrifice diversity and exploration coverage. The authors emphasize that training states must span the entire state space for the IQL loss to work well.

### Separate Predictive and Planning Representations

The authors tested using two separate representation spaces -- one for prediction (trained with pred_VCReg) and one for planning cost (trained with VF) -- but this did not improve results (0.60 accuracy on WS), suggesting that a unified value-shaped representation is preferable to a two-space approach.

---

## Comparison to Prior Work

| | **This paper (VF_quasi)** | **Sobal et al. 2025 (JEPA planning)** | **[[maes-2026-arxiv]] (LeWM)** | **[[hansen-2022-icml]] (TD-MPC)** | **[[hafner-2019-icml]] (PlaNet)** |
|---|---|---|---|---|---|
| Representation shaping | IQL value function | VCReg / EMA | SIGReg | TD-learned (reward-centric) | Reconstruction |
| Planning cost | -V(s,g) approx ||z_s - z_g|| | ||z_s - z_g|| | ||z_s - z_g|| | gamma^H * Q + sum R | CEM on reward |
| Reward required | No | No | No | Yes | Yes |
| Goal-conditioned | Yes | Yes | Yes | No (task reward) | No (task reward) |
| Planning method | MPPI | MPC | CEM | MPPI | CEM |
| Collapse prevention | IQL geometry | VCReg / EMA | SIGReg | Task reward signal | Reconstruction |

**vs [[lecun-2022-openreview]] (JEPA proposal):** LeCun's original JEPA proposal envisions a cost module that drives planning but does not specify how the representation space itself should be shaped for planning. This paper provides a concrete mechanism: use the IQL loss to make Euclidean distance in representation space reflect the goal-conditioned value function, directly implementing the idea that the cost module should guide planning.

**vs [[maes-2026-arxiv]] (LeWorldModel):** LeWM uses CEM-based MPC planning with cost = ||z_H - z_g||^2 in a SIGReg-regularized JEPA space. The planning cost is purely geometric -- it assumes Euclidean distance is a good proxy for reachability. This paper argues that assumption is flawed and proposes to explicitly shape the geometry so distance equals (negative) value. The quasi-distance extension is particularly relevant since it can capture asymmetric reachability that symmetric Euclidean distance cannot. LeWM's SIGReg ensures isotropic Gaussian embeddings (good for downstream probing), but there is no guarantee this geometry is optimal for planning. Combining SIGReg (for anti-collapse) with value-function shaping (for planning-friendly geometry) is an open direction.

**vs [[balestriero-2025-iclr]] (LeJEPA / SIGReg):** LeJEPA proves embeddings should be isotropic Gaussian for optimal downstream linear/nonlinear probing. This paper targets a different optimality criterion: embeddings should have distances proportional to reaching costs for optimal planning. These objectives may conflict -- the isotropic Gaussian prior distributes mass uniformly across dimensions, while value-function shaping concentrates structure along dynamically meaningful directions. The finding that VCReg hurts when combined with IQL loss (VF_VCReg << VF) supports this tension.

**vs [[hansen-2022-icml]] (TD-MPC):** TD-MPC also uses a value function to guide MPC planning, but in a fundamentally different way: the value function Q_theta(z, a) is used as a terminal cost added to cumulative reward predictions. The representation is shaped by reward/value gradients (task-oriented latent dynamics). This paper instead makes the representation geometry itself encode the value function via Euclidean distance -- no separate Q-network is needed at planning time. TD-MPC requires reward signals; this paper is reward-free and goal-conditioned.

**vs [[hafner-2019-icml]] (PlaNet):** PlaNet learns latent dynamics via reconstruction and plans with CEM using reward predictions. This paper eliminates both reconstruction and reward, instead using value-shaped embeddings for goal-conditioned planning in a JEPA framework.

**vs Park et al. (2024b) (Hilbert representations):** Park et al. learn structured representations where negative Euclidean distance approximates a goal-conditioned value function, but use them for policy execution. This paper adapts the same idea specifically for JEPA world models and MPC planning procedures.

---

## Strengths

- **Principled approach to a real problem**: Standard JEPA planning suffers from local minima in the MPC cost landscape. Shaping the representation space via a value function directly addresses this root cause rather than adding heuristic fixes.
- **Reward-free and offline**: The IQL loss requires only unlabeled trajectories -- no reward signals, no online interaction -- matching the JEPA philosophy of learning from passive observation.
- **Quasi-distance consistently helps**: The asymmetric quasi-distance formulation outperforms symmetric Euclidean distance across all environments, suggesting that real-world reachability is fundamentally asymmetric and representations should reflect this.
- **Clean experimental design**: Systematic comparison of 10 training approaches across 3 environments isolates the contribution of value-function shaping vs. other representation learning objectives.
- **Separate training is better**: The finding that training the encoder with the value loss alone (then training the predictor separately) outperforms joint training is practically useful and methodologically informative.

---

## Weaknesses & Limitations

- **Simple environments only**: Wall (2D navigation) and Maze environments are far from the complexity of real-world control tasks. The approach has not been tested on image-rich environments, manipulation tasks, or continuous-action-space benchmarks like DMControl.
- **Locality problem unresolved**: The value functions are inaccurate for distant state pairs due to sparse sampling and vanishing gradients. The authors acknowledge this but offer only the suggestion of hierarchical representations as future work.
- **Moderate absolute performance**: Even the best method (VF_quasi) achieves 0.71/0.96/0.63 success rates, leaving substantial room for improvement, especially on WS and Maze.
- **Separate training is a limitation in disguise**: While separate training works best, it prevents the predictor from benefiting from value-function-shaped gradients and requires a two-stage pipeline.
- **No comparison to learned reward/cost models**: The paper compares only representation-shaping methods. A comparison to planning with a learned reward predictor (as in TD-MPC or Dreamer) would clarify when value-shaped geometry outperforms explicit reward prediction.
- **Hyperparameter sensitivity**: gamma and tau require careful tuning (different optimal values for VF vs. VF_quasi), and the hyperparameter sweep in the appendix shows sensitivity to these choices.
- **No analysis of representation geometry**: The paper lacks visualization or analysis of how the learned representations differ structurally from standard JEPA representations -- such analysis would strengthen the claim that value-function shaping eliminates local minima.

---

## Key Takeaways

- **Shaping JEPA representation space with value functions significantly improves planning**: The best value-function method (VF_quasi, 0.71/0.96/0.63) substantially outperforms the best standard JEPA baseline (pred_VCReg, 0.55/0.89/0.54), validating the hypothesis that planning-friendly geometry matters.
- **Quasi-distance > Euclidean distance for value-shaped representations**: Asymmetric quasi-distance embeddings consistently outperform symmetric Euclidean distance, even when the theoretical value function is symmetric. This suggests quasi-distance enhances training expressiveness beyond its theoretical motivation.
- **VCReg and value-function shaping are in tension**: Combining VCReg with IQL loss degrades performance, implying that diversity-promoting regularizers can interfere with value-guided geometry. This is a cautionary finding for works like [[maes-2026-arxiv]] that use SIGReg with distance-based planning costs.
- **Separate encoder training is crucial**: Training the state encoder with the value loss alone, then training the predictor on top, outperforms joint training -- suggesting that the prediction objective corrupts value-function geometry when applied simultaneously.
- **Hierarchical representations are the next step**: The locality problem (inaccurate long-range value estimates) points toward hierarchical world models operating at multiple temporal scales, aligning with LeCun's H-JEPA vision in [[lecun-2022-openreview]].
- **Direct relevance to LeWorldModel**: LeWM's CEM planning uses ||z_H - z_g||^2 as cost in a SIGReg-regularized space. This paper's results suggest that explicitly shaping that space with a value function -- rather than relying on the emergent geometry of SIGReg -- could yield substantial planning improvements for LeWM-class models.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{destrade2025valueguided,
  title={Value-guided action planning with {JEPA} world models},
  author={Destrade, Matthieu and Bounou, Oumayma and Le Lidec, Quentin and Ponce, Jean and LeCun, Yann},
  booktitle={World Modeling Workshop},
  year={2025},
  note={arXiv:2601.00844}
}
```
{% endraw %}

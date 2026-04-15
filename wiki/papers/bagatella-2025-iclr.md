---
title: "TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning"
type: paper
paper_id: P023
authors:
  - "Bagatella, Marco"
  - "Pirotta, Matteo"
  - "Touati, Ahmed"
  - "Lazaric, Alessandro"
  - "Tirinzoni, Andrea"
year: 2025
venue: ICLR 2026
arxiv_id: "2510.00739"
url: "https://arxiv.org/abs/2510.00739"
pdf: "../../raw/bagatella-2025-iclr.pdf"
tags: [JEPA, zero-shot-RL, unsupervised-RL, latent-prediction, world-model, successor-features, temporal-difference, representation-learning, off-policy]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - hansen-2022-icml
  - hansen-2024-iclr
  - hafner-2023-arxiv
cited_by: []
---

# TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning

> **TD-JEPA** introduces a temporal-difference latent-predictive loss that learns state encoders, task encoders, policy-conditioned multi-step predictors, and parameterized policies entirely from offline, reward-free transitions -- enabling zero-shot optimization of any downstream reward function in latent space. The method matches or outperforms state-of-the-art zero-shot baselines across 65 tasks in ExoRL and OGBench, particularly excelling in the challenging pixel-based setting.

**Authors:** Marco Bagatella (FAIR at Meta / ETH Zurich / Max Planck Institute for Intelligent Systems, Tubingen), Matteo Pirotta (FAIR at Meta), Ahmed Touati (FAIR at Meta), Alessandro Lazaric (FAIR at Meta), Andrea Tirinzoni (FAIR at Meta) | **Venue:** ICLR 2026 | **arXiv:** [2510.00739](https://arxiv.org/abs/2510.00739)

**Correspondence:** tirinzoni@meta.com | *Work done at Meta*

---

## Problem & Motivation

Learning effective state representations that support efficient value estimation and policy optimization across tasks is a core challenge in reinforcement learning. Latent-predictive (self-predictive) representation learning -- an instance of the [[lecun-2022-openreview]] ([JEPA, LeCun 2022](../papers/lecun-2022-openreview.md)) paradigm -- jointly learns a state encoder and a predictor such that the predictor maps the representation of the current state to the representation of a future state. While several RL methods have leveraged latent prediction as an auxiliary loss for sample efficiency or exploration, existing approaches are typically limited to: (1) single-task learning, (2) one-step prediction, (3) on-policy trajectory data, or (4) single-policy training.

These limitations prevent latent-predictive methods from being used as the *core* objective in unsupervised RL, where the goal is to pre-train representations from reward-free data that can later be used for zero-shot policy optimization on any downstream task. The paper asks: can temporal-difference (TD) learning enable latent-predictive representations that are predictive of *long-term, multi-policy dynamics* from *offline, reward-free transitions*, and can these representations directly support zero-shot RL?

---

## Core Idea

TD-JEPA shows that temporal-difference learning enables latent-predictive representations that capture the long-term dynamics of multiple policies from offline data. The key insight is that a policy-conditioned multi-step predictor, trained via a TD-style latent-predictive loss, approximates the *successor features* of each policy in the learned latent space. Since successor features enable Q-value computation for any linear reward, the predictor itself becomes the mechanism for zero-shot policy optimization -- latent prediction is not an auxiliary loss but the core objective from which encoders, predictors, and zero-shot policies are all derived.

TD-JEPA trains four components end-to-end: (1) a state encoder that embeds observations for dynamics modeling, (2) a task encoder that defines the space of representable reward functions, (3) a policy-conditioned multi-step predictor, and (4) a set of parameterized latent policies. At test time, given a new reward function, the optimal policy parameter is computed via linear regression on a small labeled dataset, and the corresponding policy is returned -- entirely in latent space, with no further environment interaction.

---

## How It Works

### Background: Successor Features and Zero-Shot RL

In a reward-free MDP, executing a policy induces a *successor measure* -- the discounted distribution over future states. Given a task encoder (state features) and the successor measure, the Q-value for any linear reward decomposes as the inner product of *successor features* and a reward weight vector. Most zero-shot unsupervised RL methods (FB, HILP, RLDP) learn successor features for a set of parameterized policies and retrieve the optimal policy at test time by projecting the reward function onto the task encoder's span.

### Multi-Step Policy-Conditioned Latent Prediction (MC-JEPA)

TD-JEPA begins from a Monte Carlo formulation. Given a family of policies parameterized by latent z, the MC-JEPA loss trains a state encoder and a policy-dependent predictor to minimize:

**L_MC-JEPA** = E [ || T(phi(s), a, z) - stop-grad(phi(s+)) ||^2 ]

where s+ is sampled from the successor measure of policy pi_z. **Proposition 1** shows that this loss is equivalent (up to a constant) to minimizing the distance between the predictor output and the successor features of phi. This crucially connects multi-step latent prediction with value estimation across multiple policies.

### Temporal-Difference Formulation (TD-JEPA Loss)

The MC-JEPA loss requires on-policy sampling from the successor measure, which is impractical for offline data. Leveraging the Bellman equation for successor features, TD-JEPA defines a TD version:

**L_TD-JEPA** = E [ || T(phi(s), a, z) - stop-grad(psi(s')) - gamma * stop-grad(T(phi(s'), a', z)) ||^2 ]

where (s, a, s') are sampled from an offline dataset, a' is sampled from the policy pi_z conditioned on s', and target networks provide the stop-gradient terms. This loss only requires one-step transitions and can be estimated from off-policy, offline data.

### Separate State and Task Encoders

TD-JEPA introduces an asymmetric architecture with two distinct encoders:

- **State encoder phi:** Embeds observations into a latent space for dynamics modeling (input to the predictor).
- **Task encoder psi:** Defines the space of representable reward functions (target for the predictor).

The predictor T_phi maps phi(s) to psi-space, predicting the successor features of psi under policy pi_z. Symmetrically, a second predictor T_psi maps psi(s) to phi-space. Both are trained via the TD-JEPA loss with the roles of phi and psi swapped. This bidirectional training encourages the two representations to be mutually predictive while allowing different dimensionalities and content -- useful when state features need low-level dynamical information (joint positions, velocities) while task features should capture higher-level contextual structure.

### Zero-Shot Policy Optimization

The relationship between predictors and successor features yields a natural zero-shot RL algorithm. The policy parameter space Z is set to the task embedding space (Z = R^{d_psi}), and latent policies pi_z(phi(s)) = argmax_a T_phi(phi(s), z, a)^T z are trained to maximize the predicted successor features. Since T_phi approximates the successor features F_psi^{pi_z}, this produces optimal policies for all rewards in the span of psi.

At test time, given a small inference dataset of (state, reward) pairs, the optimal z_r is computed in closed form via linear regression: z_r = [E[psi(s)psi(s)^T]]^{-1} E[psi(s)r(s)]. The associated policy pi_{z_r} is then returned as the zero-shot solution.

### Algorithm Summary (Algorithm 1)

Each training iteration:
1. Sample a batch of transitions (s, a, s') from the offline dataset and latent codes z from Z.
2. Sample next actions a' from the actor pi(phi(s'), z).
3. Compute the TD-JEPA loss for (phi, T_phi) targeting psi, and for (psi, T_psi) targeting phi.
4. Add orthonormality regularization losses on phi and psi (covariance regularization pushing toward identity covariance).
5. Update the actor to maximize T_phi(phi(s), a, z)^T z.
6. Update target networks (phi^-, T_phi^-, psi^-, T_psi^-) via exponential moving average (EMA).

### Architecture and Training Details

- **Networks:** All successor feature estimators, predictors, and F-networks are MLPs with two layer-normalized embedding layers. State/task encoders are standard MLPs with L2-normalized output. Actor networks are Gaussian MLPs.
- **Visual observations:** 64x64 RGB images, stacked 3 frames, processed by DrQ-v2-style convolutional encoders (separate for state and task encoders).
- **State encoder output dimension d_phi:** 256 across all domains.
- **Task encoder output dimension d_psi:** 50 across all domains.
- **Discount factor gamma:** 0.98 (DMC), 0.99 (OGBench).
- **Training steps:** 2M (DMC), 1M (OGBench).
- **Batch size:** 512 (DMC_RGB), 1024 (DMC), 256 (OGBench).
- **Optimizer:** Adam, learning rate 10^{-4}.
- **EMA coefficient:** 0.001 (DMC), 0.005 (OGBench).
- **Latent codes z:** Representations of random uniform states from the dataset with probability p_goal = 0.5, otherwise sampled from the hypersphere.

---

## Theoretical Analysis

The paper provides formal guarantees for TD-JEPA in a simplified tabular setting with linear predictors, under assumptions of identity covariance (phi^T phi = psi^T psi = I), uniform state distribution, and symmetric transition kernels.

### Theorem 1 (MC-JEPA -- Gradient Matching)

For optimal predictors, the MC-JEPA loss and the successor measure approximation loss L_SM have identical gradients with respect to phi and psi. The optimal predictors recover an orthogonal projection of the successor measure onto the span of the representations. This is established via a novel "gradient matching" argument that generalizes existing results for latent-predictive representations (Tang et al. 2023, Khetarpal et al. 2025, Voelcker et al. 2024, Lawson et al. 2025).

### Theorem 2 (Non-Collapse Guarantee)

Under a continuous-time relaxation where predictors are optimized faster than representations, the covariance matrices phi^T phi and psi^T psi are constant over time. This prevents collapse to trivial solutions (phi = psi = 0) when properly initialized with unitary covariance.

### Theorem 3 (TD-JEPA -- Forward/Backward Decomposition)

The optimal predictors and gradients of the TD-JEPA loss match those of non-latent-predictive forward and backward TD losses for successor measure approximation. Unlike the MC case, the optimal predictors solve a projected Bellman operator (least-squares TD problem), yielding an *oblique* rather than orthogonal projection.

### Theorem 4 (Zero-Shot Policy Evaluation Bound)

For representations with identity covariance, the policy evaluation error (for any reward function with bounded norm) is bounded by twice the successor measure approximation loss. This directly justifies the zero-shot optimization procedure: minimizing the TD-JEPA loss indirectly minimizes the policy evaluation error for all rewards.

---

## Results

### Benchmarks

TD-JEPA is evaluated on 65 tasks across 13 datasets:
- **ExoRL / DMC:** 4 locomotion/navigation domains (walker, cheetah, quadruped, pointmass) with both proprioceptive and pixel-based observations, high-coverage reward-based datasets.
- **OGBench:** 9 navigation/manipulation domains (5 antmaze variants, cube-single, cube-double, scene, puzzle-3x3) with both proprioceptive and pixel-based observations, low-coverage goal-reaching datasets.

### Comparison to Zero-Shot Baselines (Table 1)

TD-JEPA is compared against three groups of successor-feature-based baselines:
- **Task-encoder-only methods:** Laplacian, HILP, FB (Forward-Backward).
- **Latent-predictive state encoders:** BYOL* (one-step), BYOL-gamma* (multi-step behavioral), RLDP (multi-step with contrastive loss).
- **Multi-linear decomposition:** ICVF* (value-based, expectile regression).

All methods use the same architecture and a shared explicit state encoder for fair comparison.

| Setting | TD-JEPA | Best Baseline | Second Best |
|---------|---------|---------------|-------------|
| DMC_RGB (avg) | **628.8** | 582.4 (BYOL-gamma*) | 525.7 (RLDP) |
| DMC (avg) | **661.2** | 645.4 (BYOL-gamma*) | 620.1 (ICVF*) |
| OGBench_RGB (avg) | **41.34** | 41.58 (BYOL-gamma*) | 39.89 (FB) |
| OGBench (avg) | **37.98** | 37.98 (FB) | 30.42 (BYOL-gamma*) |

TD-JEPA is on par with or better than the best performing baseline in each suite when considering suite-aggregated performance. Critically, it is the most *consistently* strong method: probability-of-improvement analysis (Figure 2) shows TD-JEPA among the top performers across all settings, whereas most baselines excel on a narrow subset while underperforming elsewhere.

### Key Domain Results

- **Pixel-based DMC:** TD-JEPA achieves 628.8 average normalized return, significantly outperforming all baselines. The advantage is particularly large on walker (738.9 vs. 648.3) and cheetah (706.0 vs. 679.8).
- **OGBench navigation (antmaze):** TD-JEPA achieves the best or competitive results across most antmaze variants, with notable strength on antmaze-mn (96.67) and antmaze-ms (84.40).
- **OGBench manipulation:** TD-JEPA leads on cube-single (34.20), scene (38.44), and puzzle-3x3 (15.60), demonstrating effectiveness on diverse task types.

### Ablation: Which Dynamics Should Be Modeled? (Section 5.2)

Comparing latent-predictive methods: BYOL* approximates one-step behavioral transitions, BYOL-gamma* approximates multi-step behavioral transitions, and TD-JEPA models multi-step transitions of the *zero-shot policies*. Directly modeling policy-conditional successor measures is on average beneficial, especially in the pixel setting (Figure 3, left).

### Ablation: Separate vs. Shared Encoders (Section 5.3)

TD-JEPA (asymmetric, separate phi and psi) vs. TD-JEPA_sym (symmetric, shared encoder):
- TD-JEPA_sym performs comparatively well, especially on proprioception.
- Distinct state and task embeddings tend to improve empirical performance more often than not.
- The advantage of separate encoders is most pronounced in pixel-based settings.

| Setting | TD-JEPA | TD-JEPA_sym | C-TD-JEPA_sym (contrastive) |
|---------|---------|-------------|----------------------------|
| DMC_RGB (avg) | **628.8** | 598.1 | 437.2 |
| DMC (avg) | **661.2** | 657.5 | 586.3 |
| OGBench_RGB (avg) | **41.34** | 39.74 | 33.93 |
| OGBench (avg) | **37.98** | 35.20 | 35.58 |

The contrastive variant (C-TD-JEPA_sym) significantly underperforms, confirming that latent prediction (non-contrastive) is preferable to contrastive objectives -- aligning with [[lecun-2022-openreview]] ([LeCun 2022](../papers/lecun-2022-openreview.md))'s arguments.

### Fast Adaptation from Pre-Trained Representations (Section 5.4)

TD-JEPA representations enable efficient downstream fine-tuning:
- Initializing TD3 agents from zero-shot policies and pre-trained encoders leads to large gains in sample efficiency over training from scratch.
- Both offline (fixed dataset) and online (growing buffer) adaptation protocols show strong improvements.
- **Frozen** pre-trained state encoders are often sufficient for downstream learning -- no fine-tuning of the encoder needed -- demonstrating that TD-JEPA learns generally useful representations.

---

## Comparison to Prior Work

| Method | Prediction Type | Policy Conditioning | Off-Policy | Separate Encoders | Core Objective |
|--------|----------------|--------------------|-----------|--------------------|---------------|
| FB (Touati & Ollivier, 2021) | Contrastive (bilinear) | Multi-policy | Yes | Task only | Contrastive SF |
| HILP (Park et al., 2024) | Goal-reaching distance | No | Partially | Task only | Temporal distance |
| RLDP (Jajoo et al., 2025) | Latent-predictive (multi-step) | Behavioral only | Yes | State only | Chained latent prediction |
| BYOL-gamma* (Lawson et al., 2025) | Latent-predictive (discounted) | Behavioral only | No (on-policy) | No | Discounted latent prediction |
| ICVF* (Ghosh et al., 2023) | Value-based (expectile) | Multi-policy | Yes | Shared | Intention-conditioned value |
| **TD-JEPA** | **Latent-predictive (TD)** | **Multi-policy (zero-shot)** | **Yes** | **State + Task** | **TD latent prediction** |

**[[lecun-2022-openreview]] ([JEPA, LeCun 2022](../papers/lecun-2022-openreview.md)):** TD-JEPA is an instantiation of the JEPA paradigm in RL. Like JEPA, it predicts in representation space rather than observation space, avoids generative decoding, and uses a non-contrastive objective. TD-JEPA extends JEPA with policy-conditioning, multi-step TD-based prediction, and successor feature recovery -- connecting abstract latent prediction to concrete zero-shot policy optimization.

**[[hansen-2022-icml]] ([TD-MPC, Hansen et al. 2022](../papers/hansen-2022-icml.md)) and [[hansen-2024-iclr]] ([TD-MPC2, Hansen et al. 2024](../papers/hansen-2024-iclr.md)):** TD-MPC uses a latent-predictive loss combined with TD learning to train a world model for planning, but it requires reward signals and operates in a single-task or multi-task reward-based setting. TD-JEPA shares the idea of combining latent prediction with TD learning, but applies it in a reward-free, unsupervised setting to learn successor features rather than Q-values directly. The appendix notes that RLDP uses a chained multi-step latent-predictive loss similar to TD-MPC's.

**[[hafner-2023-arxiv]] ([DreamerV3, Hafner et al. 2023](../papers/hafner-2023-arxiv.md)):** DreamerV3 learns a generative world model (RSSM) that decodes back to observations and requires reward signals for task-specific training. TD-JEPA operates entirely in latent space without decoding, is reward-free, and supports zero-shot transfer to arbitrary tasks. The two approaches address fundamentally different settings.

**[[maes-2026-arxiv]] ([LeWorldModel, Maes et al. 2026](../papers/maes-2026-arxiv.md)):** LeWorldModel also builds on the JEPA paradigm for world modeling. While LeWorldModel focuses on learning a general-purpose generative world model from video, TD-JEPA focuses specifically on learning representations suitable for zero-shot policy optimization in RL, with formal connections to successor features.

**[[balestriero-2025-iclr]] ([LeJEPA, Balestriero et al. 2025](../papers/balestriero-2025-iclr.md)):** LeJEPA explores the JEPA paradigm for self-supervised visual representation learning. TD-JEPA adapts the core JEPA idea (predict in representation space) to the RL domain, adding policy conditioning and TD-based multi-step prediction as key innovations.

**FB (Forward-Backward, Touati & Ollivier 2021/2023):** The most closely related zero-shot RL method. FB also learns a bilinear decomposition of successor features but uses a contrastive loss and pairwise dot products across the batch. TD-JEPA uses a latent-predictive (non-contrastive) loss, explicitly trains shared state representations, and enforces more structure in the predictor. Both achieve competitive zero-shot performance, but TD-JEPA is more consistently strong across pixel and proprioceptive settings.

---

## Strengths

- Provides a principled connection between latent-predictive (JEPA-style) representation learning and successor features, showing that the predictor directly approximates successor features -- making latent prediction not just an auxiliary loss but the core mechanism for zero-shot RL.
- Novel TD formulation enables off-policy, offline learning from one-step transitions, removing the need for on-policy rollouts or full trajectory data that limits prior latent-predictive methods.
- Strong theoretical foundations: non-collapse guarantees, gradient matching with successor measure losses, and policy evaluation error bounds that justify the zero-shot optimization procedure.
- Consistently strong empirical performance across 13 datasets spanning locomotion, navigation, and manipulation with both proprioceptive and pixel inputs -- no other baseline achieves this breadth.
- The asymmetric encoder design (separate state and task encoders) is well-motivated and empirically validated, allowing different representational requirements for dynamics modeling vs. reward specification.
- Pre-trained representations enable fast downstream adaptation, with frozen encoders often sufficient -- demonstrating practical utility beyond zero-shot evaluation.

## Weaknesses & Limitations

- Theoretical guarantees rely on strong assumptions (identity covariance, uniform state distribution, symmetric kernels) that do not hold in practice. The relaxed analysis in Appendix C requires more complex proofs and yields weaker statements.
- The formal analysis only covers the case of symmetric representations (phi = psi sharing the same space), whereas the practical algorithm uses asymmetric encoders -- the theoretical-practical gap is acknowledged but not fully bridged.
- Performance on some individual OGBench domains (e.g., antmaze-me: 0.20, cube-double: 3.00) remains very low in absolute terms, suggesting fundamental challenges in low-coverage or sparse-reward settings that TD-JEPA does not fully solve.
- The method requires sweeping orthonormal regularization strength across domains (Table 5), and performance is sensitive to this hyperparameter -- not fully hyperparameter-free.
- OGBench evaluation requires an additional behavior-cloning regularization scheme (FlowQ-like, Appendix E.6) to handle low-coverage datasets, adding complexity beyond the core TD-JEPA algorithm.
- Evaluated exclusively in simulation; no real-robot experiments. Scaling to large-scale, real robotic datasets is noted as future work.
- The zero-shot inference procedure assumes the downstream reward is linear in the task encoder's representation, limiting expressiveness for complex reward functions (though Theorem 4 notes optimality extends to non-linear rewards when the successor measure approximation is perfect).

## Key Takeaways

- **Latent prediction as the core RL objective:** TD-JEPA demonstrates that latent-predictive learning (JEPA) is not merely a useful auxiliary loss in RL but can serve as the primary objective for learning all components needed for zero-shot policy optimization -- encoders, predictors, and policies.
- **TD enables off-policy multi-step latent prediction:** The temporal-difference formulation is the key technical innovation, enabling multi-step, policy-conditioned prediction from single offline transitions, where prior methods required on-policy data or single-step prediction.
- **Predictors approximate successor features:** The formal equivalence between latent predictors and successor features (Proposition 1, Theorems 1 and 3) is the central theoretical insight, connecting two previously separate research threads -- latent-predictive representation learning and successor-feature-based zero-shot RL.
- **Separate state and task encoders help:** Using distinct encoders for dynamics modeling (phi) and task specification (psi) improves performance, especially in pixel-based settings, by allowing each encoder to specialize.
- **Latent-predictive methods are broadly preferable in visual domains:** TD-JEPA and other latent-predictive baselines (BYOL*, BYOL-gamma*) tend to outperform contrastive and task-encoder-only methods when learning from pixels -- the most challenging setting for unsupervised RL.
- **Consistent cross-domain performance matters:** While individual baselines can win on specific domains, TD-JEPA is the most consistently competitive method across all 13 datasets, suggesting it captures a more general principle for representation learning.

---

## BibTeX
{% raw %}
```bibtex
@inproceedings{bagatella2025tdjepa,
  title={{TD-JEPA}: Latent-predictive Representations for Zero-Shot Reinforcement Learning},
  author={Bagatella, Marco and Pirotta, Matteo and Touati, Ahmed and Lazaric, Alessandro and Tirinzoni, Andrea},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
{% endraw %}

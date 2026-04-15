---
title: "Predictive Coding Enhances Meta-RL To Achieve Interpretable Bayes-Optimal Belief Representation Under Partial Observability"
type: paper
paper_id: P032
authors:
  - "Kuo, Po-Chen"
  - "Hou, Han"
  - "Dabney, Will"
  - "Walker, Edgar Y."
year: 2025
venue: "NeurIPS 2025"
arxiv_id: "2510.22039"
url: "https://arxiv.org/abs/2510.22039"
pdf: "../../raw/kuo-2025-neurips.pdf"
tags: [predictive-coding, meta-RL, reinforcement-learning, belief-states, POMDP, Bayes-optimal, representation-learning, world-model, self-supervised-learning]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
cited_by: []
---

# Predictive Coding Enhances Meta-RL To Achieve Interpretable Bayes-Optimal Belief Representation Under Partial Observability

> Integrating self-supervised predictive coding modules (a VAE-based observation and reward predictor) into meta-RL produces interpretable belief representations that closely approximate Bayes-optimal belief states across six diverse POMDP tasks (bandits, Tiger, oracle bandit, latent goal cart), whereas conventional black-box meta-RL (RL^2) fails to learn the minimally sufficient representation despite achieving similar task performance, with predictive coding also enabling improved zero-shot generalization and faster transfer learning.

**Authors:** Po-Chen Kuo (UW), Han Hou (Allen Institute for Neural Dynamics), Will Dabney (Google DeepMind), Edgar Y. Walker (UW) | **Venue:** NeurIPS 2025 | **arXiv:** [2510.22039](https://arxiv.org/abs/2510.22039)

---

## Problem & Motivation

In partially observable environments (POMDPs), agents must maintain a compact representation of history to support decision-making. The gold standard is the **Bayes-optimal belief state** -- the posterior probability distribution over hidden states given the history of observations and actions. Meta-RL, particularly memory-based approaches like RL^2, can learn near-Bayes-optimal policies by encoding history in recurrent hidden states. However:

1. **Representations are not Bayes-optimal**: Even when RL^2 achieves near-optimal return, its learned representations fail to match the minimally sufficient Bayes-optimal belief states. The representations are redundant and non-injective, hindering interpretability.

2. **Inadequate representations hurt generalization**: Suboptimal representations limit the agent's ability to generalize to unseen tasks, transfer to new distributions, and explore effectively.

3. **Neuroscience motivation**: Predictive coding -- the brain's strategy of continuously predicting sensory inputs -- has been suggested as a neural implementation of Bayesian inference. If adding predictive objectives produces better belief representations, this would connect deep RL to neuroscience theory.

The paper systematically investigates whether self-supervised predictive coding modules, when integrated into meta-RL, yield more interpretable, Bayes-optimal belief representations.

---

## Core Idea

The key insight is to **decouple representation learning from policy learning** in meta-RL by introducing self-supervised predictive modules:

1. A **predictive representation module** (VAE-based) is trained with self-supervised objectives to predict future observations and rewards.
2. A **policy network** operates on the learned belief representation rather than raw RNN hidden states.

By training the representation via prediction (analogous to how the brain's predictive coding builds internal world models), the bottleneck layer is encouraged to capture the minimal sufficient statistics of the history -- exactly the Bayes-optimal belief state. The policy gradient from RL loss trains only the policy network, not the encoder, ensuring that the representation is driven purely by predictive learning.

---

## How It Works

### Architecture (Figure 1C)

The proposed meta-RL with predictive modules consists of:

1. **RNN encoder** (q_phi): Takes current observation o_t, reward r_t, and previous action a_{t-1}. Outputs a low-dimensional bottleneck m_t, which parameterizes a posterior distribution over latent belief states: b_t = p(m_t | tau_{:t}) ~ distribution.

2. **Reward decoder** (R_theta): Predicts upcoming rewards given the belief state -- akin to predictive coding's reward prediction.

3. **Observation decoder** (T_theta): Predicts upcoming observations given the belief state and action -- analogous to a transition model in world model learning.

4. **Policy network** (pi_psi): A feedforward network receiving the belief state b_t as input, producing action logits and value estimates. Trained via standard policy gradient (model-free RL).

### Training Objective

The predictive modules optimize the **evidence lower bound (ELBO)** with KL regularization:

- **Observation prediction loss**: Likelihood of next observation given belief and action.
- **Reward prediction loss**: Likelihood of next reward given belief.
- **KL regularization**: Encourages compact, informative belief representations (similar to Bayesian filtering).

Critically, the **RL policy gradient only trains the policy network**, not the RNN encoder. The encoder is trained purely by the self-supervised predictive loss. This separation ensures the representation is driven by predictive learning, not reward maximization.

### State Machine Simulation Analysis

The paper uses **state machine simulation** -- a rigorous formal method -- to evaluate whether learned representations are computationally equivalent to Bayes-optimal solutions. Two mapping functions (MLPs) are trained bidirectionally between meta-RL representations and Bayes-optimal states. Two metrics are computed:

- **State dissimilarity (D_s)**: MSE between mapped states and targets.
- **Output dissimilarity (D_o)**: Difference in return between the policy applied to mapped vs. original states.

Low values in both directions indicate the learned representation is computationally equivalent to the Bayes-optimal belief state.

### Tasks

Six diverse POMDP tasks with tractable Bayes-optimal solutions:

1. **Two-armed Bernoulli bandit**: Classic explore-exploit tradeoff.
2. **Dynamic two-armed bandit**: Arms' reward probabilities change via Markov transitions (symmetric, asymmetric reward, asymmetric transition variants).
3. **Stationary Tiger**: Sequential decision-making with information gathering (two observation accuracy levels).
4. **Dynamic Tiger**: Tiger location changes over time (two observation accuracy levels).
5. **Oracle bandit**: 11-arm bandit with an oracle arm revealing the target arm's identity -- requires active information seeking.
6. **Latent goal cart**: Continuous control task with hidden goal inference.

### Inference

After training, the belief representation b_t from the encoder bottleneck layer is fed to the policy network for action selection. The predictive modules (decoders) are not needed at inference.

---

## Results

### Representation Quality

Across all six task families, meta-RL with predictive modules consistently achieves **lower state and output dissimilarity (D_s, D_o) in both mapping directions** at the bottleneck layer compared to RL^2, indicating closer approximation to Bayes-optimal belief states.

**Two-armed Bernoulli bandit (Figure 2):**
- Both models achieve Bayes-optimal return.
- Meta-RL with predictive modules learns representations structurally more similar to Bayes-optimal states (PCA visualization).
- Bottleneck layer achieves lowest D_s and D_o in both mapping directions.

**Dynamic bandits (Figure 3):**
- Both models approach optimal return.
- Meta-RL with predictive modules achieves significantly lower D_s for the Bayes->meta-RL mapping direction, indicating higher representational equivalence.

**Tiger tasks (Figure 4):**
- In the most challenging setting (Dynamic Tiger, low observation accuracy), **only meta-RL with predictive modules** approaches Bayes-optimal return. RL^2 fails.
- Bottleneck dissimilarities are significantly lower for the predictive model across all Tiger variants.

**Oracle bandit (Figure 5):**
- **Only meta-RL with predictive modules** learns the optimal policy (sample oracle arm, use information).
- RL^2 converges to a suboptimal policy, failing to effectively use information from the oracle.
- RL^2 also fails to learn an interpretable representation (high D_s and D_o).

**Latent goal cart (Figure 6):**
- Both models approach Bayes-optimal return.
- Meta-RL with predictive modules achieves lowest state and output dissimilarities in both mapping directions.

### Generalization (Figure 7)

**Zero-shot generalization**: Models trained on Dynamic Tiger (accuracy=0.7) tested on accuracy=0.8:
- Meta-RL with predictive modules: return = -15.94 +/- 3.82 (near-optimal).
- RL^2: return = -25.56 +/- 1.80 (significantly worse).

**Transfer learning**: Models pre-trained on Oracle Bandit arms 1-5, transferred to arms 6-10:
- Meta-RL with predictive modules shows significantly faster adaptation.

### Ablation Studies (Table 1)

**No KL regularization**: Removing KL divergence increases bottleneck dissimilarities on most tasks, showing that suitable regularization promotes compact belief representations.

**Joint RL training**: Allowing RL gradients to flow through the encoder does NOT further decrease dissimilarities compared to predictive-loss-only training. This demonstrates that the enhanced representation quality is attributable to predictive learning, not reward-signal-driven learning.

---

## Comparison to Prior Work

| | **This paper** | RL^2 (Duan et al.) | SOLAR / PlaNet / Dreamer | Igl et al. (2018) |
|---|---|---|---|---|
| **Approach** | Predictive coding + meta-RL | Black-box meta-RL | Latent world models | Particle filter meta-RL |
| **Representation target** | Bayes-optimal belief states | Implicit in RNN | Latent dynamics | Approximate belief |
| **Evaluation** | State machine simulation | Return only | Return + prediction | Return + belief decoding |
| **Self-supervised module** | VAE (obs + reward prediction) | None | RSSM | Particle filter |
| **Policy gradient through encoder** | No (separated) | Yes (end-to-end) | Mixed | Yes |
| **Generalization** | Strong (zero-shot + transfer) | Weak | Not tested in meta-RL | Not tested |

**vs [[lecun-2022-openreview]]:** LeCun's framework advocates for world models that predict in abstract representation space. This paper provides empirical evidence from a different angle: predictive objectives (predicting future observations and rewards) produce representations that naturally converge to the Bayes-optimal sufficient statistic. This connects the JEPA/world-model philosophy to the classical notion of Bayesian belief states.

**vs [[hafner-2023-arxiv]] (DreamerV3):** DreamerV3 trains latent world models for control using multi-step predictive objectives. This paper complements DreamerV3 by providing the *interpretability foundation* -- explaining *why* predictive objectives improve RL performance: they produce representations closer to Bayes-optimal belief states.

---

## Strengths

- **Rigorous analysis framework**: State machine simulation goes far beyond standard decoding-based evaluation, assessing computational equivalence (not just correlation) between learned and optimal representations.
- **Systematic task coverage**: Six diverse POMDP families spanning bandits, information gathering, dynamic environments, and continuous control -- the most comprehensive evaluation of meta-RL representations to date.
- **Clean separation of predictive learning and RL**: Demonstrating that RL gradients do not further improve representations beyond what predictive learning achieves is a powerful finding that isolates the source of improvement.
- **Neuroscience connection**: Links predictive coding theory from neuroscience to concrete computational benefits in deep RL, providing evidence for predictive coding as a general principle for efficient representation learning.
- **Practical benefits beyond interpretability**: Better representations lead to better generalization (zero-shot and transfer), not just interpretability -- making the approach practically useful.

---

## Weaknesses & Limitations

- **Tractable POMDPs only**: The state machine simulation analysis requires POMDPs where Bayes-optimal solutions can be computed. Scaling to more complex environments (Atari, robotics) where ground-truth belief states are intractable is left as future work.
- **Small-scale tasks**: All tasks have small state spaces (2-11 states). It is unclear whether the representational advantages persist in high-dimensional environments.
- **Next-step prediction only**: Only single-step observation and reward prediction is used. Multi-step, multi-scale, or temporal abstraction in predictive objectives (as suggested by world model literature) is unexplored.
- **VAE-specific architecture**: The predictive module uses a specific VAE parameterization. Whether other self-supervised objectives (e.g., contrastive, JEPA-style embedding prediction) would produce similar results is not investigated.
- **No pixel observations**: All tasks use structured observations. The method's effectiveness with high-dimensional image observations (where representation learning matters most) is not tested.

---

## Key Takeaways

- **Predictive learning produces Bayes-optimal representations**: Self-supervised prediction of future observations and rewards drives meta-RL representations to converge to the minimally sufficient Bayes-optimal belief state -- something reward-only training (RL^2) fails to achieve even when it finds optimal policies.
- **Better representations lead to better behavior in hard tasks**: In challenging tasks requiring information seeking and active exploration (Oracle bandit, Dynamic Tiger with low accuracy), only the predictive model achieves Bayes-optimal policies. The representation is the bottleneck, not the policy.
- **RL gradients are unnecessary for representation quality**: The ablation showing that adding RL policy gradients to the encoder does not further improve representation quality is perhaps the paper's strongest finding. Predictive learning alone is sufficient.
- **Practical generalization benefits**: Zero-shot and transfer learning improvements demonstrate that learning better representations is not just an interpretability concern but has tangible performance benefits.
- **Connects predictive coding to Bayesian inference**: The results support the neuroscience hypothesis that predictive coding constitutes a general computational strategy for achieving Bayesian inference in neural systems.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{kuo2025predictive,
  title={Predictive Coding Enhances Meta-RL To Achieve Interpretable Bayes-Optimal Belief Representation Under Partial Observability},
  author={Kuo, Po-Chen and Hou, Han and Dabney, Will and Walker, Edgar Y.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={arXiv:2510.22039}
}
```
{% endraw %}

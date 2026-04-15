---
title: "A Path Towards Autonomous Machine Intelligence"
type: paper
paper_id: P004
authors:
  - "LeCun, Yann"
year: 2022
venue: OpenReview (position paper)
arxiv_id: ""
url: "https://openreview.net/forum?id=BZ5a1r-kVsf"
pdf: "../../raw/lecun-2022-openreview.pdf"
tags: [world-model, JEPA, self-supervised-learning]
created: 2026-04-10
updated: 2026-04-10
cites: []
cited_by:
  - assran-2023-cvpr
  - bar-2024-cvpr
  - bardes-2024-tmlr
  - ding-2024-csur
  - li-2025-arxiv
  - balestriero-2025-iclr
  - maes-2026-arxiv
  - nam-2026-arxiv
  - hauri-2026-iclrws
  - li-2025-iclr
  - terver-2025-iclr
  - chen-2025-iclr
  - destrade-2025-workshop
  - ghaemi-2025-neurips
  - zhang-2026-arxiv
  - bagatella-2025-iclr
  - assran-2025-arxiv
---

# A Path Towards Autonomous Machine Intelligence

> **Position paper** — LeCun proposes JEPA (Joint Embedding Predictive Architecture) and a six-module cognitive architecture as a roadmap to autonomous machine intelligence that learns world models through self-supervised learning rather than generative pixel prediction or contrastive methods.

**Authors:** Yann LeCun (NYU / Meta FAIR) | **Venue:** OpenReview (position paper, v0.9.2, 2022-06-27) | **OpenReview:** [BZ5a1r-kVsf](https://openreview.net/forum?id=BZ5a1r-kVsf)

---

## Problem & Motivation

Current ML systems require enormous amounts of labeled data or environment interaction to learn tasks that humans acquire with minimal exposure — a teenager learns to drive in ~20 hours while the best autonomous driving systems have consumed millions of RL trials and still fall short of human reliability. LeCun argues the key missing ingredient is an internal *world model*: a learned, configurable simulator of how the world works that allows an agent to predict consequences of actions without executing them. Three core challenges remain unsolved: (1) how machines can learn representations and world models largely by observation, (2) how machines can reason and plan in ways compatible with gradient-based learning, and (3) how machines can represent percepts and action plans hierarchically at multiple time scales. Existing approaches — purely generative models, contrastive SSL, and model-free RL — each fail to address all three simultaneously.

---

## Core Idea

LeCun argues that autonomous intelligence requires predicting in *representation space* rather than in raw observation space. Instead of training a model to reconstruct future pixels (which forces representation of irrelevant details like leaf textures), a Joint Embedding Predictive Architecture (JEPA) learns two encoders and a predictor such that the predictor maps the representation of the current observation to the representation of the future observation. Because both sides of the prediction are encoded abstractions, the model can ignore unpredictable details and focus on predictable structure. Stacking JEPAs hierarchically (H-JEPA) yields representations at multiple levels of abstraction operating over multiple time scales, enabling hierarchical planning through a cascade of abstract subgoals rather than low-level motor commands.

---

## How It Works

### Overview

The proposed architecture has six differentiable, trainable modules that interact in a closed loop:

```
Configurator → primes all modules for the task
Perception    → estimates current world state from sensors
World Model   → predicts future world states (uses JEPA / H-JEPA)
Cost Module   → computes scalar "energy" (discomfort)
Short-Term Memory → stores (state, cost) pairs for critic training
Actor         → proposes action sequences; optimizes via MPC
```

All modules are differentiable so gradients can flow between them, enabling end-to-end planning.

### Configurator Module

The configurator acts as executive control: it takes inputs from all other modules and modulates their parameters and attention circuits. Given a task, it pre-configures the perception encoder, world model predictor, and cost module to focus on task-relevant information. It can route signals through sub-networks, set subgoals, and adjust which cost sub-modules are active. Its function is analogous to the prefrontal cortex in vertebrates.

### World Model Module

The world model has two roles: (1) fill in missing information about the current state not provided by perception, and (2) predict plausible future states resulting from proposed action sequences. Because the real world is stochastic and partially observable, the world model uses latent variables to parameterize the set of plausible predictions. Predictions are performed entirely in abstract representation space — not in pixel space — via JEPA or H-JEPA (see below). The world model is the primary target of SSL training and is the most complex module in the architecture.

### Cost Module

The cost module computes a scalar "energy" measuring the agent's level of discomfort. It comprises two sub-modules:
- **Intrinsic Cost (IC):** Hard-wired and non-trainable. Encodes basic drives such as pain/pleasure, hunger, fear, curiosity. Corresponds roughly to the amygdala. Its immutability is a safety property — it prevents behavioral drift.
- **Trainable Critic (TC):** Trained to predict future intrinsic cost values given a world state. Trained by retrieving past (state, future-intrinsic-cost) pairs from short-term memory and minimizing prediction error. Its parameters are modulated by the configurator to direct attention towards subgoals.

The total cost is C(s) = IC(s) + TC(s), with each sub-module's contribution weighted by configurable coefficients.

### Actor Module and Perception-Action Modes

The actor operates in two modes:
- **Mode-1 (reactive / System 1):** A policy network directly maps perceived state to action without engaging the world model. Fast and parallelizable across tasks. Trained by distilling Mode-2 solutions.
- **Mode-2 (deliberate / System 2):** Classical Model-Predictive Control (MPC). The actor proposes a sequence of actions → the world model unrolls future states → the cost module evaluates total energy → the actor back-propagates gradients through the computational graph to find the minimum-cost action sequence. The first action is then executed. Mode-2 reasoning is onerous; only one complex task can be handled at a time.

Skills learned in Mode-2 can be compiled into Mode-1 policy networks through amortized inference, mirroring how humans automate practiced skills.

### JEPA: Joint Embedding Predictive Architecture

JEPA is the centerpiece of the paper. Given an observed variable x and a target variable y, two encoders produce representations s_x = Enc(x) and s_y = Enc(y). A predictor (which may condition on a latent variable z) maps s_x to a predicted representation s̃_y. The energy is the prediction error in representation space:

```
E_w(x, y, z) = D(s_y, Pred(s_x, z))
```

**Why not generative models?** Generative models must predict every pixel of y, forcing them to represent irrelevant stochastic details (e.g., leaf ripples in the wind). JEPA encoders can be invariant to such details, producing abstract representations where prediction is tractable.

**Handling uncertainty via latent variable z:** When the future is ambiguous (e.g., a car at a fork may go left or right), z parameterizes the set of plausible futures. By varying z over its set Z, the predictor produces the full set of plausible predictions Pred(s_x, Z).

**Preventing representational collapse:** Without a regularizer, encoders could collapse to constant representations (zero energy everywhere). Two approaches are discussed:
- *Contrastive methods* (SimCLR, MoCo, CPC, etc.): pull up energies of negative samples. LeCun argues these suffer from the curse of dimensionality — exponentially many contrastive samples are needed in high dimensions.
- *Regularized (non-contrastive) methods* (VICReg, Barlow Twins): minimize the volume of low-energy regions through variance and covariance regularizers on the representation space. LeCun argues these are strictly preferable.

**VICReg for JEPA training:** The VICReg loss enforces three criteria jointly: (1) variance of each representation dimension stays above a threshold (prevents collapse to a point), (2) covariance between different dimensions is pushed toward zero (decorrelates components), and (3) the predictor error D(s_y, s̃_y) is minimized. This is a *dimension-contrastive* rather than *sample-contrastive* method.

### Hierarchical JEPA (H-JEPA)

H-JEPA stacks multiple JEPA levels. JEPA-1 operates on raw inputs and produces low-level short-horizon predictions. JEPA-2 takes JEPA-1 representations as input and produces higher-level, longer-horizon predictions. At each level, less predictable detail is discarded; more abstract, temporally coarser representations emerge naturally from the training criteria. This yields the multi-scale world state representations necessary for hierarchical planning.

### Hierarchical Planning (Mode-2 with H-JEPA)

Planning in the H-JEPA framework proceeds top-down:
1. A high-level actor (Actor-2) infers abstract "action" targets at the coarsest level to minimize the high-level cost.
2. These high-level targets define subgoal conditions passed to lower-level cost modules.
3. A low-level actor (Actor-1) infers fine-grained action sequences that satisfy the subgoal costs.
4. The process is iterated until convergence. Joint optimization across levels is preferable to greedy top-down.

Uncertainty at each level is handled by sampling the latent variable from a Gibbs distribution defined by the regularizer. If k discrete values of z exist, the number of possible trajectories over t steps is k^t, requiring directed search (e.g., Monte Carlo Tree Search, beam search).

---

## Results / Key Findings

This is a position paper; no empirical experiments are reported. The paper's contribution is architectural and conceptual:

1. **JEPA outperforms generative models for world modeling** (theoretical argument): generative models cannot abstract away unpredictable details, whereas JEPA encoders can be invariant to them. This is the paper's central claim.
2. **Non-contrastive SSL (VICReg-style) is preferable to contrastive SSL** for training JEPAs because contrastive methods require exponentially many negatives in high dimensions.
3. **Autonomous machine intelligence does not require generative models, LLM-style language modeling, or classical symbolic AI.** The entire architecture is differentiable and trainable end-to-end.
4. **The configurator hypothesis:** Animals likely have a single configurable world model engine (in the prefrontal cortex) rather than separate models for each task. This is more data-efficient and enables transfer by analogy.
5. **Intrinsic cost immutability as a safety mechanism:** Hard-wiring the basic drives prevents behavioral collapse and enables safety constraints to be non-negotiable.

---

## Comparison to Prior Work / Related Surveys

| Approach | Prediction Space | Uncertainty | Planning | SSL Method |
|---|---|---|---|---|
| Generative models (VAE, GAN, VQ-VAE) | Pixel/observation | Latent z | Limited | Likelihood / adversarial |
| Contrastive SSL (SimCLR, MoCo, CPC) | Representation | None | No | Sample-contrastive |
| Model-free RL (PPO, SAC) | None | Value fn | Policy gradient | External reward |
| Dreamer / Ha & Schmidhuber (2018) | Latent (generative) | Stochastic z | MPC in latent | RSSM |
| **JEPA (LeCun 2022)** | Representation (non-generative) | Latent z + encoder invariance | MPC + hierarchical | Non-contrastive (VICReg) |

**[[ha-2018-neurips]] ([Ha & Schmidhuber World Models 2018](../papers/ha-2018-neurips.md)) / Dreamer:** These are the closest prior work. They also use a learned latent-space world model for MPC-style planning. The key difference is that their world models are *generative* (they decode back to observations) and use stochastic RNNs, which LeCun argues cannot abstract away irrelevant details. JEPA's encoders are non-generative and can be invariant.

**Sutton's Dyna (1991):** The Mode-2 actor using a world model for planning is conceptually equivalent to Dyna. The difference is that Dyna used tabular or function-approximated models, whereas JEPA uses a hierarchical differentiable non-generative model.

**VICReg / Barlow Twins:** These non-contrastive SSL methods are explicitly proposed as the correct training paradigm for JEPAs. The paper significantly influenced the subsequent I-JEPA (2023) and V-JEPA papers from Meta FAIR.

**Kahneman's System 1 / System 2:** Mode-1 (reactive policy) and Mode-2 (deliberate MPC) are LeCun's neural analogs to this distinction.

---

## Strengths
- Provides a coherent, unified architecture that addresses learning, representation, planning, and behavior in one framework.
- The argument against generative models and contrastive methods is clear and well-motivated — predictions in representation space is an idea that subsequently proved highly impactful (I-JEPA, V-JEPA, data2vec).
- Draws productively on cognitive science, neuroscience, and control theory to motivate design choices.
- Immutable intrinsic cost as safety constraint is a concrete, implementable safety design principle.
- The Mode-1/Mode-2 analogy to System 1/System 2 gives an intuitive handle on the architecture's two operating regimes.
- Explicitly acknowledges the role of curiosity and exploration (intrinsic cost terms) for training world models.

## Weaknesses & Limitations
- No experiments: the paper provides no empirical evidence that JEPA-based systems outperform baselines in any concrete task or domain.
- The configurator is described at a high level but the learning algorithm for the configurator itself is left unspecified ("how the configurator learns to decompose complex tasks is left open for future investigation").
- Hierarchical planning under uncertainty grows exponentially (k^t trajectories) — the paper acknowledges this requires directed search but does not provide concrete solutions beyond pointing to MCTS.
- The proposal assumes gradient-based optimization is sufficient for Mode-2 planning, but discrete or discontinuous action spaces (common in real robotics) require non-differentiable planning methods that are only briefly mentioned.
- No treatment of language, social cognition, or long-horizon semantic reasoning — the paper's scope is largely perceptual and motor.
- The claim that regularized (non-contrastive) methods are strictly preferable to contrastive ones remains an open empirical question in 2022 and beyond.

## Key Takeaways
- **JEPA predicts in abstract representation space** rather than raw observation space, avoiding the need to model unpredictable stochastic details — this is the paper's most influential architectural idea.
- **Non-contrastive SSL (VICReg/Barlow Twins) trains JEPA** by maximizing information content of representations while minimizing prediction error and latent information — avoiding the curse of dimensionality that afflicts contrastive methods.
- **H-JEPA** stacks JEPA levels to learn a hierarchy of world state representations at progressively longer time horizons and coarser abstractions, enabling hierarchical MPC-style planning.
- **The six-module cognitive architecture** (configurator, perception, world model, cost, memory, actor) provides a complete blueprint for an autonomous agent in which all components are differentiable.
- **Mode-1/Mode-2 duality** allows skills learned through deliberate world-model-based planning (Mode-2) to be compiled into fast reactive policies (Mode-1), analogous to human skill acquisition.

---

## BibTeX
```bibtex
@techreport{lecun2022path,
  title     = {A Path Towards Autonomous Machine Intelligence},
  author    = {LeCun, Yann},
  year      = {2022},
  month     = {June},
  note      = {Version 0.9.2, OpenReview preprint},
  url       = {https://openreview.net/forum?id=BZ5a1r-kVsf}
}
```

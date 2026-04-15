---
title: "SOLD: Slot Object-Centric Latent Dynamics Models for Relational Manipulation Learning from Pixels"
type: paper
paper_id: P050
authors:
  - "Mosbach, Malte"
  - "Ewertz, Jan Niklas"
  - "Villar-Corrales, Angel"
  - "Behnke, Sven"
year: 2025
venue: ICML 2025
arxiv_id: "2410.08822"
url: "https://arxiv.org/abs/2410.08822"
pdf: "../../raw/mosbach-2025-icml.pdf"
tags: [world-model, object-centric, slot-attention, latent-dynamics, model-based-RL, manipulation]
created: 2026-04-15
updated: 2026-04-15
cites: []
cited_by: []
---

# SOLD: Slot Object-Centric Latent Dynamics Models for Relational Manipulation Learning from Pixels

> **One sentence** -- SOLD is the first object-centric model-based RL algorithm that learns directly from pixel inputs, using Slot Attention for Video (SAVi) to decompose scenes into object slots and a transformer-based dynamics model to predict per-object state evolution, outperforming DreamerV3 and TD-MPC2 on multi-object robotic manipulation tasks that require relational reasoning.

**Authors:** Malte Mosbach\*, Jan Niklas Ewertz\* (equal contribution), Angel Villar-Corrales, Sven Behnke | **Venue:** ICML 2025 | **arXiv:** [2410.08822](https://arxiv.org/abs/2410.08822)

---

## Problem & Motivation

Existing model-based RL methods -- including [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)) and [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)) -- learn holistic latent representations of the environment state. While effective for many control tasks, these monolithic representations struggle when the agent must reason about relationships between multiple distinct objects. Humans naturally perceive scenes as compositions of individual objects and anticipate how actions affect specific parts of their surroundings -- a capability that is essential for robotic manipulation involving multiple interacting objects.

Despite the success of object-centric representation learning (slot-based methods) for video prediction, integrating structured object-centric representations into model-based RL has remained largely unexplored. Prior object-centric RL methods (e.g., SMORL, EIT) are model-free, while the only model-based approach (FOCUS) requires ground-truth segmentation masks and does not use object-centric states for forward prediction or action selection. SOLD addresses this gap: it is the first method that performs object-centric model-based RL directly from pixels, without any object annotations.

---

## Core Idea

SOLD structures the latent space of a world model as a set of object slots rather than a single monolithic vector. By using SAVi (Slot Attention for Video) to decompose pixel observations into per-object representations and OCVP (Object-Centric Video Prediction) to model how each object evolves under actions, SOLD provides the actor and critic with a structured input space that naturally supports relational reasoning. The key insight is that object-centric representations are not only useful for interpretable video prediction but also provide a powerful inductive bias that accelerates behavior learning -- especially on tasks requiring the agent to identify and reason about relationships between objects.

---

## How It Works

### Overview

Pixel observations --> SAVi encoder --> N object slots (Z_t) --> Transformer-based dynamics model predicts next slot set --> SAVi decoder reconstructs predicted frames. Actor and critic are trained on imagined rollouts in the structured latent space. All components are trained jointly after an initial SAVi pretraining phase.

### Three Core Components

1. **Object-Centric World Model**: Predicts the effects of actions on individual objects via slot-based dynamics
2. **Critic**: Estimates value of a given structured latent state (set of object slots)
3. **Actor**: Selects actions to maximize expected return using the structured latent space

### SAVi Encoder-Decoder (Perception)

SOLD uses SAVi (Kipf et al., 2022), an encoder-decoder architecture with a structured bottleneck of N permutation-equivariant slot embeddings. At each time step t:

- The encoder maps observation o_t to feature maps F_t
- Slot Attention performs cross-attention between previous slot representations Z_{t-1} and current features F_t, with attention coefficients normalized over the slot dimension (forcing slots to compete for feature locations)
- Attention: A = softmax_N(q(Z_{t-1}) . k(F_t)^T / sqrt(D))
- Slots are updated via a shared GRU followed by a residual MLP: Z_t = MLP(GRU(A . v(F_t), Z_{t-1}))
- The decoder independently renders each slot into per-object images and alpha masks, combined via weighted sum

**SAVi pretraining**: SAVi is pretrained on ~1 million frames from random episodes for 400K gradient steps. Critically, the encoder-decoder is NOT frozen after pretraining -- it is fine-tuned during RL to adapt to state distributions not seen during random exploration (e.g., blocks lifted in the air during Pick tasks).

**Configuration**: 2-10 slots depending on environment, 128-dimensional slot embeddings, learned slot initialization, 3 Slot Attention iterations for first frame and 1 for subsequent frames.

### Object-Centric Dynamics Model

Built on the OCVP-Seq architecture (Villar-Corrales et al., 2023), a transformer encoder that uses two specialized self-attention variants:

- **Temporal attention**: Updates each slot by aggregating information from the same slot's history across time steps -- models per-object dynamics without inter-object interaction
- **Relational attention**: Jointly processes all slots from the same time step -- models object interactions and relationships

The dynamics model autoregressively predicts the next slot set given the history of slots and actions:

```
Encoder:     Z_t = e_eta(o_t)
Decoder:     o_hat_t = d_eta(Z_t)
Dynamics:    Z_hat_{t+1} = p_psi(Z_{0:t}, a_{0:t})
Reward:      r_hat_t ~ p_zeta(r_hat_t | Z_{0:t})
```

**Training without teacher forcing**: During training, the model uses S seed frames as context and autoregressively predicts T future frames. Predictions are fed back as input (no teacher forcing), so the dynamics model learns to handle its own imperfect predictions.

**Hybrid dynamics loss**: Predicted representations are shaped by both a joint embedding loss and a reconstruction loss:

L_dyn = sum_{t=S}^{S+T-1} [ ||Z_hat_t - e_eta(o_t)||^2 + ||o_hat_t - o_t||^2 ]

**Architecture**: 4 transformer layers, 8 attention heads, 256-dimensional tokens, 512-dimensional feed-forward layers. Uses ALiBi positional encoding (linear biases in attention scores) instead of absolute position encodings, enabling generalization to longer sequences.

### Slot Aggregation Transformer (SAT)

A novel architectural backbone that aggregates information from the history of object slots to produce predictions for rewards, values, and actions. The SAT addresses the challenge that reward/value/action predictions depend on the full set of slots rather than any single one.

- Causal transformer encoder that receives flattened slot histories as input
- Learnable [out] output tokens (one per time step) aggregate information across all slots
- Learnable [reg] register tokens (4 per time step) aid computation by offloading intermediate processing
- Causal attention masks ensure tokens at time t cannot attend to output/register tokens from other time steps
- ALiBi encoding for token recency bias
- 4 layers, 8 attention heads, 256-dimensional tokens, 512-dimensional feed-forward, RMS-Normalization

The SAT produces output token representations h_t, which are fed to MLP heads for reward, value, and action predictions.

### Reward Prediction

The reward predictor maps slot representations to scalar rewards via the SAT backbone. Rather than directly predicting a scalar, it outputs a softmax distribution over K exponentially spaced bins:

b = symexp([-20, ..., +20])
r_hat_t = softmax(f_zeta^MLP(h_t))^T . b

The true reward is symlog-transformed and encoded as a two-hot target. The model is trained to maximize the log-likelihood of the encoded reward distribution:

L_rew(zeta) = -sum_{t=0}^{T-1} log p_zeta(r_t | Z_{0:t})

### Behavior Learning (Actor-Critic)

Following the Dreamer framework, the actor and critic are trained on imagined trajectories in latent space:

**Critic**: Predicts the distribution of bootstrapped lambda-returns using the SAT backbone:

R_t^lambda = r_hat_{t+1} + gamma * ((1-lambda) * R_hat_{t+1} + lambda * R_{t+1}^lambda)

Trained with categorical cross-entropy over exponentially spaced bins (same as reward prediction). Regularized toward an EMA target network (decay rate 0.98). Imagination horizon T = 15, lambda = 0.95.

**Actor**: Outputs parameters of a Gaussian action distribution N(mu_t, sigma_t | Z_{0:t}). Loss combines expected returns with entropy regularization:

L_actor(theta) = -sum_{t=0}^{T-1} [ R_hat_t^lambda / max(1, s_V) ] + eta * H(pi_theta(a_t | Z_{0:t}))

Value normalization s_V uses the EMA of the 5th-to-95th percentile range of return estimates (following DreamerV3).

### Implementation Details

- **Total parameters**: 12 million (matching DreamerV3 baseline)
- **Hardware**: Single NVIDIA A-100 GPU (40GB VRAM)
- **Optimizer**: Adam with different learning rates per component (1e-4 for dynamics/reward, 3e-5 for actor/value/SAVi fine-tuning)
- **Gradient clipping**: Max norm 0.05 (SAVi), 3.0 (transition), 10.0 (reward/value/action)
- **Training seeds**: 3 per environment

---

## Results

### Benchmark: 8 Object-Centric Robotic Environments

SOLD introduces a custom suite of 8 MuJoCo-based robotic manipulation tasks with varying levels of relational reasoning difficulty. A robot arm with 4D continuous actions (3D end-effector movement + gripper) must interact with colored objects:

- **Specific** variants: Target object is red; 0-4 distractor objects of random distinct colors
- **Distinct** variants: 3-5 objects presented; target is the odd-one-out (color differs from all others)
- **Specific-Relative** (Reach only): Target is the reddest object (perceptual CIEDE2000 distance)
- **Distinct-Groups** (Reach only): 5 targets; reach the one that appears only once

Task types: Reach (move end-effector to target), Push (slide block to goal), Pick (grasp and lift block to goal).

### Final Success Rates (%)

**(a) Specific tasks** (mainly robotic control):

| Task | DreamerV3 | TD-MPC2 | w/o OCE | SOLD |
|------|-----------|---------|---------|------|
| Reach | 87.4 +/-1 | 97.6 +/-0 | 83.2 +/-2 | **97.9 +/-0** |
| Reach-Rel. | 45.6 +/-6 | 79.1 +/-1 | 39.2 +/-3 | **91.1 +/-2** |
| Push | **97.1 +/-1** | 72.7 +/-3 | 75.2 +/-2 | 82.8 +/-2 |
| Pick | **96.7 +/-1** | 87.6 +/-2 | 22.9 +/-11 | 85.8 +/-7 |
| **Average** | 81.7 +/-21 | 84.2 +/-9 | 55.1 +/-25 | **89.4 +/-6** |

**(b) Distinct tasks** (requiring relational reasoning):

| Task | DreamerV3 | TD-MPC2 | w/o OCE | SOLD |
|------|-----------|---------|---------|------|
| Reach | 14.6 +/-6 | 31.4 +/-3 | 11.3 +/-1 | **91.8 +/-1** |
| Reach-Gr. | 13.9 +/-2 | 15.7 +/-2 | 5.1 +/-1 | **69.6 +/-2** |
| Push | 70.0 +/-5 | 12.2 +/-5 | 10.5 +/-1 | **80.6 +/-5** |
| Pick | 33.9 +/-36 | 9.8 +/-1 | 0.7 +/-0 | **56.4 +/-25** |
| **Average** | 33.1 +/-23 | 17.3 +/-8 | 6.9 +/-4 | **74.6 +/-13** |

The results reveal a dramatic performance gap on relational reasoning tasks. On Distinct tasks, SOLD achieves 74.6% average success vs. 33.1% for DreamerV3 and 17.3% for TD-MPC2. SOLD more than doubles the performance of the next-best method on these tasks.

On Specific tasks, SOLD achieves the highest average (89.4%) with substantially lower variance, though DreamerV3 wins on Push-Specific (97.1% vs. 82.8%) and Pick-Specific (96.7% vs. 85.8%) -- tasks where simple holistic representations suffice since the target is always red.

### Sample Efficiency

Training curves (Figure 4) show SOLD consistently outperforms DreamerV3 and TD-MPC2 baselines on all but the easiest Reach-Specific task, even after accounting for the samples used during SAVi pretraining. SOLD demonstrates superior sample efficiency alongside final performance.

### Object-Centric Dynamics Predictions

Open-loop predictions from a single seed frame (Figure 3) demonstrate the model's ability to:
- Generate physically accurate predictions over 50 future frames
- Maintain object-centric decomposition throughout the rollout
- Capture physical interactions (pushing, occlusions) between objects
- Predict individual object slots with sharp segmentation masks

### Interpretable Attention Patterns

Visualization of the actor's attention weights (Figure 5) shows SOLD automatically discovers task-relevant objects in an unsupervised manner. On Push-Specific, the actor's [out] token attends primarily to slots representing the robot and the target object (green cube), while largely ignoring distractor slots. The model overcomes ALiBi's recency bias when necessary, attending to a red sphere (goal indicator) that had been occluded for 15 time steps.

### Generalization to Non-Object-Centric Tasks

SOLD generalizes beyond its natural object-centric setting to environments from Meta-World and DM-Control:

| Task | SOLD Result |
|------|-------------|
| Button-Press (Meta-World) | 100% success |
| Hammer (Meta-World) | 100% success |
| Cartpole-Balance (DM-Control) | 497 return |
| Finger-Spin (DM-Control) | 645 return |

These results demonstrate that SOLD's architecture does not harm performance when object-centric reasoning is not required.

### Ablation: SAVi Fine-tuning

A key finding is that continuously fine-tuning the SAVi encoder-decoder during RL training is essential, not optional. On Pick tasks, random behaviors rarely produce blocks lifted off the table, so the pretrained SAVi lacks exposure to these configurations. The fine-tuned model accurately reconstructs lifted blocks, whereas the frozen variant fails (blocks dissolve during lifting). This resolves a common limitation of prior object-centric RL methods that freeze pretrained representations.

---

## Comparison to Prior Work

| Method | Object-Centric | Model-Based | From Pixels | No Annotations | Structured Dynamics |
|--------|---------------|-------------|-------------|----------------|-------------------|
| **SOLD** | Yes (slots) | Yes | Yes | Yes | Yes (per-object) |
| DreamerV3 | No (holistic RSSM) | Yes | Yes | Yes | No |
| TD-MPC2 | No (holistic latent) | Yes | Yes | Yes | No |
| SMORL | Yes | No (model-free) | Yes | Yes | N/A |
| EIT | Yes | No (model-free) | Yes | Yes | N/A |
| FOCUS | Yes | Yes | No (needs masks) | No (GT masks) | Partial |

**[[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)):** Both SOLD and DreamerV3 follow the Dreamer framework (actor-critic trained on imagined latent trajectories). DreamerV3 uses a holistic RSSM with categorical latent states; SOLD replaces this with object-centric slots and a transformer dynamics model. DreamerV3 performs comparably or better on simple Specific tasks but collapses on Distinct tasks requiring relational reasoning (33.1% vs. 74.6% average). SOLD matches DreamerV3's 12M parameter count.

**[[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)):** TD-MPC2 uses holistic latent dynamics with MPPI planning. On Specific tasks, TD-MPC2 is competitive (84.2% average) but struggles severely on Distinct tasks (17.3% average) -- even worse than DreamerV3. TD-MPC2's task-oriented latent space, designed to encode reward-predictive features, apparently fails to capture inter-object relationships needed for relational reasoning.

**[[hafner-2021-iclr]] ([DreamerV2](../papers/hafner-2021-iclr.md)):** SOLD builds on the Dreamer actor-critic framework introduced in DreamerV2 but replaces the RSSM encoder with SAVi and the GRU-based dynamics with a transformer. SOLD inherits the key design choices of lambda-returns, EMA target network for the critic, and entropy-regularized actor.

**[[maes-2026-arxiv]] ([LeWorldModel](../papers/maes-2026-arxiv.md)):** LeWM and SOLD both move beyond holistic world models but in different directions. LeWM uses JEPA-style prediction with SIGReg regularization for reconstruction-free, reward-free world modeling; SOLD uses slot-based decomposition with reconstruction losses. LeWM operates on CLS tokens (one embedding per frame); SOLD operates on N slot embeddings per frame. The approaches address complementary limitations -- SOLD targets relational reasoning while LeWM targets training stability and efficiency.

**FOCUS (Ferraro et al., NeurIPSW 2023):** The only prior model-based approach with object-centric representations. However, FOCUS requires ground-truth segmentation masks and does not use object-centric states for forward prediction or action selection -- only for exploration targets. SOLD requires no supervision and uses slots directly for dynamics prediction, reward estimation, and policy learning.

---

## Strengths

- First object-centric model-based RL method that works end-to-end from raw pixels without any object annotations or ground-truth segmentation
- Dramatic improvement on relational reasoning tasks: 74.6% average success on Distinct tasks vs. 33.1% (DreamerV3) and 17.3% (TD-MPC2) -- more than doubling the next-best baseline
- Interpretable behavior: attention visualizations show the actor automatically discovers and focuses on task-relevant objects, providing human-understandable decision-making traces
- SAVi fine-tuning during RL addresses a fundamental limitation of prior work (frozen pretrained representations that fail on out-of-distribution configurations like lifted blocks)
- Generalizes to non-object-centric environments (Meta-World, DM-Control) without degradation, demonstrating the architecture is not restrictive
- Compact model (12M parameters) runs on a single A-100 GPU

## Weaknesses & Limitations

- **Deterministic predictions**: The world model generates predictions deterministically, which is a drawback in stochastic or highly unpredictable environments. DreamerV3's stochastic RSSM handles this by design.
- **SAVi scalability**: SAVi performs well on the evaluated robotic tabletop environments but scaling to complex real-world scenes with diverse textures and many objects remains a significant challenge. The authors note that the core ideas are independent of the specific encoder-decoder model.
- **SAVi pretraining cost**: Requires ~1 million frames of random exploration data for pretraining (400K gradient steps), which adds to the total sample budget and compute cost. This pretraining phase is not needed by DreamerV3 or TD-MPC2.
- **Underperformance on simple tasks**: DreamerV3 outperforms SOLD on Push-Specific (97.1% vs. 82.8%) and Pick-Specific (96.7% vs. 85.8%), suggesting that when relational reasoning is not needed, holistic representations can be more efficient for learning fine motor control
- **Custom benchmark only**: Results are primarily demonstrated on a new custom benchmark introduced by the authors. Evaluation on established benchmarks (beyond the 4 Meta-World/DM-Control generalization tasks) is limited.
- **Fixed number of slots**: The number of slots N must be set per environment (2-10), requiring some domain knowledge about scene complexity

## Key Takeaways

- Object-centric representations provide a decisive advantage for relational reasoning in model-based RL: SOLD more than doubles the success rate of holistic baselines (DreamerV3, TD-MPC2) on tasks requiring the agent to identify target objects based on their relationships to other objects
- Fine-tuning the object-centric encoder-decoder during RL training is critical -- freezing pretrained representations (as in most prior object-centric RL work) fails on tasks where the policy encounters state distributions unseen during random pretraining (e.g., lifted blocks in Pick tasks)
- The Slot Aggregation Transformer (SAT) with learnable output and register tokens provides an effective architecture for mapping variable-length slot histories to scalar predictions (rewards, values) and action distributions
- SOLD's transformer-based dynamics model with temporal and relational attention produces accurate open-loop predictions over 50+ frames while maintaining interpretable object-centric decompositions
- The architecture generalizes to non-object-centric environments (Meta-World, DM-Control) without performance loss, demonstrating that structured slot representations do not restrict the model when the structure is not explicitly needed

---

## BibTeX
{% raw %}
```bibtex
@inproceedings{mosbach2025sold,
  title={{SOLD}: Slot Object-Centric Latent Dynamics Models for Relational Manipulation Learning from Pixels},
  author={Mosbach, Malte and Ewertz, Jan Niklas and Villar-Corrales, Angel and Behnke, Sven},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
{% endraw %}

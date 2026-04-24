---
title: "Causal-JEPA: Learning World Models through Object-Level Latent Interventions"
type: paper
paper_id: P047
authors:
  - "Nam, Heejeong"
  - "Le Lidec, Quentin"
  - "Maes, Lucas"
  - "LeCun, Yann"
  - "Balestriero, Randall"
year: 2026
venue: arXiv
arxiv_id: "2602.11389"
url: "https://arxiv.org/abs/2602.11389"
pdf: "../../raw/nam-2026-arxiv.pdf"
tags: [JEPA, world-model, causal, object-centric, planning]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - balestriero-2025-iclr
  - maes-2026-arxiv
  - assran-2023-cvpr
  - bardes-2024-tmlr
cited_by: []
---

# Causal-JEPA: Learning World Models through Object-Level Latent Interventions

> **Causal-JEPA (C-JEPA)** extends JEPA world models from patch-level to object-centric representations by introducing object-level masking as a structured latent intervention during training. By selectively removing individual objects' latent histories, C-JEPA forces the predictor to reason about inter-object interactions, inducing a causal inductive bias without requiring explicit causal graphs, reconstruction losses, or task-specific supervision. It achieves ~20% absolute gains on counterfactual VQA over the unmasked baseline, and matches patch-based world model performance on robotic control using only 1% of the latent tokens and 8x faster planning.

**Authors:** Heejeong Nam\* (Brown), Quentin Le Lidec\* (NYU), Lucas Maes (Mila, UdeM), Yann LeCun (NYU), Randall Balestriero (Brown) | **Venue:** arXiv (February 2026) | **arXiv:** [2602.11389](https://arxiv.org/abs/2602.11389)

---

## Problem & Motivation

World models need to capture how objects interact -- a ball bouncing off a wall, one block pushing another -- but existing approaches fail to make interaction reasoning *functionally necessary* through the learning objective itself. The paper identifies three distinct failure modes in prior work:

1. **Object-centric representations alone are insufficient.** Simply replacing patch tokens with object slots (e.g., using Slot Attention) gives a useful abstraction, but the predictor can still exploit trivial shortcuts like temporal interpolation of each object's own trajectory, ignoring other objects entirely. Without an explicit mechanism to enforce interaction modeling, models fall back on object self-dynamics or incidental correlations.

2. **Patch-based JEPA world models are computationally wasteful for planning.** Methods like DINO-WM (Zhou et al., 2025) operate on ~196 patch tokens per frame. Attention scales quadratically with token count, and model-predictive control requires repeated rollouts, making planning prohibitively slow. In contrast, a scene with 4 objects yields only 4-7 tokens per frame -- orders of magnitude fewer.

3. **Existing structured world models impose rigid assumptions.** Methods like C-SWM (Kipf et al., 2020) assume fixed relational graphs, SPARTAN (Lei et al., 2025) relies on sparse attention heuristics, and OCVP-Seq (Villar-Corrales et al., 2023) uses architectural factorization. None of these makes interaction structure emerge from the *objective* itself.

C-JEPA's key insight is that masking an object's latent history acts as a *latent intervention on observability*: it creates a counterfactual-like query during training ("what would this object do if you couldn't see its own past?") that forces the predictor to rely on other objects' states to make accurate predictions. This is a principled inductive bias that makes interaction reasoning the path of least resistance for minimizing the training loss.

---

## Core Idea

C-JEPA combines two ideas -- object-centric representations and masked joint embedding prediction -- into a single framework where the masking granularity aligns with the semantic structure of the scene. The approach takes the JEPA paradigm from [[lecun-2022-openreview]], which advocates prediction in representation space rather than pixel space, and instantiates it with object-level structure. Rather than masking random patches (as in [[assran-2023-cvpr]] I-JEPA) or spatiotemporal tubes (as in [[bardes-2024-tmlr]] V-JEPA), C-JEPA masks entire object slots across the history window. This creates a structured form of partial observability: the predictor must infer a masked object's state from the states of other objects, auxiliary variables, and a minimal identity anchor. The authors formalize this as inducing "influence neighborhoods" -- minimal sufficient sets of contextual variables for predicting each object -- and prove that optimizing the masked prediction objective forces the predictor to discover these neighborhoods, effectively learning a soft relational structure without assuming or recovering an explicit causal graph.

This paper comes from the same research group as [[maes-2026-arxiv]] (LeWorldModel), which solved end-to-end JEPA training for world models using SIGReg. C-JEPA addresses an orthogonal dimension: while LeWorldModel focuses on training stability and end-to-end learning from pixels, C-JEPA focuses on making the world model interaction-aware through object-level structure and masking. The codebase builds directly on `stable-pretraining` ([[balestriero-2025-iclr]]) and `stable-worldmodel` ([[maes-2026-arxiv]]).

---

## How It Works

### Architecture Overview

C-JEPA has three components: a **frozen object-centric encoder**, a **masking scheme**, and a **latent predictor**. The encoder converts video frames into object slots; the masking scheme selectively removes object histories; and the predictor jointly recovers masked history states and predicts future states.

### Object-Centric Encoder

A frozen object-centric encoder $g$ maps each pixel-level frame $X_t \in \mathbb{R}^{H \times W \times C}$ to a set of $N$ object-centric slot representations:

$$S_t = g(X_t) = \{s_t^1, \ldots, s_t^N\}, \quad s_t^i \in \mathbb{R}^d$$

The primary encoder is **VideoSAUR** (Zadaianchuk et al., 2023), which aggregates frozen DINOv2 ViT-S/14 patch features into object-centric slots using a SAVi-style grouping mechanism (Slot Attention with 2 iterations). Each slot has dimensionality $d = 128$. For some experiments, **SAVi** (Kipf et al., 2022), a convolutional encoder with stochastic Slot Attention, is used instead. The encoder is always frozen during C-JEPA training.

### Object-Level Masking

At each time step $\tau$ in the history window $T = \{t - T_h + 1, \ldots, t\}$, a random subset $\mathcal{M} \subset \{1, \ldots, N\}$ of object indices is selected for masking. For masked objects, the observable slot value is replaced with a **masked token** defined as:

$$\bar{z}_\tau^i = \phi(z_{t_0}^i) + e_\tau$$

where $\phi$ is a linear projection, $z_{t_0}^i$ is the slot value at the earliest time step $t_0$ (serving as an **identity anchor**), and $e_\tau$ is a learnable embedding combined with temporal positional encoding. The identity anchor preserves *which* entity is masked (addressing permutation equivariance of slots) while revealing no information about its evolving state.

Key design choices:
- **Future entity tokens are always masked** for prediction -- this is the standard forward prediction objective.
- **History tokens for selected objects are additionally masked** -- this is the novel causal intervention.
- The masking index set $\mathcal{M}$ is sampled uniformly per training example; the number of masked objects $|\mathcal{M}|$ is a hyperparameter (typically 1-4 out of 7 slots on CLEVRER, 0-2 out of 4 on Push-T).

### Auxiliary Observable Variables

Actions $a_t$ and proprioceptive signals $p_t$ are treated as **separate auxiliary tokens** $U_t = \{a_t, p_t\}$, embedded via lightweight 1D temporal convolutional encoders into 128-dimensional vectors. These are concatenated alongside object tokens as additional conditioning inputs to the predictor, rather than being fused into the object representations. This separation is important: as shown in Figure 3, treating auxiliaries as separate entities (auxiliary conditioning) consistently outperforms concatenation into object latents (e.g., 88.67% vs. 82.67% on Push-T).

### Predictor

The predictor $f$ is a **ViT-style masked Transformer** with bidirectional attention:
- 6 Transformer layers, 16 attention heads, head dimension 64, MLP hidden dimension 2048
- Input: the full sequence of entity tokens $\bar{Z}_\mathcal{T}$ (masked object slots + visible slots + auxiliary tokens) across both history and future intervals
- Bidirectional attention enables joint inference over masked history tokens and future tokens simultaneously
- Output: predicted latent states $\hat{Z}_\mathcal{T} = f(\bar{Z}_\mathcal{T})$

The choice of bidirectional (masked) prediction over autoregressive prediction is deliberate: object states do not evolve as independent first-order Markov processes, and interactions may span multiple time steps. Autoregressive predictors impose sequential dependency that can bias learning toward local self-dynamics, while masked prediction allows the model to attend jointly to the entire observable context.

### Training Objective

The masked latent prediction loss operates over the full history-future interval $\mathcal{T} = \{t - T_h + 1, \ldots, t + T_p\}$:

$$\mathcal{L}_{\text{mask}} = \mathbb{E}\left[\sum_{\tau \in \mathcal{T}} \sum_{i=1}^{N} \mathbf{1}[\bar{z}_\tau^i \neq z_\tau^i] \|\hat{z}_\tau^i - z_\tau^i\|_2^2\right]$$

This decomposes into two complementary terms:

$$\mathcal{L}_{\text{mask}} = \underbrace{\mathbb{E}\left[\|\hat{z}_\tau^i - z_\tau^i\|_2^2 \mid i \in \mathcal{M}, \tau \leq t\right]}_{\mathcal{L}_{\text{history}}} + \underbrace{\mathbb{E}\left[\|\hat{Z}_\tau - Z_\tau\|_2^2 \mid \tau > t\right]}_{\mathcal{L}_{\text{future}}}$$

- **History term** ($\mathcal{L}_{\text{history}}$): suppresses reliance on trivial self-dynamics by forcing the predictor to recover masked objects from other objects' states. This is where the causal inductive bias enters.
- **Future term** ($\mathcal{L}_{\text{future}}$): enforces alignment with standard forward world modeling, enabling downstream planning via latent rollout.

Together, object-level masking makes interaction reasoning *functionally necessary* for minimizing the prediction objective.

### Inference

At inference time, C-JEPA operates as a standard forward predictor: given a fully observable history window $S_T$ (no masking), it predicts future object states $\{S_{t+1}, \ldots, S_{t+T_p}\}$. Masking is applied only during training as a structured regularization mechanism.

### Training Details

- **CLEVRER**: history window $T_h = 6$, prediction horizon $T_p = 10$, 7 object slots, frames subsampled with stride 2, masking between 0 and 4 objects
- **Push-T**: history window $T_h = 3$, prediction horizon $T_p = 1$, 4 object slots, frame skip 5, masking between 0 and 2 objects
- All experiments conducted on a single GPU with pre-extracted object embeddings
- Trained for 30 epochs with Adam optimizer, batch size 256, learning rate $5 \times 10^{-4}$

---

## Theoretical Analysis

### Formal Assumptions

The theoretical analysis rests on four assumptions:

1. **Temporally Directed Predictive Dependencies**: Object-level state transitions are governed by time-directed predictive dependencies from past observations and auxiliary variables. No instantaneous causal effects within the same time step.
2. **Shared Transition Mechanism**: The conditional distribution of future states given a finite history is invariant across trajectories (standard stationarity).
3. **Object-Aligned Latent Representation**: Each slot corresponds to a coherent object-level state variable sufficient for reasoning about object dynamics.
4. **Finite-History Sufficiency**: A finite history window $T_h$ suffices for prediction. This is weaker than first-order Markov -- velocity, for example, requires multiple frames.

Importantly, the authors explicitly do *not* assume causal sufficiency, full observability, or global sparsity of dynamics.

### Influence Neighborhoods (Definition 1)

For a masked object state $z_t^i$, its **influence neighborhood** $\mathcal{N}_t(i)$ is defined as the minimal sufficient subset of the observable context $Z_T^{(-i)}$ (all variables except object $i$'s own history, apart from the identity anchor) such that:

$$p(z_t^i \mid Z_T^{(-i)}) = p(z_t^i \mid \mathcal{N}_t(i))$$

This captures the smallest set of contextual variables that must be consulted to predict a masked object's state. It is related to Markov blankets and causal parents but is deliberately weaker: influence neighborhoods are *predictively sufficient* sets under masking, not estimates of true causal mechanisms. They do not assume a correct causal graph, causal sufficiency, or full observability.

### Theorem 1 (Interaction Necessity under Masked History Completion)

The optimal predictor for the masked history prediction loss satisfies:

$$\hat{z}_t^{i*} = \mathbb{E}[z_t^i \mid Z_T^{(-i)}] = \mathbb{E}[z_t^i \mid \mathcal{N}_t(i)]$$

Any predictor that fails to utilize information in $\mathcal{N}_t(i)$ incurs strictly higher expected reconstruction error. This proves that *ignoring interaction-relevant variables is suboptimal* under the C-JEPA objective -- the masking makes it so.

### Corollary 1 (Discovery of Intervention-Stable Influence Neighborhoods)

Optimizing $\mathcal{L}_{\text{mask}}$ under repeated exposure to diverse object-level masking patterns encourages state-dependent attention patterns that align with the influence neighborhood $\mathcal{N}_t(i)$. This can be interpreted as a soft, local relational structure capturing predictive influence. The connection to invariant causal prediction (Peters et al., 2016) and invariant risk minimization (Arjovsky et al., 2020) is made explicit: object-level masking acts as a collection of latent interventions under which intervention-stable influence neighborhoods emerge.

### Remark 3 (Transfer of Bidirectional Training to Forward Prediction)

Because the masked prediction objective is bidirectional (attending to both past and future context), the learned influence neighborhood $\mathcal{N}_t(i)$ abstracts away temporal direction -- it captures variables jointly informative about an object's state regardless of whether the information comes from past or future observations. The transition to forward-only prediction at inference time is valid because the influence neighborhood is direction-agnostic by construction.

---

## Results

### Visual Reasoning on CLEVRER (Table 1, VQA accuracy %)

C-JEPA is evaluated on CLEVRER, a synthetic video QA benchmark with multi-object interactions, using ALOE (Ding et al., 2021) as the downstream reasoning module. Results compare C-JEPA against OC-JEPA (same architecture, no history masking) to isolate the effect of the masking objective:

| Model | \|M\| | Avg per que. (%) | Counterfact. per opt. | Counterfact. per que. |
|---|---|---|---|---|
| OC-JEPA (V) | 0 | 82.79 | 79.53 | 47.68 |
| C-JEPA (V) | 1 | 83.95 | 80.34 | 49.67 |
| C-JEPA (V) | 2 | 84.56 | 80.61 | 50.25 |
| C-JEPA (V) | 3 | 87.61 | 86.49 | 63.60 |
| C-JEPA (V) | 4 | **89.40** | **88.67** | **68.81** |
| OC-JEPA (S) | 0 | 77.28 | 76.69 | 41.10 |
| C-JEPA (S) | 2 | **83.88** | **85.16** | **60.19** |

(V) = VideoSAUR encoder, (S) = SAVi encoder.

Key findings:
- **Counterfactual reasoning sees the largest gains**: +21.13 percentage points (per question) with VideoSAUR at $|\mathcal{M}|=4$, from 47.68% to 68.81%. This aligns with the training mechanism -- object-level masking explicitly poses counterfactual-like queries.
- **Performance scales with masking budget**: More masked objects generally improves performance, confirming that the causal inductive bias strengthens with more interventions. The one exception is SAVi at $|\mathcal{M}|=4$, where excessive masking degrades performance, suggesting the optimal budget depends on encoder quality.
- **Gains come from the objective, not just the representation**: The OC-JEPA vs. C-JEPA comparison controls for representation type -- both use the same object-centric encoder. The improvement is entirely attributable to the masking objective.

### Baseline Comparison Without Reconstruction (Table 2)

Using the SAVi encoder, C-JEPA is compared against SlotFormer and OCVP-Seq under both reconstruction and non-reconstruction settings:

| Model | Avg per que. (%) | Counterfact. per opt. | Counterfact. per que. |
|---|---|---|---|
| SlotFormer | 79.44 | 79.28 | 47.29 |
| SlotFormer (-recon.) | 44.94 | 55.62 | 11.10 |
| OCVP-Seq | 83.11 | 83.21 | 56.06 |
| OCVP-Seq (-recon.) | 80.09 | 77.46 | 43.00 |
| OC-JEPA | 77.28 | 76.69 | 41.10 |
| **C-JEPA** | **83.88** | **85.16** | **60.19** |

Removing reconstruction causes SlotFormer to collapse (-34.50 points average) and OCVP-Seq to degrade moderately (-3.02 points). C-JEPA achieves the best performance *without any reconstruction*, demonstrating that the masking-based objective provides a stronger supervisory signal for interaction reasoning than pixel reconstruction.

### Predictive Control on Push-T (Table 3)

C-JEPA is evaluated on the Push-T robotic manipulation benchmark using model-predictive control (MPC) with Cross-Entropy Method (CEM) planning:

| Model | # Token x d | Success Rate (%) |
|---|---|---|
| DINO-WM | 196 x 384 | 91.33 |
| DINO-WM (reg.) | 196 x 384 | 88.00 |
| OC-DINO-WM | 6 x 128 | 60.67 (ref.) |
| OC-JEPA | 6 x 128 | 76.00 (+15.33) |
| **C-JEPA** | **6 x 128** | **88.67 (+28.00)** |

Key findings:
- **C-JEPA closes the gap with patch-based DINO-WM** (88.67% vs. 91.33%) while using **only 1.02% of the latent token space** (6 x 128 = 768 vs. 196 x 384 = 75,264 features). This represents a ~98x reduction in feature dimensionality.
- **Clear progression from patch to object-centric**: OC-DINO-WM (60.67%) shows that simply replacing patches with object slots hurts performance without an appropriate objective. Adding the JEPA predictor (OC-JEPA: 76.00%) helps, and adding object-level masking (C-JEPA: 88.67%) recovers most of the performance.
- **8x faster planning**: Under identical hardware (single L40s GPU), C-JEPA completes planning in 673 seconds (average across 3 seeds, 50 trajectories) vs. 5,763 seconds for DINO-WM, a >8x speedup. This is because predictor rollouts dominate planning cost, and attention over 6 object tokens is dramatically cheaper than over 196 patch tokens.

### Masking Strategy Ablation (Table A4)

Three masking strategies are compared on CLEVRER with matched masking budgets:

| Strategy | Budget | Avg (%) | Counterfact. per que. |
|---|---|---|---|
| Object | 1/7 | 83.95 | 49.67 |
| Object | 4/7 | 89.40 | 68.81 |
| Token | 14% | 85.69 | 56.92 |
| Token | 56% | 89.32 | 68.88 |
| Tube | 14% | 86.62 | 57.83 |
| Tube | 56% | 89.46 | 69.81 |

At matched budgets, all three strategies achieve similar top-line performance. However, object-level masking provides **more stable and controllable** behavior: token and tube masking exhibit higher sensitivity to the masking budget and can introduce unintended combinations of missing information (e.g., simultaneously masking parts of multiple objects). On Push-T (Table A5), the differences are starker: at 50% budget, object masking achieves 82.67% success vs. only 5.33% for tube masking, showing that masking entire objects is critical for inducing meaningful interaction-aware learning signals in control settings.

### Auxiliary Variable Integration (Figure 3)

Treating actions and proprioception as separate conditioning tokens (auxiliary conditioning) consistently outperforms concatenating them into object latents:

| Configuration | OC-JEPA | C-JEPA ($|\mathcal{M}|=1$) | C-JEPA ($|\mathcal{M}|=2$) |
|---|---|---|---|
| Auxiliary Conditioning | 76.00 | 77.33 | **88.67** |
| Latent Concatenation | 65.33 | 73.33 | 82.67 |

The +6 point gap at $|\mathcal{M}|=2$ validates the design of treating auxiliary variables as explicit separate entities.

---

## Comparison to Prior Work

| | **C-JEPA** | **[[maes-2026-arxiv]] LeWorldModel** | **DINO-WM** | **SlotFormer** | **OCVP-Seq** |
|---|---|---|---|---|---|
| Representation | Object slots | CLS token (single vector) | Patch tokens | Object slots | Object slots |
| Encoder | Frozen (VideoSAUR/SAVi) | End-to-end ViT | Frozen DINOv2 | Pretrained SAVi | Pretrained SAVi |
| Predictor | Masked Transformer (bidirectional) | Autoregressive Transformer | Autoregressive Transformer | Autoregressive rolloutor | Autoregressive Transformer |
| Anti-collapse | Frozen encoder | SIGReg | Frozen encoder | Reconstruction loss | Reconstruction loss |
| Masking | Object-level (structured) | None | None | None | None |
| Interaction bias | Yes (from masking objective) | No | No | No | Architectural factorization |
| Tokens per frame | ~4-7 | 1 | ~196 | ~7 | ~6 |
| Needs reconstruction | No | No | No | Yes (critical) | Yes (helpful) |
| Push-T success | 88.67% | 90% | 91.33% | -- | -- |
| Planning speed | 8x vs DINO-WM | 48x vs DINO-WM | 1x (baseline) | -- | -- |

**vs [[lecun-2022-openreview]] (JEPA position paper):** LeCun 2022 proposed predicting in representation space and outlined object-centric structure as a key ingredient for world models. C-JEPA is a direct instantiation of this vision: it operates on object-level representations, predicts in latent space, and uses structured masking to induce causal reasoning -- all properties LeCun argued for but did not implement.

**vs [[assran-2023-cvpr]] (I-JEPA):** I-JEPA applies masked prediction to image patches for self-supervised image representation learning. C-JEPA lifts the masking unit from patches to objects and shifts the domain from static images to video dynamics. The key conceptual difference is that masking a patch is a spatial operation (predict missing visual content), while masking an object is a *causal intervention* (predict an entity's state from other entities' states).

**vs [[bardes-2024-tmlr]] (V-JEPA):** V-JEPA extends masked prediction to spatiotemporal tubes in video, learning representations for action recognition. C-JEPA's masking operates on semantically meaningful units (objects) rather than geometric regions (tubes). The ablation in Table A4/A5 directly compares these: tube masking performs comparably on CLEVRER but collapses on Push-T (5.33% vs. 82.67%), showing that semantic alignment of the masking unit matters for control.

**vs [[maes-2026-arxiv]] (LeWorldModel):** LeWorldModel solves a different problem in the JEPA world model stack -- stable end-to-end training from pixels using SIGReg. C-JEPA builds on the same codebase and shares authors (Le Lidec, Maes, LeCun, Balestriero) but addresses a complementary challenge: making the world model *interaction-aware* through object-level structure and masking. LeWorldModel uses a single CLS token per frame (maximally compressed); C-JEPA uses ~4-7 object slots (structured compression). LeWorldModel trains end-to-end; C-JEPA uses a frozen encoder. The two approaches target different points in the efficiency-structure trade-off.

**vs [[balestriero-2025-iclr]] (LeJEPA):** LeJEPA provides the theoretical foundation (SIGReg, isotropic Gaussian optimality) that underlies both LeWorldModel and C-JEPA's training infrastructure. C-JEPA's codebase is built on `stable-pretraining`. While LeJEPA focuses on static image SSL, C-JEPA extends the JEPA principle to sequential, object-centric, interaction-aware world modeling.

**vs SlotFormer (Wu et al., 2023):** SlotFormer performs autoregressive rollouts over object latents with no explicit interaction constraints. It relies heavily on pixel reconstruction as a training signal -- removing reconstruction causes a 34-point collapse (Table 2). C-JEPA achieves stronger results without any reconstruction, using masking as the sole mechanism for interaction learning.

**vs OCVP-Seq (Villar-Corrales et al., 2023):** OCVP-Seq factorizes self-dynamics and interactions architecturally (at the attention level). C-JEPA makes interaction reasoning necessary through the *objective* rather than the architecture, which is more flexible and robust -- OCVP-Seq's performance degrades when reconstruction is removed, while C-JEPA never uses it.

---

## Strengths

- **Principled causal inductive bias without causal graphs**: Object-level masking induces interaction reasoning through the learning objective, avoiding the need for explicit causal graph assumptions, specialized architectures, or ground-truth interaction labels.
- **Dramatic efficiency gains for planning**: ~98x fewer latent features and >8x faster planning than patch-based DINO-WM, with only a ~3 percentage-point performance gap on Push-T (88.67% vs. 91.33%).
- **Strong counterfactual reasoning**: ~20% absolute improvement on CLEVRER counterfactual questions demonstrates that the masking objective directly strengthens causal/counterfactual understanding.
- **Reconstruction-free design**: Unlike SlotFormer and OCVP-Seq, C-JEPA requires no pixel decoder or reconstruction loss, simplifying the architecture and making it purely representation-focused.
- **Formal theoretical backing**: Theorem 1 and Corollary 1 provide principled justification for why object-level masking induces interaction learning, connecting to invariant causal prediction and Markov blankets.
- **Flexible auxiliary variable integration**: Clean separation of object slots and auxiliary tokens (actions, proprioception) as distinct entity types, validated empirically to outperform concatenation.

---

## Weaknesses & Limitations

- **Dependence on encoder quality**: Performance is bounded by the quality of the frozen object-centric encoder. The VideoSAUR and SAVi encoders provide reasonable object decompositions on CLEVRER and Push-T, but scaling to more complex real-world scenes with occlusion, deformable objects, or large numbers of entities remains untested.
- **Synthetic/simple benchmarks only**: Both CLEVRER (synthetic rigid-body physics) and Push-T (2D planar manipulation) are relatively simple environments. The method has not been validated on realistic visual environments with richer dynamics.
- **No direct validation of influence neighborhoods**: While the theoretical framework defines influence neighborhoods and proves their optimality under masking, the paper does not directly validate them on datasets with explicit temporal causal graphs. The theoretical claims are demonstrated only indirectly through downstream task performance.
- **Frozen encoder limits adaptability**: Unlike [[maes-2026-arxiv]] LeWorldModel, which trains end-to-end, C-JEPA freezes the encoder. This prevents the representation from adapting to the dynamics task, potentially leaving performance on the table.
- **Masking budget is a hyperparameter**: The optimal number of masked objects varies across tasks and encoders (e.g., $|\mathcal{M}|=4$ is best for VideoSAUR on CLEVRER but $|\mathcal{M}|=2$ is best for SAVi). No principled method for selecting the masking budget is provided.
- **Small gap to patch-based baselines on control**: C-JEPA (88.67%) does not quite match DINO-WM (91.33%) on Push-T, suggesting that the information lost in the object-centric compression is not fully recovered by the masking objective.

---

## Key Takeaways

- **Object-level masking is a latent intervention**: The central contribution is reinterpreting masked prediction at the object level as a causal intervention on observability. This is a clean, principled mechanism for inducing interaction reasoning without any of the typical machinery (causal graphs, graph neural networks, architectural factorization).
- **The objective matters more than the representation**: The OC-JEPA vs. C-JEPA comparison cleanly demonstrates that object-centric representations alone do not produce interaction-aware world models -- the masking objective is what makes the difference. This is the paper's most important empirical finding.
- **Object-centric JEPA enables efficient planning**: Reducing from ~196 patch tokens to ~4-7 object slots yields >8x speedup in model-predictive control with minimal performance loss. For real-time control applications, this trade-off is highly favorable.
- **Counterfactual reasoning is the primary beneficiary**: The largest performance gains are consistently on counterfactual questions (~20% absolute), confirming that the masking objective specifically strengthens the ability to reason about "what would happen if" -- a core requirement for causal world models.
- **Bridges JEPA and object-centric worlds**: C-JEPA is, to the authors' knowledge, the first work integrating JEPA with object-centric world modeling. Combined with [[maes-2026-arxiv]] LeWorldModel (which provides end-to-end JEPA world model training), the Balestriero/LeCun group is building a comprehensive stack for JEPA-based world models spanning from SSL foundations ([[balestriero-2025-iclr]]) to pixel-level dynamics ([[maes-2026-arxiv]]) to object-centric causal reasoning (C-JEPA).

---

## BibTeX

{% raw %}
```bibtex
@article{nam2026causaljepa,
  title={Causal-JEPA: Learning World Models through Object-Level Latent Interventions},
  author={Nam, Heejeong and Le Lidec, Quentin and Maes, Lucas and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2602.11389},
  year={2026}
}
```
{% endraw %}

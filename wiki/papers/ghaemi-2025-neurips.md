---
title: "seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models"
type: paper
paper_id: P022
authors:
  - "Ghaemi, Hafez"
  - "Muller, Eilif B."
  - "Bakhtiari, Shahab"
year: 2025
venue: NeurIPS 2025
arxiv_id: "2505.03176"
url: "https://arxiv.org/abs/2505.03176"
pdf: "../../raw/ghaemi-2025-neurips.pdf"
tags: [JEPA, world-model, autoregressive, self-supervised, invariance, equivariance]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
cited_by: []
---

# seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models

> **seq-JEPA** is a self-supervised world modeling framework that processes short sequences of action-observation pairs through a joint-embedding predictive architecture with sequential inductive biases, simultaneously learning architecturally distinct representations for invariance-demanding tasks (e.g., classification) and equivariance-demanding tasks (e.g., transformation prediction) -- without explicit equivariance losses or dual predictors.

**Authors:** Hafez Ghaemi (Universite de Montreal, Mila, CHU Sainte-Justine)\*, Eilif B. Muller (Universite de Montreal, Mila, CHU Sainte-Justine)\*{=equal contribution}, Shahab Bakhtiari (Universite de Montreal, Mila){=equal contribution} | **Venue:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) | **arXiv:** [2505.03176](https://arxiv.org/abs/2505.03176)

---

## Problem & Motivation

Joint-embedding self-supervised learning (SSL) methods typically operate on two views of an image and enforce either **invariance** or **equivariance** to the transformations applied between views. Invariant methods (SimCLR, BYOL, VICReg) learn representations that discard transformation information -- good for classification but unable to capture fine-grained distinctions (e.g., color-invariant representations cannot distinguish bird species that differ only by plumage color). Equivariant methods (SEN, EquiMod, SIE, ContextSSL) preserve transformation-specific information -- good for tasks requiring spatial or geometric awareness but often underperforming on classification.

A growing body of theoretical and empirical work highlights a **fundamental trade-off**: models that capture equivariance-related style latents do not fare well in classification, and vice versa. Existing equivariant SSL methods require specialized loss terms, dual equivariance predictors, or explicitly crafted objectives to learn equivariant representations -- adding architectural complexity and limiting flexibility.

Inspired by how humans and animals learn from *sequences* of actions and consequent observations (e.g., rotating an object to see its other side), the authors ask: can sequential processing with action conditioning naturally disentangle invariant and equivariant representations without any explicit mechanism?

---

## Core Idea

seq-JEPA combines the [[lecun-2022-openreview|JEPA]] prediction-in-representation-space paradigm with inductive biases for sequential processing. Instead of comparing two views, the model processes a **short sequence of transformed views** (observations), where each view is paired with an embedding of the **relative transformation** (action) that produced it from the previous view. A transformer encoder aggregates these action-observation pairs into a summary representation, and a predictor head conditions on the next action to predict the next observation's representation.

The key insight is that this architecture produces **two architecturally distinct representation types without being instructed to do so**:

1. **Individual encoder representations** (z_i) become transformation/action-equivariant -- they encode what transformation was applied.
2. **Aggregate representations** (z_AGG from the transformer's [AGG] token) become transformation-invariant -- aggregating multiple views eliminates transformation-specific variability.

This dual structure emerges purely from the architecture and action-conditioned sequential prediction, not from any explicit loss terms or dual predictor heads.

---

## How It Works

### Overview

seq-JEPA takes a sequence of M+1 views {x_1, ..., x_{M+1}} generated from a single sample x via transformations {t_1, ..., t_{M+1}}. The relative transformations (actions) are a_i = delta(t_i, t_{i+1}). The model predicts the representation of x_{M+1} from the sequence of preceding view-action pairs plus the final action a_M.

### Architecture Components

**Backbone encoder f:** Encodes the first M views into representations {z_1, ..., z_M}. All models use ResNet-18. The final view x_{M+1} is encoded by a target encoder (EMA of f) to produce the prediction target z_{M+1}.

**Action embeddings:** Each relative transformation a_i is encoded via a learnable linear projector into a 128-dimensional embedding (default). These are concatenated with the corresponding view representations.

**Sequence aggregator g (transformer encoder):** A lightweight transformer with 3 layers and 4 attention heads processes the sequence of (z_i, a_i) pairs for i = 1, ..., M-1 plus z_M (the last view has no associated action). It uses a learnable [AGG] token (analogous to [CLS] in ViT) that attends over all view-action pairs and produces the aggregate representation z_AGG:

```
z_AGG = g((z_1, a_1), (z_2, a_2), ..., (z_{M-1}, a_{M-1}), z_M)
```

**Predictor h (MLP):** A 2-layer MLP with 1024 hidden units and ReLU activation. Takes z_AGG concatenated with the final action embedding a_M and predicts the target representation:

```
z_hat_{M+1} = h(z_AGG, a_M)
```

### Training Objective

The ground truth target z_{M+1} is computed by the EMA target encoder and passed through a stop-gradient operator. The loss maximizes cosine similarity:

```
L_seq-JEPA = 1 - (z_hat_{M+1} / ||z_hat_{M+1}||_2) . (sg(z_{M+1}) / ||sg(z_{M+1})||_2)
```

No additional loss terms for equivariance, no dual predictors, no explicit equivariance-specific training objectives. The entire system uses a single cosine-similarity prediction loss.

### Why Dual Representations Emerge

Action conditioning in the transformer is the critical ingredient. When action embeddings are concatenated with view representations and processed sequentially, the encoder learns to encode transformation-specific information into individual z_i (equivariance), because this information is needed together with actions to predict the next observation. Meanwhile, the [AGG] token aggregates across multiple views, naturally averaging out transformation-specific variability and retaining only content information (invariance). Ablating action conditioning causes equivariance to drop sharply (R^2 from 0.71 to 0.29) while classification accuracy remains high, confirming that action conditioning is the mechanism enabling equivariant learning.

### Action and Observation Sets

The framework is evaluated across three types of action-observation setups:

**3DIEBench (3D Invariant Equivariant Benchmark):** 3D object renderings with variations in rotation, floor hue, and lighting. Actions correspond to relative differences in these three factors. Primarily used to study SO(3) rotation equivariance.

**Hand-crafted augmentations:** Standard SSL augmentations (crop, color jitter, blur) on CIFAR-100 and Tiny ImageNet. Actions correspond to the relative augmentation parameters.

**Predictive Learning across Saccades (PLS):** A biologically inspired setting where seq-JEPA learns from sequences of 32x32 patches extracted from full-resolution images (STL-10), simulating saccadic eye movements. No hand-crafted augmentations or masking are used. Actions correspond to the relative 2D positions between patch centers. Two biologically inspired sampling techniques are used:
- **Saliency-based fixation sampling:** DeepGaze IIE saliency maps guide probabilistic fixation-point selection.
- **Inhibition of Return (IoR):** Zeroing sampling probability near previously visited fixations reduces redundancy.

### Training Setup

- **Backbone:** ResNet-18 for all experiments
- **Action embedding:** Learnable linear projector, 128-d (default)
- **Sequence aggregator:** 3-layer transformer, 4 attention heads
- **Predictor:** 2-layer MLP, 1024 hidden units, ReLU
- **Optimizer:** AdamW for models with transformer projectors; Adam for ConvNet-only baselines
- **Batch size:** 512
- **Epochs:** 1000 (3DIEBench), 2000 (other datasets)
- **Target encoder:** EMA of backbone encoder

### Evaluation Protocol

- **Equivariance:** Linear regressor trained on frozen encoder representations to predict relative transformations between two views. Reported as R^2 score. Retrieval metrics (MRR, Hit@1, Hit@5) also reported.
- **Invariance:** Top-1 classification accuracy via linear probe on frozen aggregate representations (z_AGG for seq-JEPA, encoder outputs for baselines).

---

## Results

### 3DIEBench (Table 1, Figure 3)

The primary benchmark for evaluating the invariance-equivariance trade-off. Methods are conditioned on rotation.

| Method | Top-1 Acc. (%) | Rel. Rot. (R^2) | Indiv. Rot. (R^2) |
|---|---|---|---|
| *Invariant* | | | |
| BYOL | 82.90 | 0.12 | 0.25 |
| SimCLR | 81.13 | 0.35 | 0.54 |
| VICReg | 80.48 | 0.20 | 0.36 |
| VICRegTraj | 81.26 | 0.27 | 0.43 |
| *Equivariant* | | | |
| SEN | 83.43 | 0.35 | 0.57 |
| EquiMod | 84.29 | 0.32 | 0.55 |
| SIE | 77.49 | 0.58 | 0.62 |
| Conditional BYOL | 82.61 | 0.31 | 0.47 |
| ContextSSL (c=126) | 80.40 | **0.74** | **0.78** |
| *Invariant + Equivariant* | | | |
| seq-JEPA (1,1) | 84.08 | 0.65 | 0.69 |
| seq-JEPA (1,3) | 85.31 | 0.65 | 0.69 |
| seq-JEPA (3,3) | 86.14 | 0.71 | 0.74 |
| **seq-JEPA (3,5)** | **87.41** | 0.71 | 0.74 |
| seq-JEPA (no act cond) | 86.05 | 0.29 | 0.37 |

Subscripts denote (M_tr, M_val) -- training and inference sequence lengths. seq-JEPA achieves the best classification accuracy (87.41%) while matching the best rotation R^2 scores. Invariant methods (BYOL, VICReg) achieve decent classification but poor equivariance. Equivariant methods (SIE, ContextSSL) achieve good rotation prediction but lower classification. seq-JEPA occupies the Pareto-optimal corner in Figure 3, strong in both simultaneously.

Increasing inference sequence length (M_val) improves classification accuracy, as more views contribute to a more invariant aggregate representation. Ablating action conditioning retains classification but destroys equivariance.

### Hand-Crafted Augmentations (Table 2)

On CIFAR-100 and Tiny ImageNet, seq-JEPA is evaluated with conditioning on crop, color jitter, blur, or all three.

Key findings:
- seq-JEPA consistently achieves higher equivariance (R^2) than both invariant and equivariant baselines across all transformation types.
- Best equivariance for a given augmentation is achieved when the model is specialized and conditioned only on that augmentation.
- Classification performance is competitive with or exceeds baselines.
- Ablating actions causes equivariance to collapse across all transformations.

### Predictive Learning across Saccades (Table 3)

On STL-10, seq-JEPA learns visual representations from patch sequences without any hand-crafted augmentations -- a biologically inspired setting.

| Method | Top-1 Acc. (%) | Position (R^2) |
|---|---|---|
| SimCLR (augmentations) | 85.23 | -0.06 |
| Conv-JEPA (M_val = 4) | 80.04 | 0.80 |
| seq-JEPA | 70.45 | 0.38 |
| seq-JEPA (position cond.) | 83.44 | 0.80 |
| seq-JEPA (M_val = 6) | 84.12 | 0.80 |
| seq-JEPA (w/o saliency & IoR) | 79.85 | 0.88 |
| seq-JEPA (w/o IoR) | 77.97 | 0.85 |

With position conditioning and increased sequence length, seq-JEPA reaches 84.12% top-1, approaching SimCLR's 85.23% -- notably without any hand-crafted augmentations. Saliency-based sampling and IoR are critical for high classification accuracy. 2D positional equivariance is essential for forming semantic representations across saccades.

### Path Integration (Figure 6)

seq-JEPA naturally supports path integration -- predicting cumulative transformations from action sequences -- in both:
- **Visual path integration** (saccades): R^2 remains high across increasing observation distances, degrading gracefully.
- **Angular path integration** (3D rotations): Strong performance over multiple rotated views.

Ablating action conditioning causes path integration to fail entirely, while ablating the visual stream has only minor impact -- confirming actions are the dominant signal for this capability.

### Action Conditioning Ablations (Section 4.5)

| Ablation | Top-1 Acc. (%) | Rel. Rot. (R^2) |
|---|---|---|
| Full seq-JEPA (3,3) | 86.14 | 0.71 |
| No action conditioning | 86.05 | 0.29 |
| Action in transformer only | ~intermediate | ~intermediate |
| Action in predictor only | ~intermediate | ~intermediate |

Predictor conditioning is more critical than transformer conditioning for equivariance. Action embedding dimensionality saturates around 128-d; even 16-d or 64-d captures the rotation structure adequately.

### Scaling Properties (Section 4.6)

Both training and inference sequence lengths improve performance:
- Longer training sequences (M_tr) yield better equivariance.
- Longer inference sequences (M_val) yield better classification (more invariant aggregate representations).
- This mirrors the context-length scaling behavior observed in foundation models for text and vision.

---

## Comparison to Prior Work

| | **seq-JEPA** | **JEPA** ([[lecun-2022-openreview]]) | **LeWorldModel** ([[maes-2026-arxiv]]) | **LeJEPA** ([[balestriero-2025-iclr]]) | **DreamerV3** ([[hafner-2023-arxiv]]) |
|---|---|---|---|---|---|
| Core paradigm | Sequential action-conditioned JEPA | Conceptual framework for prediction in repr. space | End-to-end JEPA with SIGReg for planning from pixels | Provably collapse-free JEPA via SIGReg | Generative world model (RSSM) for RL |
| Action role | Explicit action embeddings drive equivariance | Actions used by world model for planning | Actions are continuous control inputs for MPC | No action concept (static SSL) | Actions are discrete/continuous RL controls |
| Invariance/equivariance | Both simultaneously via architecture | Not addressed (conceptual) | Not addressed (planning focus) | Invariance only (isotropic Gaussian target) | Neither (latent-state prediction) |
| Collapse prevention | EMA target encoder + stop-gradient | VICReg proposed | SIGReg (provable) | SIGReg (provable) | Reconstruction loss + symlog |
| Prediction target | Next view in latent space | Next state in latent space | Next state in latent space | Same-image multi-view prediction | Next state in RSSM |
| Sequential processing | Yes (transformer over action-obs pairs) | Yes (proposed conceptually) | Yes (multi-step rollouts) | No (two-view comparison) | Yes (RSSM recurrence) |
| Backbone | ResNet-18 | Unspecified | ViT (various scales) | Any (ViT, ResNet, etc.) | CNN encoder |
| Scale of experiments | ResNet-18, 3DIEBench/CIFAR/STL-10 | None (position paper) | ViT up to 300M, DMControl/Atari | Up to ViT-g (1.8B) on ImageNet | Various RL benchmarks |

**vs [[lecun-2022-openreview]] (JEPA/H-JEPA position paper):** LeCun 2022 proposed JEPA as the conceptual framework for world models that predict in representation space and use actions to condition predictions. seq-JEPA is an explicit realization of this vision in the SSL domain: it processes action-observation sequences through a joint-embedding predictive architecture. However, seq-JEPA focuses on representation learning rather than planning -- it does not implement the full six-module cognitive architecture or hierarchical planning. The sequential prediction with action conditioning is directly aligned with LeCun's proposal that world models should predict future states conditioned on actions.

**vs [[maes-2026-arxiv]] (LeWorldModel):** LeWorldModel builds on LeJEPA's SIGReg to create an end-to-end JEPA world model for RL with multi-step latent rollouts and MPC planning. Both papers implement the JEPA world model concept, but from different angles: LeWorldModel targets control and planning in RL environments, while seq-JEPA targets representation learning and the invariance-equivariance trade-off in SSL. seq-JEPA's action conditioning creates equivariant representations as an emergent property; LeWorldModel's action conditioning serves the planning objective.

**vs [[balestriero-2025-iclr]] (LeJEPA):** LeJEPA solves the collapse problem for JEPAs with a provably sufficient regularizer (SIGReg) but operates in the standard two-view SSL setting with no sequential or action-conditioned processing. It targets invariant representations only. seq-JEPA uses the traditional EMA + stop-gradient approach to collapse prevention (not provably sufficient) but gains the ability to learn equivariant representations simultaneously through its sequential architecture. The two approaches are complementary: one could potentially combine SIGReg with seq-JEPA's sequential architecture.

**vs [[hafner-2023-arxiv]] (DreamerV3):** DreamerV3 is a generative world model (RSSM) trained with reconstruction losses for model-based RL. seq-JEPA is a non-generative (joint-embedding) world model trained with prediction-in-representation-space losses for self-supervised representation learning. The key conceptual difference is exactly what LeCun 2022 argued: seq-JEPA avoids pixel-space reconstruction and instead learns abstract representations. DreamerV3 learns action-conditioned dynamics for planning; seq-JEPA learns action-conditioned representations for downstream tasks. DreamerV3 operates on much more complex environments (Atari, DMControl, Minecraft) but does not explicitly study representation quality via linear probes.

---

## Strengths

- **Elegant architectural solution to the invariance-equivariance trade-off:** Rather than introducing loss terms, dual predictors, or explicit group-theoretic constraints, seq-JEPA achieves both properties through architectural inductive biases alone -- action conditioning plus sequential aggregation.
- **Strong empirical performance on both invariance and equivariance tasks simultaneously:** On 3DIEBench, seq-JEPA is the only method occupying the Pareto-optimal corner (best classification + best rotation prediction).
- **Biologically inspired saccade-based learning:** The PLS setting demonstrates that seq-JEPA can learn visual representations from sequences of patches without any hand-crafted augmentations, drawing a connection to how primates build representations through active exploration.
- **Natural support for path integration:** The ability to predict cumulative transformations over sequences emerges naturally from the sequential architecture, without task-specific modifications.
- **Scaling with sequence length:** Performance improves with both training and inference sequence lengths, offering a principled knob for trading compute for quality.
- **Thorough ablation studies:** The paper systematically ablates action conditioning (in transformer vs. predictor), action embedding dimensionality, sequence lengths, saliency sampling, and IoR, providing clear understanding of each component's contribution.

---

## Weaknesses & Limitations

- **Limited backbone scale:** All experiments use ResNet-18, a relatively small backbone. It is unclear how the invariance-equivariance trade-off and the architectural disentanglement behave at ViT-Large or ViT-Huge scale, where most modern SSL methods are evaluated.
- **Small-scale datasets only:** Evaluations are on 3DIEBench, CIFAR-100, Tiny ImageNet, and STL-10. No experiments on ImageNet-1K or other large-scale benchmarks, making it difficult to compare absolute numbers with state-of-the-art SSL methods.
- **No provable collapse prevention:** Unlike [[balestriero-2025-iclr|LeJEPA]], seq-JEPA relies on the standard EMA target encoder + stop-gradient approach, which has no formal guarantee against representational collapse.
- **Limited action/transformation types:** The equivariance evaluation focuses on SO(3) rotations, 2D positions, and standard augmentations. It is unclear how the framework handles more complex or compositional transformations.
- **Classification gap in saccade setting:** In PLS without augmentations, seq-JEPA (84.12%) trails SimCLR with augmentations (85.23%), suggesting that augmentation-free learning from saccades does not yet fully close the gap.
- **Computational overhead of sequential processing:** Processing sequences of M views plus a transformer encoder adds overhead compared to standard two-view SSL methods. The paper does not report training time comparisons.
- **No downstream fine-tuning results:** All evaluations use frozen features with linear probes or regressors. Fine-tuning performance is not assessed.

---

## Key Takeaways

- **Action conditioning is the key to emergent equivariance:** Without explicit equivariance losses, the simple act of concatenating action embeddings with view representations and processing them sequentially causes individual encoder representations to become equivariant while aggregated representations remain invariant.
- **Sequential processing resolves the invariance-equivariance trade-off:** By processing variable-length sequences rather than fixed two-view pairs, seq-JEPA can simultaneously excel at tasks requiring either invariance or equivariance -- a capability no prior SSL method achieved.
- **The predictor is more important than the transformer for equivariance:** Ablations show that action conditioning in the MLP predictor contributes more to equivariance than action conditioning in the transformer aggregator.
- **Augmentation-free representation learning is viable through saccade simulation:** seq-JEPA demonstrates that SSL can work without hand-crafted augmentations by simulating biologically plausible sequential observation strategies, opening a path toward more naturalistic self-supervised learning.
- **Sequence length is a new scaling dimension for SSL:** Analogous to context length in language models, both training and inference sequence lengths offer a way to improve representation quality, with longer inference sequences yielding more invariant (classification-friendly) representations.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{ghaemi2025seqjepa,
  title={seq-{JEPA}: Autoregressive Predictive Learning of Invariant-Equivariant World Models},
  author={Ghaemi, Hafez and Muller, Eilif B. and Bakhtiari, Shahab},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={arXiv:2505.03176}
}
```
{% endraw %}

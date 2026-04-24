---
title: "HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling"
type: paper
paper_id: P060
authors:
  - "Jaber, Jaber"
  - "Jaber, Osama"
year: 2026
venue: arXiv
arxiv_id: "2603.29090"
url: "https://arxiv.org/abs/2603.29090"
pdf: "../../raw/jaber-2026-arxiv.pdf"
tags: [world-model, object-centric, slot-attention, ssm, causal-discovery, hierarchical-dynamics, jepa]
created: 2026-04-15
updated: 2026-04-15
cites:
  - hafner-2023-arxiv
  - bardes-2024-tmlr
  - lecun-2022-openreview
cited_by: []
---

# HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling

> **One sentence.** HCLSM is a five-layer world model architecture that unifies object-centric decomposition (slot attention), hierarchical temporal dynamics (SSM + sparse event transformer + goal transformer), and causal structure learning (GNN with DAG regularization) in a single differentiable model, achieving 0.008 MSE next-state prediction on PushT with emerging spatial decomposition and learned event boundaries.

**Authors:** Jaber Jaber, Osama Jaber (RightNow AI) | **Venue:** arXiv 2026 | **arXiv:** [2603.29090](https://arxiv.org/abs/2603.29090)

---

## Problem & Motivation

Current world models that predict future latent states from video (V-JEPA, DreamerV3) use flat latent representations that entangle all objects, ignore causal structure, and collapse temporal dynamics into a single scale. This has three consequences: (1) the model cannot represent individual objects separately, preventing counterfactual reasoning ("what if the gripper had pushed harder?"); (2) different timescales of dynamics -- continuous physics, discrete events, strategic goals -- are forced into a single temporal mechanism; (3) without explicit causal structure, the model cannot distinguish correlation from causation in object interactions. No prior system combines all three -- object decomposition, hierarchical temporal dynamics, and causal reasoning -- in a single differentiable architecture.

---

## Core Idea

HCLSM's key insight is that "structure must precede prediction": if all losses are active from the start of training, the JEPA prediction objective dominates and prevents slots from specializing to individual objects. The solution is a two-stage training protocol inspired by biological visual development -- first learn what things are (spatial reconstruction via a broadcast decoder), then learn what they do (temporal dynamics via JEPA-style prediction). This allows slot attention to develop object-specific representations before dynamics prediction begins. The architecture itself stacks five layers: perception, object decomposition, hierarchical dynamics (three temporal scales), causal reasoning, and continual memory.

---

## How It Works

### Overview

Video frames (B, T, C, H, W) -> frozen ViT encoder (Layer 1) -> slot attention with spatial broadcast decoder (Layer 2) -> per-object SSM + sparse event transformer + goal transformer (Layer 3) -> causal GNN with DAG regularizer (Layer 4) -> Hopfield memory + EWC (Layer 5). The spatial broadcast decoder provides the training signal in Stage 1; JEPA prediction loss is added in Stage 2.

### Layer 1: Perception

A Vision Transformer encoder processes video frames into patch embeddings (B, T, M, d_model) where M = (H/p)^2 patches (p=16). Temporal position embeddings are added per-frame. A linear projection maps from d_model to d_world, the unified representation dimension.

### Layer 2: Object Decomposition

**Slot Attention with Dynamic Birth/Death:** N_max = 32 slot proposals initialized from a learned Gaussian. Slots compete for patch tokens through softmax attention over the slot dimension for K iterations. Each slot gets an existence head predicting p_alive in [0,1]; dormant slots can be "born" when residual attention energy (uncaptured tokens) exceeds a threshold.

**Spatial Broadcast Decoder (SBD):** Each slot is independently broadcast to a 14x14 spatial grid with (x,y) coordinates, decoded by a 4-layer CNN into feature predictions plus an alpha mask. Alpha masks are softmax-normalized over alive slots, creating pixel-level competition. The reconstruction target is frozen ViT features from an EMA target encoder (DINOSAUR-style), providing a semantic signal rather than pixel-level targets:

L_SBD = sum_n sum_p alpha_{n,p} * ||f_hat_{n,p} - f*_p||^2

**Relation Graph:** A GNN processes all-pairs edge features [o_i; o_j; o_i - o_j; o_i * o_j] through an edge MLP, producing weighted messages per-node. For N > 32 slots, chunked computation processes edges in blocks of 16 to limit memory.

### Layer 3: Hierarchical Dynamics

**Level 0 -- Selective SSM (Continuous Physics):** Each object gets its own SSM track with shared parameters. The selective scan h_t = exp(Delta_t * A) * h_{t-1} + B_t * x_t, y_t = C_t^T * h_t processes per-object continuous dynamics. A custom Triton kernel achieves 38x speedup over sequential PyTorch.

**Level 1 -- Sparse Event Transformer:** An event detector monitors multi-scale temporal features (frame differences at scales 1, 2, 4) through causal dilated convolutions. When the event score exceeds a learned threshold, that timestep is gathered into a dense event tensor. A standard transformer with SwiGLU processes only the K << T event timesteps, with cost O(K * N^2) instead of O(T * N^2).

**Level 2 -- Goal Compression Transformer:** Learned summary queries cross-attend to the event sequence, compressing it into n_summary abstract state tokens. A goal-level transformer processes these, optionally conditioned on language/goal embeddings.

**Hierarchy Manager:** Vectorized gather/scatter operations combine three levels with learned per-level gating weights.

### Layer 4: Causal Structure

A causal adjacency matrix W in R^{N x N} is learned with Gumbel-softmax binary sampling, L1 sparsity, and a NOTEARS DAG constraint h(A) = tr(e^{A*A}) - N = 0 enforced via augmented Lagrangian optimization. GNN edge weights provide the primary causal structure signal.

### Layer 5: Continual Memory

Modern Hopfield networks for content-addressable episodic storage plus Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention.

### Two-Stage Training Protocol

- **Stage 1 (first 40% of training):** Only SBD reconstruction loss + diversity regularizer. Prediction loss is computed for monitoring but does not produce gradients. This forces slots to specialize spatially.
- **Stage 2 (remaining 60%):** Full JEPA prediction loss is activated alongside SBD (now as a regularizer with weight 1.0 instead of 5.0). Dynamics model learns to predict futures of already-decomposed objects.

### Training Details

- **Model size:** 68M parameters (HCLSM Small)
- **Data:** PushT from LeRobot / Open X-Embodiment (206 episodes, 25,650 frames)
- **Compute:** NVIDIA H100 80GB GPUs, batch size 4, lr = 1.5e-4 (cosine + 2K warmup), bfloat16, 50K steps (~6 hours per run)
- **Stability fixes:** Replace x**2 with x*x (bf16 PowBackward0 NaN), initialize SSM A_log in [-0.5, 0], clamp activations to [-50, 50], disable GradScaler
- **GPU-native slot tracking:** Differentiable Sinkhorn-Knopp replaces CPU Hungarian matching

---

## Results

### Quantitative (PushT, 68M params, 50K steps)

| Configuration | Pred. loss | Track. loss | Diversity loss | SBD loss | Total loss | Speed |
|---|---|---|---|---|---|---|
| HCLSM (no SBD) | **0.002** | **0.001** | 0.154 | -- | 0.100 | 2.3 sps |
| HCLSM (two-stage) | 0.008 | 0.016 | **0.132** | **0.008** | 0.262 | 2.9 sps |

Without SBD, prediction loss is lower (0.002) because all 32 slots encode the scene distributively -- an easier prediction target. With two-stage training, the SBD loss reaches 0.008, indicating individual slots have learned to reconstruct specific spatial regions. Diversity loss is lower with SBD (0.132 vs 0.154), confirming slots are more differentiated.

### Spatial Decomposition

The spatial broadcast decoder's alpha masks show that different slots claim different spatial regions. The decomposition is not yet clean (each object is split across multiple slots for a 3-object scene), but represents the first emergence of object-aware representations in the model.

### Event Detection

The learned event detector identifies 2-3 events per 16-frame PushT sequence, corresponding to moments of significant state change (e.g., contact between robot and T-block).

### Triton SSM Kernel Performance

| Config | B x N | T | Sequential | Triton | Speedup |
|---|---|---|---|---|---|
| Tiny | 128 | 16 | 6.22 ms | 0.16 ms | **39.3x** |
| Base | 512 | 16 | 69.64 ms | 1.83 ms | **38.0x** |

---

## Comparison to Prior Work

| System | Objects | Hierarchy | Causal | SSM | Real Data |
|---|---|---|---|---|---|
| V-JEPA 2 | No | No | No | No | Yes |
| **[[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md))** | No | No | No | Yes | Yes |
| SlotFormer | Yes | No | No | No | No |
| DINOSAUR | Yes | No | No | No | Yes |
| **HCLSM (this)** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

**vs [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)):** DreamerV3 uses flat latent states (RSSM) without object decomposition or explicit causal structure. HCLSM decomposes scenes into object slots and models their interactions through a causal GNN, enabling structured world knowledge that DreamerV3's monolithic latent state cannot provide. Both handle real-world data; HCLSM adds structural inductive biases at the cost of complexity.

**vs [[bardes-2024-tmlr]] ([V-JEPA](../papers/bardes-2024-tmlr.md)):** V-JEPA predicts latent video representations via masking, treating the scene as a flat vector. HCLSM extends the JEPA prediction objective to per-object slot representations with hierarchical temporal dynamics, making it a structured JEPA world model rather than a flat one.

**vs [[lecun-2022-openreview]] ([H-JEPA](../papers/lecun-2022-openreview.md)):** LeCun's Hierarchical JEPA vision proposes multi-level temporal abstraction in a JEPA framework. HCLSM implements a concrete instantiation with three explicit temporal levels (SSM for continuous physics, sparse transformer for events, goal transformer for abstract planning), though the hierarchy is not yet deeply validated.

---

## Strengths

- **First system to combine all three:** Object-centric decomposition, hierarchical temporal dynamics, and causal structure learning in a single differentiable architecture
- **Principled two-stage training:** Biologically-inspired protocol that solves the practical problem of prediction loss dominating over slot specialization
- **Impressive engineering:** Custom Triton SSM kernel (38x speedup), GPU-native Sinkhorn slot tracking, comprehensive bf16 stability fixes -- all open-sourced (8,478 lines, 51 modules, 171 tests)
- **Emerging spatial decomposition:** SBD successfully drives slot spatial specialization on real robot data despite only 206 episodes
- **Learned event detection:** Automatically identifies discrete event boundaries without supervision

---

## Weaknesses & Limitations

- **Slot count does not adapt:** All 32 slots remain alive; the existence head fails to kill unused slots. A 3-object scene uses 32 slots, splitting objects across multiple slots rather than clean 1-to-1 assignment
- **Causal discovery does not work:** The explicit causal adjacency matrix collapses to zero under sparsity regularization. Joint training with dynamics at bf16 causes NaN. The GNN edge weights provide implicit structure but have not been verified against ground-truth causal relationships
- **Only tested on PushT:** A single task with 206 episodes is insufficient to validate claims about hierarchical temporal dynamics and causal reasoning
- **Half the runs failed:** 4 runs launched, only 2 survived due to seed-dependent NaN at bf16 -- training stability remains a significant issue
- **No planning or control evaluation:** The model is a world model but is not used for downstream planning, policy learning, or counterfactual reasoning
- **Bootstrapped nature:** The authors are transparent that this is early-stage research with significant limitations

---

## Key Takeaways

- Structure must precede prediction: activating all losses simultaneously causes the JEPA prediction loss to dominate, preventing slot specialization -- a two-stage protocol (40% reconstruction-only, then 60% reconstruction + prediction) is required
- HCLSM is the first architecture to combine object-centric slots, three-level temporal hierarchy (SSM + sparse event transformer + goal transformer), and causal GNN in a single differentiable model
- Emerging spatial decomposition on real robot data (PushT) demonstrates the viability of the approach, but clean object assignment and causal discovery remain unsolved challenges
- A custom Triton kernel for the selective SSM scan achieves 38x speedup, making per-object temporal processing practical; open-source implementation at [github.com/rightnow-ai/hclsm](https://github.com/rightnow-ai/hclsm)

---

## BibTeX

{% raw %}
```bibtex
@article{jaber2026hclsm,
  title     = {{HCLSM}: Hierarchical Causal Latent State Machines for Object-Centric World Modeling},
  author    = {Jaber, Jaber and Jaber, Osama},
  journal   = {arXiv preprint arXiv:2603.29090},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.29090}
}
```
{% endraw %}

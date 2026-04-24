---
title: "Evaluating World Models"
type: topic
tags: [world-model, evaluation, benchmarks, myhill-nerode, physics, 3D, survey, metrics]
created: 2026-04-10
updated: 2026-04-10
papers: [vafa-2024-neurips, ding-2024-csur, kong-2025-arxiv]
---

# Evaluating World Models

> Evaluating whether a generative model has learned a genuine world model — as opposed to a sophisticated pattern matcher — is a distinct and underappreciated problem. Standard metrics (next-token accuracy, task success rate, FID/FVD) are insensitive to the structural properties that world models require: causal consistency, state representation completeness, physical plausibility, and counterfactual correctness.

## Background

The evaluation problem for world models has two dimensions. The first is *functional*: does the model's output support downstream tasks like planning, policy training, or data generation? This is addressed by task-specific benchmarks (Atari 100k, DMControl, autonomous driving closed-loop metrics). The second is *structural*: does the model's internal behavior reflect a valid, consistent model of the world — i.e., does it correctly track world state rather than pattern-match on training distributions?

[[vafa-2024-neurips]] ([Evaluating WM](../papers/vafa-2024-neurips.md)) addresses the structural question directly, proving that standard metrics fail to detect world model learning and introducing a theoretically grounded alternative based on the Myhill-Nerode theorem. The core finding is striking: a transformer achieving 1.00 next-token test accuracy on NYC shortest-path routing has sequence compression precision of only 0.10, meaning it fails in 90% of cases to produce the same continuation distributions from two prefixes that arrive at the same intersection.

[[ding-2024-csur]] ([Comprehensive Survey](../papers/ding-2024-csur.md)) surveys the broader evaluation landscape, identifying the lack of a canonical benchmark spanning both implicit (MBRL) and generative (video) world models as one of the field's most pressing open problems. Specialized benchmarks (Physics-IQ, T2VPhysBench, VBench-2.0, EWMBench) expose specific failure modes but lack cross-domain coverage.

[[kong-2025-arxiv]] ([3D/4D Survey](../papers/kong-2025-arxiv.md)) addresses the evaluation of 3D and 4D world understanding — geometric consistency, dynamic scene reconstruction, and spatial reasoning — highlighting additional failure modes that 2D image/video metrics miss entirely.

## Key Approaches

### Myhill-Nerode Behavioral Evaluation

Grounded in formal language theory, this framework evaluates whether a generative model's output distribution satisfies the necessary and sufficient conditions for world model learning: (1) *sequence compression* — prefixes reaching the same world state produce identical continuation distributions; (2) *sequence distinction* — prefixes reaching different world states produce different continuation distributions. Both metrics are model-agnostic (require only sampling from the model), and can reconstruct the model's implicit world graph for topological comparison against ground truth.

Key papers: [[vafa-2024-neurips]] ([Evaluating WM](../papers/vafa-2024-neurips.md))

### Physics Evaluation

The [[ding-2024-csur]] ([Comprehensive Survey](../papers/ding-2024-csur.md)) highlights Physics-IQ as a key benchmark revealing that video world models achieve high visual realism scores while systematically failing physical laws: gravity, fluid dynamics, and thermal processes are incorrectly modeled even by Sora-class systems. The survey argues that data-driven scaling cannot recover physical laws and calls for hybrid physics-data approaches.

Key papers: [[ding-2024-csur]] ([Comprehensive Survey](../papers/ding-2024-csur.md))

### 3D / 4D Spatial Evaluation

[[kong-2025-arxiv]] ([3D/4D Survey](../papers/kong-2025-arxiv.md)) argues that world models must ultimately represent 3D structure and 4D (spatio-temporal) dynamics, not just 2D video sequences. It surveys evaluation approaches for geometric reconstruction quality, novel-view synthesis consistency, dynamic scene understanding, and spatial reasoning in open-world environments.

Key papers: [[kong-2025-arxiv]] ([3D/4D Survey](../papers/kong-2025-arxiv.md))

### Downstream Task Benchmarks

For RL world models, standard evaluation is downstream policy performance: Mean Human Normalized Score on Atari 100k (see [atari-100k benchmark](../benchmarks/atari-100k.md)), normalized reward on DMControl (see [dmcontrol benchmark](../benchmarks/dmcontrol.md)), closed-loop success rate for autonomous driving. These are task-effective but structurally blind: a model can score well here while failing Myhill-Nerode criteria.

Key papers: [[ding-2024-csur]] ([Comprehensive Survey](../papers/ding-2024-csur.md))

## Open Questions

- **Structural vs. functional trade-off**: Can a model achieve high downstream task performance while failing structural world model criteria? Evidence from Vafa et al. suggests yes — implying that task benchmarks underestimate evaluation requirements.
- **Extending Myhill-Nerode to continuous worlds**: The DFA formulation requires a finite state space. Extending compression/distinction metrics to physical systems with continuous state is an open theoretical problem.
- **Canonical cross-domain benchmark**: No single benchmark evaluates world models across MBRL, video generation, navigation, and robotics domains simultaneously.
- **Counterfactual evaluation**: World models should support not just forward prediction but counterfactual reasoning ("what would have happened if I had turned right?"). No systematic benchmark for this exists.
- **Training distribution vs. world model quality**: The Othello experiment in Vafa et al. shows that training data distribution (uniform random vs. championship games) determines world model quality more than model capacity. How to curate training data for world model learning is unresolved.
- **Evaluation of foundation world models**: As world models scale to billions of parameters trained on internet video (Cosmos, Sora), evaluation must generalize beyond held-out clips from the same distribution to genuinely novel physical scenarios.

## Timeline

- **2022**: LeCun argues for representation-space prediction as a proxy for world model quality — [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md))
- **2024**: Myhill-Nerode framework for behavioral world model evaluation — [[vafa-2024-neurips]] ([Evaluating WM](../papers/vafa-2024-neurips.md))
- **2024**: Comprehensive survey of evaluation benchmarks and their gaps — [[ding-2024-csur]] ([Comprehensive Survey](../papers/ding-2024-csur.md))
- **2025**: 3D/4D world model evaluation landscape — [[kong-2025-arxiv]] ([3D/4D Survey](../papers/kong-2025-arxiv.md))

## Papers

| Paper | Year | Key Idea |
|-------|------|----------|
| [[vafa-2024-neurips]] ([Evaluating WM](../papers/vafa-2024-neurips.md)) | 2024 | Myhill-Nerode compression/distinction; next-token accuracy ≠ world model; GPT-4 compression 0.21 |
| [[ding-2024-csur]] ([Comprehensive Survey](../papers/ding-2024-csur.md)) | 2024 | Benchmark gap; physics failure modes; understanding vs. prediction taxonomy |
| [[kong-2025-arxiv]] ([3D/4D Survey](../papers/kong-2025-arxiv.md)) | 2025 | 3D/4D geometric consistency evaluation; spatial reasoning; dynamic scenes |

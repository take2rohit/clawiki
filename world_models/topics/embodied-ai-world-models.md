---
title: "Embodied AI World Models"
type: topic
tags: [world-model, embodied-ai, robotics, navigation, manipulation, sim-to-real, JEPA, foundation-model]
created: 2026-04-10
updated: 2026-04-10
papers: [li-2025-arxiv, mazzaglia-2024-neurips, bar-2024-cvpr, lecun-2022-openreview]
---

# Embodied AI World Models

> Embodied AI world models enable robots and navigation agents to reason about physical consequences of their actions without executing them in the real world. They range from compact implicit representations learned for policy optimization (as in model-based RL) to generative video simulators that produce realistic action-conditioned future observations for robot skill learning and sim-to-real transfer.

## Background

Embodied AI has historically relied on classical physics simulators (MuJoCo, PyBullet, Isaac Gym) for training robot policies. While effective in narrow domains, these simulators struggle to capture contact-rich manipulation, cloth physics, and the visual diversity of real environments. Learned world models offer a data-driven alternative that can generalize to visually complex, unstructured settings.

Two architectural paradigms dominate the space. The first is the implicit representation approach: world models in model-based RL (see the RL World Models topic) build compact latent dynamics models from real interaction data and use them for planning or imagination-based policy training. [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md)) extends this paradigm by grounding world model latent states in a pretrained VLM, enabling multi-task and data-free policy learning from language or visual prompts without language annotations.

The second paradigm, proposed most influentially by LeCun in [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md)), argues that world models should predict in abstract representation space rather than pixel space. The JEPA (Joint Embedding Predictive Architecture) learns encoders that are invariant to irrelevant details, predicting future representations rather than future pixels. This is argued to be superior to generative models (which must model noise and detail) and to contrastive SSL (which requires exponentially many negatives in high dimensions). V-JEPA (Meta, 2024) and V-JEPA 2 realize this vision for video understanding.

For navigation specifically, [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md)) demonstrates that a single action-conditioned video predictor trained on diverse environments (indoor, outdoor, driving) can serve as a general-purpose navigation planner.

[[li-2025-arxiv]] ([Embodied AI Survey](../papers/li-2025-arxiv.md)) surveys the full landscape of world models for embodied intelligence, covering manipulation, locomotion, and navigation.

## Key Approaches

### Implicit World Models for Robot Policy Learning

Following the Dreamer paradigm, compact latent-space world models can be trained on robot interaction data and used to optimize policies in imagination. This avoids expensive real robot trials and can incorporate diverse sensory modalities. The key innovation in [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md)) is connecting the latent space to a pretrained VLM so that task goals can be specified via language, images, or video without domain-specific language annotations — a major practical bottleneck in robotics.

Key papers: [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md))

### JEPA and Non-Generative World Models

LeCun's [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md)) proposes a fundamentally different world model architecture: instead of decoding predictions back to pixel space (as Dreamer and Ha & Schmidhuber do), JEPA predicts in abstract representation space. Encoders learn to discard irrelevant stochastic details, producing representations where future prediction is tractable. H-JEPA stacks multiple JEPA levels to yield hierarchical, multi-scale world state representations suitable for hierarchical planning. Non-contrastive SSL (VICReg/Barlow Twins) prevents representational collapse without requiring explicit negative samples.

Key papers: [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md))

### Navigation World Models

Navigation agents require a different type of world model: given an egocentric observation and a navigation action (turn, go forward), predict the resulting observation. [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md)) trains a single action-conditioned video predictor across diverse real-world environments including indoor scenes, outdoor urban areas, and driving scenarios. The resulting model can be used to plan multi-step navigation sequences by imagining action outcomes without access to a map.

Key papers: [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md))

### Survey of Embodied AI World Models

[[li-2025-arxiv]] ([Embodied AI Survey](../papers/li-2025-arxiv.md)) provides a taxonomy of world models for embodied intelligence, organizing approaches by the type of implicit representation learned, whether the model generates future environments, and how it supports sim-to-real transfer. Key systems covered include DayDreamer (real-world locomotion learned in hours), UniSim (dynamic action-conditioned video), and V-JEPA 2 (MPC-based control from JEPA representations).

Key papers: [[li-2025-arxiv]] ([Embodied AI Survey](../papers/li-2025-arxiv.md))

## Open Questions

- **Sim-to-real transfer**: Generative video world models trained on real data narrow the gap vs. physics simulators, but contact forces, deformable objects, and lighting physics remain challenging.
- **Language-vision alignment without annotations**: GenRL shows one path (connector-aligner with VLMs), but the modality gap between internet VLMs and embodied observation spaces is a persistent challenge.
- **JEPA vs. generative models in practice**: LeCun's argument for representation-space prediction is theoretically compelling but empirical comparisons in embodied domains at scale are limited.
- **Long-horizon planning with hierarchical world models**: H-JEPA proposes hierarchical prediction but practical implementations at multiple timescales remain nascent.
- **Contact-rich manipulation**: Current video world models produce plausible manipulation videos but do not reliably model forces, deformation, or tool use at the level needed for planning fine manipulation.
- **Open-world generalization**: World models trained on specific robot platforms and tasks do not easily generalize to new embodiments or scenes.

## Timeline

- **2022**: JEPA cognitive architecture and representation-space world models — [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md))
- **2024**: Action-conditioned navigation world models (NWM) — [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md))
- **2024**: VLM-grounded world model for multi-task embodied RL (GenRL) — [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md))
- **2025**: Embodied AI world model survey — [[li-2025-arxiv]] ([Embodied AI Survey](../papers/li-2025-arxiv.md))

## Papers

| Paper | Year | Key Idea |
|-------|------|----------|
| [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md)) | 2022 | Prediction in representation space; H-JEPA; non-contrastive SSL; 6-module architecture |
| [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md)) | 2024 | Action-conditioned video prediction for navigation across diverse environments |
| [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md)) | 2024 | VLM connector-aligner; data-free policy learning; 0.80 on 35 tasks |
| [[li-2025-arxiv]] ([Embodied AI Survey](../papers/li-2025-arxiv.md)) | 2025 | Survey of world models for embodied intelligence (manipulation, locomotion, navigation) |

---
title: "Foundation World Models"
type: topic
tags: [world-model, foundation-model, video-generation, scaling, physical-ai, JEPA, open-weight, diffusion, autoregressive]
created: 2026-04-10
updated: 2026-04-10
papers: [agarwal-2025-arxiv, lecun-2022-openreview, bar-2024-cvpr, mazzaglia-2024-neurips]
---

# Foundation World Models

> Foundation world models are large-scale, general-purpose world models pre-trained on massive, diverse datasets — analogous to foundation models in language and vision — intended to be fine-tuned or prompted for a wide range of downstream physical AI tasks. They represent the convergence of two previously distinct research threads: the compact latent dynamics models of model-based RL and the generative video simulators of the video generation community.

## Background

The term "foundation model" entered the world model vocabulary around 2023–2025 as it became clear that world model capabilities scale with data and compute in ways analogous to language models. Three influences drove this:

1. **Scale of video generation**: Sora (OpenAI, 2024), Cosmos (NVIDIA, 2025), and similar systems demonstrated that training on tens of millions of hours of video with transformer or diffusion architectures produces systems with emergent physical intuition — objects persist, scenes are 3D-consistent, and physics is roughly maintained.

2. **LeCun's JEPA blueprint**: [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md)) argued that the correct foundation for a world model is not pixel generation but prediction in abstract representation space, learned via non-contrastive self-supervised learning. This influenced I-JEPA, V-JEPA, and V-JEPA 2 from Meta, and positioned SSL as a viable pre-training objective for world models — without requiring domain-specific rewards.

3. **Multi-task scaling in RL world models**: [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md)) showed that a generative world model for RL can be connected to a pretrained VLM via a small connector-aligner module, enabling zero-shot and data-free policy learning for new tasks without language annotations. This demonstrated that foundation VLMs can serve as a generalizing backbone for task specification in embodied RL.

[[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md)) showed that a single action-conditioned video predictor trained across diverse environments generalizes to new scene types at test time — an early example of a "foundation" video world model for navigation.

## Key Approaches

### Large Pre-trained Video World Models (Physical AI)

[[agarwal-2025-arxiv]] ([Cosmos](../papers/agarwal-2025-arxiv.md)) is the most comprehensive current example: a platform with two model families (diffusion: 7B/14B; autoregressive: 4B/12B), a curated 100M-clip training dataset from 20M hours of diverse video, an open-weight release, and post-training recipes for three Physical AI domains (camera control, robotic manipulation, autonomous driving). The key design choices are: (1) a causal tokenizer that enables joint image+video training; (2) separate diffusion and autoregressive families to trade off quality vs. controllability; (3) post-training to specialize for specific physical domains.

Key papers: [[agarwal-2025-arxiv]] ([Cosmos](../papers/agarwal-2025-arxiv.md))

### Representation-Space Foundation (JEPA)

[[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md)) proposes that foundation world models should predict in latent representation space rather than pixel space. The JEPA encoder learns to discard unpredictable details (lighting, texture) while preserving predictable structure (object positions, trajectories). Hierarchical H-JEPA stacks these encoders to produce multi-scale world representations. The configurator module provides a blueprint for how a single pre-trained world model engine can be repurposed for diverse tasks — a direct precursor to the foundation model paradigm.

Key papers: [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md))

### VLM-Grounded Foundation World Models for RL

[[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md)) demonstrates a practical path to foundation world models in embodied RL: pre-train a generative world model on diverse interaction data, then connect it to a pre-trained VLM (InternVideo2) via learned connector-aligner modules using only visual data. The resulting Multimodal-Foundation World Model (MFWM) enables data-free policy learning for new tasks within 30 minutes by imagining latent states internally without any real environment interaction.

Key papers: [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md))

### Diverse-Domain Action-Conditioned Models

[[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md)) demonstrates generalization across indoor, outdoor, and driving environments from a single action-conditioned video predictor. This is foundational in the sense that the model is not specialized to any single domain and can be queried for navigation planning in novel environments at test time.

Key papers: [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md))

## Open Questions

- **Scaling laws for world models**: Do world model capabilities (causal reasoning, physical correctness, counterfactual accuracy) scale predictably with data and compute, as language capabilities do in LLMs? Early evidence from Cosmos and DreamerV3 is encouraging but incomplete.
- **Evaluation at scale**: FID/FVD measure perceptual quality, not physical correctness. No consensus evaluation suite exists for foundation world models spanning physical plausibility, causal reasoning, and counterfactual correctness.
- **Pixel generation vs. representation prediction**: LeCun's JEPA argues against generative (pixel-space) world models, while Cosmos and similar systems double down on pixel generation. The empirical question of which paradigm produces better physical AI has not been definitively resolved.
- **Compositionality and scene graph grounding**: Foundation world models must represent multiple interacting objects. Current video generators lack reliable object permanence; compositional scene understanding is an open problem.
- **Fine-tuning efficiency**: Cosmos requires post-training on domain-specific data; the amount of domain data needed to specialize a 7B foundation model for a new robotic task is not well characterized.
- **Open-source vs. proprietary gap**: Cosmos is open-weight but required 10,000 H100 GPUs; academic groups cannot replicate the pre-training. Bridging the resource gap is a structural challenge for the field.

## Timeline

- **2022**: JEPA blueprint for representation-space foundation world model — [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md))
- **2024**: Cross-domain action-conditioned navigation world model — [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md))
- **2024**: VLM-grounded MFWM for data-free embodied RL (GenRL) — [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md))
- **2025**: Open-weight Physical AI foundation world model platform (Cosmos) — [[agarwal-2025-arxiv]] ([Cosmos](../papers/agarwal-2025-arxiv.md))

## Papers

| Paper | Year | Key Idea |
|-------|------|----------|
| [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md)) | 2022 | Representation-space prediction; H-JEPA; non-contrastive SSL; configurable world model |
| [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md)) | 2024 | Single action-conditioned model generalizing across diverse environments |
| [[mazzaglia-2024-neurips]] ([GenRL](../papers/mazzaglia-2024-neurips.md)) | 2024 | VLM-grounded world model; data-free learning; connector-aligner modules |
| [[agarwal-2025-arxiv]] ([Cosmos](../papers/agarwal-2025-arxiv.md)) | 2025 | Open-weight WFM platform; 20M hours training; diffusion + AR; post-training for Physical AI |

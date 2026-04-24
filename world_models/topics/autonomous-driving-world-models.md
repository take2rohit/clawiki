---
title: "Autonomous Driving World Models"
type: topic
tags: [world-model, autonomous-driving, video-generation, bev, multi-view, trajectory-conditioning, simulation]
created: 2026-04-10
updated: 2026-04-10
papers: [hu-2023-arxiv, min-2024-cvpr, feng-2025-arxiv, agarwal-2025-arxiv, bar-2024-cvpr]
---

# Autonomous Driving World Models

> Autonomous driving (AD) world models learn to simulate realistic, controllable driving scenarios — either as latent representations for planning, or as video generators producing photorealistic future frames conditioned on ego-vehicle trajectories, map data, and sensor inputs. They serve two complementary purposes: generating synthetic training data to reduce reliance on expensive real-world collection, and acting as a learned simulator for closed-loop policy evaluation.

## Background

Traditional autonomous driving stacks decompose the problem into perception → prediction → planning → control, each module trained independently. World models offer a unified alternative: a single learned model of driving dynamics that can either generate realistic synthetic scenarios (data engine role) or predict future states for planning (agent brain role). The key challenge is producing temporally consistent, multi-view-coherent, physically plausible video conditioned on controllable signals (ego trajectory, weather, traffic agents).

[[hu-2023-arxiv]] ([GAIA-1](../papers/hu-2023-arxiv.md)) from Wayve was among the first to demonstrate a large-scale generative world model for autonomous driving, using an autoregressive transformer conditioned on video, text, and action tokens to produce realistic driving videos. [[min-2024-cvpr]] ([DriveWorld](../papers/min-2024-cvpr.md)) introduced a multi-view world model with 4D pre-training to provide structured spatial representations for end-to-end planning. [[agarwal-2025-arxiv]] ([Cosmos](../papers/agarwal-2025-arxiv.md)) from NVIDIA provides the most recent large-scale open-weight platform, post-training multi-view diffusion models on driving data to achieve FID 32.16 and FVD 210.23 vs. VideoLDM-MultiView's 60.84/884.46. [[feng-2025-arxiv]] ([AD Survey](../papers/feng-2025-arxiv.md)) surveys the breadth of generation paradigms and downstream applications.

For navigation (a related but simpler domain), [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md)) learns action-conditioned video prediction in diverse real-world environments including driving scenarios.

## Key Approaches

### Large-Scale Generative Driving Models (Video World Models)

These approaches train large transformer or diffusion models on real driving video corpora to produce temporally coherent, multi-view future frames. The generation is conditioned on structured inputs: ego-vehicle trajectory (GPS/IMU), weather/lighting, text descriptions, and sometimes LiDAR or BEV maps.

Key papers: [[hu-2023-arxiv]] ([GAIA-1](../papers/hu-2023-arxiv.md)), [[agarwal-2025-arxiv]] ([Cosmos](../papers/agarwal-2025-arxiv.md))

### Multi-View Spatial World Models

Multi-camera rigs (typically 6 surrounding cameras on AV platforms) require the world model to maintain geometric consistency across viewpoints over time. DriveWorld introduces 4D pre-training — jointly learning 3D spatial structure and temporal dynamics from multi-camera video — providing richer representations for downstream planning compared to single-view generation.

Key papers: [[min-2024-cvpr]] ([DriveWorld](../papers/min-2024-cvpr.md))

### Trajectory-Conditioned Simulation

A critical use case is post-training a general video world model with trajectory conditioning: given a planned ego trajectory, the model generates what the driver would see. This enables closed-loop evaluation of planning algorithms without real-world driving. [[agarwal-2025-arxiv]] ([Cosmos](../papers/agarwal-2025-arxiv.md)) demonstrates trajectory-conditioned models achieving trajectory following error (TFE) of 20.20 cm, only 7 cm worse than real videos.

Key papers: [[agarwal-2025-arxiv]] ([Cosmos](../papers/agarwal-2025-arxiv.md)), [[hu-2023-arxiv]] ([GAIA-1](../papers/hu-2023-arxiv.md))

### Navigation World Models

At a coarser granularity than full AV stacks, navigation world models learn to predict egocentric video given navigation actions in diverse indoor and outdoor environments. These can be used to plan sequences of actions without a map by imagining the outcome of each candidate trajectory.

Key papers: [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md))

### Survey and Taxonomy

[[feng-2025-arxiv]] ([AD Survey](../papers/feng-2025-arxiv.md)) provides a comprehensive taxonomy of autonomous driving world models, organizing approaches by generation paradigm (autoregressive, diffusion, hybrid) and downstream application (data augmentation, simulation, planning).

Key papers: [[feng-2025-arxiv]] ([AD Survey](../papers/feng-2025-arxiv.md))

## Open Questions

- **Physical grounding**: Current video world models produce visually plausible but physically approximate results. Gravity, occlusion, contact dynamics, and wet/icy road behavior are not reliably modeled.
- **Closed-loop evaluation validity**: Does high video generation quality (FID/FVD) correlate with policy improvement when using the model as a simulator? This link has not been firmly established.
- **Long-horizon consistency**: Generating 10+ seconds of consistent multi-view driving video without object permanence failures remains difficult at production scale.
- **Sensor modality integration**: Most world models operate on RGB cameras; integrating LiDAR, radar, and HD maps into a unified generative model is an open problem.
- **Real-time inference**: Diffusion-based generation is expensive; making world model simulation fast enough for online planning or closed-loop training at scale requires architectural innovation.
- **Safety-critical scenario coverage**: Rare events (pedestrians in fog, sudden cut-ins) are underrepresented in training data and therefore in the learned world model distribution.

## Timeline

- **2023**: Large-scale generative driving world model (GAIA-1) — [[hu-2023-arxiv]] ([GAIA-1](../papers/hu-2023-arxiv.md))
- **2024**: Multi-view 4D spatial world model for planning (DriveWorld) — [[min-2024-cvpr]] ([DriveWorld](../papers/min-2024-cvpr.md))
- **2024**: Action-conditioned navigation world models — [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md))
- **2025**: Open-weight WFM platform with trajectory conditioning (Cosmos) — [[agarwal-2025-arxiv]] ([Cosmos](../papers/agarwal-2025-arxiv.md))
- **2025**: Comprehensive AD world model survey — [[feng-2025-arxiv]] ([AD Survey](../papers/feng-2025-arxiv.md))

## Papers

| Paper | Year | Key Idea |
|-------|------|----------|
| [[hu-2023-arxiv]] ([GAIA-1](../papers/hu-2023-arxiv.md)) | 2023 | Autoregressive transformer; video+text+action conditioning; Wayve |
| [[min-2024-cvpr]] ([DriveWorld](../papers/min-2024-cvpr.md)) | 2024 | Multi-view 4D pre-training; spatial world model for end-to-end planning |
| [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md)) | 2024 | Action-conditioned video prediction for navigation planning |
| [[agarwal-2025-arxiv]] ([Cosmos](../papers/agarwal-2025-arxiv.md)) | 2025 | Open-weight WFM; 7B/14B diffusion; trajectory conditioning; FID 32.16 |
| [[feng-2025-arxiv]] ([AD Survey](../papers/feng-2025-arxiv.md)) | 2025 | Survey of AD world models by generation paradigm and application |

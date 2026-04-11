---
title: "A Survey of World Models for Autonomous Driving"
type: paper
paper_id: P007
authors:
  - "Feng, Tuo"
  - "Wang, Wenguan"
  - "Yang, Yi"
year: 2025
venue: arXiv
arxiv_id: "2501.11260"
url: "https://arxiv.org/abs/2501.11260"
pdf: "../../raw/feng-2025-arxiv.pdf"
tags: [world-model, survey, autonomous-driving, occupancy-grid, point-cloud, diffusion, self-supervised-learning]
created: 2026-04-10
updated: 2026-04-10
cites:
  - ha-2018-neurips
  - hu-2023-arxiv
  - ding-2024-csur
  - li-2025-arxiv
  - kong-2025-arxiv
cited_by: []
---

# A Survey of World Models for Autonomous Driving

> **Survey** — Feng, Wang, and Yang organize 253+ papers on autonomous driving world models under a three-tier taxonomy (future physical world generation, behavior planning, and prediction-planning interaction), providing the most structurally detailed coverage of this domain including nine benchmark tables across 4D scene generation, occupancy forecasting, and motion planning.

**Authors:** Tuo Feng, Wenguan Wang (corresponding), Yi Yang (Zhejiang University / CCAI) | **Venue:** arXiv (submitted September 2025, v4) | **arXiv:** [2501.11260](https://arxiv.org/abs/2501.11260)

---

## Problem & Motivation

Autonomous driving requires fusing heterogeneous sensor streams (cameras, LiDAR, radar, HD maps) into a representation that supports real-time decision-making under adversarial conditions — sensor degradation in rain, unpaved roads, aggressive maneuvers, long-tail safety events. Prior surveys either treat autonomous driving as one application among many general world model domains, or provide coarse coverage that focuses only on world simulation while neglecting the interaction between prediction and planning. A dedicated, structured survey is needed that: (1) develops a comprehensive taxonomy covering all four generative output modalities (image, BEV, occupancy grid, point cloud); (2) classifies behavior planning methods (rule-based, learning-based, RL/MPC, LLM-based); and (3) addresses the critical but underexplored interaction between future prediction and behavior planning in closed-loop systems.

---

## Core Idea

A world model for autonomous driving is defined as a generative spatio-temporal neural system that encodes the physical environment into a compact latent state (jointly capturing geometry, semantics, and causal context), then rolls it forward under hypothetical actions. The formal core task is: given multi-view images I and LiDAR points P from previous T frames, predict coupled ego-trajectory τ^(T+1) and surrounding scene z^(T+1). The survey organizes the full landscape into three tiers corresponding to this definition's components: how the physical world evolves (generation), how an agent should act within it (behavior planning), and how prediction and planning interact bidirectionally (the closed-loop regime).

---

## How It Works

### Overview

The survey's eight sections follow the structure in Figure 1:
- §2: Background and formal problem definition
- §3: Taxonomy (three tiers)
- §4: Data and training paradigms (SSL, pretraining, data generation)
- §5: Application areas (scene understanding, motion prediction, simulation, end-to-end driving)
- §6: Performance comparison (nine benchmark tables)
- §7: Future research directions
- §8: Conclusion

Nine detailed tables catalogue methods with architecture, input/output modality, conditioning, and training datasets.

### Tier 1: Generation of Future Physical World (§3.1)

Four output modality tracks, shown in Figure 2:

**3.1.1 Image-based Generation**

Synthesizing photorealistic future driving video. Methods split into two families:

- *Dreamer series*: DriveDreamer (ECCV'24, nuScenes, 2D image), DriveDreamer-2 (AAAI'25, LLM-prompted diversity), DriveDreamer4D (CVPR'25, 4D spatial-temporally coherent), ReconDreamer (CVPR'25, online restoration), WorldDreamer (arxiv'24, masked-token multimodal). Trend: 2D → 4D, closed → open-ended conditioning.
- *Diffusion models*: BEVControl (ICCV'24), DrivingDiffusion (CVPR'24), Drive-WM (CVPR'24), Vista (NeurIPS'24), GEM (CVPR'24, ego-vision multimodal), BevWorld, DrivePhysica. Advance along two axes: *controllable generation* (BEV layout-guided, text-guided, optical-flow-guided) and *high-fidelity spatio-temporal modelling* (long-horizon roll-outs at higher resolution).
- *Transformer-based*: HoloDrive (multi-camera camera+LiDAR+text), BEVGen, DrivingWorld (VideoGPT-style, multimodal token sequence), CarFormer (object-centric slots). These evolve from single-view renderers to *holistic, action-aware world models*.

Table 1 catalogues 40+ image/BEV generation methods with architecture, input modality, output modality, and training dataset.

**3.1.2 BEV-based Generation**

Projecting all sensors into a bird's-eye-view lattice:
- *Early probabilistic*: FIERY (ICCV'21, first probabilistic BEV forecaster), StretchBEV, MiLi, 2DCNN.
- *Object-centric*: CarFormer (self-driving with learned object-centric representations).
- *Generative*: GenAD (CVPR'24, ego-conditioned future BEV sampling), BEVControl (editable BEV sketches for safety auditing), ViDAR (geometry-aware pre-training across board).
- Trend: from deterministic BEV prediction → multi-agent, trajectory-conditioned, efficient, user-controllable world models.

**3.1.3 OG-based Generation (Occupancy Grid)**

Divides the scene into 3D voxels, assigns occupancy probabilities. Provides richer geometric detail than BEV but at higher compute cost.
- *CNN-based*: Occ4cast, Cam4DOcc, PreWorld (semi-supervised two-stage).
- *Transformer-based*: MUVO (camera+LiDAR voxel-level Transformer), OccWorld (ECCV'24, autoregressive token-based), DFIT-OccWorld (decouples dynamic/static warping), DriveWorld (memory-augmented pretraining for planning), OccProphet, OccLLaMA, Occ-LLM (instruction-driven reasoning), GaussianWorld (Gaussian splatting), T³Former (triplane sparse attention, real-time camera-only), I²-World (intra/inter tokenization).
- *Diffusion-based*: OccSora (trajectory-resampling control), UniScene (compact 4D tokens + DiT), DynamicCity (HexPlane + DiT), COME (occupancy+flow+BEV layout).
- Table 2 catalogues 35+ OG/PC generation methods.

**3.1.4 PC-based Generation (Point Cloud / LiDAR)**

Predicts future 3D LiDAR sweeps:
- *CNN-based*: PCP (3D spatio-temporal volume), 4DOcc, PCPNet.
- *Transformer-based*: ViDAR (vision-LiDAR autoregressive pretraining), HERMES (multi-view + text → BEV → point cloud via differentiable voxel rendering).
- *Diffusion-based*: LiDARGen (score-matching on range images), CopilotD4 (VQ-VAE + discrete diffusion), RangeLDM (latent denoising with Hough-voting), LiDARCrafter (4D world modeling from text → ego-centric scene graph).
- *Other*: NeRF-LiDAR, NFL, UltraLiDAR (discrete latent tokens).

### Tier 2: Behavior Planning for Intelligent Agents (§3.2)

**3.2.1 Learning-based Planning**

Modern multimodal architectures ingest LiDAR, RADAR, GPS, camera frames to output lane-change choices, future state distributions, reference paths, or direct control commands.

- *RL & MPC Planners*: Model-free RL (PFBD using TD3), adaptive world model RL (AdaptiveDriver — closed-loop social world model + MPC; AdaWM, AdAWM), uncertainty-aware variants, latent-space planners (ThinkDrive). Model-based RL variants (Raw2Drive — dual-stream privileged/raw world models; PIWM — DreamerV3-style individual world model with inter-agent intentions). VLM integration: VL-SAFE (offline WM-RL guided by VLM safety scores), IRL-VLA (reward world model via inverse RL replacing simulator rewards).
- *LLM-based Planners*: Autoregressive multimodal transformers (DrivingGPT, VaVAM — next-token prediction over image+action tokens), BEV-centric (WoTE — BEV predictor scoring sampled trajectories online), sparse-token (SSR — 16 navigation-guided tokens), visual CoT (FSDrive — VLM imagines future frame + 3D boxes, then inverse dynamics).
- *Volume-based Planners*: Cost-volume methods (BSDNet, ST-P3, MP3, NEAT, NMP, PPF) sample kinematically feasible trajectories and rank via composite cost from learned occupancy/segmentation maps. Occupancy-volume variants (OccWorld+planning head) replace heuristic costs with learnable occupancy forecasts.

Table 3 catalogues 30+ learning-based planners.

**3.2.2 Rule-based Planning**

Four families (Table 4): car-following models (IDM and variants), sampling-based planners (RRT, RRG, Conformal Lattice, Discrete Terminal Manifold), continuous optimisation (Apollo EM Planner, Continuous-Curvature Spline), artificial potential-field navigation. Modern stacks combine these for interpretability; limitations include brittleness in dense traffic and inability to provide formal safety guarantees.

**3.2.3 Search-based Planning**

Graph-search on motion primitives (Dijkstra, A* family, Hybrid-State A*). Model-predictive A*, adaptive fidelity models, multi-heuristic search. Extends search-based planning from highways to complex urban/off-road scenarios.

### Tier 3: Interaction between Planning and Prediction (§3.3)

This tier examines the feedback loop between behavior planning and future prediction — the critical axis for closed-loop deployment. Three evolutionary regimes (Figure 3):

- *Open-loop regime*: Generative methods (DriveGAN, MagicDrive, DriveDreamer) synthesize scene videos for data augmentation but replay fixed futures without responding to online control. NAVSIM is quasi-closed-loop. Limitation: breaks the causal link between actions and subsequent observations.
- *Uncontrollable closed-loop regime*: Autoregressive driving world models (DriveDreamer, GAIA-1, Vista, WorldDreamer, OccWorld, many others) generate futures conditioned on ego actions, unifying prediction and planning. Supplies diverse data, captures uncertainty. But *latent physics are opaque* — users cannot inject rare events, edit traffic rules, or verify safety guarantees; compounding distribution drift threatens reliability.
- *Controllable closed-loop regime*: Neural-hybrid simulators (UniSim, OASim), game-engine-based (CARLA, MetaDrive, Waymax), physics-linked (Dreamland = physics simulator + video generator), multi-agent aware (Sky-Drive, DriveArena), latent-space control (World4Drive — intention-aware latent, LAW — plan-conditioned latent). Goal: fully differentiable sandbox with editable physics, multi-agent uncertainty, auto-curricula.

### Training Paradigms (§4)

**4.1 Self-Supervised Learning**

Two branches for occupancy-centric self-supervision:
- *Camera-to-3D lifting*: RenderOcc, SelfOcc, OccNeRF, H3O, World4Drive — lift multi-view images into volumetric space, learn from 2D renders + occupancy cues.
- *LiDAR self-labeling*: UnO, EO — derive self-labels by contrasting predicted 4D fields with future scans; RenderWorld, UniPAD, PreWorld — couple differentiable rendering with multi-view cameras.
- *Tokenization+diffusion*: COPILOT4D, BEVWorld — discretise multimodal data into vocabulary codes, learn discrete diffusion; AD-L-JEPA (arxiv'25) — eliminates contrastive/generative heads, directly forecasts BEV embeddings.

Drawbacks: accuracy lags fully-supervised 3D/4D baselines; volumetric rendering is memory-heavy; principled label-free 4D occupancy forecasting remains largely unexplored.

**4.2 Large-Scale Pretraining**

Three paradigms: (i) Universal vision/multimodal frameworks (DriveWorld, UniPAD, BEVWorld, ViDAR, etc.) — millions of image-LiDAR sequences in unified BEV/voxel space; (ii) LiDAR/occupancy-centric self-supervision (Occupancy-MAE, AD-L-JEPA, UnO); (iii) Foundation-level generative world models (GAIA-1, GAIA-2) — scale to hundreds of millions of frames, controllable multi-camera generation.

**4.3 Data Generation for Training**

Generator-based data pipelines offer three benefits over hand-coded simulators: coverage (extrapolates to corner cases absent from logs), fidelity (cross-view consistency, metre-accurate depth), efficiency (amortises fleet costs). Table 5 summarises six key methods (DriveDreamer-2, BEVControl, SimGen, GAIA-2, GEM, Vista, DrivingWorld, MiLA, CoGen).

---

## Results / Key Findings

**Benchmark results (§6):**

- **4D Scene Generation** (Table 6, CarlaSC / Occ3D-Waymo): DynamicCity consistently outperforms OccSora on both 2D metrics (IS↑, FID↓, KID↓) and 3D metrics (P↑, R↑). FID improves from 32.94 → 12.92 on Occ3D-Waymo.
- **Point Cloud Forecasting** (Table 7, OpenScene mini): DFIT-OccWorld-O achieves 0.70m² average Chamfer Distance vs. ViDAR's 1.58m², representing a 2.3× improvement. Token-based diffusion outperforms range-image CNN/LSTM.
- **4D Occupancy Forecasting** (Table 8, Occ3D-nuScenes): I²-World-O (SD-Occ) and T³Former-O (SD-Occ) achieve state-of-art with 49.80 and 40.40 average mIoU respectively. Camera-only T³Former-F remains competitive without 3D occupancy supervision.
- **Motion Planning** (Table 9, nuScenes): UniAD+DriveWorld reduces average L2 error by 33% and collision rate by 39% vs. UniAD alone. T³Former-O achieves best occupancy-input performance (1.00m avg L2, 0.30 avg collision). FSDrive achieves 0.28m/0.10% but requires privileged ego status supervision.

Key empirical trends:
1. **Richer supervision = better planning**: Adding world-model pretraining to perception-only planners consistently reduces L2 and collision rates by 30–40%.
2. **Occupancy-centric models remain competitive with video**: Camera-only T³Former-F rivals SD-Occ-based models, validating the trend toward annotation-efficient occupancy pipelines.
3. **Open-loop metrics can overstate safety**: Open-loop benchmarks (NAVSIM) cannot expose failures from distribution shift when the agent's own actions are not fed back to the environment.

---

## Comparison to Prior Work / Related Surveys

| Survey | Scope | Taxonomy Depth | Planning Coverage | Benchmarks |
|---|---|---|---|---|
| [[ding-2024-csur]] ([Ding et al., P006](../papers/ding-2024-csur.md)) | General (4 domains) | Medium (2 tiers) | Light | 8 tables descriptive |
| [[li-2025-arxiv]] ([Li et al., P008](../papers/li-2025-arxiv.md)) | Embodied AI / robotics | Medium (3 tiers) | Moderate | Qualitative |
| [[kong-2025-arxiv]] ([Kong et al., P009](../papers/kong-2025-arxiv.md)) | 3D/4D scenes | High (spatial) | None | Per-domain |
| **Feng et al. (P007)** | Autonomous driving | High (3 tiers × 4 modalities) | Deep (4 planners × rule/learning/RL/LLM) | 9 quantitative tables |

**vs. [[ding-2024-csur]] ([Ding et al., P006](../papers/ding-2024-csur.md))**: Feng et al. provides significantly deeper technical treatment of autonomous driving — four generation modalities vs. one section in Ding; dedicated behavior planning tier absent from Ding; nine quantitative benchmark tables vs. Ding's descriptive approach.

**vs. Guan et al. (2024, early survey)**: The survey cites Guan et al. [62] as an initial survey that categorises studies coarsely and often focuses solely on world simulation or lacks discussion on planning-prediction interaction. Feng et al. addresses both gaps.

---

## Strengths
- Most technically detailed autonomous driving world model survey; the four-modality taxonomy (image/BEV/OG/PC) is a unique organizing contribution.
- Nine quantitative benchmark tables provide side-by-side performance numbers across 4D scene generation, occupancy forecasting, and motion planning — rare in survey papers.
- Tier 3 (prediction-planning interaction) and the three-regime framework (open-loop → uncontrollable closed-loop → controllable closed-loop) is an original analytical contribution with practical deployment implications.
- Clear identification of four frontier challenges: self-supervised world models, multimodal fusion, advanced simulation, efficient architectures.
- Submitted September 2025, v4 — highly current, covering systems through mid-2025.

## Weaknesses & Limitations
- Autonomous driving only; does not address transfer of world model knowledge to other domains.
- Behavior planning section's rule-based and search-based coverage is thorough but these methods are unlikely to be competitive in the next generation of deployed systems — depth may be disproportionate.
- Physics fidelity limitations of video-based world models are acknowledged but not deeply analyzed; no dedicated section on hybrid physics-data approaches (cf. Ding et al. §6.1).
- Social and pedestrian modeling is mentioned but not systematically covered — crucial for urban deployment.
- Limited discussion of safety certification and formal verification, which are critical for regulatory approval.
- No discussion of LLM/VLM world knowledge integration (covered by Ding et al.) beyond their use as planners.

## Key Takeaways
- **The closed-loop gap is the central unsolved problem**: Open-loop benchmarks significantly overstate safety; controllable closed-loop regimes where agents can inject rare events and verify safety guarantees are the target, but current systems remain in the "uncontrollable closed-loop" regime.
- **Four output modalities serve different needs**: Image generation is richest for perception training; BEV is planning-friendly; occupancy grids provide fine-grained 3D geometry; point clouds are geometry-first. Unified multimodal systems that serve all four simultaneously are emerging (HERMES, DrivingWorld).
- **Self-supervised world model pretraining reduces annotation costs by 30–40%** and consistently improves planning performance when added to perception-only pipelines — making it a standard training recipe.
- **LLM-based planners struggle with long-horizon consistency and safety certification**: Despite competitive results on nuPlan, they demand high GPU compute and lack formal guarantees required for production deployment.
- **Data generation from world models is becoming central**: Generator-based synthetic data pipelines that condition on HD maps, trajectories, and physics priors now produce training data rivaling fleet logs for rare scenarios, making them a pillar of the autonomous driving training stack.

---

## BibTeX
```bibtex
@article{feng2025survey,
  title     = {A Survey of World Models for Autonomous Driving},
  author    = {Feng, Tuo and Wang, Wenguan and Yang, Yi},
  journal   = {arXiv preprint arXiv:2501.11260},
  year      = {2025},
  month     = {September},
  url       = {https://arxiv.org/abs/2501.11260}
}
```

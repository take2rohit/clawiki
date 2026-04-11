---
title: "A Comprehensive Survey on World Models for Embodied AI"
type: paper
paper_id: P008
authors:
  - "Li, Xinqing"
  - "He, Xin"
  - "Zhang, Le"
  - "Wu, Min"
  - "Li, Xiaoli"
  - "Liu, Yun"
year: 2025
venue: arXiv
arxiv_id: "2510.16732"
url: "https://arxiv.org/abs/2510.16732"
pdf: "../../raw/li-2025-arxiv.pdf"
tags: [world-model, survey, embodied-AI, robotics, autonomous-driving, rssm, 3DGS, NeRF, JEPA, self-supervised-learning]
created: 2026-04-10
updated: 2026-04-10
cites:
  - ha-2018-neurips
  - hafner-2019-icml
  - hafner-2021-iclr
  - hafner-2023-arxiv
  - lecun-2022-openreview
  - ding-2024-csur
  - feng-2025-arxiv
  - kong-2025-arxiv
cited_by: []
---

# A Comprehensive Survey on World Models for Embodied AI

> **Survey** — Li et al. propose a novel three-axis taxonomy (functionality × temporal modeling × spatial representation) for world models in embodied AI, covering 255+ papers across robotics and autonomous driving with eight quantitative benchmark tables spanning pixel generation, 4D occupancy forecasting, DMC control, manipulation, and motion planning.

**Authors:** Xinqing Li, Xin He, Le Zhang, Min Wu (Senior Member IEEE), Xiaoli Li (Fellow IEEE), Yun Liu (corresponding; Nankai University / A*STAR / SUTD) | **Venue:** arXiv (v2, November 2025) | **arXiv:** [2510.16732](https://arxiv.org/abs/2510.16732)

---

## Problem & Motivation

Embodied AI systems must perceive complex multimodal environments, act within them, and anticipate how actions reshape future world states — three capabilities that jointly define the frontier of robotics. World models serve as internal simulators that capture environment dynamics, enabling forward and counterfactual rollouts for perception, prediction, and decision-making. Rapid growth in both the MBRL world model literature (Dreamer series) and the large-scale generative world model literature (Sora, V-JEPA 2) has created inconsistencies in terminology, taxonomy, and evaluation standards across sub-communities. Existing surveys either take a function-oriented perspective (Ding et al.) or focus narrowly on autonomous driving (Feng et al., Guan et al.), leaving a unified treatment of embodied AI — which spans manipulation, navigation, locomotion, and driving — as an open gap. Two specific problems are unaddressed: (1) the lack of a unified dataset spanning all embodied AI domains constrains generalization, and (2) existing evaluation metrics (FID, FVD) assess pixel fidelity while ignoring physical consistency, causality, and long-horizon dynamics.

---

## Core Idea

The survey introduces a three-axis taxonomy that decomposes any world model along three orthogonal design dimensions: (1) **Functionality** — whether the model is decision-coupled (task-specific, optimized for a particular control problem) or general-purpose (task-agnostic, focused on broad predictive and generative capabilities); (2) **Temporal Modeling** — whether the model uses sequential simulation/inference (autoregressive rollout) or global difference prediction (parallel prediction of entire future states); (3) **Spatial Representation** — one of four strategies: global latent vector, token feature sequence, spatial latent grid, or decomposed rendering representation (NeRF/3DGS). This 2×2×4 structure provides a principled organizing framework that maps naturally onto concrete design choices.

---

## How It Works

### Overview

The survey follows seven sections:
- §II: Background — POMDP formulation, core concepts, mathematical framework (ELBO for world model training)
- §III: Three-axis taxonomy with detailed per-axis breakdown and two summary tables
- §IV: Data resources (simulation platforms, interactive benchmarks, offline datasets, real-world robot platforms) and evaluation metrics
- §V: Quantitative performance comparison (8 tables)
- §VI: Challenges and future directions (3 axes)
- §VII: Conclusion

Two comprehensive tables (Table I for robotics/general-purpose, Table II for autonomous driving) catalogue 80+ methods with taxonomy labels, publication venue, dataset platforms, and input modalities.

### Mathematical Framework

World models are formalized as POMDPs. The joint distribution over observations and latent states factorizes as:
```
p(o_{1:T}, z_{0:T} | a_{0:T-1}) = p(z_0) ∏_{t=1}^{T} p(z_t|z_{t-1},a_{t-1}) p(o_t|z_t)
```
Three components: (i) **Dynamics Prior** p_θ(z_t | z_{t-1}, a_{t-1}); (ii) **Filtered Posterior** q_φ(z_t | z_{t-1}, a_{t-1}, o_t); (iii) **Reconstruction** p_θ(o_t | z_t). Training via ELBO maximization: reconstruction term (faithful observation prediction) + KL regularization (align posterior with dynamics prior). This formulation admits instantiation with RNNs, Transformers, or diffusion decoders.

### Axis 1: Functionality

**A. Decision-Coupled World Models**

Task-specific models that learn dynamics optimized for a particular decision-making task.

*Sequential Simulation and Inference — Global Latent Vector:*
- Early decision-coupled models combined RNNs with global latent states. RSSM (PlaNet) = deterministic memory + stochastic components for robust long-horizon imagination. Dreamer/V2/V3 series.
- RSSM extensions for transferability: PreLAR (implicit action abstractions, bridges video pretraining and fine-tuning), Wang et al. (optical flow as embodiment-agnostic action representation), SENSEI (VLM → semantic rewards → RSSM reward propagation), SR-AIF (sparse-reward robotics), ReDRAW (sim-to-real with reward-free data), AdaWM (adaptive world model RL for planning), CALL (multi-agent RSSM with ego-centric information sharing).
- Variants: TransDreamer (Transformer replaces recurrent core), GLAM (Mamba parallel framework), GLAMOR (object-conditioned IDM), Iso-Dream (decomposes controllable/uncontrollable dynamics).

*Sequential Simulation and Inference — Token Feature Sequence:*
- MWM (masked world model — decouples visual tokens from RSSM dynamics), WISTER (action-conditioned contrastive predictive coding → TSSM), TWM (Transformer aligns multimodal tokens during training), Inner Monologue (closed-loop feedback into LLMs for reasoning/deliberation), EvoAgent (LLM guides low-level actions + regularizes RSSM updates), RoboHorizon (dense rewards + masked autoencoder).

*Sequential — Spatial Latent Grid:*
DriveDreamer/GenAD/OccWorld (GRU-based dynamics on grid/occupancy tokens), DriveWorld (RSSM on BEV), Raw2Drive (dual-stream privileged/raw world models for CARLA E2E driving), FASTTopoWM (align fast/slow systems from vehicle poses), WoTE (simulate candidate trajectories in BEV → reward model).

*Sequential — Decomposed Rendering Representation:*
ManiGaussian (per-point Gaussian motion attributes with diffusion), GAF (Gaussian action field for robotic manipulation), ManiGaussian++ (hierarchical leader-follower multi-body design), DreMa (Gaussian splat + physics simulator for digital twins in imitation learning), DrivePhysica (3DGS + cross-view point map alignment), DriveDreamer4D (complex lane-change trajectories → 4DGS), ReconDreamer (online restoration module for Gaussian-rendered artifacts).

**B. General-Purpose World Models**

Task-agnostic simulators for broad prediction and generalization.

*Sequential — Token Feature Sequence:*
iVideoGPT (pretrained on action-free videos → downstream control), Genie (discrete latent actions + spatiotemporal tokens for interactive environments), PACT (causal Transformer for unified perception+action representation), DINO-world (predicts DINOv2 feature temporal evolution for zero-shot planning), WorldVLA (tokenizes environmental states as discrete symbolic tokens for VLA agents), DCWM/TrajWorld, Statler (structured world-state maintenance + LLM reader/writer), Inner Monologue (closed-loop reasoning), NavCoT (LLM-based VLN via reasoning), MineDreamer (Chain-of-Imagination, multimodal LLM imagines future observations → guides diffusion → IDM for planning), FSDrive (FutureSightDrive — visual spatio-temporal CoT).

*Sequential — Spatial Latent Grid:*
EmbodiedDreamer (differentiable physics + video diffusion for photorealistic and physically consistent futures), TesserAct (jointly generates RGB + depth + normal for IDM-based action learning), DFIT-OccWorld (decoupled voxel warping + image-assisted single-stage training), RoboDreamer (decomposes instructions into low-level video diffusion primitives), ManipDreamer (action-tree prior + depth/semantic guidance).

*Sequential — Decomposed Rendering:*
GAF (Gaussian action field), DriveDreamer4D, AETHER (dynamic 4D reconstruction + action-conditioned video prediction + vision-based planning on synthetic 4D data → zero-shot real generalization), MiniGaussian++ (multi-body deformations).

*Global Difference Prediction — Token Feature Sequence:*
TOKEN (tokenizes traffic scenes into object-level tokens for long-tail autonomous driving), GeoDrive (extracts 3D representations, renders trajectory-conditioned views), FLARE (aligns diffusion policies with latent future representations, avoiding pixel-space video), LaDi-WM (interactive diffusion in latent space aligned with visual foundation models), villa-X (latent actions aligned with ego-centric forward dynamics via joint diffusion), VidMan (adapts pretrained video diffusion via self-attention adapter for accurate action prediction).

*Global — Spatial Latent Grid:*
V-JEPA/V-JEPA 2 (JEPA extended to video — predicts latent features of occluded spatiotemporal regions without pixel reconstruction; V-JEPA 2 scales pretraining to 22M videos + 15 robot datasets, transfers to robotic planning), AD-L-JEPA (JEPA adapted to BEV LiDAR for AD self-supervised pretraining), WorldDreamer (masked visual sequence prediction to learn physics+motion for diverse video generation).

*Global — Decomposed Rendering:*
Sora/video world models with DiT (unified spacetime patches), ForeDiff (deterministic predictive stream + denoising), DynamicCity (4D occupancy → HexPlane + DiT), GaussianWorld (ego-motion + object dynamics + newly observed regions via 3DGS updating), InfiniCube (hybrid VQ-based + video synthesis + dynamic Gaussian reconstruction for large-scale driving scenes).

### Axis 2: Temporal Modeling Details

- **Sequential Simulation and Inference**: Autoregressive rollout, one step at a time. RNN-based (compact, efficient for real-time), Transformer-based (captures long-range dependencies), Mamba/SSM-based (linear-time complexity for long-horizon). Drawback: error accumulation over long rollouts.
- **Global Difference Prediction**: Predicts entire future sequence in parallel. Reduces error accumulation, improves multi-step coherence. But requires heavier compute and weaker closed-loop interactivity. Token-based (TOKEN, villa-X) and rendering-based (Sora, DynamicCity, GaussianWorld) variants.

### Axis 3: Spatial Representation Details

- **Global Latent Vector**: Compact, efficient for real-time on-device. Limited in capturing fine-grained spatiotemporal detail. Best for decision-coupled MBRL (Dreamer series).
- **Token Feature Sequence**: Captures complex spatial, temporal, and cross-modal dependencies. Enables multimodal integration and LLM reuse. Dominant for general-purpose models (iVideoGPT, Genie, TOKEN).
- **Spatial Latent Grid**: Geometry-aligned (BEV, voxel grids). Incorporates spatial inductive biases. Enables efficient convolutional/attention updates and streaming rollouts. Essential for autonomous driving occupancy models.
- **Decomposed Rendering Representation**: Decomposes 3D scenes into explicit primitives (3DGS, NeRF). Enables view-consistent forecasts, object-level compositionality, physics priors, and seamless digital twin integration. Highest fidelity; scales poorly in dynamic scenes.

### Data Resources (§IV)

Table III provides a unified data resource overview across four categories:

- **Simulation Platforms**: MuJoCo (2012, physics engine), NVIDIA Isaac (E2E GPU-accelerated), CARLA (2017, urban AD), Habitat (2019, 3D indoor).
- **Interactive Benchmarks**: Atari/Atari100k, DMC (MuJoCo-based, pixel+state), Meta-World (50 manipulation tasks), RLBench (100 manipulation tasks), nuPlan (AD closed-loop), LIBERO (130 lifelong manipulation tasks).
- **Offline Datasets**: SSv2 (220k videos), nuScenes (1k scenes, 360° sensor suite), Waymo (1.15k scenes, 10Hz, 5 LiDARs + 5 cameras), HM3D (1k indoor scenes), RT-1 (130k real robot trajectories), Occ3D (1.9k scenes voxel-level), OXE (1M+ cross-embodiment trajectories from 22 robots), OpenDV (2k+ hours driving video), VideoMix22M (22M+ samples, V-JEPA 2 pretraining).
- **Real-world Robot Platforms**: Franka Emika (7-DoF, 1kHz torque control), Unitree Go1 (quadruped, panoramic depth, 1.5 TFLOPS), Unitree G1 (humanoid, 43-DoF, 120 N·m knee torque, 3D LiDAR+depth cameras).

---

## Results / Key Findings

**Table IV — Video Generation on nuScenes (FID/FVD):**
- DrivePhysica achieves best visual fidelity; MiLA achieves strongest temporal coherence (FVD 14.9), setting new state-of-art in tandem.
- MiLA (arXiv'25, 360×640) achieves FID 4.1 / FVD 14.9 — substantially better than DriveDreamer (FID 52.6 / FVD 452.0).
- Resolution is a major confound: models operating at 576×1024 (Vista, DrivingPhysica) benefit from higher pixel budget.

**Table V — 4D Occupancy Forecasting on Occ3D-nuScenes (mIoU):**
- COME (Camera, GT ego) achieves best average mIoU 44.13 and IoU 38.36 — outperforms all other camera-based methods substantially.
- With occupancy-input ground truth: DTT-O achieves 74.58 mIoU avg; COME achieves competitive 34.23.
- Methods using occupancy inputs dominate camera-only variants; adding auxiliary supervision (GT ego trajectory) further mitigates performance decay at 2–3s.

**Table VI — DMC Benchmark (Episode Return):**
- DreamerV3 (Nature'25): 820 steps average, competitive across all tasks.
- DiWM: 870 steps on Reacher-Easy/Cheetah-Run/Finger-Spin/Walker-Walk.
- Recent models achieving strong performance in far fewer training steps, reflecting data efficiency gains.

**Table VII — Manipulation on RLBench:**
- VidMan achieves best average across 5 tasks (63/10 avg across tasks including Stacking Blocks 63%, Open Jar 48%, Slide Block 98%).
- IDM is a promising architectural direction (identified by survey).
- ManiGaussian++ supports bimanual manipulation while others do not.

**Table VIII — Open-loop Planning on nuScenes (L2/Collision):**
- UniAD+DriveWorld: best average L2 (0.69m avg) and collision rate (0.12% avg).
- SSR: best collision rate (0.10% avg at 1s) without auxiliary supervision — most practical for deployment.
- Camera-based models now surpass occupancy-input models on L2, reflecting E2E planning maturity.

Key empirical patterns:
1. **Auxiliary supervision consistently helps** but GT ego trajectory is privileged information — models trained without it (SSR, camera-only OccWorld-F) are more deployment-realistic.
2. **3DGS-based spatial representations** (ManiGaussian, DreMa) achieve highest manipulation fidelity and are becoming competitive with token-based approaches.
3. **Global difference prediction** (TOKEN, villa-X) reduces error accumulation in long-horizon tasks but trades off closed-loop interactivity — a fundamental tension.
4. **JEPA-style models** (V-JEPA 2 with 22M-sample pretraining + 15 robot datasets) demonstrate that non-generative self-supervised pretraining can match or exceed generative methods for embodied control transfer.

---

## Comparison to Prior Work / Related Surveys

| Survey | Taxonomy Axes | Robotics | Driving | Manipulation | Key Distinction |
|---|---|---|---|---|---|
| [[ding-2024-csur]] ([Ding et al., P006](../papers/ding-2024-csur.md)) | Understanding vs. prediction | Light | Moderate | None | Cross-domain (social, gaming included) |
| [[feng-2025-arxiv]] ([Feng et al., P007](../papers/feng-2025-arxiv.md)) | Generation modality × planning | None | Deep | None | AD-specific, quantitative benchmarks |
| [[kong-2025-arxiv]] ([Kong et al., P009](../papers/kong-2025-arxiv.md)) | 3D/4D × static/dynamic | None | Some | None | Spatial emphasis, novel view synthesis |
| **Li et al. (P008)** | Functionality × temporal × spatial | Deep | Deep | Deep | Unified embodied AI, POMDP formalism |

**vs. [[ding-2024-csur]] ([Ding et al., P006](../papers/ding-2024-csur.md))**: Li et al. provides deeper coverage of robotics/manipulation (Tables I and VII absent from Ding et al.) and introduces a formal POMDP mathematical framework for unifying all world model variants. Ding et al. covers social simulacra and gaming domains absent from Li et al.

**vs. [[feng-2025-arxiv]] ([Feng et al., P007](../papers/feng-2025-arxiv.md))**: Li et al. and Feng et al. share overlapping autonomous driving coverage but from different perspectives. Feng et al.'s four-modality generation taxonomy is more granular for AD; Li et al.'s three-axis framework is more general and maps better to robotics.

**vs. [[ha-2018-neurips]] ([Ha & Schmidhuber 2018](../papers/ha-2018-neurips.md))**: Li et al. frames Ha & Schmidhuber as the seminal crystallization of RSSM-based decision-coupled world models under the Global Latent Vector / Sequential Simulation quadrant of their taxonomy.

**vs. [[lecun-2022-openreview]] ([LeCun 2022, JEPA](../papers/lecun-2022-openreview.md))**: V-JEPA/V-JEPA 2 appears under the General-Purpose / Global Difference Prediction / Token Feature Sequence cell, validating LeCun's non-generative prediction-in-representation-space proposal. AD-L-JEPA extends this to BEV LiDAR.

---

## Strengths
- The three-axis taxonomy is the most principled organizational framework in the world model survey literature — it maps directly to concrete architectural design choices.
- Formal POMDP treatment with ELBO derivation is unique among world model surveys and grounds the taxonomy in a unified mathematical framework.
- Best coverage of manipulation and robotics subtasks (ManiGaussian series, ParticleFormer, 3DFlowAction) absent from other surveys.
- Eight quantitative benchmark tables provide genuinely comparable numbers across pixel generation, occupancy, DMC control, manipulation, and motion planning.
- Table III (data resources with 25+ platforms/benchmarks/datasets) is the most comprehensive resource overview in the domain.
- V-JEPA 2 (22M-sample pretraining) coverage is timely and significant — first survey to place JEPA explicitly within the three-axis taxonomy.
- Curated bibliography at GitHub (github.com/Li-Zn-H/AwesomeWorldModels) provides ongoing update mechanism.

## Weaknesses & Limitations
- Functionality axis (decision-coupled vs. general-purpose) is the least rigorous axis — the boundary is fuzzy (DreamerV3 "masters diverse domains" but is trained with environment-specific rewards) and could have been defined more precisely.
- Gaming and social simulacra domains (covered by Ding et al.) are absent.
- The three-axis framework can create awkward categorizations: some models span multiple cells and the survey occasionally places methods in multiple categories without clear guidance.
- The challenge discussion (§VI) is shorter than the taxonomy section and does not propose concrete research directions with the same precision.
- No analysis of the trade-off between spatial representation fidelity and computational cost for real-time deployment — critical for physical robots.
- Coverage of physics-hybrid approaches (hard constraint + neural) is thinner than Ding et al. §6.1, despite being directly relevant to embodied AI.

## Key Takeaways
- **The three-axis framework reveals clear architecture trends**: decision-coupled systems use global latent vectors + sequential rollouts for efficiency; general-purpose systems use token sequences + global prediction for fidelity; the most advanced systems (V-JEPA 2, AETHER, DriveDreamer4D) combine spatial latent grids or decomposed rendering with global prediction for closed-loop interactivity.
- **3DGS is displacing NeRF as the decomposed rendering primitive of choice** for embodied AI world models, offering dynamic scene modeling (ManiGaussian, DreMa, DriveDreamer4D) that static NeRFs cannot match.
- **V-JEPA 2 establishes non-generative JEPA pretraining as competitive with generative approaches** for embodied control transfer across 15 robot datasets — directly validating LeCun's 2022 JEPA proposal at scale.
- **The eval gap is the field's most pressing problem**: FID/FVD measure pixel fidelity while embodied AI needs physical consistency, causal fidelity, and long-horizon accuracy. New benchmarks (EWM-Bench) begin to address this but cross-domain standards remain lacking.
- **Unified cross-domain datasets are the critical missing infrastructure**: current fragmentation across manipulation (SSv2, RT-1), driving (nuScenes, Waymo), and navigation (HM3D) prevents the development of world models that genuinely generalize across embodied AI tasks.

---

## BibTeX
```bibtex
@article{li2025comprehensive,
  title     = {A Comprehensive Survey on World Models for Embodied {AI}},
  author    = {Li, Xinqing and He, Xin and Zhang, Le and Wu, Min and Li, Xiaoli and Liu, Yun},
  journal   = {arXiv preprint arXiv:2510.16732},
  year      = {2025},
  month     = {November},
  url       = {https://arxiv.org/abs/2510.16732}
}
```

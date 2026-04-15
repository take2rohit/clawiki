---
title: '3D and 4D World Modeling: A Survey'
type: paper
paper_id: P009
authors:
- Kong, Lingdong
- Yang, Wesley
- Mei, Jianbiao
- Liu, Youquan
- Liang, Ao
- Zhu, Dekai
- Lu, Dongyue
- Yin, Wei
- Hu, Xiaotao
- Jia, Mingkai
- Deng, Junyuan
- Zhang, Kaiwen
- Wu, Yang
- Yan, Tianyi
- Gao, Shenyuan
- Wang, Song
- Li, Linfeng
- Pan, Liang
- Liu, Yong
- Zhu, Jianke
- Ooi, Wei Tsang
- Hoi, Steven C. H.
- Liu, Ziwei
year: 2025
venue: arXiv
arxiv_id: '2509.07996'
url: https://arxiv.org/abs/2509.07996
pdf: ../../raw/kong-2025-arxiv.pdf
tags:
- world-model
- survey
- occupancy-grid
- point-cloud
- video-generation
- autonomous-driving
- robotics
- NeRF
- 3DGS
- diffusion
created: 2026-04-10
updated: 2026-04-10
cites:
- ding-2024-csur
- feng-2025-arxiv
- ha-2018-neurips
- hu-2023-arxiv
- li-2025-arxiv
cited_by:
- ding-2024-csur
- feng-2025-arxiv
- li-2025-arxiv

---

# 3D and 4D World Modeling: A Survey

> **Survey** — Kong et al. present the first comprehensive review explicitly dedicated to native 3D and 4D world modeling, organizing 325+ methods under a modality-first taxonomy (VideoGen / OccGen / LiDARGen) × four functional roles (Data Engine / Action Interpreter / Neural Simulator / Scene Reconstructor), with 14 evaluation tables covering generation quality, occupancy forecasting, and planning across the full 3D/4D spectrum.

**Authors:** Lingdong Kong (project lead), Wesley Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Dongyue Lu, Wei Yin, Xiaotao Hu, Mingkai Jia, Junyuan Deng, Kaiwen Zhang, Yang Wu, Tianyi Yan, Shenyuan Gao, Song Wang, Linfeng Li, Liang Pan, Yong Liu, Jianke Zhu, Wei Tsang Ooi, Steven C. H. Hoi, Ziwei Liu (WorldBench Team, NUS / NTU and collaborators) | **Venue:** arXiv (v3, December 2025) | **arXiv:** [2509.07996](https://arxiv.org/abs/2509.07996) | **Project Page:** [worldbench.github.io/survey](https://worldbench.github.io/survey)

---

## Problem & Motivation

World modeling research has largely been centered on 2D data — images and videos — even as the most critical real-world applications (autonomous driving, robotics, digital twins) inherently require 3D and 4D spatial representations. Native 3D/4D data modalities — RGB-D imagery, volumetric occupancy grids, and LiDAR point clouds — provide explicit geometry, causality, and multi-view consistency that 2D projections cannot recover. Yet no prior survey has specifically targeted this body of work, leaving it fragmented across sub-communities using inconsistent terminology. Meanwhile, the term "world model" itself has become ambiguous: some researchers define it as a generative model for sensory data; others as a predictive simulator for decision-making. This ambiguity makes it difficult to compare methods or identify open gaps. This survey addresses these issues with the first review explicitly dedicated to 3D/4D world modeling and generation, providing standardized definitions, a structured taxonomy, comprehensive dataset coverage, and rigorous evaluation benchmarks.

---

## Core Idea

The survey separates two dimensions that prior work conflates: (1) **what a model consumes** (the input representation modality: video, occupancy, or LiDAR), and (2) **what a model does** (its functional role: data engine, action interpreter, neural simulator, or scene reconstructor). This 3×4 cross-product gives a taxonomy where any 3D/4D world model can be placed unambiguously. The taxonomy is further grounded by a formal distinction between two paradigms:
- **Generative World Models**: G(x_i, C_geo, C_act, C_sem) → S_g (synthesize plausible scenes from conditions)
- **Predictive World Models**: P(x_i^{-t:0}, C_act) → S_p^{1:k} (forecast k future steps from history and actions)

Conditioning signals are standardized as geometric C_geo (camera pose, depth, BEV, HD map, bounding boxes, flow, past occupancy, LiDAR pattern, partial point cloud, RGB frame, surface mesh), action-based C_act (ego-trajectory, ego-velocity, ego-acceleration, ego-steering, ego-command, route plan, action token, scan path), and semantic C_sem (semantic mask, text prompt, scene graph, object label, weather tag, material tag).

---

## How It Works

### Overview

The survey is organized into seven sections following the pipeline in Figure 1:
- §2: Preliminaries (representations, definitions, generative model families)
- §3: Three-track methodology taxonomy (VideoGen, OccGen, LiDARGen), each with three functional subtypes
- §4: Datasets & Evaluations (five evaluation dimensions, 14 benchmark tables)
- §5: Applications (autonomous driving, robotics, games/XR, digital twins, emerging domains)
- §6: Challenges & Future Directions (five axes)
- §7: Conclusion

Four main tables (Tables 2–4) catalogue 63 VideoGen, 40 OccGen, and 34 LiDARGen methods with their datasets, architectures, input/output modalities, conditioning signals, sequence length, and functional category.

### 3D/4D Representations (§2.1)

- **Video Streams**: x_v ∈ R^{T×H×W×C} — temporal video, emphasizes geometric coherence and temporal consistency.
- **Occupancy Grids**: x_o ∈ {0,1}^{X×Y×Z} (static) or x_o^t ∈ {0,1}^{T×X×Y×Z} (4D dynamic) — voxelized geometry enforcing spatial constraints.
- **LiDAR Point Clouds**: x_l = {(x_i,y_i,z_i)}_{i=1}^N; sequential x_l^t further records timestamps — geometry-direct, robust to texture/lighting/weather.
- **Neural Representations**: NeRF (continuous volumetric fields) and Gaussian Splatting (explicit Gaussian primitives with position, covariance, color) — temporal extensions add dynamic components.

### Four Functional Types (§2.2.2)

| Type | Inputs | Output | Purpose |
|---|---|---|---|
| **Data Engines** | C_geo (+ optional C_act, C_sem) | Generated scene S_g | Plausibility + diversity for large-scale data augmentation |
| **Action Interpreters** | Historical x_i^{-t:0} + C_act | Predicted sequence S_p^{1:k} | Action-aware forecasting for trajectory planning + policy evaluation |
| **Neural Simulators** | Current state S_g^t + agent policy π | Next state S_g^{t+1} | Closed-loop interactive simulation |
| **Scene Reconstructors** | Partial observations x_i^p + optional C_geo | Completed scene S_g | High-fidelity mapping, digital twin restoration |

### Track 1: VideoGen (§3.1)

63 video-based generation methods in Table 2. Three functional subtypes:

**Data Engines (§3.1.1):**
- *Perception data augmentation*: BEVGen (AR Transformer for cross-view consistent BEV-aligned surround images), BEVControl (SD for boosting data with editable BEV sketches, +1.2× mNDS improvement), MagicDrive (3D geometry + camera pose → high-fidelity images), MagicDrive-V2 (DiT), SyntheOcc (3D semantic multi-plane images), PerspLDiff (perspective-layout SD), Panacea/Panacea+ (SD for diverse long-tail augmentation), WoVoGen (cross-sensor consistency via world volume-aware synthesis), NoiseController (multi-level noise decomposition), SimGen (sim-to-real via cascade SD), DrivePhysica (CARLA representation learning + flow guidance).
- *Planning-oriented data mining*: Delphi (failure-case-driven SD for E2E planning), DriveDreamer-2 (LLM → user queries → agent trajectories → HDMaps → traffic-compliant videos), Nexus (fine-grained tokens + independent noise for controllable corner-case generation), Challenger (physics-aware trajectory refinement for adversarial behavior synthesis).
- *Scene editing & style transfer*: GeoDrive (3D geometry + dynamic editing), SyntheOcc (occupancy-guided occlusion-aware editing).

**Action Interpreters (§3.1.2):**
- *Action-guided generation*: GAIA-1 (AR — fuses video, text, action → realistic driving scenarios, pioneered the paradigm), GAIA-2 (adds agent configs, environmental factors, road semantics), GenAD (SD → zero-shot, language/action-conditioned, OpenDV dataset), Vista (robust multi-scenario action conditioning, NeurIPS'24), GEM (CVPR'25, precise ego-motion multimodal control), MaskGWM (mask-based DiT for fidelity + long-horizon), InfinityDrive/Epona (memory injection + chain-of-forward for long-horizon error mitigation), DrivingWorld (GPT-style from predefined trajectories), DriVerse/MiLA/PosePilot/LongDWM (trajectory alignment, temporal stability, pose control, depth-free guidance).
- *Forecasting-driven action planning*: Drive-WM (SD video rollouts for trajectory selection), DriveDreamer (ActionFormer for ego-environment interaction), ADriver-I (multimodal LLM + AR control + world evolution), DrivingGPT/DrivingWorld (GPT-style for unified perception/prediction/planning), Doe-1 (MLLM for closed-loop), VaVAM (MLLM bridges video diffusion + action expert), ProphetDWM (latent action learning + state forecasting).

**Neural Simulators (§3.1.3):**
- *Generation-driven*: DriveArena (first closed-loop framework: TrafficManager + WorldDreamer), DreamForge (object position encoding + temporal attention for long-term), DrivingSphere (4D semantic occupancy + visual synthesis for spatio-temporal multi-view consistency), UMGen (behavioral interaction simulation between ego and user-defined agents), Nexus (closed-loop via nuPlan benchmarks), GeoDrive (geometry-aware trajectory optimization for VLA systems).
- *Reconstruction-centric*: NeRF/3DGS-based neural environments (StreetGaussian, HUGSIM, UniSim, Uni-Gaussians, OmniRe, ReconDreamer, Stage-1). StreetGaussian represents dynamic urban streets as point clouds with semantic logits + 3DGS per-vehicle/background. HUGSIM integrates physical constraints with 3DGS for aggressive behavior synthesis.

### Track 2: OccGen (§3.2)

40 occupancy-based generation methods in Table 3. Three functional subtypes:

**Scene Representors (§3.2.1):**
- *3D perception robustness enhancement*: SSD (discrete + latent SD for 3D categorical data), SemCity (CVPR'24, adds manipulation functions for scene editing), DriveWorld (MSSM — joint occupancy + image/LiDAR pretraining), UniScene (DiT generalization across modalities), UrbanDiff (semantic occupancy as geometric prior for 3D-aware image synthesis), WoVoGen (4D temporal occupancy for cross-sensor consistency), DrivingSphere (4D occupancy → temporally consistent video).
- *Generation consistency guidance*: OccScene (CVPR'25, denoising 3D occupied cells), SSD/UniScene/UrbanDiff (occupancy as structural prior for video synthesis).

**Occupancy Forecasters (§3.2.2):**
- *Predictive model pretraining*: Emergent-Occ (differentiable rendering for self-supervised 4D forecasting from raw sequences), UnO (CVPR'24, continuous 4D occupancy for joint perception+forecasting), UniWorld/UniScene/DriveWorld (large-scale pretraining combining image+LiDAR for generalizable occupancy models).
- *Ego-conditioned forecasting*: OccWorld (ECCV'24, jointly models ego motion + surrounding 3D occupancy), OccSora (trajectory-conditioned 4D occupancy over 16s horizons), OccProphet (observer-forecaster-refiner, ICLR'25), DFIT-OccWorld, Drive-OccWorld (OccWorld variants with efficiency/E2E improvements), Cam4DOcc (CVPR'24, camera-only 4DOcc pipeline), OccLLaMA (LLM + occupancy for instruction-driven E2E), Occ-LLM, I²World (intra/inter tokenization, ICCV'25), T³Former (triplane sparse attention, real-time camera-only, arxiv'25), COME (DiT + occupancy/flow/BEV layout conditioning), UniOcc (benchmark).

**Autoregressive Simulators (§3.2.3):**
- *Scalable open-world generation*: PDD (scale-varied diffusion: coarse → fine 4D), XCube (hierarchical voxel-based latent diffusion for multi-resolution 4D), SemCity (scene editing), InfiniCube/X-Scene (voxel-based occupancy + consistent visual synthesis for editable open-world simulation).
- *Long-horizon dynamic simulation*: OccSora (trajectory-conditioned sequences over 16s, 4D autoregressive), DynamicCity (ICLR'25, HexPlane + DiT, layout-aware + command-conditioned for controllable scene synthesis + agent interaction), DrivingSphere (static backgrounds + dynamic objects for closed-loop), UniScene (layout-conditioned 4D with semantic + geometric detail).

### Track 3: LiDARGen (§3.3)

34 LiDAR-based generation methods in Table 4. Three functional subtypes:

**Data Engines (§3.3.1):**
- *Perception data augmentation*: DUSty (GAN-based, disentangles depth map from measurement uncertainty), DUSty v2 (implicit neural for arbitrary resolution), LiDARGen (Langevin dynamics, first Diffusion-based), R2DM (DDPM + positional encoding), R2Flow (flow matching, accelerated generation), LiDM/RangeLDM/3DiSS (latent diffusion on compressed raw scans), LiDARGRIT (VQ-VAE + AR Transformer), LiDARGRIT+ (adds raydrop estimation), SDS (multi-view simultaneous diffusion for geometric consistency), SPIRAL (NeurIPS'25, first segmentation-labeled LiDAR generation, novel closed-loop inference), La La LiDAR (layout-guided + scene graph-based diffusion with foreground-aware control injector), Veila (conditional diffusion for panoramic LiDAR from monocular RGB).
- *Scene completion*: UltraLiDAR (discrete VQ-VAE for sparse-to-dense completion), LiDiff/DiffSSC (DDPM repositioning for completion), LiDAR-EDIT (flexible editing including removal/insertion), LiDPM/Distillation-DPO (generation from pure Gaussian noise + novel synthesis), SuperPC (unified transforms-to-representations framework).
- *Rare condition modeling*: Text2LiDAR (Transformer + text for adverse weather LiDAR), WeatherGen (rainy/snowy/foggy LiDAR), OLiDM (two-stage: foreground objects first → scene generation), LOGen (object-level traffic participant synthesis).
- *Multimodal generation*: X-Drive (CVPR'25, dual-branch diffusion for aligned LiDAR + multi-view camera, cross-modality epipolar condition), DriveX (decoupled spatial + future latent, multi-modal outputs including point clouds, camera images, semantic maps).

**Action Forecasters (§3.3.2):**
- Copilot4D (ICLR'24, VQ-VAE tokenization + masked generative image Transformer as discrete diffusion → 1–3s future LiDAR from ego actions), ViDAR (CVPR'24, camera → future LiDAR pretraining for perception/prediction/planning), BEVWorld (multi-modal tokenizer for surround-view + LiDAR), DriveX, HERMES (ICCV'25, LLMs + multiview BEV → textual descriptions + LiDAR generation).

**Autoregressive Simulators (§3.3.3):**
- HoloDrive (arXiv'24, AR framework jointly generating multi-view camera + LiDAR via depth prediction branch for 2D↔3D alignment), LiDARCrafter (AAAT'26, extends La La LiDAR to 4D domain with AR LiDAR sequence generator for fine-grained control + long-term coherence), LidarDM (ICRA'25, constructs mesh grids from point clouds → diffusion conditioned on BEV layout → dynamic objects via motion trajectories → ray projection for long sequential LiDAR synthesis), OpenDWM.

### Datasets & Evaluation (§4)

**Table 5 (25 datasets)** — comprehensive dataset overview covering nuScenes (1M frames, 6 cams, 400k occupancy), Waymo Open (1.15k scenes, 1M frames), KITTI/KITTI-360, SemanticKITTI, OpenDV-YouTube (60M frames, 2139 scenes), Occ3D-nuScenes (240k, 40k scenes), NAVSIM (115k scenes, 920k frames), CarlaSC, Argoverse 2, PandaSet, KITTI-360, OpenCOOD, SSCBench, DrivingDojo, OmniDrive, EUVS, Pi3DET.

**Evaluation framework** (Table 14, 5 perspectives):
1. **Generation Quality**: FID, FVD, FRD (Fréchet Range Distance for LiDAR), FPD (Fréchet Point Cloud Distance), FSVD (Fréchet Sparse Volume Distance), FPVD, F3D, S-FRD/S-FPD (semantic), KID, IS, IQ; Statistical: PR, SWD, JSD, MMD, COV, 1-NNA, Diversity; Spatial: VCS (View Consistency Score via LoFTR keypoints).
2. **Forecasting Quality**: IoU (occupancy), Chamfer Distance (point cloud), FVD/FID (temporal frames), TTCE/CTC (temporal LiDAR consistency).
3. **Planning-Centric**: L2 error, collision rate, PDMS/ADMS (nuPlan closed-loop scores), NAVSIM score.
4. **Reconstruction-Centric**: PSNR, SSIM, LPIPS, Novel Trajectory Agent IoU.
5. **Downstream Evaluation**: mAP/NDS (detection), mIoU (segmentation/occupancy), MOTA/MOTP (tracking), Success Rate.

---

## Results / Key Findings

**Table 6 — VideoGen on nuScenes (FID/FVD):**
- Single-view: MaskGWM (FID 4.0, FVD 59.4) and GeoDrive (FID 4.1, FVD 61.6) achieve state-of-art. Vista achieves FID 6.9/FVD 89.4 at 576×1024 resolution.
- Multi-view: DiST-4D (FVD 22.67) and UniScene (FVD 71.94) achieve best balance, with FVD <80 at ≥512×512. Early BEV-based models (BEVControl FID 24.85, BEVGen FID 24.54) lag substantially.
- Key insight: resolution and frame rate are major confounds. Temporal coherence (FVD) benefits most from explicit geometry and spatiotemporal alignment.

**Tables 7–8 — VideoGen Downstream Evaluation (Detection + Segmentation + Planning on nuScenes):**
- Best generation-based detection: DrivePhysica (35.5 mAP, 43.7 NDS on BEVFusion), Glad (36.5 mAP).
- Best segmentation: UniMLVG (70.8% road, 32.1% vehicle mIoU), CogDriving (65.7% road, 32.1%).
- Best closed-loop planning: DrivingArena and DreamForge show non-trivial success rates (PDMS 0.81, 0.76 respectively). Real data upper bound: 37.9 mAP, 49.9 NDS, 1.05 L2.
- Photorealistic generation alone is insufficient; explicit geometry, temporal consistency, and motion dynamics are crucial.

**Table 9 — OccGen Reconstruction Quality (mIoU/IoU on nuScenes VAE):**
- T³Former achieves 85.50 mIoU with Triplane-VAE (100,100,16,8), best among VQVAE methods.
- X-Scene achieves state-of-art 92.40 mIoU / 85.60 IoU with Triplane-VAE — significantly outperforms DOME/UniScene VAE baselines (~77–83 mIoU).
- Triplane factorization is decisive: enforces geometric consistency and enables finer spatial detail.

**Table 10 — OccGen 4D Occupancy Forecasting (mIoU/IoU at 1s/2s/3s on nuScenes):**
- I²World achieves best performance: 47.62/38.58/32.98 mIoU, 54.29/49.43/45.69 IoU — 3× better than baselines (GaussianAD 6.29/5.36/4.58 mIoU).
- T³Former also excels at 3-second horizon: 46.32/33.23/28.73 mIoU.
- Naive autoregressive approaches deteriorate rapidly at longer horizons; triplane factorization substantially improves spatial fidelity.

**Table 11 — OccGen Motion Planning Quality (L2/Collision on nuScenes):**
- Occ-LLM achieves best planning results: 0.12m/0.24m/0.49m L2, with competitive collision rates.
- GaussianAD and T³Former balance error and safety well. Drive-OccWorld: 0.32m/0.75m/1.49m L2 with 0.05/0.17/0.64 collision rate.
- Integrating occupancy world models into planning pipelines consistently outperforms pure trajectory-based methods.

**Tables 12–13 — LiDARGen Benchmarks:**
- Table 12 (Perceptual Fidelity on SemanticKITTI): WeatherGen achieves best across all metrics (FRD 184.11, FPD 11.42, JSD 0.0290, MMD 3.80×10⁻⁵) using Mamba backbone. R2DM and Text2LiDAR substantially outperform early baselines (LiDARGen FRD 681.37).
- Table 13 (4D LiDAR Generation Consistency on nuScenes): UniScene and OpenDWM-DiT demonstrate best short-horizon temporal consistency (TTCE 2.74/2.71, CTC 0.90/0.89 at 1 frame). LiDARCrafter achieves strong performance (TTCE 2.65, CTC 1.12) with fine-grained control.

Key cross-track findings:
1. **Photorealistic generation ≠ task utility**: generative fidelity (FID/FVD) has weak correlation with downstream perception and planning gains. Explicit geometry and temporal structure are decisive.
2. **Triplane representations are the key architectural breakthrough** for occupancy: enabling spatial consistency and fine-grained detail (X-Scene 92.40 vs. VQVAE baselines ~27–80 mIoU).
3. **Temporal error accumulation is the central bottleneck**: naive autoregressive models rapidly degrade at >2s horizons; structured priors (triplane, geometry constraints, memory mechanisms) are essential.
4. **Cross-modal coherence is unsolved**: most methods optimize one modality; systems like X-Drive (LiDAR + camera) and HERMES (video + LiDAR + text) represent early attempts at genuine multi-modal consistency.

---

## Comparison to Prior Work / Related Surveys

| Survey | Explicit 3D/4D Focus | Modality Coverage | LiDAR Coverage | Benchmark Depth |
|---|---|---|---|---|
| [[ding-2024-csur]] ([Ding et al., P006](../papers/ding-2024-csur.md)) | No (mentions but broad) | Video + occupancy | Light | Descriptive |
| [[feng-2025-arxiv]] ([Feng et al., P007](../papers/feng-2025-arxiv.md)) | Partial (occupancy/point cloud) | Video + OG + PC | Moderate (Table 4) | 9 quantitative tables |
| [[li-2025-arxiv]] ([Li et al., P008](../papers/li-2025-arxiv.md)) | Partial (SLG, DDR spatial axes) | Mixed | Light | 8 quantitative tables |
| **Kong et al. (P009)** | Yes (explicit scope) | VideoGen + OccGen + LiDARGen | Deep (34 methods, Tables 4,12,13) | 14 quantitative tables |

**vs. [[feng-2025-arxiv]] ([Feng et al., P007](../papers/feng-2025-arxiv.md))**: Both cover autonomous driving world models in depth. Kong et al. is explicitly scoped to native 3D/4D and provides more LiDAR-specific coverage (34 methods vs. ~15 in Feng et al.). Feng et al. has deeper behavior planning coverage; Kong et al. has deeper reconstruction and scene editing coverage.

**vs. [[li-2025-arxiv]] ([Li et al., P008](../papers/li-2025-arxiv.md))**: Li et al.'s "Decomposed Rendering Representation" axis partially overlaps with Kong et al.'s NeRF/3DGS reconstruction focus. Kong et al. goes significantly deeper on occupancy and LiDAR modalities; Li et al. covers broader robot domains (manipulation, DMC).

**vs. [[ding-2024-csur]] ([Ding et al., P006](../papers/ding-2024-csur.md))**: Kong et al.'s native 3D/4D scope is distinct — Ding et al. treats video generation and occupancy grids as subcategories of a broader survey, while Kong et al. treats them as first-class primary tracks.

**Unique contribution**: Table 14's comprehensive evaluation metric taxonomy (30+ metrics organized across 5 dimensions with formal definitions) is the most systematic treatment of 3D/4D evaluation metrics in the world model literature.

---

## Strengths
- First survey explicitly dedicated to native 3D/4D world modeling — fills a clear gap left by video-centric and domain-specific surveys.
- Highly systematic: 14 evaluation tables, 4 method tables (63+40+34 = 137 methods catalogued with full details), Table 14's metric taxonomy with 30+ metrics formally defined.
- The conditions taxonomy (Table 1, 20 geometric/action/semantic conditions) is the most precise treatment of conditioning signals in any world model survey — directly useful for practitioners designing systems.
- VideoGen neural simulator subtraction into generation-driven vs. reconstruction-centric (NeRF/3DGS) is unique and accurate — other surveys blur this distinction.
- Applications coverage extends beyond autonomous driving to games/XR, digital twins, scientific discovery, healthcare, and industrial simulation — the broadest application scope among the five surveys.
- Project page and GitHub repo ensure ongoing maintainability of the method catalogue.

## Weaknesses & Limitations
- Autonomous driving centric: despite claiming broad scope, 85%+ of methods are from autonomous driving; manipulation, locomotion, and indoor robotics are mentioned in applications (§5.2) but not covered systematically in the taxonomy.
- No coverage of decision-making/MBRL world models (Dreamer series, RSSM) — the implicit representation branch from Ding et al.'s taxonomy. This is an explicit scope choice but means the survey misses a major portion of the world model landscape.
- Physics and causality limitations are identified in §6.3 but the discussion of hybrid physics-data solutions is thinner than Ding et al. §6.1 — surprising given the native 3D/4D focus where physics should be most tractable.
- Table 6's VideoGen benchmark is incomplete for newer methods (many cells have no FVD scores) because temporal consistency evaluation requires standardized protocols that many methods do not follow.
- LiDAR benchmark (Table 12) uses SemanticKITTI while most perception benchmarks use nuScenes, creating disconnect with Tables 7–11.

## Key Takeaways
- **Native 3D/4D representations (occupancy, LiDAR, RGB-D) provide the inductive biases needed for physical plausibility**, multi-view consistency, egocentric causality, and map/topology adherence that 2D video generation cannot provide — making them essential for safety-critical embodied AI.
- **Triplane factorization is the breakthrough for occupancy generation**: T³Former and X-Scene demonstrate 92%+ mIoU in reconstruction and 3× better long-horizon forecasting than VQVAE baselines — the geometry-enforcing inductive bias of triplanes is decisive.
- **Functional role (data engine / action interpreter / neural simulator / scene reconstructor) should drive architecture design choices**: data engines need diversity and coverage; action interpreters need temporal coherence; neural simulators need closed-loop stability; scene reconstructors need geometric fidelity. Conflating these roles leads to underspecified models.
- **WeatherGen's Mamba backbone sets new state-of-art in LiDAR generation** across all fidelity metrics (FRD, FPD, JSD, MMD on SemanticKITTI), suggesting that linear-complexity state-space models are particularly well-suited for the sequential scan structure of LiDAR data.
- **Five critical axes for future progress**: (1) standardized 3D/4D benchmarks with closed-loop evaluation, (2) long-horizon high-fidelity generation with structured priors, (3) physical realism + fine-grained controllability + generalization, (4) efficient real-time architectures (sparse computation, inference acceleration), (5) cross-modal coherence among video, occupancy, and LiDAR streams.

---

## BibTeX
{% raw %}
```bibtex
@article{kong2025survey3d4d,
  title     = {{3D} and {4D} World Modeling: {A} Survey},
  author    = {Kong, Lingdong and Yang, Wesley and Mei, Jianbiao and Liu, Youquan and Liang, Ao and Zhu, Dekai and Lu, Dongyue and Yin, Wei and Hu, Xiaotao and Jia, Mingkai and Deng, Junyuan and Zhang, Kaiwen and Wu, Yang and Yan, Tianyi and Gao, Shenyuan and Wang, Song and Li, Linfeng and Pan, Liang and Liu, Yong and Zhu, Jianke and Ooi, Wei Tsang and Hoi, Steven C. H. and Liu, Ziwei},
  journal   = {arXiv preprint arXiv:2509.07996},
  year      = {2025},
  month     = {December},
  url       = {https://arxiv.org/abs/2509.07996}
}
```
{% endraw %}

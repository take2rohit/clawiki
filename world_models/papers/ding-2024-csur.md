---
title: Understanding World or Predicting Future? A Comprehensive Survey of World Models
type: paper
paper_id: P006
authors:
- Ding, Jingtao
- Zhang, Yunke
- Shang, Yu
- Feng, Jie
- Zhang, Yuheng
- Zong, Zefang
- Yuan, Yuan
- Su, Hongyuan
- Li, Nian
- Piao, Jinghua
- Deng, Yucheng
- Sukiennik, Nicholas
- Gao, Chen
- Xu, Fengli
- Li, Yong
year: 2024
venue: ACM Computing Surveys (2025)
arxiv_id: '2411.14499'
url: https://arxiv.org/abs/2411.14499
pdf: ../../raw/ding-2024-csur.pdf
tags:
- world-model
- survey
- video-generation
- self-supervised-learning
- autonomous-driving
- robotics
- embodied-AI
- model-based-rl
created: 2026-04-10
updated: 2026-04-10
cites:
- feng-2025-arxiv
- ha-2018-neurips
- hafner-2019-icml
- hafner-2021-iclr
- hafner-2023-arxiv
- kong-2025-arxiv
- lecun-2022-openreview
- li-2025-arxiv
cited_by:
- feng-2025-arxiv
- kong-2025-arxiv
- li-2025-arxiv

---

# Understanding World or Predicting Future? A Comprehensive Survey of World Models

> **Survey** — Ding et al. organize 338+ papers on world models under a binary taxonomy — implicit representation (understanding the world) vs. future prediction — and map coverage across four application domains: generative games, embodied intelligence, autonomous driving, and social simulacra.

**Authors:** Jingtao Ding, Yunke Zhang, Yu Shang, Jie Feng, Yuheng Zhang, Zefang Zong, Yuan Yuan, Hongyuan Su, Nian Li, Jinghua Piao, Yucheng Deng, Nicholas Sukiennik, Chen Gao, Fengli Xu, Yong Li (Tsinghua University / BNRist) | **Venue:** ACM Computing Surveys (2025, accepted Dec 2024) | **arXiv:** [2411.14499](https://arxiv.org/abs/2411.14499)

---

## Problem & Motivation

The 2024 emergence of multimodal large language models (GPT-4, Sora) and video generation models has reignited debate about what "world model" means and whether these systems qualify. There is no unified definitional framework — some researchers (Ha & Schmidhuber 2018) treat world models as internal representations for decision-making, while others (LeCun 2022, Sora) emphasize future prediction. Prior surveys focus narrowly on video generation or autonomous driving without cross-domain synthesis. This survey addresses the gap by offering the first comprehensive treatment of world models across both functions and four major application domains, clarifying what each domain actually needs from a world model.

---

## Core Idea

World models serve two primary functions: (1) constructing implicit internal representations of the external world that enable understanding and decision-making, and (2) predicting future states of the external world, primarily through visual simulation. These functions are not mutually exclusive — the most capable systems pursue both — but different application domains emphasize one over the other. The survey adopts this binary as its organizing principle and maps the entire landscape of 2018–2025 world model research onto it.

---

## How It Works

### Coverage & Methodology

The survey covers ~338 papers published from 2018 through mid-2025. Organization follows Figure 2's framework:
- Section 3: Implicit representation (decision-making world models + world knowledge in LLMs)
- Section 4: Future prediction (video generation + embodied environments)
- Section 5: Applications (gaming, embodied intelligence, autonomous driving, social simulacra)
- Section 6: Open problems and future directions

Fourteen summary tables catalogue representative methods with year, venue, modality, task, and technique.

### Category 1: Implicit Representation of the External World (Section 3)

**3.1 World Models in Decision-Making (Model-Based RL)**

The survey frames MBRL world models as learned environment transition functions M(s, a) → s'. Key approaches include:
- *Deterministic transition models* (MSE prediction): PETS, MBPO, PILCO
- *Probabilistic transition models* (KL divergence): Dreamer V1/V2/V3. DreamerV3 solves 150+ tasks including Minecraft diamond collection without domain-specific tuning.
- *Transformer-based unified models*: Janner et al.'s Trajectory Transformer, TD-MPC2 (integrates trajectory optimization in latent space).
- *LLM-backbone world models*: LLMs directly generating actions (navigation, web agents); modular approaches pairing GPT-4 with PDDL planners (Guan et al.); WebDreamer using specialized LLMs as web simulators; WorldCoder building/refining Python programs as world models.

MPC-based policy generation: planning a τ-step reward-maximizing action sequence using the world model. MCTS extensions for continuous action spaces. AlphaGo/AlphaZero as discrete-space MCTS applications.

**3.2 World Knowledge Learned by Models**

LLMs acquire three categories of world knowledge through pretraining:
- *Common sense and general knowledge*: benchmarked by KoLA, EWOK, Geometry of Concepts, BLEnD.
- *Knowledge of the global physical world*: spatial/temporal neurons in LLaMA2; GeoLLM for geospatial prediction; UrbanLLaVA for urban spatial understanding.
- *Knowledge of local physical world*: cognitive map emergence; WM-ABench; Spatial457 spatial reasoning benchmark.
- *Knowledge of human society*: Theory of Mind (Testing ToM, High-order ToM, MuMA-ToM); LLM-ToM; SafeWorld; cultural knowledge (100-language benchmark).

Table 1 summarizes 30+ representative works across these four knowledge categories.

### Category 2: Future Prediction of the Physical World (Section 4)

**4.1 World Models as Video Generation**

Video world models generate sequences of future frames representing how the world will evolve. Key capabilities required:
- *Long-term prediction*: NUWA-XL (coarse-to-fine diffusion), LWM (long video + language transformers), StreamingT2V (autoregressive with short/long-term memory blocks).
- *Multi-modal integration*: 3D-VLA (integrates 3D perception, reasoning, action), Pandora (world-state simulation with free-text).
- *Interactivity and action-conditioning*: VideoDecision, iVideoGPT (visual + action + reward), NWM (navigation-conditioned), Aether (camera-trajectory-conditioned).
- *Temporal consistency*: WorldGPT (multimodal refined keyframe generation), WorldMem (integrated memory mechanism), ConsistI2V.
- *Physical law adherence*: Cosmos (NVIDIA, 2025; physics pre-training breakthrough); Genesis (re-engineered physics core).

Sora (OpenAI, 2024) is the watershed moment — minute-long temporally coherent video from text, suggesting emergent physics. But studies reveal persistent failures: inaccurate gravity/fluid/thermal dynamics, poor causal reasoning (actions don't reliably alter events).

Table 2 catalogues 20+ video world model systems by category, backbone, and technique.

**4.2 World Models as Embodied Environments**

Transitioning from video generation to fully interactive simulation. Three environment types (Table 3, 30+ systems):
- *Indoor environments* (static/structured): AI2-THOR, Matterport 3D, Virtual Home, Habitat, SAPIEN, iGibson, AVLEN, ProTHOR, Holodeck, AnyHome, LEGENT.
- *Outdoor environments* (large-scale, variable): MetaUrban, UrbanWorld (generative 3D urban scenes), MineDOJO (Minecraft open-world).
- *Dynamic environments* (generative, action-conditioned): UniSim (dynamically generates robot manipulation video), Pandora, AVID, Streetscapes (urban video diffusion with weather/traffic), Roboscape, TesserAct (normal-map physics constraints), Deepverse (4D autoregressive), Aether.

Trend: from static asset-based environments → first-person dynamic generative environments. Physics constraints (depth maps, normal maps, geometric predictions) increasingly embedded in generation.

### Application Domains (Section 5)

**5.1 Game Intelligence**

Games as ideal testbeds (well-defined rules, clear action-consequence structure). Three capability dimensions:
- *Interactivity*: GameNGen (real-time neural game engine, 20 fps); GameGen-X (multi-modal game control signals); Matrix-Game (17B parameter fine-grained keyboard/mouse control).
- *Consistency*: MineWorld (visual-action autoregressive transformer); WHAM (consistent gameplay with user modifications).
- *Generalization*: GameFactory (scene-generalizable action control, open-domain generative priors); "generative infinite game" concept.

**5.2 Embodied Intelligence (Robotics)**

Three learning tasks (Table 4):
- *Learning implicit representation*: CNN/ViT visual representations; RoboCraft (GNN particle-based system representation); PointNet/SpatialLM (3D point cloud processing); BC-Z, Text2Motion, Gensim (task representation via language).
- *Predicting future environment*: UniPi, VIPER, GR-2, IRASim, VPP, DreamGen, Roboscape, EVAC, V-JEPA 2 — video prediction for robot action guidance and synthetic data augmentation.
- *Sim-to-real transfer*: DayDreamer (real-world locomotion in hours); SWIM (30-min human video fine-tuning); NeBula (structured belief space for diverse morphologies).

Key insight: generative video world models bridge the sim-to-real gap by learning generalized representations of real-world dynamics.

**5.3 Urban Intelligence**

Two sub-domains:
- *Autonomous driving*: Four-component pipeline (perception → prediction → planning → control). Table 5 catalogues scene understanding (Faster RCNN, MultiNet, BEVFormer, Transfusion) and driving world simulators (GAIA-1, DriveDreamer, Drive-WM, OccWorld, OccSora, CopilotD4). Trend from geometric-space trajectory simulation → video-based world simulators.
- *Autonomous logistics and urban analytics*: Micromobility (NWM, URBAN-SIM, Vid2Sim, CityWalker), aerial vehicles (AirScape, CityNavAgent), urban knowledge (CityGPT, UrbanLLaVA, GeoLLM), mobility prediction (AgentMove, CAMS, PIGEON). Table 6 summarizes 16+ systems.

**5.4 Societal Intelligence (Social Simulacra)**

LLM-driven social simulacra as explicit world models of human society. Two roles (Table 7):
- *Mirroring real-world society*: Generative Agents (AI Town), S3 (social networks), EconAgent (macroeconomics), SRAP-Agent (policy evaluation), AgentSociety (most advanced large-scale platform), SocioVerse.
- *Understanding the external world (agent's implicit world model)*: Agent-Pro (interaction → structured beliefs), GovSim (sustainable cooperation via collective cognition), AgentGroupChat (group deliberation).

**5.5 World Model Functions: Cloud vs. Edge**

- *Cloud-side (data engine)*: video generation for training data synthesis, RL environments, policy evaluation.
- *Edge-side (agent brain)*: compressed latent-space world model for on-device planning (V-JEPA 2 with MPC).

---

## Results / Key Findings

1. **The understanding vs. prediction divide is real but porous**: MBRL systems (DreamerV3) use implicit world models for decision-making without pixel generation; video generators (Sora, Cosmos) prioritize future prediction but lack causal/counterfactual reasoning. The most capable future systems will need both.

2. **Video generation ≠ world simulation**: Sora produces visually coherent videos but fails systematically on physics (gravity, fluid dynamics, thermal processes). Physics-IQ benchmark shows current video generators achieve visual realism while failing physics understanding. Data-driven scaling alone cannot recover robust physical laws.

3. **LLMs as world model backbones**: LLMs have emergent spatial/temporal neurons and geographic/social knowledge, but urban knowledge is coarse and inaccurate (Feng et al.). LLMs partially converge to representations isomorphic to vision model representations (Li et al.).

4. **Generative embodied environments are the new frontier**: Transition from static asset-based simulators (AI2-THOR, Matterport3D) to action-conditioned dynamic video simulators (UniSim, Pandora, Streetscapes) dramatically reduces environment setup effort.

5. **Physics integration is the critical missing ingredient**: Hybrid hard+soft approaches (Genesis with re-engineered physics, PhysGen coupling rigid-body + diffusion, physics-informed PDE residual losses) significantly outperform pure data-driven approaches for counterfactual reasoning.

6. **Social simulacra at scale**: AgentSociety demonstrates societal-scale simulation (polarization, policy interventions). LLM-agent social simulacra successfully reproduce emergent network dynamics, strategic behavior, and macroeconomic trends.

7. **Benchmarking gap**: No single canonical benchmark exists. Specialized benchmarks (Physics-IQ, T2VPhysBench, VBench-2.0, EAI, EWMBench) expose specific failure modes. Standardizing evaluation across understanding and prediction tasks remains a critical open problem.

---

## Comparison to Prior Work / Related Surveys

| Survey | Scope | Year | Domains | Key Taxonomy |
|---|---|---|---|---|
| [[feng-2025-arxiv]] ([Feng et al., P007](../papers/feng-2025-arxiv.md)) | Autonomous driving only | 2025 | AD | Generation paradigm × downstream task |
| [[li-2025-arxiv]] ([Li et al., P008](../papers/li-2025-arxiv.md)) | Embodied AI only | 2025 | Robotics | Representation × model type |
| [[kong-2025-arxiv]] ([Kong et al., P009](../papers/kong-2025-arxiv.md)) | 3D/4D world modeling | 2025 | 3D/4D scenes | Reconstruction vs. generation |
| **Ding et al. (P006)** | General world models | 2024 | Gaming, AD, robotics, social | Understanding vs. prediction |

**vs. Domain-specific surveys**: Ding et al. is uniquely cross-domain, connecting techniques that appear independently in robotics, autonomous driving, and social AI literature. The gaming and social simulacra coverage is absent from other surveys.

**vs. [[lecun-2022-openreview]] ([LeCun 2022, JEPA](../papers/lecun-2022-openreview.md))**: Ding et al. adopts LeCun's JEPA framework as one approach under the implicit representation category, but maps the full landscape rather than advocating a single architecture. Notably, they cover the video generation branch (Sora, Cosmos) that LeCun's position paper would consider misguided.

---

## Strengths
- Only survey that spans all four major application domains (gaming, embodied AI, autonomous driving, social simulacra) under a unified taxonomy.
- Extensive tabular coverage: 8 tables, 14 domain-specific summaries cataloguing 300+ systems.
- Clear historical roadmap (Figure 1) tracing world model development from Minsky's frames (1960s) through Sora (2024) to Genie 3 (2025).
- Balanced treatment of both implicit (MBRL) and generative (video) paradigms without privileging one.
- Societal intelligence coverage (social simulacra) is unique among world model surveys.
- Timely identification of physics failure modes and the need for hybrid physics-data approaches.

## Weaknesses & Limitations
- Limited technical depth on any single approach — breadth comes at the cost of mechanistic understanding.
- The understanding/prediction taxonomy is intuitive but not always crisp: DreamerV3's RSSM generates latent predictions but is primarily a decision-making model.
- Benchmarking section is descriptive rather than comparative — no meta-analysis of cross-benchmark performance trends.
- Social simulacra section is thinner than other domains; the connection between agent-level world models and societal-level world models deserves deeper treatment.
- Published December 2024; rapidly superseded by early 2025 systems (V-JEPA 2, Genie 3, Cosmos) which appear only in brief mentions.
- Coverage of 3D/4D spatial world models is limited (addressed more thoroughly by Kong et al., P009).

## Key Takeaways
- **The field has bifurcated**: MBRL systems build compact implicit world models for planning; generative video models build expressive future simulators. Both are "world models" but serve fundamentally different needs, and no current system fully integrates both.
- **Video generation is necessary but not sufficient**: Sora-class models demonstrate impressive temporal coherence but lack causal/counterfactual reasoning and fail basic physics tests — the core capabilities required for decision-making world models.
- **Physics integration is the next frontier**: Pure data-driven scaling cannot recover physical laws; hybrid approaches embedding physics simulators or PDE constraints are emerging as the path forward.
- **LLMs bring world knowledge but not world simulation**: LLMs have useful implicit representations of geographic, social, and common-sense knowledge but lack the dynamic simulation capabilities needed for embodied control.
- **Societal and urban applications are underserved**: Social simulacra and urban analytics represent high-impact application domains where world model capabilities are still nascent, offering the greatest opportunities for future research.

---

## BibTeX
```bibtex
@article{ding2024understanding,
  title     = {Understanding World or Predicting Future? {A} Comprehensive Survey of World Models},
  author    = {Ding, Jingtao and Zhang, Yunke and Shang, Yu and Feng, Jie and Zhang, Yuheng and Zong, Zefang and Yuan, Yuan and Su, Hongyuan and Li, Nian and Piao, Jinghua and Deng, Yucheng and Sukiennik, Nicholas and Gao, Chen and Xu, Fengli and Li, Yong},
  journal   = {ACM Computing Surveys},
  year      = {2024},
  month     = {December},
  note      = {arXiv:2411.14499},
  url       = {https://arxiv.org/abs/2411.14499}
}
```

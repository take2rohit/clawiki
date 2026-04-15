---
title: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
type: paper
paper_id: P024
authors:
  - "Assran, Mahmoud"
  - "Bardes, Adrien"
  - "Fan, David"
  - "Garrido, Quentin"
  - "Howes, Russell"
  - "Komeili, Mojtaba"
  - "Muckley, Matthew"
  - "Rizvi, Ammar"
  - "Roberts, Claire"
  - "Sinha, Koustuv"
  - "Zholus, Artem"
  - "Arnaud, Sergio"
  - "Gejji, Abha"
  - "Martin, Ada"
  - "Hogan, Francois Robert"
  - "Dugas, Daniel"
  - "Bojanowski, Piotr"
  - "Khalidov, Vasil"
  - "Labatut, Patrick"
  - "Massa, Francisco"
  - "Szafraniec, Marc"
  - "Krishnakumar, Kapil"
  - "Li, Yong"
  - "Ma, Xiaodong"
  - "Chandar, Sarath"
  - "Meier, Franziska"
  - "LeCun, Yann"
  - "Rabbat, Michael"
  - "Ballas, Nicolas"
year: 2025
venue: arXiv (Meta FAIR)
arxiv_id: "2506.09985"
url: "https://arxiv.org/abs/2506.09985"
pdf: "../../raw/assran-2025-arxiv.pdf"
tags: [V-JEPA, self-supervised, video, world-model, robotics, planning]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
cited_by: []
---

# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

> **V-JEPA 2** is a 1B-parameter self-supervised video model that scales the JEPA framework to 1M hours of internet video, achieving 77.3 top-1 on SSv2, 39.7 recall-at-5 on EK100 action anticipation (44% relative improvement over prior SOTA), state-of-the-art Video QA at the 8B model class (84.0 PerceptionTest, 76.9 TempCompass), and zero-shot robotic pick-and-place on Franka arms across two labs -- all without task-specific training or reward.

**Authors:** Mahmoud Assran\*, Adrien Bardes\*, David Fan\*, Quentin Garrido\*, Russell Howes\*, Mojtaba Komeili\*, Matthew Muckley\*, Ammar Rizvi\*, Claire Roberts\*, Koustuv Sinha\*, Artem Zholus\*, et al. (29 authors; \*core team) | **Venue:** arXiv (Meta FAIR, June 2025) | **arXiv:** [2506.09985](https://arxiv.org/abs/2506.09985)

---

## Problem & Motivation

A core challenge in AI is building systems that can *understand* the physical world, *predict* future states, and *plan* actions -- like humans do by maintaining an internal world model learned largely through observation. Existing approaches face key limitations:

1. **Reward-dependent world models** (e.g., Dreamer, TD-MPC) require explicit reward signals and train task-specific models from interaction data, which is scarce in the real world. They do not scale to internet video.
2. **Video generation models** (e.g., Cosmos, Sora) learn pixel-level predictions that waste capacity on unpredictable visual details (every blade of grass, every leaf on a tree). They emphasize visual fidelity over planning utility, and are computationally expensive at inference.
3. **Vision-Language-Action models** (e.g., RT-2, Octo, pi-0) rely on behavior cloning from high-quality expert demonstrations and lack an explicit predictive model of the world. They cannot reason about unseen situations or leverage failed demonstrations.

The JEPA framework ([[lecun-2022-openreview]](../papers/lecun-2022-openreview.md)) proposed predicting in *representation space* rather than pixel space, focusing on predictable aspects of a scene while ignoring stochastic details. V-JEPA (Bardes et al., 2024) demonstrated this for short video clips. The question is: can this approach scale to build a world model that is simultaneously useful for understanding, prediction, and planning in the physical world?

---

## Core Idea

V-JEPA 2 demonstrates that a single self-supervised video encoder, trained with the JEPA mask-denoising objective on internet-scale data, produces representations rich enough to support three distinct capabilities:

1. **Understanding** -- When evaluated with frozen probes or aligned with an LLM, V-JEPA 2 representations achieve state-of-the-art performance on video classification, action recognition, and video question answering.
2. **Prediction** -- The same representations enable state-of-the-art action anticipation on EK100, far surpassing specialized supervised models.
3. **Planning** -- By post-training a lightweight action-conditioned predictor (V-JEPA 2-AC) on only 62 hours of unlabeled robot video, the model enables zero-shot robotic manipulation via model-predictive control in new environments.

The key insight is that self-supervised learning on internet video can serve as a foundation for a world model, and a small amount of interaction data is sufficient to ground this world model in the action space of a physical robot.

---

## How It Works

### Stage 1: Self-Supervised Video Pretraining (V-JEPA 2)

**Objective.** The mask-denoising objective from V-JEPA (Bardes et al., 2024) and I-JEPA (Assran et al., 2023): given a video, randomly drop patches to create a masked view *x*, and train a predictor to reconstruct the representation of the unmasked video *y* in the latent space of an EMA encoder:

```
minimize_{theta, phi, Delta_y}  || P_phi(Delta_y, E_theta(x)) - sg(E_bar{theta}(y)) ||_1
```

where Delta_y is a learnable mask token, sg is stop-gradient, and E_bar{theta} is the exponential moving average encoder. This prevents representation collapse without negative samples or contrastive losses.

**Architecture.** Both encoder E and predictor P are Vision Transformers (ViT). V-JEPA 2 scales to ViT-g (1B parameters). Key architectural changes from V-JEPA:
- **3D-RoPE** replaces absolute sincos positional embeddings -- the feature dimension is partitioned into three segments for temporal, height, and width axes, each receiving 1D rotary embeddings. This stabilizes training at scale.
- **Tubelet size:** 2 x 16 x 16 (T x H x W) with the same multiblock masking strategy as V-JEPA.
- Predictor is a ViT-small, shared across all encoder scales.

**Four scaling ingredients** take V-JEPA to V-JEPA 2, each contributing additive gains (cumulative +4.0 points average accuracy on 6 tasks):

| Ingredient | Change | Gain |
|------------|--------|------|
| Data scaling | 2M to 22M videos (VideoMix22M) | +1.0 |
| Model scaling | ViT-L (300M) to ViT-g (1B) | +1.5 |
| Longer training | 90K to 252K iterations (warmup-constant-decay schedule) | +0.8 |
| Higher resolution | 256 to 384 spatial; 16 to 64 frames | +0.7 |

**Dataset: VideoMix22M (VM22M).** Combines publicly available sources:

| Source | Samples | Type | Hours |
|--------|---------|------|-------|
| SSv2 | 168K | EgoVideo | 168 |
| Kinetics | 733K | ExoVideo | 614 |
| HowTo100M | 1.1M | ExoVideo | 134K |
| YT-Temporal-1B (curated) | 19M | ExoVideo | 1.6M |
| ImageNet | 1M | Images | n/a |

YT-Temporal-1B is curated using retrieval-based filtering (cluster-based selection toward a target distribution), yielding +1.4 point improvement over uncurated data.

**Progressive resolution training.** Training at full resolution (64 frames x 384 x 384) for the entire run would cost ~60 GPU-years for ViT-g. Instead, V-JEPA 2 trains at 16 frames / 256 x 256 for the warmup and constant phases (240K iterations), then increases resolution during the 12K-iteration cooldown phase. This achieves 8.4x reduction in GPU time while matching full-resolution performance.

### Stage 2: Action-Conditioned World Model (V-JEPA 2-AC)

After pretraining, the V-JEPA 2 encoder is **frozen**. A new action-conditioned predictor P_phi is trained on top to predict future frame representations given past frames, actions, and end-effector states.

**Data.** 62 hours of unlabeled video from the Droid dataset (Khazatsky et al., 2024) -- teleoperated Franka Panda robot arm. No reward, no task labels, no success/failure annotations. Only raw video + 7D end-effector state (3D position, 3D orientation as Euler angles, 1D gripper state). Actions are computed as the change in end-effector state between consecutive frames.

**Architecture.** The predictor is a ~300M parameter transformer (24 layers, 16 heads, 1024 hidden dim, GELU activations) with block-causal attention: each patch feature at time step t can attend to patches, actions, and states from t and all previous time steps. Feature maps from the frozen ViT-g encoder are 16 x 16 x 1408 per frame. Actions, states, and flattened feature maps are projected via separate learnable affine transformations to the predictor's hidden dimension. 3D-RoPE encodes spatiotemporal positions for patch tokens; only temporal RoPE is applied to action and state tokens.

**Training objective.** The sum of a teacher-forcing loss and a rollout loss:

```
L(phi) = L_teacher-forcing(phi) + L_rollout(phi)
```

- **Teacher forcing:** L1 loss between predicted and actual next-frame representations, averaged over T=15 steps per clip. At each step, the predictor receives the ground-truth encoding of the current frame.
- **Rollout loss:** L1 loss between the model's own autoregressive rollout (T=2 steps, differentiating through one recurrent step) and the ground-truth future representation. This improves robustness to error accumulation during multi-step planning.

### Stage 3: Planning via Model Predictive Control

Given a goal image x_g, V-JEPA 2-AC plans at each time step by finding an action sequence that minimizes an energy function in representation space:

```
E(a_{1:T}; z_k, s_k, z_g) = || P(a_{1:T}; s_k, z_k) - z_g ||_1
```

where z_k = E(x_k) is the current frame representation and z_g = E(x_g) is the goal representation. Optimization uses the **Cross-Entropy Method** (CEM): sample action sequences from Gaussians, evaluate energy, update distribution toward top-k performers, repeat for several refinement iterations. Only the first action is executed (receding horizon control).

For prehensile manipulation (pick-and-place), the model is given sub-goal images: (1) object grasped, (2) object near target location, (3) object placed. The planner optimizes with respect to each sub-goal in sequence.

---

## Results

### Understanding: Probe-based Classification (Table 4)

Frozen V-JEPA 2 encoder + 4-layer attentive probe, evaluated on 6 tasks:

| Method | Params | Avg. | SSv2 | Diving-48 | Jester | K400 | COIN | IN1K |
|--------|--------|------|------|-----------|--------|------|------|------|
| DINOv2 | 1.1B | 81.1 | 50.7 | 82.5 | 93.4 | 83.6 | 90.7 | 86.1 |
| PE_core G | 1.9B | 82.3 | 55.4 | 76.9 | 90.0 | 88.5 | **95.3** | 87.6 |
| SigLIP2 | 1.2B | 81.1 | 49.9 | 75.3 | 91.0 | 87.3 | 95.1 | **88.0** |
| V-JEPA ViT-H | -- | 85.2 | 74.3 | 87.9 | 97.7 | 84.5 | 87.1 | 80.0 |
| InternVideo2s2-1B | 1B | 87.0 | 69.7 | 86.4 | 97.0 | **89.4** | 93.8 | 85.8 |
| **V-JEPA 2 ViT-g** | 1B | **87.5** | 75.3 | 90.1 | 97.7 | 86.6 | 90.7 | 84.6 |
| **V-JEPA 2 ViT-g_384** | 1B | **88.2** | **77.3** | **90.2** | **97.8** | 87.3 | 91.1 | 85.1 |

V-JEPA 2 achieves the best average performance across all six tasks. It significantly outperforms all other encoders on motion understanding (SSv2: 77.3, Diving-48: 90.2, Jester: 97.8) while remaining competitive on appearance tasks.

### Prediction: Human Action Anticipation on EK100 (Table 5)

| Method | Params | Verb R@5 | Noun R@5 | Action R@5 |
|--------|--------|----------|----------|------------|
| InAViT | 160M | 51.9 | 52.0 | 25.8 |
| Video-LLaMA | 7B | 52.9 | 52.0 | 26.0 |
| PlausiVL | 8B | 55.6 | 54.2 | 27.6 |
| **V-JEPA 2 ViT-g_384** | **1B** | **63.6** | **57.1** | **39.7** |

V-JEPA 2 achieves 39.7 action recall-at-5 -- a +12.1 point (44% relative) improvement over PlausiVL (8B), using only 1B parameters and a frozen backbone with an attentive probe.

### Understanding: Video Question Answering (Table 8)

V-JEPA 2 aligned with Llama 3.1 8B, using 88.5M alignment samples:

| Method | Enc/LLM | Avg. | PercepTest | MVP | TempCompass | TemporalBench | TOMATO | TVBench | MVBench |
|--------|---------|------|------------|-----|-------------|--------------|--------|---------|---------|
| InternVL-2.5 | 300M/7B | 52.1 | 68.9 | 39.9 | 68.3 | 24.3 | 29.4 | 61.6 | 72.6 |
| Qwen2VL | 675M/7B | 47.0 | 66.9 | 29.2 | 67.9 | 20.4 | 31.5 | 46.0 | 67.0 |
| Qwen2.5VL | 1B/7B | 49.7 | 70.5 | 36.7 | 71.7 | 24.5 | 24.6 | 50.5 | 69.6 |
| PLM 8B | 1B/8B | 56.7 | 82.7 | 39.7 | 72.7 | 28.3 | 33.2 | **63.5** | **77.1** |
| **V-JEPA 2 ViT-g_384** | **1B/8B** | **59.5** | **84.0** | **44.5** | **76.9** | **36.7** | **40.3** | 60.6 | 73.5 |

V-JEPA 2 sets new state-of-the-art in the 8B class on PerceptionTest (+1.3 over PLM), MVP (+4.8), TempCompass (+4.2), TemporalBench (+8.4), and TOMATO (+7.1). This is the first demonstration that a video encoder pretrained *without any language supervision* can achieve state-of-the-art VidQA when aligned with an LLM.

### Planning: Zero-Shot Robot Manipulation (Table 2)

Deployed zero-shot on Franka Emika Panda arms with RobotiQ grippers in two different labs (neither in Droid training set):

| Method | Reach | Grasp (Cup/Box) | Reach w/ Obj (Cup/Box) | Pick-&-Place (Cup/Box) |
|--------|-------|-----------------|------------------------|------------------------|
| Octo (avg) | 100% | 15% / 0% | 15% / 70% | 15% / 10% |
| **V-JEPA 2-AC (avg)** | **100%** | **65% / 25%** | **75% / 75%** | **80% / 65%** |

V-JEPA 2-AC achieves the highest success rate across all tasks. On the full pick-and-place task, it reaches 80% (cup) and 65% (box) vs. 15% and 10% for Octo.

### Planning: V-JEPA 2-AC vs. Cosmos (Table 3, Lab 2)

| Method | Samples | Iter. | Horizon | Time/action | Reach | Grasp (Cup/Box) | Pick-&-Place (Cup/Box) |
|--------|---------|-------|---------|-------------|-------|-----------------|------------------------|
| Cosmos | 80 | 10 | 1 | 4 min | 80% | 0% / 20% | 0% / 0% |
| **V-JEPA 2-AC** | **800** | **10** | **1** | **16 sec** | **100%** | **60% / 20%** | **80% / 50%** |

V-JEPA 2-AC is 15x faster per action (16 sec vs. 4 min) and achieves substantially higher success rates. Planning in representation space is both faster and more effective than planning through a video generation model.

---

## Comparison to Prior Work

| Method | Paradigm | Pretraining | Action-Conditioned | Internet Video Scale | Zero-Shot Robot Control |
|--------|----------|-------------|-------------------|---------------------|------------------------|
| V-JEPA (Bardes et al., 2024) | JEPA mask-denoising | SSL, 2M videos | No | No | No |
| I-JEPA (Assran et al., 2023) | JEPA mask-denoising | SSL, images only | No | No | No |
| Dreamer V3 / TD-MPC2 | Latent WM + RL | Reward-based | Yes | No | No |
| Cosmos | Video generation | SSL + action fine-tune | Yes | Yes (20M hours) | Limited |
| Octo | VLA (behavior cloning) | Supervised | Yes | No | Yes (Droid) |
| NWM ([[bar-2024-cvpr]](../papers/bar-2024-cvpr.md)) | Pixel-space diffusion WM | SSL video + actions | Yes | Partially (Ego4D) | Navigation only |
| LeWM ([[maes-2026-arxiv]](../papers/maes-2026-arxiv.md)) | End-to-end JEPA WM | SSL + actions | Yes | No (single tasks) | Simulated only |
| **V-JEPA 2 / 2-AC** | **JEPA mask-denoising + AC predictor** | **SSL, 1M hours** | **Yes (post-training)** | **Yes** | **Yes (manipulation)** |

**[[lecun-2022-openreview]](../papers/lecun-2022-openreview.md) (JEPA position paper).** V-JEPA 2 is a direct realization of the JEPA world model vision: learning representations through self-supervised prediction in latent space, then using those representations for planning via energy minimization. The V-JEPA 2-AC planning loop mirrors LeCun's proposed architecture where a world model predicts future states and an actor optimizes actions to minimize a cost (energy) function. V-JEPA 2 validates the core thesis that representation-space prediction, not pixel reconstruction, is the path to world models that understand, predict, and plan.

**I-JEPA (Assran et al., 2023) and V-JEPA (Bardes et al., 2024).** V-JEPA 2 directly extends V-JEPA by scaling the encoder from ViT-H (600M) to ViT-g (1B), the dataset from 2M to 22M videos, and training duration from 90K to 252K iterations. Key architectural changes include 3D-RoPE (replacing absolute sincos embeddings) and progressive resolution training. V-JEPA 2 also extends the framework beyond pure understanding to prediction (action anticipation) and planning (robotic control) -- capabilities not demonstrated in V-JEPA or I-JEPA.

**[[bar-2024-cvpr]](../papers/bar-2024-cvpr.md) (Navigation World Models).** Both NWM and V-JEPA 2-AC use world models for robotic planning with CEM optimization. However, NWM generates pixel-level predictions (diffusion in pixel/latent image space), while V-JEPA 2-AC plans entirely in representation space, making it 15x faster per planning step. NWM focuses on navigation (3-DoF), while V-JEPA 2-AC tackles manipulation (7-DoF). NWM's CDiT architecture is bespoke; V-JEPA 2 reuses a general-purpose self-supervised video encoder.

**[[maes-2026-arxiv]](../papers/maes-2026-arxiv.md) (LeWorldModel).** Both pursue JEPA-based world models, but from different starting points. LeWM trains a 15M-parameter model end-to-end from pixels on single tasks (DMControl), using SIGReg for anti-collapse. V-JEPA 2 uses a two-stage approach: massive SSL pretraining (frozen 1B encoder) followed by a lightweight action-conditioned predictor. V-JEPA 2 demonstrates real-world robot control; LeWM remains in simulation. The approaches are complementary: LeWM's end-to-end training could potentially be combined with V-JEPA 2's scale.

---

## Strengths

- **Unified framework for three capabilities.** A single pretrained encoder supports understanding (classification, VQA), prediction (action anticipation), and planning (robot manipulation) -- validating the JEPA world model thesis more comprehensively than any prior work.
- **Remarkable data efficiency for robotics.** Only 62 hours of unlabeled robot video (no rewards, no task labels, no success annotations) enables zero-shot manipulation in novel environments. This is orders of magnitude less interaction data than behavior cloning or RL approaches.
- **First language-free encoder to achieve SOTA VidQA.** V-JEPA 2 is pretrained without any language supervision, yet when aligned with an LLM it outperforms vision encoders trained with language (SigLIP, PE) on temporal understanding benchmarks. This challenges the conventional wisdom that language-supervised pretraining is necessary for VidQA.
- **Efficient planning in representation space.** V-JEPA 2-AC requires only 16 seconds per action on a single RTX 4090, compared to 4 minutes for Cosmos. Planning in abstract representation space is both faster and more effective than planning through video generation.
- **Strong scaling behavior.** Performance improves consistently with data scale (2M to 22M videos), model scale (300M to 1B), training duration, and resolution. No saturation is observed, suggesting further gains are achievable.
- **Progressive resolution training.** Achieves 8.4x compute reduction while matching full-resolution performance -- a practical contribution for training large video models.
- **Open-source.** Code released at github.com/facebookresearch/vjepa2.

## Weaknesses & Limitations

- **Camera positioning sensitivity.** V-JEPA 2-AC must implicitly infer the action coordinate axis from the monocular RGB camera. The robot base is often not visible, making this ill-defined. The authors manually tried different camera positions before finding one that worked, which limits practical deployment.
- **Short planning horizon.** The model is limited to predictions ~16 seconds into the future. Error accumulation in autoregressive rollouts degrades accuracy over longer horizons. Pick-and-place succeeds because it can be decomposed into short sub-goals, but more complex multi-step tasks remain out of reach.
- **Image-only goal specification.** Tasks must be specified as goal images (or sequences of sub-goal images). Language-based goal specification would be far more practical for real deployment. The authors note that aligning V-JEPA 2-AC with a language model is future work.
- **Limited manipulation complexity.** Demonstrated tasks (reach, grasp, pick-and-place) are relatively simple. No deformable objects, no multi-object rearrangement, no tool use. Success rates on grasping boxes (25% average) highlight that fine manipulation remains challenging.
- **No end-to-end training of action-conditioned model.** The encoder is frozen during action-conditioned training. This means the representations are not optimized for control -- they must happen to be useful. An end-to-end approach (as in [[maes-2026-arxiv]](../papers/maes-2026-arxiv.md)) could potentially learn more action-relevant features.
- **Scale ceiling unknown.** The largest model is 1B parameters -- well below frontier vision encoders (20B+). Whether V-JEPA 2's recipe continues to scale is an open question.
- **VidQA gaps.** V-JEPA 2 does not outperform PLM 8B on TVBench and MVBench, suggesting that some appearance-heavy or complex reasoning benchmarks still favor language-supervised encoders.

## Key Takeaways

- V-JEPA 2 is the most comprehensive validation of the JEPA world model thesis to date: a single self-supervised video encoder achieves SOTA on video understanding, prediction, AND enables zero-shot robotic planning.
- Scaling self-supervised video pretraining (data, model, resolution, training duration) yields consistent gains across all capabilities, with no sign of saturation at 1B parameters.
- Only 62 hours of unlabeled robot video are needed to post-train an action-conditioned world model that generalizes zero-shot to new environments, new objects, and new labs.
- Planning in representation space (V-JEPA 2-AC) is 15x faster and more effective than planning through video generation (Cosmos), providing strong evidence that JEPA-style latent prediction is preferable to pixel-space generation for robotic control.
- An encoder pretrained without any language supervision can achieve state-of-the-art VidQA when aligned with an LLM, overturning the assumption that language-supervised pretraining is necessary.
- The energy landscape induced by V-JEPA 2-AC is smooth and locally convex (Figure 9), suggesting that the learned representation space has favorable geometry for gradient-free planning optimization.

---

## BibTeX

{% raw %}
```bibtex
@article{assran2025vjepa2,
  title={V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and Arnaud, Sergio and Gejji, Abha and Martin, Ada and Hogan, Francois Robert and Dugas, Daniel and Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and Rabbat, Michael and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```
{% endraw %}

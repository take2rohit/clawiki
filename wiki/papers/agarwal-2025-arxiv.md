---
title: "Cosmos World Foundation Model Platform for Physical AI"
type: paper
paper_id: P017
authors:
  - "Agarwal, Niket"
  - "et al. (77 authors, NVIDIA)"
year: 2025
venue: arXiv (NVIDIA)
arxiv_id: "2501.03575"
url: "https://arxiv.org/abs/2501.03575"
pdf: "../../raw/agarwal-2025-arxiv.pdf"
tags: [world-model, foundation-model, video-generation, diffusion, autoregressive, robotics, autonomous-driving]
created: 2026-04-10
updated: 2026-04-10
cites:
  - hu-2023-arxiv
cited_by: []
---

# Cosmos World Foundation Model Platform for Physical AI

> **Cosmos** — NVIDIA's open-weight World Foundation Model (WFM) platform trained on 20M hours of video using 10,000 H100 GPUs over three months, providing pre-trained diffusion (7B/14B) and autoregressive (4B/12B) models that, when post-trained on driving data, achieve FID of 32.16 and FVD of 210.23 compared to VideoLDM-MultiView's 60.84/884.46.

**Authors:** Niket Agarwal et al. (77 authors, NVIDIA) | **Venue:** arXiv 2025 | **arXiv:** [2501.03575](https://arxiv.org/abs/2501.03575)

---

## Problem & Motivation

Physical AI (robots, autonomous vehicles) cannot be safely trained by direct real-world interaction — the data scaling problem is severe, and unsafe exploratory actions can cause real damage. A World Foundation Model (WFM) acting as a digital twin of the physical environment would allow policy training, evaluation, and data synthesis without physical risk. Existing approaches are either too narrow (domain-specific simulators), too low-fidelity (game engines), or too expensive to fine-tune (proprietary video generation models). Furthermore, video tokenization — the compression step that makes large-scale WFM training tractable — has been limited by poor reconstruction quality at high compression ratios. No existing platform provides open-weight pre-trained WFMs, efficient causal tokenizers, a curated 100M-clip training pipeline, and post-training recipes for downstream Physical AI tasks.

---

## Core Idea

Cosmos provides a complete, open-source infrastructure stack for building customized World Foundation Models. The core insight is that a single, large pre-trained WFM can serve as a general visual physics prior that is then fine-tuned (post-trained) with small domain-specific datasets to specialize for different Physical AI applications. The platform provides two complementary WFM families — diffusion-based (for quality) and autoregressive-based (for controllability and LLM interoperability) — along with causal video tokenizers that unify image and video tokenization in a single architecture.

---

## How It Works

### Overview

Raw Internet video (20M hours) → Video Curator (split, filter, annotate, dedup, shard) → ~100M clips → Cosmos Tokenizer (causal continuous/discrete encoder) → Pre-trained WFM (diffusion or autoregressive transformer) → Post-training on domain dataset → Specialized Physical AI WFM → downstream tasks (camera control, robotics, autonomous driving).

### Video Curator

A 5-stage pipeline: (1) **Split** — TransNetV2 neural shot boundary detection (F1=0.967 on BBC); clips 2–60s, H.264/NVENC re-encoded. (2) **Filter** — motion filtering via ViT classifier on optical flow (removes static/shaky videos), visual quality filtering via DOVER (removes bottom 15%), text overlay detection via InternVideo2+MLP, video type filtering via VLM-labeled taxonomy. (3) **Annotate** — VILA 13B VLM generates ~559-character captions per clip (10× throughput via FP8 TRT-LLM). (4) **Dedup** — SemDeDup with k=10,000 GPU-accelerated k-means clustering removes ~30% duplicates. (5) **Shard** — packaged as WebDatasets by resolution/aspect ratio. Dataset composition: driving (11%), manipulation (16%), human motion (10%), spatial navigation (16%), first-person POV (8%), nature (20%), camera dynamics (8%), synthetic (4%), other (7%).

### Cosmos Tokenizer

An attention-based encoder-decoder that supports both continuous (for diffusion WFMs) and discrete (for autoregressive WFMs) tokenization of images and videos in a single unified architecture. The key design features are:

- **Temporal causality:** inputs processed through a 3D Haar wavelet transform (grouping frames as {x_0, x_{1:4}, x_{5:8}, ...}), followed by causal residual blocks with left-padding and causal spatio-temporal self-attention. This ensures no future frames influence current token computation — critical for Physical AI's causal real-world requirement.
- **Compression factors:** spatial s_HW and temporal s_T. Trained at: CI/DI 8×8 and 16×16 (image); CV/DV 4×8×8, 8×8×8, 8×16×16 (video).
- **Continuous tokenizer:** vanilla autoencoder formulation, latent dimension C=16.
- **Discrete tokenizer:** Finite-Scalar Quantization (FSQ) with 6 levels (8,8,8,5,5,5), vocabulary size 64,000.
- **Training losses:** Stage 1 — L1 pixel loss + VGG perceptual loss. Stage 2 — adds optical flow loss (temporal smoothness) + Gram-matrix loss (sharpness) + adversarial loss.

**Results:** Cosmos-Tokenize1-CV4×8×8 achieves PSNR 35.85 / SSIM 0.920 / rFVD 10.057 on DAVIS, outperforming all prior tokenizers. Runtime: 2×–12× faster than prior art on A100 80GB.

### Diffusion-based WFM (Cosmos-Predict1)

Architecture builds on DiT (Peebles & Xie, 2023) with adaptations for video:

- **3D patchification:** latent tokens of shape T×C×H×W flattened via non-overlapping 3D patches (p_t=1, p_h=p_w=2).
- **FPS-aware 3D RoPE:** 3D-factorized Rotary Position Embedding with temporal frequencies rescaled by video FPS — enables joint image+video training at varying rates and resolutions.
- **Conditioning:** T5 text encoder + cross-attention; noise level via adaptive layer normalization (scale, shift, gate) with scale-shift-gate modulation.
- **Uncertainty weighting:** MLP-parameterized u(σ) function weights loss contribution by noise level, treating denoising at different scales as multi-task learning.
- **Model family:** Cosmos-Predict1-7B-Text2World → fine-tuned to Cosmos-Predict1-7B-Video2World. Also 14B variants.
- **Prompt upsampler:** Cosmos-UpsamplePrompt1-12B-Text2World (Mistral-NeMo-12B based) converts brief human prompts to VLM-style detailed captions preferred by the diffusion WFM.

### Autoregressive-based WFM

- Llama3-style GPT models trained from scratch: Cosmos-Predict1-4B and Cosmos-Predict1-12B (base); Cosmos-Predict1-5B-Video2World and Cosmos-Predict1-13B-Video2World (video-conditioned).
- Uses Cosmos-Tokenize1-DV8×16×16 discrete tokens.
- T5 text embeddings injected via cross-attention layers in transformer blocks.
- **Diffusion decoder:** Cosmos-Predict1-7B-Decoder fine-tuned to map DV8×16×16 discrete tokens → CV8×8×8 continuous tokens (reduces artifacts from heavy discrete compression).

### Training

- **Compute:** 10,000 NVIDIA H100 GPUs, three months total for all WFMs reported.
- **Pre-training stages:** Text2World (text → video from scratch) → Video2World (text + past video → future video).
- **Two-stage pre-training for diffusion:** (1) Text2World generation, (2) fine-tune for Video2World conditioning.

### Inference

Diffusion WFMs: DDPM/EDM sampling from Gaussian noise. Autoregressive WFMs: next-token prediction with causal attention. Long videos via sliding window. Post-trained models conditioned on action trajectories, camera poses, or robot end-effector positions depending on downstream task.

---

## Results

### Cosmos Tokenizer (DAVIS dataset, continuous video CV4×8×8)

| Tokenizer | PSNR ↑ | SSIM ↑ | rFVD ↓ |
|-----------|--------|--------|--------|
| CogVideoX-Tokenizer 4×8×8 | 29.20 | 0.864 | 19.58 |
| Omni-Tokenizer 4×8×8 | 22.23 | 0.713 | 117.66 |
| **Cosmos-Tokenize1-CV4×8×8** | **35.85** | **0.920** | **10.057** |

Even at 2× higher compression (CV8×8×8), Cosmos-Tokenizer (30.61 PSNR) still outperforms CogVideoX at 4×8×8.

### Autonomous Driving Post-training (Multi-View)

| Method | FID ↓ | FVD ↓ | TSE ↓ | CSE ↓ |
|--------|-------|-------|-------|-------|
| VideoLDM-MultiView | 60.84 | 884.46 | 1.24 | 6.48 |
| **Cosmos-Predict1-7B-Text2World-MultiView** | **32.16** | **210.23** | 0.68 | 2.11 |
| + TrajectoryConditioned | - | - | **0.59** | **2.02** |
| Real Videos | - | - | 0.69 | 1.71 |

The post-trained Cosmos WFM outperforms VideoLDM-MultiView by 1.9× on FID and 4.2× on FVD. Multi-view consistency (CSE) approaches real video levels (2.02 vs. 1.71).

### Trajectory Consistency (Autonomous Driving, Table 25)

| Method | TAE-ATE ↓ | TAE-RPE-R ↓ | TAE-RPE-t ↓ | TFE ↓ (cm) |
|--------|-----------|------------|------------|-----------|
| VideoLDM-MultiView | 0.88 | 22.94 | 0.77 | - |
| Cosmos-Text2World-MultiView | 0.77 | 4.25 | 0.29 | - |
| **+ TrajectoryConditioned** | **0.54** | 4.31 | **0.18** | **20.20** |
| Real Videos | 0.49 | 4.60 | 0.14 | 13.49 |

Trajectory-conditioned Cosmos WFM achieves trajectory consistency within ~7cm (TFE) of real videos — sufficient for training autonomous driving agents.

### Object Permanence (Qualitative)

In 20 randomly sampled generated driving videos with 157 tracked objects (YOLOv11x), zero objects exhibited physically impossible tracking merges, demonstrating physical consistency.

---

## Comparison to Prior Work

| | Cosmos | GAIA-1 | Sora (OpenAI) | VideoLDM |
|---|--------|--------|---------------|----------|
| Open-weight | Yes | No | No | Partial |
| Tokenizer family | Continuous + Discrete | Discrete only | Unknown | Continuous |
| WFM family | Diffusion + AR | AR + Diffusion decoder | Diffusion | Diffusion |
| Physical AI post-training | Yes (3 domains) | No | No | Partial |
| Guardrail system | Yes | No | Unknown | No |
| Scale | 7B/14B (diff), 4B/12B (AR) | 6.5B (WM) + 2.6B (dec) | ~unknown | ~1B |

**[[hu-2023-arxiv]] ([GAIA-1](../papers/hu-2023-arxiv.md))** is a predecessor covering automotive generation with LLM-style scaling, but uses a proprietary dataset and models. Cosmos generalizes across robotics and driving with open weights.

**VideoLDM-MultiView** is the strongest baseline for multi-view driving generation; Cosmos post-training surpasses it by 4.2× on FVD.

**Sora** is a general-purpose video generation model without Physical AI post-training recipes or open weights.

---

## Strengths
- Only open-weight WFM platform with both diffusion and autoregressive families — democratizes Physical AI research.
- Cosmos Tokenizer achieves PSNR 35.85 on DAVIS at 4×8×8 compression, outperforming all prior tokenizers including those with 2× less compression.
- Unified causal architecture enables joint image+video training — images are just single-frame videos.
- Post-training demonstrations across three physically distinct domains (camera control, robotic manipulation, autonomous driving).
- Trajectory-conditioned driving WFM achieves TFE within 7cm of real videos — practical for AV policy training.
- Guardrail system (pre-Guard: Aegis LLM + keyword blocking; post-Guard: video safety classifier + face blur) enables responsible deployment.

## Weaknesses & Limitations
- Models still lack reliable object permanence in challenging scenarios; contact-rich dynamics (pushing, grasping forces) are often inaccurate.
- Autoregressive WFMs currently underperform diffusion WFMs in visual quality (diffusion models win on 3D consistency and controllability).
- Physical grounding (gravity, fluid dynamics, lighting physics) is often approximate.
- Evaluation relies partly on FID/FVD which do not fully capture Physical AI task performance.
- 10,000 H100 GPU-months of compute is inaccessible to academic groups — open weights help, but training from scratch is not feasible without industrial resources.
- Post-training datasets (RDS for driving, robot datasets) are proprietary; only post-training recipes and model weights are released.

## Key Takeaways
- Cosmos Tokenizer achieves +4 dB PSNR over prior art at equal compression (4×8×8), and at 8×16×16 compression (4× more aggressive), still matches or exceeds prior art at 4×8×8.
- Cosmos post-trained driving WFM achieves FID 32.16 / FVD 210.23 vs. VideoLDM-MultiView's 60.84 / 884.46 — a 1.9× / 4.2× improvement.
- Trajectory-conditioned model achieves TFE of 20.20cm, only ~7cm worse than real videos (13.49cm) — validated for AV agent training.
- 100M training clips curated from 20M hours via a 5-stage pipeline that removes ~30% near-duplicates and bottom-15%-quality content.
- The platform trained on 10,000 H100 GPUs for three months; the open-weight release makes these models accessible for Physical AI research without retraining.

---

## BibTeX
```bibtex
@article{agarwal2025cosmos,
  title={Cosmos World Foundation Model Platform for Physical {AI}},
  author={Agarwal, Niket and others},
  journal={arXiv preprint arXiv:2501.03575},
  year={2025}
}
```

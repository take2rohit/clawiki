---
title: "ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model"
type: paper
paper_id: P052
authors:
  - "Zhang, Haichao"
  - "Li, Yijiang"
  - "He, Shwai"
  - "Nagarajan, Tushar"
  - "Chen, Mingfei"
  - "Lu, Jianglin"
  - "Li, Ang"
  - "Fu, Yun"
year: 2026
venue: arXiv
arxiv_id: "2603.22281"
url: "https://arxiv.org/abs/2603.22281"
pdf: "../../raw/hzhang-2026-arxiv.pdf"
tags: [jepa, vlm, world-model, latent-prediction, dual-pathway, reasoning]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - bardes-2024-tmlr
cited_by: []
---

# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model

> **One sentence.** ThinkJEPA combines a dense JEPA branch for fine-grained frame-level dynamics with a VLM "thinker" branch for long-horizon semantic guidance via a hierarchical pyramid representation extraction module, outperforming both VLM-only and JEPA-only baselines on hand-manipulation trajectory prediction.

**Authors:** Haichao Zhang (Northeastern), Yijiang Li (UCSD), Shwai He (UMD), Tushar Nagarajan (UT Austin), Mingfei Chen (UW), Jianglin Lu (Northeastern), Ang Li (UMD), Yun Fu (Northeastern) | **Venue:** arXiv 2026 | **arXiv:** [2603.22281](https://arxiv.org/abs/2603.22281)

**Note:** Only 1 page of this PDF is available (abstract only). The following wiki page is based on the abstract. Sections are abbreviated accordingly.

---

## Problem & Motivation

Dense JEPA-style latent world models (e.g., V-JEPA 2) predict future latent states from video with dense frame-by-frame observations, which limits temporal context and can bias predictors toward local, low-level extrapolation rather than capturing long-horizon semantics. VLMs provide strong semantic grounding and general knowledge but are suboptimal as standalone dense predictors due to compute-driven sparse sampling, a language-output bottleneck that compresses fine-grained interaction states, and a data-regime mismatch when adapting to small action-conditioned datasets. Neither approach alone captures both fine-grained dynamics and long-horizon reasoning.

---

## Core Idea

ThinkJEPA proposes a dual-temporal pathway framework: a dense JEPA branch processes every frame for fine-grained motion and interaction cues, while a VLM "thinker" branch processes uniformly sampled frames at a larger temporal stride for knowledge-rich long-horizon semantic guidance. A hierarchical pyramid representation extraction module aggregates multi-layer VLM representations into guidance features that are compatible with the JEPA latent prediction, effectively transferring the VLM's progressive reasoning signals to the latent world model.

---

## How It Works

### Overview (from abstract)

- **Dense JEPA branch:** Processes dense frame sequences for fine-grained motion and interaction modelling
- **VLM thinker branch:** Processes uniformly sampled frames with larger temporal stride for long-horizon semantic guidance and general knowledge
- **Hierarchical pyramid representation extraction:** Aggregates multi-layer VLM representations into guidance features compatible with JEPA latent prediction
- **Task:** Hand-manipulation trajectory prediction

### Training

Details not available from the abstract. The framework combines JEPA-style latent prediction with VLM guidance, likely using a joint or staged training procedure.

### Inference

At test time, both pathways contribute to latent prediction: the JEPA branch provides frame-level dynamics while the VLM branch provides semantic context for long-horizon rollout stability.

---

## Results

### Main Results (EgoDex — Hand Manipulation Trajectory Prediction)

| Method | ADE ↓ | FDE ↓ | Acc ↑ | FD ↓ | SL1 ↓ | CD ↓ |
|--------|-------|-------|-------|------|-------|------|
| Qwen3-VL Thinking | 0.142 | 0.144 | 0.084 | 99.54 | 1.656 | 0.615 |
| V-JEPA Predictor | 0.071 | 0.066 | 0.471 | 74.22 | 1.252 | 0.317 |
| **ThinkJEPA** | **0.061** | **0.056** | **0.596** | **74.03** | **1.248** | **0.315** |

ThinkJEPA reduces ADE by 14% over V-JEPA Predictor and boosts accuracy from 47.1% to 59.6% — a 26% relative improvement. The VLM-only baseline (Qwen3-VL) performs poorly on fine-grained trajectory metrics (ADE 0.142 vs 0.061), confirming that VLMs alone cannot handle dense dynamics prediction.

### EgoExo4D (Cross-Domain Generalization)

| Method | ADE ↓ | FDE ↓ | Acc ↑ | FD ↓ | SL1 ↓ | CD ↓ |
|--------|-------|-------|-------|------|-------|------|
| Qwen3-VL Thinking | 0.661 | 0.690 | 0.038 | 104.55 | 1.756 | 0.690 |
| V-JEPA Predictor | 0.659 | 0.636 | 0.074 | 89.24 | 1.520 | 0.469 |
| **ThinkJEPA** | **0.622** | **0.597** | **0.171** | **79.65** | **1.364** | **0.359** |

Gains are even larger on EgoExo4D: accuracy jumps from 7.4% (V-JEPA) to 17.1% (ThinkJEPA) — a 2.3× improvement — suggesting VLM guidance is most valuable for diverse, out-of-domain scenarios.

### Recursive Rollout Stability (EgoDex)

| Model | A@4 | A@8 | A@16 | A@32 |
|-------|-----|-----|------|------|
| Qwen3-VL | 0.140 | 0.819 | 1.375 | 1.026 |
| V-JEPA Predictor | 0.121 | 0.126 | 0.134 | 0.142 |
| **ThinkJEPA** | **0.071** | **0.078** | **0.092** | **0.111** |

ThinkJEPA maintains lower error at all horizons. The VLM-only baseline explodes at horizon 16 (A@16 = 1.375), while ThinkJEPA stays at 0.092 — demonstrating the robustness of hybrid VLM+JEPA over pure VLM rollouts.

### Ablations

**Hierarchical pyramid (all-layer) is critical:** Using only last-layer or mid-layer VLM features gives ADE 0.128; aggregating all layers drops it to 0.061 — a 52% reduction. This is the single most important design choice.

**Dual-temporal pathway matters:** Without separate dense/sparse branches, accuracy drops from 59.6% to 9.9%, confirming that the VLM thinker branch provides essential long-horizon context.

**ThinkJEPA vs trajectory baselines:** ThinkJEPA (ADE 0.061) outperforms all decoder-only and encoder-decoder baselines (best: BC at 0.077), despite not being specifically designed for trajectory output.

---

## Comparison to Prior Work

**vs [[lecun-2022-openreview]] ([H-JEPA](../papers/lecun-2022-openreview.md)):** LeCun's hierarchical JEPA envisions multi-level temporal abstraction. ThinkJEPA implements a concrete dual-pathway instantiation where the VLM provides the higher-level semantic abstraction that guides the lower-level JEPA dynamics predictor.

**vs [[bardes-2024-tmlr]] ([V-JEPA](../papers/bardes-2024-tmlr.md)):** V-JEPA uses dense frame masking for latent video prediction. ThinkJEPA extends this by adding a VLM guidance branch that provides long-horizon semantic context, addressing V-JEPA's limitation of biasing toward local extrapolation.

**vs [[maes-2026-arxiv]] ([LeWorldModel](../papers/maes-2026-arxiv.md)):** LeWorldModel is a JEPA world model using SIGReg for collapse prevention and CEM for planning. ThinkJEPA takes a different approach to improving JEPA prediction quality by leveraging VLM reasoning rather than regularisation techniques, and operates in a dual-pathway architecture rather than a single JEPA pipeline.

---

## Strengths

- **Principled dual-pathway design:** Separates fine-grained dynamics (JEPA) from semantic reasoning (VLM), allowing each component to operate at its natural temporal scale
- **Hierarchical pyramid extraction:** A novel module for transferring multi-layer VLM reasoning signals into JEPA-compatible guidance features
- **Addresses a real limitation:** Dense JEPA predictors struggle with long-horizon semantics; VLMs struggle with fine-grained dynamics -- combining both is well-motivated

---

## Weaknesses & Limitations

- **Only abstract available:** Full methodology, architecture details, training procedures, and quantitative results cannot be evaluated from the 1-page PDF
- **Single task evaluation (per abstract):** Only hand-manipulation trajectory prediction is mentioned; generality to other domains is unknown
- **VLM compute overhead:** Running a VLM branch alongside JEPA likely adds significant inference cost, though the sparse sampling may mitigate this
- **Unclear training details:** How the VLM and JEPA branches are jointly or separately trained is not described in the abstract

---

## Key Takeaways

- ThinkJEPA proposes a dual-pathway latent world model that combines dense JEPA dynamics with VLM semantic reasoning through a hierarchical pyramid representation extraction module
- The approach addresses complementary limitations: JEPA's bias toward local extrapolation and VLM's limitations as a dense predictor (sparse sampling, language bottleneck, data-regime mismatch)
- Results on hand-manipulation trajectory prediction show improvements over both VLM-only and JEPA-only baselines, with more robust long-horizon rollout behavior
- This represents an emerging trend of combining JEPA-style latent prediction with large language/vision-language model reasoning (see also [[lecun-2022-openreview]])

---

## BibTeX

{% raw %}
```bibtex
@article{zhang2026thinkjepa,
  title     = {{ThinkJEPA}: Empowering Latent World Models with Large Vision-Language Reasoning Model},
  author    = {Zhang, Haichao and Li, Yijiang and He, Shwai and Nagarajan, Tushar and Chen, Mingfei and Lu, Jianglin and Li, Ang and Fu, Yun},
  journal   = {arXiv preprint arXiv:2603.22281},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.22281}
}
```
{% endraw %}

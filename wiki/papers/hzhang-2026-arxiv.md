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

From the abstract: experiments on hand-manipulation trajectory prediction show that ThinkJEPA outperforms both a strong VLM-only baseline and a JEPA-predictor baseline, yielding more robust long-horizon rollout behavior. Specific numbers are not available from the abstract.

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

---
title: "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
type: paper
paper_id: P025
authors:
  - "Chen, Delong"
  - "Shukor, Mustafa"
  - "Moutakanni, Theo"
  - "Chung, Willy"
  - "Yu, Jade"
  - "Kasarla, Tejaswi"
  - "Bang, Yejin"
  - "Bolourchi, Allen"
  - "LeCun, Yann"
  - "Fung, Pascale"
year: 2025
venue: ICLR 2026
arxiv_id: "2512.10942"
url: "https://arxiv.org/abs/2512.10942"
pdf: "../../raw/chen-2025-iclr.pdf"
tags: [JEPA, vision-language, multimodal, self-supervised, selective-decoding, vqa, video-classification, retrieval]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
cited_by: []
---

# VL-JEPA: Joint Embedding Predictive Architecture for Vision-language

> **VL-JEPA** is the first non-generative vision-language model built on a Joint Embedding Predictive Architecture, predicting continuous embeddings of target text rather than autoregressively generating tokens; it outperforms standard token-generative VLMs with 50% fewer trainable parameters and natively supports selective decoding that reduces decoding operations by ~2.85x, while achieving state-of-the-art zero-shot classification/retrieval and competitive VQA performance across 20 benchmarks.

**Authors:** Delong Chen\*, Mustafa Shukor\*, Theo Moutakanni\*, Willy Chung\* (equal contribution), Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung — Meta FAIR, HKUST, Sorbonne Universite, NYU | **Venue:** ICLR 2026 | **arXiv:** [2512.10942](https://arxiv.org/abs/2512.10942)

---

## Problem & Motivation

The dominant approach to vision-language tasks is to use large generative Vision Language Models (VLMs) that take visual input and a textual query and autoregressively generate a textual response token by token. This paradigm suffers from two fundamental limitations:

1. **Modeling waste**: VLMs are forced to learn both task-relevant semantics and task-irrelevant surface-level linguistic features (word choice, paraphrasing style). During training, the model must reconstruct every surface detail of the target text Y in token space, even though many different phrasings are equally valid answers. In the raw one-hot token space, semantically equivalent answers (e.g., "the lamp is turned off" vs. "room will go dark") are orthogonal, making the prediction task inherently ill-posed and forcing the model to fit multiple disjoint high-density regions.

2. **Latency and efficiency**: Autoregressive decoding must complete token-by-token before the semantic content of the answer is available. This is particularly problematic for real-time streaming applications (live action tracking, smart glasses, robotics) where the system must continuously update its understanding as new video frames arrive. VLMs cannot selectively decode only when semantics change — they must run the full decoder at every step.

The JEPA framework proposed by [[lecun-2022-openreview]] ([LeCun 2022](lecun-2022-openreview.md)) advocates prediction in abstract representation space rather than data space. Prior JEPA instantiations like [[assran-2023-cvpr]] (I-JEPA) and [[bardes-2024-tmlr]] (V-JEPA) demonstrated this for vision-only self-supervised learning. VL-JEPA extends the paradigm to the multimodal vision-language setting.

---

## Core Idea

Instead of generating tokens, VL-JEPA learns to predict continuous embeddings of the target text. The model maps visual inputs and textual targets into a shared abstract embedding space, where the predictor learns the mapping from visual embeddings (conditioned on a textual query) to target text embeddings. This design has three key consequences:

1. **Simplified target distribution**: Semantically equivalent answers that are orthogonal in token space get mapped to nearby points in embedding space, collapsing the multi-modal prediction problem into a unimodal one.
2. **No heavy decoder during training**: The text decoder (Y-Decoder) is not involved in the main training loop — only invoked at inference when human-readable text is needed.
3. **Native selective decoding**: Because VL-JEPA produces a continuous embedding stream non-autoregressively, decoding can be triggered only when the predicted embedding changes significantly, reducing computation by ~2.85x without quality loss.

---

## How It Works

### Architecture Components

VL-JEPA consists of four components (Figure 1):

1. **X-Encoder** (X_V -> S_V): A frozen V-JEPA 2 ViT-L (304M parameters) that compresses visual inputs (single image or video frames) into a sequence of visual embeddings — analogous to "visual tokens" in classical VLMs. Video inputs are uniformly sampled at 256x256 resolution; images are duplicated to match the video frame shape.

2. **Predictor** ((S_V, X_Q) -> S_hat_Y): The core component. Initialized with the last 8 Transformer layers of Llama-3.2-1B (490M trainable parameters). Takes visual embeddings S_V and textual query embeddings X_Q as input, jointly attended without causal masking. The predictor outputs are pooled and projected to produce the predicted target embedding S_hat_Y. Linear projections connect the predictor with both the vision and text embedding spaces.

3. **Y-Encoder** (Y -> S_Y): Initialized from EmbeddingGemma-300M. Embeds the textual target Y into a continuous latent embedding that serves as the prediction target. Maximum context length of 512 tokens. A learning rate multiplier of 0.05x is applied to text encoder parameters to prevent degradation of the pretrained embedding quality early in training.

4. **Y-Decoder** (S_hat_Y -> Y_hat): Not used during training. At inference time, translates the predicted embedding into human-readable text when text output is needed. Lightweight — invoked only when necessary.

### Training Objective

The training loss is defined in embedding space:

```
L_VL-JEPA = D(S_hat_Y, S_Y)
```

using bi-directional **InfoNCE loss**, which jointly trains the Predictor and Y-Encoder. InfoNCE decomposes into: (1) an **alignment** term that minimizes distance between normalized prediction and target embeddings, and (2) a **uniformity** term that pushes embeddings apart to avoid representation collapse. Both Predictor and Y-Encoder are trained jointly with bi-directional loss so they mutually learn from each other.

The shared embedding space has 1,536 dimensions, with linear projection heads applied to both Predictor and Y-Encoder outputs.

### Multi-tasking

VL-JEPA handles diverse tasks through a single unified architecture:

- **Vision-text-to-text generation** (captioning, open-ended VQA): Query X_Q is a captioning prompt; the predictor predicts S_hat_Y, which is decoded to text via Y-Decoder with selective decoding.
- **Open-vocabulary classification**: Candidate label texts are encoded via Y-Encoder; the predicted embedding S_hat_Y is compared to each candidate, selecting the nearest match. No decoder needed.
- **Discriminative VQA**: Candidate answers are encoded and compared to the predicted embedding. No decoder needed.
- **Text-to-video retrieval**: Candidate videos are mapped to embeddings via a retrieval captioning prompt; these are ranked by similarity to the encoded textual retrieval query.

### Two-Stage Training

**Stage 1 — Large-scale Pretraining** (VL-JEPA_BASE):
- **Data**: Datacomp + YFCC-100M (image-text), Action100M (video-text with action descriptions and captions from HowTo100M)
- **Schedule**: Image-only training for 100k iterations (1 frame/input, batch size 24k, seeing 2B samples, reaching 61.6% ImageNet zero-shot); then video pretraining for 60k iterations at 8 frames + 10k iterations at 32 frames
- **Compute**: 4 weeks on 24 nodes with 8x NVIDIA H200 GPUs each; learning rate 5x10^-5
- **Evaluation**: Zero-shot classification and text-to-video retrieval

**Stage 2 — Supervised Finetuning** (VL-JEPA_SFT):
- **Data**: PLM data mixture — 25M VQA samples, 2.8M captioning samples, 1.8M classification samples, plus downsampled pretraining data to prevent catastrophic forgetting
- **Schedule**: 83k steps with batch size 3,072 (~2.5 days on 24 nodes), cosine learning rate annealing
- **Evaluation**: VQA capabilities, classification, retrieval

### Selective Decoding

For streaming/real-time applications, VL-JEPA produces a continuous stream of predicted embeddings S_hat_Y. The Y-Decoder is invoked only when a significant semantic shift is detected in the embedding stream (e.g., when the local window variance exceeds a threshold). This reduces decoding operations by ~2.85x while maintaining similar CIDEr scores on captioning, compared to uniform (every-frame) decoding.

### Inference

At inference time, behavior depends on the task:
- **Classification/retrieval**: Direct embedding comparison — no decoder needed, single forward pass.
- **Text generation**: Predictor outputs S_hat_Y, then Y-Decoder translates to text. For streaming, selective decoding applies.
- **Discriminative VQA**: Candidate answers encoded by Y-Encoder, nearest-neighbor selection against S_hat_Y.

---

## Results

### Zero-Shot Classification and Retrieval (Table 1)

VL-JEPA_BASE (1.6B total params, 3.3B samples seen) vs. generalist foundation models (CLIP, SigLIP2, PE-Core):

| Model | Params | Samples | Avg Classification (8 datasets) | Avg Retrieval Recall@1 (8 datasets) |
|---|---|---|---|---|
| CLIP (ViT-L) | 380M | 12.8B | 30.7 | 35.3 |
| SigLIP2 (ViT-L) | 882M | 40B | 38.6 | 41.6 |
| PE-Core (ViT-L) | 671M | 58B | 42.9 | 48.9 |
| **VL-JEPA_BASE** | **1.6B** | **3.3B** | **52.5** | **63.7** |
| **VL-JEPA_SFT** | **1.6B** | — | **75.4** | **63.8** |

VL-JEPA_BASE achieves higher average classification accuracy (52.5 vs. 44.7 for the best baseline PE-Core-G) and higher average retrieval recall@1 (63.7 vs. 58.1). Notably, VL-JEPA has seen substantially fewer vision-language pairs (3.6B vs. PE-Core-G's 86B).

Per-dataset analysis: VL-JEPA_BASE is particularly strong on **motion-centric** benchmarks (SSv2, EK-100, EgoExo4D) and step recognition (COIN, CrossTask), while relatively weaker on appearance-centric benchmarks (Kinetics-400) where it has seen fewer training pairs. After SFT, VL-JEPA_SFT approaches the performance of specialist models individually optimized per dataset.

### Visual Question Answering (Table 2)

VL-JEPA_SFT on four discriminative VQA benchmarks:

| Benchmark | VL-JEPA_SFT (1.6B) | Best Comparable VLM |
|---|---|---|
| GQA (compositional reasoning) | 61.5 | InternVLM-Chat (Vicuna-13B): 66.6 |
| TallyQA (complex object counting) | 69.9 | PaLI-17B: 71.9 |
| POPE (object hallucination) | 85.7 | SmoIVLM-2B: 87.5 |
| POPEv2 (object hallucination) | 86.3 | Qwen2-VL-2B: 91.3 |

VL-JEPA_SFT achieves competitive results with established VLM families (InstructBLIP, Qwen-VL, LLaVA-1.5) despite being a 1.6B non-generative model. It approaches but does not surpass the largest specialist models.

### WorldPrediction-WM (Table 3)

VL-JEPA is evaluated on the WorldPrediction benchmark, where the model identifies which action explains a state transition between initial and final images:

| Model | Accuracy |
|---|---|
| GPT-4o | 53.3 |
| Claude-3.5-sonnet | 55.6 |
| Gemini-2.0 | N/A |
| **VL-JEPA_BASE (1.6B)** | **63.9** |
| **VL-JEPA_SFT (1.6B)** | **65.7** |

VL-JEPA establishes a new state of the art at 65.7%, surpassing large VLMs and frontier LLMs (GPT-4o, Claude-3.5-sonnet) on this world-state understanding task. This result connects to [[maes-2026-arxiv]] ([LeWorldModel](maes-2026-arxiv.md)), which also targets world modeling with JEPA — VL-JEPA approaches it from the vision-language side rather than pixel-level latent dynamics.

### Action Anticipation (Tables 4 and 5)

Fine-tuned VL-JEPA on EPIC-KITCHENS-100 and COIN next-step forecasting:

**EPIC-KITCHENS-100** (Recall@5 at different anticipation times):

| Model | Vision Encoder | t=1s | t=2s | t=4s | t=10s |
|---|---|---|---|---|---|
| V-JEPA2 | ViT-g-384px | 39.7 | 28.6 | 19.3 | 8.2 |
| V-JEPA2 | ViT-L-256px | 32.7 | 23.4 | 15.9 | 7.1 |
| **VL-JEPA** | ViT-L-256px | **34.2** | **26.0** | **19.4** | **11.7** |

With the same ViT-L-256px encoder, VL-JEPA outperforms V-JEPA2 at all anticipation times, with particularly notable advantages at longer horizons (t=10s: 11.7 vs. 7.1). This suggests the language-conditioned embedding space captures temporal dynamics more effectively for long-range prediction.

**COIN next-step forecasting**: VL-JEPA achieves 56.2%, establishing a new SOTA over VideoLLM (49.7%), ProVideLLM (53.6), and others.

### Controlled Comparison: Embedding vs. Token Prediction

Under strictly matched conditions (same vision encoder, spatial resolution, frame rate, training data, batch size, iterations — only differing in whether the objective is in embedding space or token space), VL-JEPA delivers consistently higher performance on zero-shot captioning and classification while using roughly **50% fewer trainable parameters**.

### Selective Decoding

VL-JEPA reduces the number of decoding operations by ~2.85x while maintaining similar CIDEr scores compared to non-adaptive uniform decoding. This is possible because the embedding stream changes only when the visual content's semantics actually shift.

---

## Comparison to Prior Work

| | **VL-JEPA** | **Classical VLMs** (LLaVA, InstructBLIP, Qwen-VL) | **CLIP / SigLIP2** | **V-JEPA / V-JEPA 2** |
|---|---|---|---|---|
| Prediction space | Continuous embeddings | Discrete tokens (autoregressive) | Contrastive embeddings | Self-supervised visual embeddings |
| Language involvement | Query-conditioned prediction of text embeddings | Full token generation | Text-image contrastive alignment | None (vision-only) |
| Decoder in training | No (Y-Decoder unused) | Yes (full LM decoder) | No | No |
| Selective decoding | Native | Not supported | N/A | N/A |
| Trainable params | ~490M (predictor) | Typically 7B+ | 300M-900M | 300M+ |
| Task coverage | Classification, retrieval, VQA, captioning | All language-output tasks | Classification, retrieval only | Vision-only tasks |

**vs [[lecun-2022-openreview]] ([LeCun 2022 — JEPA position paper](lecun-2022-openreview.md)):** LeCun proposed JEPA as a framework for learning world models through prediction in abstract representation space, with a six-module cognitive architecture as the long-term vision. VL-JEPA is the first realization of this framework for multimodal vision-language understanding, demonstrating that embedding-space prediction works for language-conditioned tasks, not only vision-only self-supervised learning.

**vs [[assran-2023-cvpr]] (I-JEPA):** I-JEPA applies joint embedding prediction to images, using masked patch prediction in latent space with a teacher-student architecture. VL-JEPA extends the JEPA paradigm from single-modality (image patches predicting other image patches) to cross-modality (visual inputs predicting text embeddings). VL-JEPA replaces the teacher-student asymmetry with a bi-directional InfoNCE loss and adds language conditioning via the predictor.

**vs [[bardes-2024-tmlr]] (V-JEPA):** V-JEPA extends I-JEPA to video via spatiotemporal masked prediction, still in a vision-only self-supervised paradigm. VL-JEPA uses V-JEPA 2 as its frozen X-Encoder but adds the language branch — predictor, Y-Encoder, and Y-Decoder — enabling text-conditioned tasks. On EPIC-KITCHENS-100 with the same ViT-L-256px encoder, VL-JEPA outperforms V-JEPA2 at all anticipation times, suggesting that the language-conditioned embedding space captures richer temporal semantics.

**vs [[maes-2026-arxiv]] ([LeWorldModel](maes-2026-arxiv.md)):** LeWorldModel applies JEPA to pixel-level world modeling for robotic control with SIGReg regularization. VL-JEPA approaches world modeling from the language side — its strong WorldPrediction-WM results (65.7% SOTA) show that embedding-space prediction captures physical state transitions even without explicit dynamics modeling. The two approaches are complementary: LeWorldModel for action-conditioned latent planning, VL-JEPA for language-conditioned world understanding.

**vs Generative VLMs (InstructBLIP, Qwen-VL, LLaVA):** These models autoregressively generate tokens, requiring expensive decoders during both training and inference. VL-JEPA matches their VQA performance (GQA 61.5, TallyQA 69.9, POPE 85.7, POPEv2 86.3) with 1.6B parameters vs. their typical 7B-13B, and without a decoder during training. The trade-off is that VL-JEPA cannot produce free-form open-ended text as flexibly — it excels at discriminative and embedding-based tasks.

---

## Strengths

- **Paradigm shift**: First successful non-generative vision-language model that handles classification, retrieval, VQA, captioning, and world-state understanding through a single unified architecture operating entirely in embedding space.
- **Efficiency**: 50% fewer trainable parameters than matched token-generative VLMs with stronger performance; Y-Decoder absent during training; selective decoding reduces inference cost by ~2.85x for streaming.
- **Simplified learning**: Embedding-space targets collapse semantically equivalent answers to nearby points, transforming an ill-posed multi-modal prediction problem into a well-posed unimodal one.
- **Real-time streaming**: Non-autoregressive prediction enables continuous semantic monitoring with on-demand decoding — a capability fundamentally unavailable to autoregressive VLMs.
- **Broad evaluation**: Validated across 8 classification datasets, 8 retrieval datasets, 4 VQA benchmarks, WorldPrediction-WM, and 2 action anticipation benchmarks — not a single-task demonstration.
- **World understanding**: 65.7% on WorldPrediction-WM surpasses GPT-4o and Claude-3.5-sonnet, suggesting JEPA's embedding space captures causal world dynamics, aligning with the vision in [[lecun-2022-openreview]].

---

## Weaknesses & Limitations

- **Free-form generation limited**: VL-JEPA's strength is in discriminative and embedding-based tasks. For open-ended generative tasks requiring novel, detailed textual outputs, the lightweight Y-Decoder is a bottleneck compared to full autoregressive VLMs.
- **VQA gap to largest models**: On GQA (61.5) and POPEv2 (86.3), VL-JEPA trails the largest specialist VLMs (InternVLM-Chat at 66.6, Qwen2-VL-2B at 91.3) — competitive but not yet dominant on knowledge-intensive benchmarks.
- **Appearance-centric weakness**: VL-JEPA_BASE underperforms on appearance-centric benchmarks like Kinetics-400 and task recognition on COIN/CrossTask, attributed to having seen fewer vision-language pairs (3.6B vs. 86B for PE-Core-G). The model may require more data to close this gap.
- **Frozen vision encoder**: The X-Encoder (V-JEPA 2) is frozen, meaning VL-JEPA inherits its representation biases and cannot adapt the visual backbone to language-specific requirements. End-to-end encoder training could improve results but at higher cost.
- **Y-Decoder quality**: The paper provides limited analysis of the Y-Decoder's text quality for captioning tasks. The selective decoding experiments preserve CIDEr scores, but broader generation quality metrics (fluency, detail, hallucination rate) are not thoroughly evaluated.
- **Two-stage training complexity**: The pretraining-then-SFT pipeline, while standard, requires substantial compute (4 weeks on 192 H200 GPUs for pretraining alone) and careful data mixture design.

---

## Key Takeaways

- **Embedding prediction beats token prediction**: Under strictly controlled conditions, predicting continuous embeddings outperforms generating tokens for vision-language tasks, with half the trainable parameters. This validates a central thesis of the JEPA framework for multimodal learning.
- **Non-generative models can be general-purpose**: VL-JEPA handles classification, retrieval, VQA, captioning, and world understanding through one architecture — proving that embedding-space models are not limited to discriminative-only applications.
- **Selective decoding is a natural JEPA advantage**: The non-autoregressive embedding stream enables decoding only when semantics change, a ~2.85x efficiency gain that autoregressive models cannot achieve by design. This is critical for real-time streaming applications.
- **JEPA captures world dynamics**: The 65.7% SOTA on WorldPrediction-WM — surpassing GPT-4o and Claude-3.5 — provides direct evidence that JEPA's embedding space encodes causal physical relationships, connecting to the broader world-model agenda in [[lecun-2022-openreview]] and [[maes-2026-arxiv]].
- **Bridge between vision-only JEPA and language**: VL-JEPA demonstrates that the JEPA paradigm, proven for vision-only SSL in I-JEPA and V-JEPA, extends naturally to multimodal settings. V-JEPA 2 serves as a frozen backbone; the language branch (predictor + Y-Encoder) adds multimodal capability on top.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{chen2025vljepa,
  title={{VL-JEPA}: Joint Embedding Predictive Architecture for Vision-language},
  author={Chen, Delong and Shukor, Mustafa and Moutakanni, Th{\'e}o and Chung, Willy and Yu, Jade and Kasarla, Tejaswi and Bang, Yejin and Bolourchi, Allen and LeCun, Yann and Fung, Pascale},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  note={arXiv:2512.10942}
}
```
{% endraw %}

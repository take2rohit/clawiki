---
title: "LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures"
type: paper
paper_id: P042
authors:
  - "Huang, Hai"
  - "LeCun, Yann"
  - "Balestriero, Randall"
year: 2025
venue: "arXiv"
arxiv_id: "2509.14252"
url: "https://arxiv.org/abs/2509.14252"
pdf: "../../raw/huang-2025-arxiv.pdf"
tags: [JEPA, LLM, language-model, self-supervised-learning, fine-tuning, pretraining, representation-learning, NLP]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
  - balestriero-2025-iclr
cited_by: []
---

# LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures

> **LLM-JEPA** introduces the first JEPA-based training objective for Large Language Models, adding an embedding-space prediction loss (using cosine similarity between encoder representations of two views of the same knowledge, e.g., natural language description and code) alongside the standard next-token prediction loss, yielding significant accuracy improvements across Llama3, Gemma2, OpenELM, and OLMo models on NL-RX, GSM8K, Spider, and other benchmarks while maintaining generative capabilities and resisting overfitting.

**Authors:** Hai Huang (Atlassian), Yann LeCun (NYU), Randall Balestriero (Brown University) | **Venue:** arXiv 2025 | **arXiv:** [2509.14252](https://arxiv.org/abs/2509.14252)

---

## Problem & Motivation

Representation learning in vision has shown that embedding-space training objectives (as in JEPA) are far superior to input-space reconstruction for perception tasks. However, Large Language Models (LLMs) remain firmly anchored to **next-token prediction** -- a purely input-space, generative objective. This creates a fundamental mismatch:

1. **LLMs' tasks involve perception and reasoning**, where JEPA is known to be preferable, yet their training objectives only optimize reconstruction.
2. **Input-space reconstruction is sub-optimal** for learning abstract representations -- a finding well-established in vision but unexplored for language.
3. **The lack of JEPA for language** reflects the challenge of adapting embedding-space objectives to autoregressive models where the primary use case (text generation) requires maintaining generative capabilities.

The key question is: *can language training methods learn from the success of JEPA in vision?* The challenge is that datasets with natural multi-view structure (where the same knowledge has different representations, like text descriptions and corresponding code) are needed for JEPA objectives.

---

## Core Idea

LLM-JEPA augments the standard LLM training loss with a JEPA objective that operates on natural multi-view data. The core insight is that many NLP datasets contain pairs that are **two views of the same underlying knowledge** -- for example:

- Natural language descriptions and regular expressions (NL-RX)
- Natural language questions and SQL queries (Spider)
- Math word problems and their solutions (GSM8K)
- Issue descriptions and code diffs

By treating these as view pairs (analogous to different augmented views of the same image in vision JEPA), the model can learn a JEPA objective that encourages the internal representations of the two views to be predictable from each other, while preserving generative capabilities through the retained next-token prediction loss.

Two key principles guide the design:
1. **Preserve generative capabilities**: The standard LLM loss is always retained.
2. **Improve abstraction**: The JEPA objective forces representations to align across views, improving the embedding space structure.

---

## How It Works

### The LLM-JEPA Objective

The complete loss combines two terms:

**L_LLM-JEPA = sum_l L_LLM(Text_{1:l-1}, Text_l) + lambda * d(Pred(Enc(Text)), Enc(Code))**

- **L_LLM**: Standard next-token prediction (cross-entropy) applied to the Text portion, preserving generative capabilities.
- **JEPA term**: Cosine similarity distance between the predicted embedding of Text (after passing through a predictor) and the embedding of Code.

### The Encoder

The LLM itself serves as the encoder. The embedding is the `hidden_state` of the last token from the final layer -- the standard approach used for LLM probing. Text and Code are packed into a single context window with a custom attention mask to prevent cross-referencing.

### The Predictor

A **tied-weights predictor** that appends k predictor tokens (special [PRED] tokens) to the input prompt and uses the LLM's own self-attention mechanism to produce the prediction Pred(Enc(.)). The embedding of the last predictor token is the output. When k=0, the predictor is trivial: Pred(x) = x. The predictor reuses the LLM's internal weights, greatly reducing training overhead.

### Implementation with Custom Attention Mask

A critical implementation challenge: the two views cannot be obtained in a single causal forward pass. LLM-JEPA solves this with a custom self-attention mask:

- Text and Code are concatenated in one sequence.
- A block-causal attention mask ensures: (1) Text tokens attend only to other Text tokens (causally), (2) Code tokens attend only to other Code tokens (causally), (3) Neither can attend to the other.
- This yields embeddings for both views in two forward passes total (one for LLM loss on Text, one for JEPA embeddings of both views).

### The Metric

Cosine similarity is used as the distance function d, following standard practice in vision JEPAs. MSE, L2 norm, and InfoNCE were evaluated as alternatives but cosine similarity performed best.

### Training

- **Fine-tuning setup**: Hyperparameter search over learning rate lr in {1e-5, 2e-5, 4e-5, 8e-5}, and a 2D grid over (k, lambda) in {0,1,2,3,4} x {0.5,1,2,4}. Train for 4-6 epochs. Five fixed random seeds per experiment for statistical significance.
- **Models tested**: Llama-3.2-1B-Instruct, gemma-2-2b-it, OpenELM-1_1B-Instruct, OLMo-2-0425-1B-Instruct.
- **Datasets**: NL-RX-SYNTH, NL-RX-TURK, GSM8K, Spider, RottenTomatoes, NQ-Open, HellaSwag, cestwc/paraphrase.
- **Pretraining setup**: From randomly initialized weights on NL-RX-SYNTH, lr=8e-5, lambda=2, k=3.

### Inference

At inference time, only the standard autoregressive generation pipeline is used -- the JEPA objective is training-only. No extra forward passes are needed, so there is zero inference overhead.

---

## Results

### Fine-tuning Across Models and Datasets (Figure 1, Table 12 in Appendix)

LLM-JEPA consistently improves over standard fine-tuning:

| Model | Dataset | Baseline | LLM-JEPA | Improvement |
|---|---|---|---|---|
| Llama-3.2-1B | NL-RX-SYNTH | ~55% | ~70% | +15% |
| gemma-2-2b | NL-RX-SYNTH | ~48% | ~62% | +14% |
| OpenELM-1.1B | NL-RX-SYNTH | ~35% | ~45% | +10% |
| Llama-3.2-1B | Spider | ~33% | ~42% | +9% |
| Llama-3.2-1B | GSM8K | ~27% | ~38% | +11% |

### Resistance to Overfitting (Figure 1, right)

Over 6 training epochs on NL-RX-SYNTH/TURK, standard fine-tuning accuracy plateaus and begins to degrade, while LLM-JEPA accuracy continues to improve, demonstrating robust resistance to overfitting.

### Pretraining Results (Table 2)

| Model | Method | Accuracy (%) |
|---|---|---|
| Llama-3.2-1B | L_LLM | 54.38 +/- 1.70 |
| Llama-3.2-1B | L_LLM-JEPA (ours) | **60.59 +/- 1.01** |

LLM-JEPA improves pretraining accuracy by +6.21% (p-value = 2.94e-4).

### Extension to QA and Reasoning (Tables 4, 5)

| Dataset | Baseline | LLM-JEPA | p-value |
|---|---|---|---|
| NQ-Open | 20.12 +/- 0.41 | **21.59 +/- 0.40** | 2.44e-3 |
| HellaSwag | 69.40 +/- 0.99 | **70.51 +/- 1.20** | 0.0136 |
| GSM8K (Qwen3-1.7B) | 44.32 +/- 0.39 | **45.00 +/- 0.40** | 0.0115 |
| GSM8K (R1-Distill-Qwen-1.5B) | 13.87 +/- 1.01 | **15.04 +/- 0.15** | 0.0396 |

LLM-JEPA improves reasoning models (Qwen3, DeepSeek-R1-Distill) on GSM8K, extending benefits to LRMs.

### JEPA-Loss Dropout (Table 6, Figure 5)

Random JEPA-loss dropout (LD) randomly skips the JEPA computation for a fraction of batches, reducing compute:

- LD=0.5 with lambda=2 achieves comparable accuracy to LD=0 (regular LLM-JEPA) with **25% fewer FLOPs**.
- LD=0.75 with lambda=4 maintains performance with **37.5% fewer FLOPs**.

### Ablation Study (Table 3)

| Method | Accuracy (%) |
|---|---|
| Baseline (NTP only) | 57.29 +/- 5.32 |
| **LLM-JEPA (cosine sim)** | **71.46 +/- 1.34** |
| L2-norm | 2.22 +/- 0.07 |
| MSE | 70.64 +/- 2.05 |
| Prepend [PRED] | 68.07 +/- 2.57 |
| Code -> Text direction | 65.70 +/- 2.63 |
| InfoNCE loss | 34.40 +/- 6.10 |

Cosine similarity is the clear winner. L2-norm fails catastrophically. The Text->Code prediction direction is better than Code->Text.

---

## Comparison to Prior Work

| | **LLM-JEPA** | SimCSE | Sentence-BERT | data2vec |
|---|---|---|---|---|
| **Modality** | Language (LLMs) | Language (BERT) | Language (BERT) | Multi-modal |
| **Generative capability** | Preserved | No | No | No |
| **Training paradigm** | JEPA + NTP | Contrastive | Supervised pairs | JEPA (masked prediction) |
| **View construction** | Natural multi-view data | Dropout augmentation | Supervised pairs | Masking |
| **Applicable to** | Finetuning + pretraining | Embedding only | Embedding only | Pretraining only |

**vs [[lecun-2022-openreview]]:** LLM-JEPA is the first concrete realization of LeCun's JEPA framework for language models. The key adaptation is treating natural multi-view data (text/code pairs) as the equivalent of different views of the same image. This requires datasets with inherent two-view structure, unlike vision where augmentations create views.

**vs [[assran-2023-cvpr]] (I-JEPA):** I-JEPA predicts masked image patch representations. LLM-JEPA instead predicts cross-view representations (Text -> Code) within an autoregressive framework. The tied-weights predictor reuses the LLM backbone, unlike I-JEPA's separate predictor network.

**vs [[balestriero-2025-iclr]] (LeJEPA):** LLM-JEPA does not use LeJEPA's SIGReg regularization. The autoregressive NTP loss appears sufficient to prevent collapse in the language setting, unlike vision where explicit collapse prevention is needed.

### Key Finding: NTP Does Not Implicitly Minimize JEPA

A critical empirical finding (Figure 3, right): minimizing the standard next-token prediction loss L_LLM does NOT implicitly minimize the JEPA prediction loss D(Pred(Enc(Text)), Enc(Code)). This proves the JEPA term provides genuinely new learning signal beyond what NTP already captures.

---

## Strengths

- **First JEPA for LLMs**: Opens a new direction by bridging the well-established gap between vision and language self-supervised learning paradigms.
- **Maintains generative capabilities**: Unlike prior embedding-focused approaches (SimCSE, Sentence-BERT), LLM-JEPA preserves the LLM's text generation ability -- a critical practical requirement.
- **Extensive empirical validation**: Tested across 4 model families, 8+ datasets, 4 model sizes, and both fine-tuning and pretraining, with rigorous statistical testing (5 seeds, paired t-tests).
- **Resistance to overfitting**: LLM-JEPA continues improving where standard fine-tuning degrades, suggesting the JEPA term acts as an implicit regularizer.
- **Structured representations**: t-SNE visualizations and SVD analysis demonstrate that LLM-JEPA induces clear geometric structure in the representation space, with near-linear mappings between Text and Code embeddings.
- **Practical compute savings**: JEPA-loss dropout reduces overhead while maintaining gains, making the approach practical.

---

## Weaknesses & Limitations

- **Requires natural multi-view data**: LLM-JEPA needs datasets where two views of the same knowledge exist (text/code, question/SQL). General text datasets without this structure cannot directly benefit, significantly limiting applicability.
- **2x compute overhead during training**: Without loss dropout, each batch requires an extra forward pass for the JEPA term. While loss dropout mitigates this, it remains a non-trivial cost.
- **Hyperparameter sensitivity**: The optimal (k, lambda) configuration varies across model/dataset combinations and may occur anywhere in the grid, making tuning expensive.
- **Modest absolute improvements on some benchmarks**: NQ-Open (+1.47%), HellaSwag (+1.11%), and GSM8K gains on reasoning models (+0.68% to +1.17%) are statistically significant but small in absolute terms.
- **No large-scale pretraining**: Pretraining experiments are limited to Llama-3.2-1B on small datasets. It remains unclear whether LLM-JEPA scales to full pretraining of large models on massive corpora.
- **Limited to tasks with clear view pairs**: The paper acknowledges that developing a data-augmentation analog for language (to create views for arbitrary text) would be needed for universal applicability.

---

## Key Takeaways

- **JEPA principles transfer from vision to language**: Adding an embedding-space prediction loss to LLM training improves both fine-tuning and pretraining accuracy, validating that the benefits of JEPA observed in vision extend to language.
- **NTP and JEPA are complementary, not redundant**: Minimizing next-token prediction does not implicitly minimize the JEPA objective. The JEPA term provides genuinely different learning signal that improves the LLM's internal representations.
- **Natural multi-view data is the key enabler**: The success of LLM-JEPA depends on datasets where two representations of the same knowledge exist. This constraint is both the method's strength (no artificial augmentation needed) and its limitation (restricts applicability).
- **LLM-JEPA induces structured representations**: The JEPA loss creates a near-linear mapping between Text and Code embeddings in the LLM's representation space, suggesting it learns more geometrically organized internal representations.
- **Overfitting resistance is a valuable side benefit**: The JEPA term acts as an implicit regularizer, allowing longer training without degradation -- important for fine-tuning on small datasets.

---

## BibTeX

{% raw %}
```bibtex
@article{huang2025llm,
  title={LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures},
  author={Huang, Hai and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2509.14252},
  year={2025}
}
```
{% endraw %}

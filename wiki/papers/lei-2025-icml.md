---
title: "M3-JEPA: Multimodal Alignment via Multi-gate MoE based on JEPA"
type: paper
paper_id: P028
authors:
  - "Lei, Hongyang"
  - "Cheng, Xiaolong"
  - "Qin, Qi"
  - "Wang, Dan"
  - "Kun, Fan"
  - "Huang, Huazhen"
  - "Gu, Qingqing"
  - "Wu, Yetao"
  - "Jiang, Zhonglin"
  - "Chen, Yong"
  - "Ji, Luo"
year: 2025
venue: "ICML 2025"
arxiv_id: "2409.05929"
url: "https://arxiv.org/abs/2409.05929"
pdf: "../../raw/lei-2025-icml.pdf"
tags: [JEPA, multimodal, mixture-of-experts, contrastive-learning, cross-modal-alignment, vision-language, audio]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
cited_by: []
---

# M3-JEPA: Multimodal Alignment via Multi-gate MoE based on JEPA

> **M3-JEPA** extends the Joint-Embedding Predictive Architecture to any-to-any multimodal alignment by using a Multi-gate Mixture-of-Experts (MMoE) predictor that projects input embeddings into a shared latent space, trained with contrastive + regularization losses and optimized via Alternating Gradient Descent (AGD), achieving state-of-the-art on vision-language retrieval, audio-text retrieval, and image classification with only 140M trainable parameters.

**Authors:** Hongyang Lei\*, Xiaolong Cheng\* (Geely AI Lab, Zhejiang, China), Qi Qin\* (Peking University), Dan Wang, Fan Kun (Geely AI Lab), Huazhen Huang (Shenzhen Institutes of Advanced Technology, CAS), Qingqing Gu, Yetao Wu, Zhonglin Jiang, Yong Chen, Luo Ji (Geely AI Lab) -- \*equal contribution | **Venue:** ICML 2025 (PMLR 267) | **arXiv:** [2409.05929](https://arxiv.org/abs/2409.05929)

---

## Problem & Motivation

Current multimodal learning strategies primarily optimize in the original token space, making them easy to integrate with large language model backbones but susceptible to **modality collapse** during cross-modal alignment. This collapse arises from conflicting gradients, missing modality labels, and mismatched data distributions between modalities. In particular, self-supervised learning (SSL) on discrete token spaces struggles in continuous domains (image, video, audio), causing cross-modal alignment to be difficult and prone to losing key information, especially under uncertainty, redundancy, or ambiguity.

The authors propose to leverage the [[lecun-2022-openreview]] ([JEPA](../papers/lecun-2022-openreview.md)) paradigm for multimodal tasks: instead of aligning in token/pixel space, align in a latent embedding space via a predictor. This design filters modality noise and captures core cross-modal information. However, no prior work has generalized JEPA to arbitrary any-to-any modality combinations. M3-JEPA fills this gap.

---

## Core Idea

M3-JEPA reformulates multimodal alignment as a JEPA prediction problem: given an observed modality x, predict the embedding of an unobserved modality y in latent space. The key innovations are:

1. **Multi-gate MoE predictor**: A lightweight Mixture-of-Experts module with separate gates for different loss terms (contrastive and regularization), disentangling modality-specific and shared information through the gating function.
2. **Energy-based formulation**: The total loss combines a contrastive loss (separating negatives, maximizing mutual information) and an L2 regularization loss (regressing positives, minimizing conditional entropy), forming an energy function.
3. **Alternating Gradient Descent (AGD)**: Instead of jointly optimizing all multimodal tasks simultaneously (which causes gradient conflicts), AGD switches between different directional tasks (e.g., image->text, text->image) at each training step.
4. **Information-theoretic optimality**: The loss is shown to be equivalent to maximizing mutual information minus alpha times conditional entropy, with optimal alpha=0.5 derived from both free-energy analogy and convergence analysis.

---

## How It Works

### Architecture

1. **Modality encoders**: Pretrained unimodal encoders (LLaMA-3-8B for text, DINOv2-Large for images, LanguageBind for audio). Most parameters are frozen; only 3 layers are fine-tuned via LoRA (rank=64).
2. **MMoE predictor**: Randomly initialized, fully trained. Contains N=12 experts per modality (total M*N experts where M is number of modalities), K=4 top-K selection, L=2 gates (one for contrastive loss, one for regularization loss). Inner hidden dimension h=2048, dropout=0.1.
3. **Gating function**: G = softmax(g * [e_x concatenated with e_m]), where e_m are learnable modality-specific embeddings and g is a shared learnable gate matrix -- task-agnostic routing.

### Losses (Section 3.2)

**Regularization loss** (L2 regression):
```
L_reg = |e_{x->y} - e_y|^2_2
```

**Contrastive loss** (in-batch negatives):
```
L_cl = (1/B) * sum_i [-log(exp(sim(e^i_{x->y}, e^i_y)/tau) / sum_j exp(sim(e^j_{x->y}, e^j_y)/tau))]
```

**Total loss**:
```
L(e_{x->y}, e_y) = alpha * L_reg + (1 - alpha) * L_cl
```

The optimal alpha=0.5 is derived theoretically and validated empirically (Figure 4).

### Alternating Gradient Descent (Section 3.4)

With T tasks (multimodal directions), AGD updates at step i:
```
theta(i+1) = theta(i) - eta * grad_theta L^t(MMoE-K(e^t_x(i)), e^t_y(i))
    if mod(i, T) = t,  for t = 1, 2, ..., T
```

This decouples updates for different tasks, avoiding gradient conflicts that arise from simultaneously optimizing e.g., image-to-text and text-to-image alignments through the same connector. Convergence to a local optimum is guaranteed when subtask losses are independent and convex.

### Information-Theoretic Analysis (Section 4.1)

The total loss is shown equivalent to:
```
theta* = argmin_theta  -I(x; y) + alpha * (H(y|x) + H(x|y))
```

Where I(x;y) is mutual information and H(y|x), H(x|y) are conditional entropies. The contrastive loss maximizes mutual information (reducing redundancy via negatives), while the regularization loss minimizes conditional entropy (reducing uncertainty via positive regression). The connection to free energy: F = U - TS, where the critical temperature T_c at energy-entropy balance is alpha=0.5.

### Inference

For retrieval: modality encoders precompute embeddings offline; the lightweight MMoE predictor runs online for cross-modal alignment. Retrieval time on COCO is 0.02 seconds -- 8x faster than CLIP (0.16s) and 2.5x faster than BLIP-2 (0.05s).

---

## Results

### Vision-Language Retrieval (Table 1, Flickr30K and COCO)

| Method | #Trainable Params | Flickr30K I->T R@1 | Flickr30K T->I R@1 | COCO I->T R@1 | COCO T->I R@1 |
|---|---|---|---|---|---|
| CLIP | 428M | 88.0 | 68.7 | -- | -- |
| BLIP | 446M | 97.1 | 86.7 | 82.4 | 65.1 |
| BLIP-2 (ViT-L) | 474M | 96.9 | 88.6 | 83.5 | 66.3 |
| BLIP-2 (ViT-g) | 1.2B | 97.6 | 89.7 | 83.4 | 68.3 |
| BEiT-3 | 1.9B | 94.9 | 81.5 | 84.8 | 67.2 |
| **M3-JEPA** | **140M** | **97.8** | **97.8** | **87.7** | **89.7** |

M3-JEPA achieves R@1 of 97.8 on Flickr30K image-to-text and 87.7 on COCO image-to-text with only 140M trainable parameters -- significantly fewer than BLIP-2 (474M-1.2B).

### Audio-Text Retrieval (Table 2, Zero-shot)

| Method | Clotho A->T R@1 | Audiocaps A->T R@1 |
|---|---|---|
| LanguageBind | 16.1 | 17.8 |
| **M3-JEPA** | **17.0** | **20.4** |

Superior zero-shot audio-text performance despite the framework being primarily designed for vision-language tasks.

### Image Classification on ImageNet-1K (Table 3)

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| CLIP-ViT | 82.1 | 82.4 | 82.0 | 82.0 |
| DINOv2 | 83.2 | 83.5 | 83.3 | 83.1 |
| **M3-JEPA** | **86.6** | **86.9** | **86.6** | **86.5** |

M3-JEPA outperforms both CLIP and DINOv2 on ImageNet classification by treating class labels as a separate "modality" encoded via one-hot vectors.

### VQA (Table 4)

| Method | VQAv2 test-dev | VQAv2 test-std | NLVR-2 dev | NLVR-2 test-P |
|---|---|---|---|---|
| BEiT-3 | 84.2 | **84.0** | **91.5** | **92.6** |
| **M3-JEPA** | 82.3 | 82.5 | 86.8 | 87.6 |

M3-JEPA performs second-best on VQAv2 and trails BEiT-3 on NLVR-2, potentially due to BEiT-3's much larger pretraining corpus (MSCOCO + Visual Genome).

### Ablation: MoE and AGD (Table 5, COCO)

| MoE | AGD | I->T R@1 | T->I R@1 |
|---|---|---|---|
| No | Yes | 74.4 | 82.3 |
| Yes | No | 68.2 | 74.2 |
| **Yes** | **Yes** | **87.7** | **89.7** |

Both MoE and AGD are essential. Removing MoE (using MLP instead) drops I->T R@1 by 13.3 points. Removing AGD drops it by 19.5 points.

---

## Comparison to Prior Work

| | **M3-JEPA** | **CLIP** | **BLIP-2** | **BEiT-3** |
|---|---|---|---|---|
| Architecture | Frozen encoders + MoE predictor | Dual encoder | Q-Former + frozen LLM | Unified multimodal transformer |
| Alignment space | Latent (JEPA) | Embedding | Embedding + LLM | Token space |
| Modalities | Any-to-any (text, image, audio, one-hot) | Image-text | Image-text | Image-text |
| Trainable params | 140M | 428M (all) | 474M | 1.9B |
| Gradient conflict handling | AGD (alternating tasks) | None | None | None |
| COCO I->T R@1 | **87.7** | -- | 83.5 | 84.8 |

**vs [[lecun-2022-openreview]] ([JEPA position paper](../papers/lecun-2022-openreview.md)):** LeCun 2022 proposed JEPA for learning world models through prediction in latent space. M3-JEPA is the first work to generalize this framework to any-to-any multimodal alignment, demonstrating that the latent-space prediction principle extends beyond single-modality SSL.

**vs [[assran-2023-cvpr]] ([I-JEPA](../papers/assran-2023-cvpr.md)) and [[bardes-2024-tmlr]] ([V-JEPA](../papers/bardes-2024-tmlr.md)):** These apply JEPA to images and video respectively but remain within a single modality. M3-JEPA extends the JEPA paradigm across modalities using MoE as the cross-modal predictor.

**vs CLIP/BLIP-2:** CLIP trains dual encoders from scratch with contrastive loss. BLIP-2 uses a Q-Former as a lightweight connector. M3-JEPA uses an MoE predictor that simultaneously handles contrastive and regularization objectives with separate gates, achieving better retrieval with fewer trainable parameters.

---

## Strengths

- **First generalized JEPA for any-to-any multimodal alignment**: Handles arbitrary modality combinations (vision-language, audio-text, image classification, VQA) with a single architecture.
- **Highly parameter-efficient**: Only 140M trainable parameters (MoE predictor + 3 LoRA layers) -- 3-14x fewer than competing methods -- while achieving state-of-the-art retrieval.
- **Strong theoretical grounding**: Information-theoretic analysis connects the loss to mutual information and conditional entropy; free-energy analogy derives optimal loss weight; AGD convergence is formally discussed.
- **Fast inference**: 0.02s retrieval time on COCO via modality precomputing + lightweight MoE, 8x faster than CLIP.
- **Effective disentanglement**: The multi-gate design separates modality-specific and shared information, with dual gates for contrastive vs. regularization objectives.

---

## Weaknesses & Limitations

- **Not a generative model**: JEPA's latent-space prediction reformulates alignment as energy minimization, filtering modality noise but precluding generation tasks. The paper acknowledges this as a limitation (Appendix C).
- **Not truly modality-agnostic**: Adding a new modality requires manually selecting an encoder, adding modality-specific experts, and redesigning the training pipeline. The framework is not plug-and-play for arbitrary new modalities.
- **Simple multimodal fusion**: Input modalities are concatenated before the MoE predictor. For VQA tasks requiring fine-grained cross-modal reasoning, this underperforms cross-attention approaches (BEiT-3).
- **Heavy encoder backbone**: Total parameter count is ~8.5B (LLaMA-3-8B + DINOv2-Large + LanguageBind), even though only 140M are trained. Inference requires running these large encoders.
- **AGD convergence assumption**: The convergence guarantee assumes each subtask loss is independent and convex, which may not hold for deep neural networks in practice.

---

## Key Takeaways

- **JEPA generalizes effectively to multimodal alignment**: Predicting in latent embedding space (rather than token space) mitigates modality collapse and achieves state-of-the-art cross-modal retrieval.
- **Multi-gate MoE is an effective cross-modal predictor**: Separate gates for contrastive and regularization losses disentangle modality-specific from shared information, and the design is both lightweight and powerful.
- **Alternating Gradient Descent is essential for multi-directional multimodal tasks**: Jointly optimizing all task directions simultaneously causes gradient conflicts; AGD decouples them and is critical for performance (ablating AGD drops COCO R@1 by 19.5 points).
- **The optimal loss weight is alpha=0.5**: Derived from both information-theoretic analysis (critical temperature in free-energy analogy) and convergence analysis, confirmed empirically.
- **Parameter efficiency does not require performance sacrifice**: 140M trainable parameters surpasses models with 3-14x more parameters on vision-language retrieval.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{lei2025m3jepa,
  title={{M3-JEPA}: Multimodal Alignment via Multi-gate {MoE} based on the Joint-Embedding Predictive Architecture},
  author={Lei, Hongyang and Cheng, Xiaolong and Qin, Qi and Wang, Dan and Kun, Fan and Huang, Huazhen and Gu, Qingqing and Wu, Yetao and Jiang, Zhonglin and Chen, Yong and Ji, Luo},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  series={PMLR},
  volume={267},
  year={2025},
  note={arXiv:2409.05929}
}
```
{% endraw %}

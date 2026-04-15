---
title: "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning"
type: paper
paper_id: P034
authors:
  - "Mur-Labadia, Lorenzo"
  - "Muckley, Matthew"
  - "Bar, Amir"
  - "Assran, Mido"
  - "Sinha, Koustuv"
  - "Rabbat, Mike"
  - "LeCun, Yann"
  - "Ballas, Nicolas"
  - "Bardes, Adrien"
year: 2026
venue: "arXiv (Meta FAIR)"
arxiv_id: "2603.14482"
url: "https://arxiv.org/abs/2603.14482"
pdf: "../../raw/mur-labadia-2026-arxiv.pdf"
tags: [JEPA, self-supervised-learning, video-representation, dense-features, vision-transformer, depth-estimation, semantic-segmentation, world-model, multi-modal]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
  - assran-2025-arxiv
cited_by: []
---

# V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning

> **V-JEPA 2.1** extends the V-JEPA family with four innovations -- dense predictive loss on all tokens (masked and visible), deep self-supervision at intermediate layers, multi-modal tokenizers for images and videos, and VisionMix-163M data scaling -- producing representations with both fine-grained spatial structure and global semantic understanding, achieving SOTA on dense tasks (47.9 mIoU ADE20K, 0.307 RMSE NYUv2) and global tasks (77.7% SSv2, 85.5% IN1K) with a frozen ViT-G backbone.

**Authors:** Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, Adrien Bardes (FAIR at Meta, Universidad de Zaragoza) | **Venue:** arXiv 2026 | **arXiv:** [2603.14482](https://arxiv.org/abs/2603.14482) | **Code:** [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)

---

## Problem & Motivation

V-JEPA and V-JEPA 2 ([bardes-2024-tmlr](../papers/bardes-2024-tmlr.md), [assran-2025-arxiv](../papers/assran-2025-arxiv.md)) have demonstrated strong *global* video understanding through masked feature prediction. However, the learned representations lack *dense* spatial structure -- PCA visualizations of V-JEPA 2 patch features are noisy and fragmented, and linear probing on dense tasks gives poor results (22.2 mIoU on ADE20K, 0.682 RMSE on NYUv2).

The root cause: in standard V-JEPA, the prediction loss is applied **only to masked tokens**. The visible context tokens have no explicit self-supervision signal, so the model has no incentive to encode local spatial information in them. Context tokens can instead act as global aggregators (similar to register tokens), discarding fine-grained spatial grounding.

Meanwhile, other SSL approaches like DINOv3 yield high-quality dense features but are primarily image-based and do not learn temporal dynamics from video. The challenge is to combine the best of both: **dense local features** (for segmentation, depth, tracking) and **global semantic understanding** (for recognition, prediction, planning) within a single self-supervised video model.

---

## Core Idea

Apply the self-supervised predictive loss to **all tokens** -- both masked and visible -- forcing every patch embedding to encode its local spatial content rather than serving as a global information aggregator. Combine this with deep self-supervision at multiple intermediate encoder layers, modality-specific tokenizers for native image/video processing, and large-scale data and model scaling, to produce unified image-video representations with both dense and global capabilities.

---

## How It Works

### Architecture

V-JEPA 2.1 follows the JEPA framework from [lecun-2022-openreview](../papers/lecun-2022-openreview.md) with key extensions:

- **Multi-Modal Tokenizer:** A 3D convolution (16x16x2) for video and a 2D convolution (16x16) for images, plus a learnable modality embedding added to both encoder and predictor inputs. This replaces V-JEPA 2's single 3D conv that wastefully duplicated images 16 times temporally.
- **x-encoder** $E_\theta$: A ViT (up to ViT-G, 2B params) that processes visible context tokens and outputs multi-level representations from multiple intermediate layers plus the final layer.
- **Multi-level Fusion MLP:** Concatenates representations from three intermediate encoder blocks and the final output along the channel axis, then reduces dimensionality via MLP before feeding to the predictor.
- **Multi-level Predictor** $P_\phi$: Processes the fused multi-level context tokens concatenated with learnable mask tokens and produces multi-level predictions for each token.
- **y-encoder** $\bar{E}_\theta$: EMA copy of the x-encoder processing the full unmasked input, producing multi-level target representations.

### Training: Dense Predictive Loss

The key innovation is the **dense prediction loss** $\mathcal{L}_\text{dense} = \mathcal{L}_\text{predict} + \mathcal{L}_\text{ctx}$:

1. **Prediction loss** $\mathcal{L}_\text{predict}$: Standard L1 loss on masked tokens (same as V-JEPA):
   $$\mathcal{L}_\text{predict} = \frac{1}{|M|}\sum_{i \in M} \|P_\phi(E_\theta(x), \Delta_y)_i - \text{sg}(\bar{E}_\theta(y))_i\|_1$$

2. **Context loss** $\mathcal{L}_\text{ctx}$: A distance-weighted L1 loss on visible context tokens:
   $$\mathcal{L}_\text{ctx} = \frac{1}{|C|}\sum_{i \in C} \lambda_i \|P_\phi(E_\theta(x), \Delta_y)_i - \text{sg}(\bar{E}_\theta(y))_i\|_1$$

   The weighting $\lambda_i = \lambda / \sqrt{d_\text{min}(i, M)}$ emphasizes context patches near masked regions, enforcing local continuity between masked and context areas while maintaining a good trade-off between segmentation and classification performance.

### Training: Deep Self-Supervision

Both $\mathcal{L}_\text{predict}$ and $\mathcal{L}_\text{ctx}$ are applied at **four encoder levels** (three intermediate blocks + final output), not just the final layer. This provides training signals throughout the network, allowing local information to flow toward final layers and eliminating the need for intermediate-layer extraction at inference time.

### Data Scaling: VisionMix-163M

The training data is expanded from VideoMix-22M to VisionMix-163M by replacing ImageNet-1M with LVD-142M curated images, and rebalancing video sampling weights (YT-1B weight increased from 0.188 to 0.720, SSv2 from 0.056 to 0.170). Images and videos are trained on separate workers within each batch, with gradients aggregated before updating.

### Training Schedule

- **Phase 1:** 135K iterations at 256x256 resolution (16 frames for video), warmup-constant LR schedule.
- **Phase 2 (Cool-down):** 12K iterations at higher resolution (384x384 for video with 64 frames, 512x512 for images), with decaying LR.

### Model Distillation

The ViT-G (2B) model is distilled into ViT-B (80M) and ViT-L (300M) variants. The distillation uses a frozen teacher, an EMA student copy (not used in loss but serves as final model), and only last-layer supervision with a 12-block predictor.

### Inference

The frozen encoder processes images or videos natively (no temporal duplication for images). For downstream tasks, representations are evaluated via linear probing (dense tasks), attentive probing (global tasks), or as frozen features for world-model planning.

---

## Results

### Dense Tasks (Frozen ViT-G Backbone)

| Task | V-JEPA 2 | V-JEPA 2.1 | Improvement | Previous SOTA |
|---|---|---|---|---|
| ADE20K Semantic Seg. (mIoU) | 24.4 | 47.9 | +96% | 54.8 (DINOv3) |
| NYUv2 Depth (RMSE) | 0.642 | 0.307 | +52% | 0.309 (DINOv3) |
| Ego4D Short-Term OI (mAP) | 6.02 | 7.71 | +28% | -- |
| DAVIS Object Tracking (J&F) | 52.5 | 71.1 | +35% | 69.9 (DINOv3) |

### Global Tasks (Frozen ViT-G Backbone)

| Task | V-JEPA 2 | V-JEPA 2.1 | Previous SOTA |
|---|---|---|---|
| SSv2 Action Recognition (%) | 77.3 | 77.7 | 71.1 (InternVideo2s) |
| K400 Action Recognition (%) | 87.3 | 87.7 | 89.4 (InternVideo2s) |
| IN1K Image Classification (%) | 85.1 | 85.5 | 88.1 (DINOv3) |
| EK Action Anticipation (Recall@5) | 27.6 | 40.8 | 39.7 (PlausVL) |

### World Modeling and Robotics

- **Robot Grasping:** +20% success rate over V-JEPA 2 AC on real Franka arms in zero-shot settings.
- **Robot Navigation:** 5.687 ATE on Tartan Drive, SOTA with 10x faster planning speed vs. prior work.
- **Video Object Segmentation:** 72.7 J&F-Mean on YouTube-VOS.
- **VQA:** 83.1% accuracy on PerceptionTest.

### Ablation: Cumulative Impact (ViT-L, Table 1)

| Component | IN1K | SSv2 | NYU RMSE | ADE20K mIoU |
|---|---|---|---|---|
| V-JEPA 2 baseline | 82.2 | 72.8 | 0.682 | 22.2 |
| + Context Loss | 72.6 | 62.5 | 0.474 | 33.8 |
| + Deep Self-Supervision | 80.8 | 72.1 | 0.463 | 38.6 |
| + VisionMix Data | 81.6 | 72.6 | 0.418 | 40.8 |
| + Multi-modal Tokenizer | 81.6 | 72.6 | 0.415 | 41.4 |
| + Model Scaling (ViT-G) | 84.8 | 76.1 | 0.365 | 47.1 |
| + Cool-down | 85.5 | 77.7 | 0.307 | 47.9 |

The context loss alone causes a large drop in global tasks (82.2 -> 72.6 IN1K) but dramatic dense improvement (22.2 -> 33.8 mIoU). Deep self-supervision recovers global performance while maintaining dense gains.

---

## Comparison to Prior Work

**vs [bardes-2024-tmlr](../papers/bardes-2024-tmlr.md) (V-JEPA):** V-JEPA established feature prediction from video but had no dense feature capability and used a single-level predictor. V-JEPA 2.1 adds dense prediction loss, multi-level supervision, and multi-modal tokenization -- transforming JEPA from a global-only to a unified dense+global representation learner.

**vs [assran-2025-arxiv](../papers/assran-2025-arxiv.md) (V-JEPA 2):** V-JEPA 2 scaled V-JEPA to larger models and added action-conditioned prediction for robotics, but its dense features remained fragmented. V-JEPA 2.1 specifically addresses this with the context loss and deep self-supervision, improving ADE20K mIoU from 22.2 to 47.9 and NYUv2 RMSE from 0.682 to 0.307.

**vs [assran-2023-cvpr](../papers/assran-2023-cvpr.md) (I-JEPA):** I-JEPA demonstrated JEPA for images; V-JEPA 2.1 extends the framework to joint image-video training with modality-specific tokenizers and dense feature extraction.

**vs DINOv3:** DINOv3 remains the reference for dense features (54.8 mIoU ADE20K), and V-JEPA 2.1 narrows this gap substantially (47.9) while greatly exceeding DINOv3 on video understanding tasks (SSv2: 77.7 vs. not reported for DINOv3). V-JEPA 2.1 surpasses DINOv3 on depth estimation (0.307 vs. 0.309 RMSE).

**vs [balestriero-2025-iclr](../papers/balestriero-2025-iclr.md) (LeJEPA):** LeJEPA provides provable collapse avoidance via SIGReg for image JEPA. V-JEPA 2.1 continues to use the EMA-based collapse prevention but focuses on the orthogonal problem of dense feature quality through its novel context loss.

---

## Strengths

- **Solves a fundamental limitation of V-JEPA features:** The context loss is an elegant insight -- by supervising visible tokens, the model is forced to encode local spatial information, producing dense features with clear semantic structure.
- **Unified dense + global representations:** A single frozen backbone achieves SOTA on both dense (depth, segmentation, tracking) and global (recognition, anticipation) tasks.
- **Native multi-modal training:** Modality-specific tokenizers allow proper treatment of images and videos within a shared encoder, eliminating the computational waste of temporal duplication.
- **Strong world-modeling performance:** Translates directly to +20% improvement in real robot grasping and SOTA navigation, validating that better dense features improve embodied tasks.
- **Thorough ablation study:** Each component is ablated independently with clear contribution analysis.

---

## Weaknesses & Limitations

- **Still trails DINOv3 on pure dense tasks:** ADE20K mIoU of 47.9 vs. 54.8 for DINOv3, suggesting room for improvement in spatial feature quality.
- **Massive compute requirements:** ViT-G (2B params) trained on VisionMix-163M requires substantial GPU resources. The distilled smaller models (ViT-B, ViT-L) are not fully evaluated across all benchmarks.
- **Context loss weighting is non-trivial:** The distance-weighted scheme with progressive warmup adds complexity; naive application of the context loss degrades global performance significantly.
- **No formal collapse guarantee:** Like V-JEPA 2, relies on EMA + stop-gradient heuristics for collapse prevention, unlike [balestriero-2025-iclr](../papers/balestriero-2025-iclr.md) which provides provable guarantees.
- **Limited evaluation beyond vision:** No evaluation on language-vision tasks or multi-modal understanding beyond VQA.

---

## Key Takeaways

- **Supervising visible context tokens is the key to dense JEPA features:** The standard JEPA loss on masked tokens alone provides no incentive for the encoder to maintain local spatial structure in context tokens. Adding an explicit context loss transforms the quality of dense representations.
- **Deep self-supervision at intermediate layers recovers global performance:** When the context loss degrades classification accuracy, applying the loss at multiple encoder levels restores it by allowing local information to flow toward final layers.
- **V-JEPA 2.1 advances the JEPA framework toward LeCun's vision of world models:** The combination of dense spatial features, temporal understanding, and demonstrated robot grasping/navigation improvements shows JEPA-based representations are increasingly suitable for embodied intelligence.
- **The recipe scales with data and model size:** Each component (context loss, deep supervision, data scaling, model scaling, cool-down) contributes incrementally, and they compose well together.

---

## BibTeX

{% raw %}
```bibtex
@article{murlabadia2026vjepa21,
  title={{V-JEPA} 2.1: Unlocking Dense Features in Video Self-Supervised Learning},
  author={Mur-Labadia, Lorenzo and Muckley, Matthew and Bar, Amir and Assran, Mido and Sinha, Koustuv and Rabbat, Mike and LeCun, Yann and Ballas, Nicolas and Bardes, Adrien},
  journal={arXiv preprint arXiv:2603.14482},
  year={2026}
}
```
{% endraw %}

---
title: "Revisiting Feature Prediction for Learning Visual Representations from Video"
type: paper
paper_id: P029
authors:
  - "Bardes, Adrien"
  - "Garrido, Quentin"
  - "Ponce, Jean"
  - "Chen, Xinlei"
  - "Rabbat, Michael"
  - "LeCun, Yann"
  - "Assran, Mahmoud"
  - "Ballas, Nicolas"
year: 2024
venue: TMLR 2024
arxiv_id: "2404.08471"
url: "https://arxiv.org/abs/2404.08471"
pdf: "../../raw/bardes-2024-tmlr.pdf"
tags: [JEPA, self-supervised-learning, video-representation, feature-prediction, masked-modeling, vision-transformer]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
cited_by:
  - balestriero-2025-iclr
  - maes-2026-arxiv
---

# Revisiting Feature Prediction for Learning Visual Representations from Video

> **V-JEPA** applies the Joint-Embedding Predictive Architecture to video by predicting masked spatio-temporal feature representations (not pixels), trained on 2M videos without any pretrained image encoders, text, negative examples, or reconstruction -- achieving 72.2% on Something-Something-v2 and 81.9% on Kinetics-400 with a frozen ViT-H/16 backbone, outperforming all prior video SSL methods and narrowing the gap with large-scale image models.

**Authors:** Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas (FAIR at Meta, Inria, ENS, NYU) | **Venue:** TMLR 2024 | **arXiv:** [2404.08471](https://arxiv.org/abs/2404.08471) | **Code:** [github.com/facebookresearch/jepa](https://github.com/facebookresearch/jepa)

---

## Problem & Motivation

Self-supervised learning from video has been dominated by two paradigms: (1) **pixel-reconstruction methods** (VideoMAE, OmniMAE, Hiera) that predict missing pixel values via masked autoencoders, and (2) **contrastive/invariance methods** that enforce similarity between augmented views. Both have significant drawbacks for video:

- **Pixel prediction wastes capacity on low-level detail.** Masked autoencoders must dedicate model capacity to reconstructing every pixel, including texturally unpredictable regions (e.g., background noise, fine textures). This is particularly problematic in video where spatial and temporal redundancy is high -- the model can "cheat" by copying nearby visible patches rather than learning semantic representations.
- **Contrastive methods require negative samples or hand-crafted augmentations.** Methods like VideoCLIP rely on text supervision, while augmentation-based methods require careful design of spatio-temporal transformations that may not transfer across domains.
- **Image models miss temporal dynamics.** State-of-the-art image SSL models (DINOv2, OpenCLIP, I-JEPA) produce excellent appearance features but fundamentally cannot learn motion and temporal reasoning from static images. On Something-Something-v2, which requires understanding fine-grained temporal dynamics (e.g., "pushing something from left to right"), these models score 20+ points below video-pretrained models.

The JEPA framework proposed by [[lecun-2022-openreview]] ([LeCun 2022](../papers/lecun-2022-openreview.md)) argued that prediction in representation space -- rather than pixel space -- should produce more abstract, useful representations. I-JEPA (Assran et al., 2023) validated this for images, but the question remained: **how effective is feature prediction as a standalone objective for unsupervised learning from video**, using modern tools (ViTs, large datasets, multi-block masking)?

---

## Core Idea

Instead of predicting what the missing pixels look like, predict what the missing regions *mean*. V-JEPA masks out large spatio-temporal blocks of a video, encodes only the visible parts, and trains a predictor network to guess the latent representations of the masked regions. The targets come from an exponential moving average (EMA) encoder that processes the full video. Because the loss operates entirely in representation space, the model is free to ignore unpredictable low-level detail and focus on high-level semantic and temporal structure. No pixel decoder is ever needed, no pretrained image encoder is required, and no text, labels, or negative examples are used.

---

## How It Works

### Architecture Overview (JEPA)

V-JEPA follows the Joint-Embedding Predictive Architecture from [[lecun-2022-openreview]] ([LeCun 2022](../papers/lecun-2022-openreview.md)):

- **x-encoder** $E_\theta(\cdot)$: A Vision Transformer (ViT) that processes only the *visible* (unmasked) tokens of a video clip. Trained via gradient descent.
- **y-encoder** $\bar{E}_\theta(\cdot)$: An identical ViT that processes the *full* (unmasked) video clip to produce target representations. Its parameters are an exponential moving average (EMA) of the x-encoder -- never trained directly by gradients. The EMA momentum starts at 0.998 and linearly increases to 1.0 during training.
- **Predictor** $P_\phi(\cdot, \cdot)$: A narrow transformer (12 blocks, embedding dimension 384) that takes the x-encoder output tokens concatenated with learnable mask tokens (carrying positional information about which patches are masked) and predicts the representation of each masked token.

### Training Objective

The loss is an L1 regression between predictor outputs and y-encoder targets:

$$\text{minimize}_{\theta, \phi} \quad \|P_\phi(E_\theta(x), \Delta_y) - \text{sg}(\bar{E}_\theta(y))\|_1$$

where $\text{sg}(\cdot)$ is stop-gradient and $\Delta_y$ encodes the spatio-temporal positions of the masked (target) tokens. The L1 loss was chosen over L2 for stability. Collapse is prevented by the asymmetry: the y-encoder (EMA) evolves slower than the x-encoder, and the predictor must bridge the gap. The authors provide a theoretical motivation adapted from BYOL showing that when the predictor is optimal, the encoder gradient equals the gradient of the median absolute deviation of the target conditioned on the encoder output -- forcing the encoder to capture as much information as possible.

### Masking Strategy: Multi-Block Masking

The masking strategy is critical. V-JEPA uses a **3D multi-block** approach with two complementary mask types per video clip:

1. **Short-range masks:** Union of 8 randomly sampled target blocks, each covering 15% of each frame's spatial extent, repeated across the entire temporal dimension. Aspect ratio randomly chosen in (0.75, 1.5).
2. **Long-range masks:** Union of 2 randomly sampled target blocks, each covering 70% of each frame's spatial extent, repeated across the entire temporal dimension.

The union of these masks yields an average masking ratio of ~90%. The context x is the complement of the masked region. The masks span the full temporal extent of the clip, making the prediction task harder and preventing the model from exploiting temporal redundancy.

Key ablation finding: multi-block masking (random spatio-temporal blocks from the entire video) substantially outperforms both random-tube masking (removing random spatial patches across all frames) and causal multi-block masking (restricting context to the first few frames).

### Network Parameterization

- **Backbone:** Standard ViT (L/16, H/16, or H/16_384). Video clips are split into a 3D grid of spatio-temporal patches (16x16 pixels, 2 consecutive frames each), flattened into a 1D token sequence of length 1568 (for 224x224 resolution, 16 frames). 3D sin-cos absolute positional embeddings are added. No [CLS] token is used.
- **Predictor:** 12 transformer blocks with embedding dimension 384. Takes x-encoder outputs concatenated with learnable mask tokens (parameterized as shared learnable vector + absolute 3D positional embedding). Number of attention heads matches the backbone.
- **Multi-mask prediction:** For efficiency, two different masks (short-range and long-range) are sampled per clip. The x-encoder and predictor are run separately for each mask, but the y-encoder representation is computed only once and shared.

### Pretraining Data and Setup

- **Dataset (VideoMix2M):** ~2 million videos from HowTo100M, Kinetics-400/600/700, and Something-Something-v2, with validation set overlap removed.
- **Input:** 16 frames sampled with temporal stride 4 (~3 second clips at 30fps), resized to 224x224 (or 384x384 for ViT-H/16_384).
- **Optimization:** AdamW, batch size 3072 (2400 for H/16_384), 90K iterations total, cosine LR schedule with 12K warmup, weight decay 0.04 to 0.4.
- **Hardware:** A100 80GB GPUs, bfloat16 precision.

### Evaluation: Attentive Probing

A key design choice for frozen evaluation: since the JEPA loss is unnormalized, there is no reason the encoder's feature space should be linearly separable. Rather than simple average pooling + linear probe, V-JEPA uses **attentive probing** -- a learned cross-attention layer with a single learnable query token that pools over all output tokens, followed by a residual connection, LayerNorm, 2-layer MLP with GeLU, and a linear classifier. This non-linear pooling yields +17 points on K400 and +16 points on SSv2 compared to average pooling. Importantly, the encoder weights remain frozen; only the probe is trained.

### Inference

For downstream evaluation:
- **Video tasks (K400, SSv2):** Multiple clips are sampled (8 for K400, 2 for SSv2), each processed independently by the frozen encoder. Feature maps are concatenated and fed to the attentive probe. Multiple spatial crops (3) are averaged at test time.
- **Image tasks (IN1K, Places205, iNat21):** Input images are duplicated to create a 2-frame "video" and processed through the video tokenizer (3D conv with temporal stride 2 produces identical token count to a single-frame 2D tokenizer).

---

## Results

### Feature Prediction vs. Pixel Prediction (Table 1)

All models: ViT-L/16, 90K iterations on VideoMix2M, multi-block masking, frozen backbone with attentive probe.

| Target | K400 | SSv2 | IN1K | K400-ft |
|---|---|---|---|---|
| Pixels (MAE-style) | 68.6 | 66.0 | 73.3 | 85.4 |
| **Features (V-JEPA)** | **73.7** | **74.8** | **74.8** | **85.6** |

Feature prediction consistently outperforms pixel prediction in both frozen evaluation and fine-tuning, with the largest gains on SSv2 (+8.8 points), the task most dependent on temporal understanding.

### Comparison with Pixel Prediction Methods (Table 5)

All ViT-L/16 models, frozen evaluation with attentive probing:

| Method | #Samples Seen | K400 | SSv2 | AVA | IN1K | Places205 | iNat21 | K400-ft | SSv2-ft |
|---|---|---|---|---|---|---|---|---|---|
| OmniMAE | 2400M | 65.6 | 60.6 | 14.4 | **75.1** | 59.8 | 66.1 | 84.0 | 74.2 |
| VideoMAE | 410M | 77.8 | 63.5 | 21.6 | 71.1 | 59.3 | 64.6 | 85.4 | 74.3 |
| Hiera | 770M | 75.5 | 64.2 | 15.8 | 68.9 | **58.5** | 56.9 | **87.3** | **75.1** |
| **V-JEPA** | **270M** | **80.8** | **69.5** | **25.6** | 74.8 | 60.3 | **67.8** | 85.6 | **75.1** |

V-JEPA outperforms all pixel-prediction baselines on every downstream task except ImageNet (where OmniMAE, trained directly on images, achieves 75.1% vs. 74.8%). Critically, V-JEPA achieves this while seeing **an order of magnitude fewer samples** (270M vs. 770M--2400M). V-JEPA also matches Hiera-L on SSv2 fine-tuning (75.1%) despite Hiera's hierarchical architectural prior.

### State-of-the-Art Comparison (Table 6)

Frozen evaluation with attentive probing across all model scales:

| Method | Arch. | Params | Data | K400 | SSv2 | AVA | IN1K | Places205 | iNat21 |
|---|---|---|---|---|---|---|---|---|---|
| *Image-pretrained:* | | | | | | | | | |
| I-JEPA | ViT-H/16_512 | 630M | IN22K | 79.7 | 50.0 | 19.8 | 84.4 | 66.5 | 85.7 |
| OpenCLIP | ViT-G/14 | 1800M | LAION | 81.8 | 34.8 | 23.2 | 85.3 | **70.2** | 83.6 |
| DINOv2 | ViT-g/14 | 1100M | LVD-142M | **83.4** | 50.6 | 24.3 | **86.2** | 68.4 | **88.8** |
| *Video-pretrained:* | | | | | | | | | |
| MVD | ViT-L/16 | 200M | IN1K+K400 | 79.4 | 66.5 | 19.7 | 73.3 | 59.4 | 65.7 |
| OmniMAE | ViT-H/16 | 630M | IN1K+SSv2 | 71.4 | 65.4 | 16.0 | 76.3 | 60.6 | 72.4 |
| VideoMAE | ViT-H/16 | 630M | K400 | 79.8 | 66.2 | 20.7 | 72.3 | 59.1 | 65.5 |
| VideoMAEv2 | ViT-g/14 | 1100M | Un.Hybrid | 71.2 | 61.2 | 12.9 | 71.4 | 60.6 | 68.3 |
| Hiera | Hiera-H | 670M | K400 | 77.0 | 64.7 | 17.5 | 71.4 | 59.5 | 61.7 |
| *V-JEPA:* | | | | | | | | | |
| V-JEPA | ViT-L/16 | 200M | VideoMix2M | 80.8 | 69.5 | 25.6 | 74.8 | 60.3 | 67.8 |
| V-JEPA | ViT-H/16 | 630M | VideoMix2M | **82.0** | **71.4** | **25.8** | 75.9 | 61.7 | 67.9 |
| V-JEPA | ViT-H/16_384 | 630M | VideoMix2M | 81.9 | **72.2** | 25.0 | **77.4** | **62.8** | **72.6** |

**Key takeaways from Table 6:**
- V-JEPA H/16 outperforms *every* prior video SSL model on *every* video and image task with a notable margin: +5 points on SSv2, +2 on K400, +5 on AVA, +1 on IN1K, +2 on Places205, and +0.2 on iNat21 compared to the best video baseline.
- On SSv2 (motion understanding), V-JEPA achieves +21 points over DINOv2, +21 over OpenCLIP, and +22 over I-JEPA -- demonstrating that video pretraining with feature prediction uniquely captures temporal dynamics.
- On K400 (appearance-heavy), V-JEPA H/16 (82.0%) approaches DINOv2 g/14 (83.4%) despite being trained on 2M videos vs. 142M images with a smaller model.
- V-JEPA H/16_384 achieves 77.4% on ImageNet-1K with a frozen backbone, narrowing the gap with image models despite never being trained on images.

### Label Efficiency (Table 7)

When labeled data is scarce, V-JEPA's advantage grows:

| Method | K400 5% | K400 10% | K400 50% | SSv2 5% | SSv2 10% | SSv2 50% |
|---|---|---|---|---|---|---|
| MVD | 62.6 | 68.3 | 77.2 | 42.9 | 49.5 | 61.0 |
| VideoMAE | 62.3 | 68.5 | 78.2 | 41.4 | 48.1 | 60.5 |
| VideoMAEv2 | 37.0 | 48.8 | 67.8 | 28.0 | 37.3 | 54.0 |
| **V-JEPA H/16** | **67.0** | **72.1** | **80.2** | **51.9** | **57.5** | **67.3** |
| **V-JEPA H/16_384** | **68.2** | **72.8** | **80.6** | **54.0** | **59.3** | **67.9** |

Reducing available labels by 10x (from ~287 to ~29 per class on K400), V-JEPA drops only 12% while VideoMAEv2 drops 30%, VideoMAE drops 16%, and MVD drops 15%. V-JEPA is significantly more label-efficient because its frozen representations are already semantically rich.

### Training Efficiency (Figure 5)

V-JEPA achieves approximately **2x speedup** in pretraining wallclock time compared to pixel-prediction methods (VideoMAE, VideoMAEv2) for the same frozen SSv2 accuracy. This is because feature prediction does not require a pixel decoder, reducing compute per iteration.

### Ablation: Pretraining Data (Table 2)

Average performance increases monotonically with dataset size. ViT-H/16 on VideoMix2M (2M videos) achieves the best average across tasks (72.8%), but task-specific peaks sometimes favor smaller, more targeted datasets (e.g., K710 alone is best for K400-specific performance).

### Ablation: Masking Strategy (Table 4)

| Masking | K400 | SSv2 | IN1K |
|---|---|---|---|
| random-tube[0.9] | 51.5 | 46.4 | 55.6 |
| causal multi-block[6] | 61.3 | 49.8 | 66.9 |
| causal multi-block[12] | 71.9 | 63.6 | 72.2 |
| **multi-block** | **72.9** | **67.4** | **72.8** |

Multi-block masking (random spatio-temporal blocks from the entire video) dramatically outperforms random-tube (21 points on K400) and causal masking. This validates that forcing the model to predict across large, structured missing regions -- rather than scattered tubes -- produces better semantic representations. The causal restriction (context from first frames only) hurts because it limits the information available for prediction.

### Qualitative Analysis (Section 6)

A conditional diffusion decoder was trained to project V-JEPA's predicted feature representations back to pixel space (without access to the unmasked video). The visualizations show that V-JEPA predictions capture correct object identity, spatial layout, and consistent motion across time -- confirming that the feature-space predictions are semantically grounded. The predictions also exhibit positional uncertainty (objects appear at slightly varying locations across samples), consistent with predicting at an abstract level rather than exact pixel locations.

---

## Comparison to Prior Work

**vs [[lecun-2022-openreview]] ([JEPA position paper](../papers/lecun-2022-openreview.md)):** LeCun 2022 proposed the JEPA framework and argued that predicting in representation space avoids the pitfalls of pixel prediction and contrastive learning. V-JEPA is the first large-scale *video* instantiation of this framework, validating the core hypothesis that feature prediction produces versatile representations across both motion-understanding and appearance-based tasks without reconstruction, text, or negative examples.

**vs I-JEPA (Assran et al., 2023):** I-JEPA applied the JEPA framework to static images using multi-block masking and an EMA target encoder. V-JEPA extends this to video by (a) introducing 3D spatio-temporal multi-block masking with short-range and long-range masks, (b) using video-specific tokenization (3D convolution with temporal stride), and (c) training on video datasets. The core architecture and loss are conceptually identical. On tasks requiring temporal understanding (SSv2), V-JEPA dramatically outperforms I-JEPA (72.2% vs. 50.0%), confirming that video pretraining is essential for learning temporal dynamics. On appearance tasks (K400), V-JEPA H/16 (82.0%) also surpasses I-JEPA H/16 (79.7%).

**vs [[balestriero-2025-iclr]] ([LeJEPA](../papers/balestriero-2025-iclr.md)):** LeJEPA provides a provably collapse-free alternative to the EMA-based collapse prevention used in V-JEPA. V-JEPA relies on the asymmetric teacher-student (EMA) architecture with stop-gradients -- heuristic mechanisms that work empirically but lack theoretical guarantees. LeJEPA replaces these with SIGReg regularization. However, V-JEPA operates on video while LeJEPA experiments are image-only.

**vs [[maes-2026-arxiv]] ([LeWorldModel](../papers/maes-2026-arxiv.md)):** LeWorldModel builds on the JEPA lineage (incorporating LeJEPA's SIGReg) and extends it to end-to-end world modeling from pixels with latent planning. V-JEPA is a representation learning method (no planning, no action-conditioning), while LeWorldModel targets decision-making. V-JEPA's contribution is upstream: showing that feature prediction from video produces excellent general-purpose visual representations.

**vs VideoMAE / OmniMAE / Hiera (pixel prediction):** V-JEPA consistently outperforms all pixel-prediction video methods in frozen evaluation while processing 4-10x fewer samples during pretraining. The feature prediction objective eliminates the need for a pixel decoder, providing both better representations and faster training. The gap is especially large on temporally demanding tasks (SSv2: 69.5% vs. 66.5% for VideoMAE at ViT-L scale).

**vs DINOv2 / OpenCLIP (image models):** On appearance-heavy benchmarks (K400, IN1K), large image models remain competitive or superior -- DINOv2 ViT-g/14 achieves 83.4% on K400 vs. V-JEPA's 82.0%. However, image models fail catastrophically on temporal tasks: DINOv2 scores 50.6% on SSv2 vs. V-JEPA's 72.2% (+21.6 points). V-JEPA closes the gap on image tasks (77.4% on IN1K) while being dramatically better on video tasks.

---

## Strengths

- **Clean validation of the JEPA principle for video:** Demonstrates that feature prediction alone -- without reconstruction, text, contrastive losses, or pretrained encoders -- is a sufficient objective for learning versatile video representations. This is a conceptually important result.
- **Versatile frozen representations:** A single frozen backbone performs well on both motion-based (SSv2) and appearance-based (K400, IN1K) tasks, unlike prior video models that excel on one or the other.
- **Sample and compute efficiency:** Achieves SOTA video results while seeing an order of magnitude fewer samples than competing methods (270M vs. 1600M for VideoMAEv2) and training roughly 2x faster than pixel prediction methods.
- **Excellent label efficiency:** The performance gap between V-JEPA and baselines *widens* as labeled data decreases, making V-JEPA especially attractive for low-resource downstream tasks.
- **Thorough ablation study:** Systematic investigation of feature vs. pixel prediction, data distribution, pooling strategy, masking strategy, and 26 masking ablation experiments provide actionable design principles.
- **Attentive probing protocol:** The cross-attention pooling strategy is shown to benefit all evaluated models (including baselines), providing a fairer and more informative evaluation protocol for unnormalized representations.

---

## Weaknesses & Limitations

- **No theoretical guarantee against collapse:** V-JEPA relies on the EMA teacher + stop-gradient heuristic for collapse prevention, with no formal proof of stability. The theoretical motivation (Section 3.1) adapts BYOL analysis but remains informal. Subsequent work ([[balestriero-2025-iclr]] ([LeJEPA](../papers/balestriero-2025-iclr.md))) addresses this gap.
- **Still trails image models on image tasks:** V-JEPA ViT-H/16_384 achieves 77.4% on ImageNet-1K vs. DINOv2's 86.2%. The authors hypothesize this is due to limited visual diversity in video datasets compared to internet-scale image datasets, but the gap remains significant.
- **No action-conditioned prediction or planning:** V-JEPA learns representations but does not model dynamics in a way that supports planning or control -- it is not a world model in the sense of [[lecun-2022-openreview]] ([LeCun 2022](../papers/lecun-2022-openreview.md))'s full cognitive architecture.
- **Evaluation limited to classification-style tasks:** All downstream evaluations are classification (action recognition, image recognition) or detection (AVA). There is no evaluation on dense prediction tasks (segmentation, depth estimation, tracking) or generation tasks.
- **VideoMix2M is not publicly reproducible as a single dataset:** The pretraining dataset is assembled from multiple public sources with specific filtering, making exact reproduction nontrivial.
- **Attentive probing adds complexity:** The non-linear attentive probe complicates fair comparison with methods evaluated using simple linear probes. While the authors show it helps baselines too, it introduces a trainable component that blurs the line between "frozen" and "adapted" evaluation.
- **Limited exploration of model scale:** The largest model is ViT-H/16 (630M params), while competing image models reach ViT-g/14 (1.1B+). Scaling V-JEPA to billion-parameter models remains untested.

---

## Key Takeaways

- **Feature prediction is a better learning objective than pixel prediction for video SSL:** Across all controlled comparisons (same architecture, same data, same compute), predicting in representation space outperforms predicting in pixel space, with the gap largest on tasks requiring temporal reasoning (+8.8 points on SSv2).
- **Video pretraining uniquely captures temporal dynamics that image models cannot learn:** The +21 point advantage over DINOv2 and OpenCLIP on Something-Something-v2 demonstrates that motion understanding requires training on video, and feature prediction is particularly effective at extracting it.
- **Multi-block masking with high masking ratio (~90%) is critical:** Masking large contiguous spatio-temporal blocks forces the model to make high-level semantic predictions rather than exploiting local redundancy. Random tube masking and causal masking both produce substantially worse representations.
- **V-JEPA achieves SOTA video SSL with an order of magnitude fewer training samples:** The combination of feature prediction (no decoder overhead) and multi-block masking yields both better representations and faster training than pixel-based alternatives.
- **The JEPA framework scales from images to video with minimal modification:** The core architecture (encoder + EMA target encoder + predictor with L1 loss) transfers directly from I-JEPA to V-JEPA; the key adaptations are 3D tokenization, 3D positional embeddings, and spatio-temporal masking.

---

## BibTeX

{% raw %}
```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Chen, Xinlei and Rabbat, Michael and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2024},
  note={arXiv:2404.08471}
}
```
{% endraw %}

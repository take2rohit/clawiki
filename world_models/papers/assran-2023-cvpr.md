---
title: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
type: paper
paper_id: P030
authors:
- Assran, Mahmoud
- Duval, Quentin
- Misra, Ishan
- Bojanowski, Piotr
- Vincent, Pascal
- Rabbat, Michael
- LeCun, Yann
- Ballas, Nicolas
year: 2023
venue: CVPR 2023
arxiv_id: '2301.08243'
url: https://arxiv.org/abs/2301.08243
pdf: ../../raw/assran-2023-cvpr.pdf
tags:
- JEPA
- self-supervised-learning
- vision-transformer
- masked-prediction
- representation-learning
created: 2026-04-15
updated: 2026-04-15
cites:
- lecun-2022-openreview
cited_by:
- balestriero-2025-iclr
- chen-2024-iclr
- chen-2025-iclr
- drozdov-2024-arxiv
- gogl-2026-arxiv
- huang-2025-arxiv
- huang-2026-arxiv
- kenneweg-2025-esann
- kobanda-2026-arxiv
- kuang-2026-arxiv
- lei-2025-icml
- maes-2026-arxiv
- mo-2024-neurips
- mur-labadia-2026-arxiv
- nam-2026-arxiv
- parthasarathy-2025-arxiv
- yu-2025-neurips
- zhu-2025-aaai
- zhu-2026-arxiv

---

# Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

> **I-JEPA (Image-based Joint-Embedding Predictive Architecture)** learns semantic image representations without hand-crafted data augmentations by predicting the representations of masked target blocks from a single context block in abstract representation space, achieving 81.1% ImageNet-1K linear-probe top-1 accuracy with a ViT-H/16 while requiring 2.5x--10x less compute than comparable view-invariance and reconstruction-based methods.

**Authors:** Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas (Meta AI / FAIR, McGill University, Mila, NYU) | **Venue:** CVPR 2023 | **arXiv:** [2301.08243](https://arxiv.org/abs/2301.08243)

---

## Problem & Motivation

Self-supervised learning from images has been dominated by two families of methods, each with fundamental limitations:

1. **Invariance-based methods** (SimCLR, BYOL, DINO, iBOT) train encoders to produce similar embeddings for augmented views of the same image. They learn high-level semantic features but encode strong inductive biases from hand-crafted augmentations (random cropping, color jitter, flipping). These biases are **detrimental for tasks whose invariances differ** from those hard-coded by augmentations -- for example, instance segmentation requires sensitivity to spatial layout that random crops destroy. The augmentation recipes also do not straightforwardly transfer across domains (e.g., from natural images to audio or medical imaging).

2. **Generative/reconstruction-based methods** (MAE, BEiT, SimMIM) predict missing pixels or tokens from masked inputs. They need less prior knowledge and generalize across modalities, but reconstructing pixel-level details forces the model to spend capacity on low-level texture and noise. As a result, their off-the-shelf representations (via linear probing) are **typically of a lower semantic level** and require extensive fine-tuning to match invariance-based methods on semantic classification tasks.

The core tension is: invariance-based methods learn semantic features but are biased and costly (multi-view processing), while generative methods are flexible but learn less semantic features. Neither approach fully realizes the self-supervised learning framework proposed in [[lecun-2022-openreview]] ([A Path Towards Autonomous Machine Intelligence](../papers/lecun-2022-openreview.md)), which advocated for Joint-Embedding Predictive Architectures (JEPAs) that predict in representation space rather than pixel space.

---

## Core Idea

The authors realized that the shortcomings of both families can be overcome by a single architectural change: **predict missing information in abstract representation space rather than in input/pixel space, and do so without requiring hand-crafted view augmentations**. Concretely, given a single context block of an image, the model predicts the representations of several other (target) blocks of the same image. Because the targets are representations computed by a learned target encoder (not raw pixels), the model is free to discard irrelevant low-level details and learn an abstract, semantic prediction target.

Two design choices proved essential for making this work:

- **Sample sufficiently large target blocks** (semantic scale), so the prediction task is inherently about high-level structure rather than local texture.
- **Use a sufficiently informative (spatially distributed) context block**, so the predictor has enough information to make non-trivial predictions without relying on simple spatial interpolation.

The resulting multi-block masking strategy, combined with prediction in representation space, produces representations that are simultaneously semantic (useful for classification) and spatially informative (useful for object counting and depth prediction) -- bridging the gap between invariance-based and generative approaches.

---

## How It Works

### Architecture Overview

I-JEPA consists of three components, all based on Vision Transformer (ViT) architectures:

1. **Context encoder** (f_theta): A standard ViT that processes only the visible context patches of the image. Its parameters theta are learned through gradient-based optimization.

2. **Target encoder** (f_theta_bar): An identical ViT architecture whose weights theta_bar are an exponential moving average (EMA) of the context encoder weights. Given the full (unmasked) image y, it produces patch-level representations s_y = {s_{y_1}, ..., s_{y_N}} for all N patches.

3. **Predictor** (g_phi): A narrow (bottlenecked) ViT that takes the context encoder output and a set of positional mask tokens, and predicts the target representations at specified spatial locations. Its embedding dimension is fixed at 384 (narrower than the encoder), while matching the encoder's number of self-attention heads. Predictor depth is 6 for ViT-B/16 and 12 for ViT-L/16, ViT-H/16, and ViT-H/14.

I-JEPA is pretrained **without a [cls] token**. For evaluation, the target encoder's output is average-pooled to produce a global image representation.

### Multi-Block Masking Strategy

The masking strategy is a core design contribution. For each image:

1. **Target blocks**: Randomly sample M = 4 (possibly overlapping) rectangular blocks from the target-encoder output. Each block has a random scale in the range (0.15, 0.2) relative to the image and a random aspect ratio in the range (0.75, 1.5). Crucially, masking is applied to the **output of the target encoder**, not its input -- the target encoder always sees the full image. This ensures that target representations are of a high semantic level.

2. **Context block**: Sample a single large block with random scale in the range (0.85, 1.0) and unit aspect ratio. Then **remove any regions that overlap with the target blocks**, producing a spatially distributed, informative context. The resulting context block typically covers about 75% of the image patches while being non-overlapping with all targets.

This design is the opposite of typical reconstruction approaches: instead of a large context predicting a small missing region, I-JEPA uses a large, sparse context to predict multiple small-but-semantic target regions scattered across the image.

### Prediction

Given the masked context block x, the context encoder produces s_x = {s_{x_j}}_{j in B_x}. For each of the M target blocks, the predictor g_phi takes s_x and a set of learnable mask tokens (one per patch in the target block, each augmented with a positional embedding specifying the target location) and outputs predicted patch-level representations:

s_hat_y(i) = g_phi(s_x, {m_j}_{j in B_i})

The predictor is applied M times, once for each target block, conditioned on different positional mask tokens each time.

### Loss

The loss is the average L2 distance between predicted and target patch representations across all M target blocks:

L = (1/M) * sum_{i=1}^{M} D(s_hat_y(i), s_y(i)) = (1/M) * sum_{i=1}^{M} sum_{j in B_i} ||s_hat_{y_j} - s_{y_j}||_2^2

### Training

- **Optimizer**: AdamW
- **Batch size**: 2048
- **Learning rate**: Linearly warmed from 10^-4 to 10^-3 over 15 epochs, then cosine-decayed to 10^-6
- **Weight decay**: Linearly increased from 0.04 to 0.4 throughout pretraining
- **EMA momentum**: Starts at 0.996, linearly increased to 1.0
- **Dataset**: ImageNet-1K (1.28M images), resolution 224x224 (or 448x448 for high-resolution variants)
- **No augmentations**: No random cropping, flipping, color jitter, or any other hand-crafted data augmentations

The mask sampler is implemented efficiently in a few lines of PyTorch using a batch-collator function in the data loader. Each iteration returns a mini-batch of images plus context and target masks for each image.

### Inference

For evaluation, the pretrained target encoder is frozen. Its output is average-pooled (or concatenated across last 4 layers) to produce a global image representation. A linear classifier or linear probe is trained on top for downstream tasks. No special augmentations are needed at evaluation time either.

---

## Results

### ImageNet-1K Linear Evaluation (Table 1)

The primary benchmark: freeze the pretrained encoder, train a linear classifier on ImageNet-1K.

| Method | Arch. | Epochs | Top-1 (%) |
|---|---|---|---|
| **Methods without view data augmentations** | | | |
| data2vec | ViT-L/16 | 1600 | 77.3 |
| MAE | ViT-B/16 | 1600 | 68.0 |
| MAE | ViT-L/16 | 1600 | 76.0 |
| MAE | ViT-H/14 | 1600 | 77.2 |
| CAE | ViT-B/16 | 1600 | 70.4 |
| CAE | ViT-L/16 | 1600 | 78.1 |
| **I-JEPA** | **ViT-B/16** | **600** | **72.9** |
| **I-JEPA** | **ViT-L/16** | **600** | **77.5** |
| **I-JEPA** | **ViT-H/14** | **300** | **79.3** |
| **I-JEPA** | **ViT-H/16_448** | **300** | **81.1** |
| **Methods using extra view data augmentations** | | | |
| SimCLR v2 | RN152 (2x) | 800 | 79.1 |
| DINO | ViT-B/8 | 300 | 80.1 |
| iBOT | ViT-L/16 | 250 | 81.0 |

Key findings: I-JEPA significantly outperforms all augmentation-free methods (MAE, data2vec, CAE) while using fewer pretraining epochs. At scale (ViT-H/16 at 448x448 resolution), I-JEPA matches the performance of view-invariance methods like DINO and iBOT that rely on hand-crafted augmentations.

### Semi-Supervised ImageNet-1% (Table 2)

Using only 1% of ImageNet labels (~12-13 images per class):

| Method | Arch. | Epochs | Top-1 (%) |
|---|---|---|---|
| **Methods without view data augmentations** | | | |
| data2vec | ViT-L/16 | 1600 | 73.3 |
| MAE | ViT-L/16 | 1600 | 67.1 |
| MAE | ViT-H/14 | 1600 | 71.5 |
| **I-JEPA** | **ViT-L/16** | **600** | **69.4** |
| **I-JEPA** | **ViT-H/14** | **300** | **73.3** |
| **I-JEPA** | **ViT-H/16_448** | **300** | **77.3** |
| **Methods using extra view data augmentations** | | | |
| iBOT | ViT-B/16 | 400 | 69.7 |
| DINO | ViT-B/8 | 300 | 70.0 |
| SimCLR v2 | RN151 (2x) | 800 | 70.2 |
| BYOL | RN200 (2x) | 800 | 71.2 |
| MSN | ViT-B/4 | 300 | 75.7 |

I-JEPA outperforms MAE across comparable architectures. At resolution 448, I-JEPA surpasses all augmentation-based methods except MSN (which uses a smaller patch size ViT-B/4 and hand-crafted augmentations).

### Transfer Learning (Table 3)

Linear-probe transfer to downstream classification tasks:

| Method | Arch. | CIFAR100 | Places205 | iNat18 |
|---|---|---|---|---|
| **Without augmentations** | | | | |
| data2vec | ViT-L/16 | 81.6 | 54.6 | 28.1 |
| MAE | ViT-H/14 | 77.3 | 55.0 | 32.9 |
| **I-JEPA** | **ViT-H/14** | **87.5** | **58.4** | **47.6** |
| **With augmentations** | | | | |
| DINO | ViT-B/8 | 84.9 | 57.9 | 55.9 |
| iBOT | ViT-L/16 | 88.3 | 60.4 | 57.3 |

I-JEPA dramatically outperforms augmentation-free methods on all three benchmarks. It surpasses DINO on CIFAR100 and Places205 even though DINO uses hand-crafted augmentations. The gap to iBOT narrows substantially.

### Low-Level Prediction Tasks (Table 4)

Linear-probe transfer on Clevr object counting and depth prediction:

| Method | Arch. | Clevr/Count | Clevr/Dist |
|---|---|---|---|
| **Without augmentations** | | | |
| data2vec | ViT-L/16 | 85.3 | 71.3 |
| MAE | ViT-H/14 | 90.5 | 72.4 |
| **I-JEPA** | **ViT-H/14** | **86.7** | **72.4** |
| **With augmentations** | | | |
| DINO | ViT-B/8 | 86.6 | 53.4 |
| iBOT | ViT-L/16 | 85.7 | 62.8 |

I-JEPA matches or exceeds view-invariance methods on low-level tasks, outperforming both DINO and iBOT by a large margin on depth prediction (Clevr/Dist). This confirms that I-JEPA captures local spatial information that invariance-based methods discard.

### Scalability (Section 7, Figure 5)

I-JEPA is highly compute-efficient:
- Pretraining a ViT-H/14 on ImageNet requires less than 1200 GPU hours.
- This is **2.5x faster** than a ViT-S/16 pretrained with iBOT.
- It is **over 10x more efficient** than a ViT-H/14 pretrained with MAE.
- I-JEPA introduces ~7% overhead per iteration compared to MAE (for computing target representations in representation space), but converges in ~5x fewer iterations, yielding large net savings.
- A huge I-JEPA model (ViT-H/14) uses less total compute than iBOT's smallest model (ViT-S/16).

### Dataset and Model Scaling (Table 5)

| Pretrain Data | Arch. | CIFAR100 | Place205 | iNat18 | Clevr/Count | Clevr/Dist |
|---|---|---|---|---|---|---|
| IN1K | ViT-H/14 | 87.5 | 58.4 | 47.6 | 86.7 | 72.4 |
| IN22K | ViT-H/14 | 89.5 | 57.8 | 50.5 | 88.6 | 75.0 |
| IN22K | ViT-G/16 | 89.5 | 59.1 | 55.3 | 86.7 | 73.0 |

Moving from ImageNet-1K to ImageNet-22K improves performance across most tasks. Scaling to ViT-G/16 further improves semantic tasks (iNat18, Places205) but not low-level tasks -- the larger patch size (16 vs 14) may be detrimental for local prediction.

### Fine-Tuning on Full ImageNet (Table 15, Appendix D)

| Method | Arch. | Epochs | Top-1 (%) |
|---|---|---|---|
| MAE | ViT-H/14_448 | 1600 | 87.8 |
| **I-JEPA** | **ViT-H/16_448** | **300** | **87.1** |

When fine-tuned end-to-end, I-JEPA achieves 87.1% top-1, within 0.7% of MAE despite 5.3x fewer pretraining epochs.

### Ablations

**Predicting in representation space vs. pixel space (Table 7):** Changing the loss to predict in pixel space instead of representation space degrades the 1% ImageNet linear-probe accuracy from 66.9% to 40.7% (ViT-L/16, 500-800 epochs). This confirms that abstract prediction targets are essential for learning semantic representations.

**Masking strategy (Table 6):** The proposed multi-block masking dramatically outperforms alternatives on 1% ImageNet (ViT-B/16, 300 epochs):
- multi-block: **54.2%**
- rasterized (predict 3 quadrants from 1): 15.5%
- block (single target block): 20.2%
- random (random patch mask): 17.6%

**Target block scale (Table 8):** The optimal target scale is (0.15, 0.2). Too small (0.075, 0.2) drops to 19.2%; too large (0.2, 0.3) drops to 33.6%. The sweet spot produces blocks that are large enough to be semantic but small enough to create a non-trivial prediction task.

**Context block scale (Table 9):** Larger context is always better: scale (0.85, 1.0) yields 54.2% vs 31.2% for (0.40, 1.0). A spatially distributed, informative context is crucial.

**Number of target blocks (Table 10):** More targets consistently improve performance: 1 target gives 9.0%, 2 gives 22.0%, 3 gives 48.5%, and 4 gives 54.2%.

**Output masking vs. input masking (Table 11):** Masking the target-encoder output (I-JEPA's approach) yields 67.3% vs 56.1% for masking the input. Masking the output ensures the target encoder always processes the full image, producing more semantic targets.

**Predictor depth (Table 12):** Deeper predictor (12 layers) significantly outperforms shallower (6 layers): 66.9% vs 64.0% on 1% ImageNet with ViT-L/16.

**Predictor width (Table 14):** A bottlenecked predictor (384 dim) outperforms a full-width predictor (1024 dim): 70.7% vs 68.4%. The bottleneck forces the predictor to rely on the context encoder's representations rather than memorizing low-level details.

---

## Comparison to Prior Work

| | **I-JEPA** | MAE | data2vec | iBOT | DINO |
|---|---|---|---|---|---|
| **Architecture family** | Joint-Embedding Predictive | Generative (pixel reconstruction) | Joint-Embedding Predictive | Hybrid (JEA + reconstruction) | Joint-Embedding (distillation) |
| **Prediction space** | Representation (L2 in latent) | Pixel (MSE in input) | Representation (L2 in latent) | Representation + pixel | Representation (cross-entropy) |
| **Views / augmentations** | Single view, no augmentations | Single view, no augmentations | Single view, no augmentations | Multi-view, hand-crafted augs | Multi-view, hand-crafted augs |
| **Collapse prevention** | Asymmetric architecture + EMA target encoder | N/A (generative) | EMA target encoder | EMA teacher + centering | EMA teacher + centering |
| **ImageNet linear (best)** | 81.1% (ViT-H/16_448, 300ep) | 77.2% (ViT-H/14, 1600ep) | 77.3% (ViT-L/16, 1600ep) | 81.0% (ViT-L/16, 250ep) | 80.1% (ViT-B/8, 300ep) |
| **Low-level tasks** | Strong (72.4 Clevr/Dist) | Strong (72.4 Clevr/Dist) | Moderate (71.3 Clevr/Dist) | Weak (62.8 Clevr/Dist) | Weak (53.4 Clevr/Dist) |
| **Compute efficiency** | High (< 1200 GPU-hrs for ViT-H/14) | Low (> 12000 GPU-hrs for ViT-H/14) | Moderate | Low (multi-view overhead) | Low (multi-view overhead) |

**vs MAE (He et al., 2022):** Both MAE and I-JEPA use a single image view and mask-then-predict. The critical difference is that MAE reconstructs pixels while I-JEPA predicts in representation space. This architectural distinction explains I-JEPA's substantial advantage on linear probing (+4.1% for ViT-L/16, +2.1% for ViT-H/14) despite using 2.7x fewer epochs. MAE's pixel reconstruction forces capacity allocation to low-level details; I-JEPA's abstract targets let the target encoder filter out irrelevant information. However, MAE performs slightly better on Clevr/Count (90.5 vs 86.7), suggesting its pixel-level targets still capture certain fine-grained local patterns better.

**vs data2vec (Baevski et al., 2022):** data2vec also predicts masked-patch representations from an EMA target encoder, making it architecturally the closest method to I-JEPA. Key differences: data2vec uses all unmasked patches as context, while I-JEPA uses a single informative context block; I-JEPA's multi-block masking strategy with carefully sized targets produces substantially better off-the-shelf representations (77.5% vs 77.3% for ViT-L/16, but with 600 vs 1600 epochs). data2vec also does not report strong results on local prediction tasks.

**vs iBOT (Zhou et al., 2022):** iBOT combines data2vec-style patch-level reconstruction loss with DINO's view-invariance loss, requiring multi-view processing with hand-crafted augmentations. I-JEPA matches iBOT's linear-probe performance (81.1% vs 81.0%) without any augmentations and with substantially less compute. A ViT-Huge/14 I-JEPA model uses less compute than iBOT's smallest ViT-S/16 model.

**vs DINO (Caron et al., 2021):** DINO learns view-invariant representations via self-distillation with an EMA teacher. It achieves strong semantic features but discards spatial information (53.4% on Clevr/Dist vs 72.4% for I-JEPA). I-JEPA surpasses DINO on CIFAR100 and Places205 transfer despite using no augmentations.

**vs [[lecun-2022-openreview]] ([A Path Towards Autonomous Machine Intelligence](../papers/lecun-2022-openreview.md)):** LeCun's 2022 position paper proposed the JEPA framework and argued that predicting in representation space (rather than pixel space) would lead to more useful and abstract representations by allowing the model to eliminate unpredictable information. I-JEPA is the first concrete instantiation of this vision for images. The paper empirically validates LeCun's core hypothesis: representation-space prediction does learn more semantic features than pixel-space reconstruction, and the masking strategy determines what level of abstraction the representations capture.

**Cited by [[balestriero-2025-iclr]] ([LeJEPA](../papers/balestriero-2025-iclr.md)):** LeJEPA identifies a key limitation of I-JEPA -- its reliance on EMA, stop-gradients, and asymmetric architecture to prevent representation collapse, none of which have theoretical guarantees. LeJEPA replaces these heuristics with SIGReg, a provably collapse-free regularizer derived from proving that optimal JEPA embeddings should follow an isotropic Gaussian distribution.

**Cited by [[maes-2026-arxiv]] ([LeWorldModel](../papers/maes-2026-arxiv.md)):** LeWorldModel extends the JEPA prediction-in-representation-space principle to a temporal world model that predicts future latent states from pixels, building directly on I-JEPA's validation of LeCun's framework.

---

## Strengths

- **Eliminates hand-crafted augmentations**: I-JEPA achieves competitive or superior performance to augmentation-heavy methods without encoding any view-invariance biases, making it more broadly applicable across domains and tasks.
- **Bridges semantic and spatial quality**: Unlike invariance-based methods that discard spatial information or generative methods that spend capacity on pixel details, I-JEPA learns representations that are strong on both classification (81.1% ImageNet linear) and local prediction tasks (72.4% Clevr/Dist -- matching MAE and far exceeding DINO/iBOT).
- **Exceptional compute efficiency**: Requires 2.5-10x less GPU hours than comparable methods. A ViT-H/14 I-JEPA costs less to train than a ViT-S/16 iBOT. This makes large-scale SSL pretraining accessible.
- **Clean, principled design**: The three-component architecture (context encoder, target encoder, predictor) with a simple L2 prediction loss is easy to understand, implement, and extend. The masking strategy has clear ablation-backed justification.
- **Thorough ablation study**: Every design choice is ablated -- prediction space, masking strategy, target scale, context scale, number of targets, output vs. input masking, predictor depth/width, weight decay, and dataset/model scaling. This provides actionable guidance for practitioners.
- **Visualization of learned representations**: The RCDM-based visualizations in Figures 6-8 provide qualitative evidence that I-JEPA captures high-level object structure and pose while discarding low-level background details, validating the design rationale.

---

## Weaknesses & Limitations

- **No theoretical collapse guarantee**: I-JEPA relies on an asymmetric architecture (EMA target encoder, bottlenecked predictor) to prevent representation collapse. Unlike [[balestriero-2025-iclr]] ([LeJEPA](../papers/balestriero-2025-iclr.md)), there is no formal proof that these heuristics prevent collapse under all conditions. The EMA momentum schedule, predictor bottleneck width, and other hyperparameters must be tuned carefully.
- **Single-modality evaluation**: All experiments are on images. The paper argues the framework generalizes to other modalities (audio, text) but provides no cross-modal evidence. (V-JEPA later extended the approach to video.)
- **ImageNet-centric benchmarks**: While transfer learning is evaluated on CIFAR100, Places205, iNat18, and Clevr, the pretraining is always on ImageNet. The paper does not explore pretraining on non-natural-image domains where the absence of augmentations would be most valuable.
- **Gap on object counting**: I-JEPA underperforms MAE on Clevr/Count (86.7 vs 90.5), suggesting that predicting in representation space may sacrifice some fine-grained local information that pixel reconstruction preserves.
- **No end-to-end fine-tuning results in main paper**: Full fine-tuning on ImageNet is only reported in Appendix D (87.1% vs 87.8% for MAE). The linear-probing focus, while highlighting the quality of frozen representations, does not show whether I-JEPA is competitive when the full pipeline is optimized.
- **Fixed masking hyperparameters**: The target scale (0.15, 0.2), context scale (0.85, 1.0), and number of targets (M=4) are hand-tuned on ImageNet. It is unclear whether these transfer well to images at different resolutions or with different structural properties.

---

## Key Takeaways

- **Predicting in representation space is strictly better than pixel space for learning semantic features**: The ablation showing 66.9% vs 40.7% (Table 7) is perhaps the paper's most important result. It empirically validates LeCun's JEPA hypothesis that abstract prediction targets, by letting the target encoder filter out irrelevant details, produce fundamentally more useful representations.
- **The masking strategy determines the semantic level of learned representations**: Multi-block masking with sufficiently large targets and a spatially distributed context is essential. Switching to random, block, or rasterized masking drops performance from 54.2% to 15-20%. This finding is actionable: practitioners designing new JEPA variants should invest heavily in masking strategy design.
- **Augmentation-free SSL can match augmentation-heavy methods at scale**: I-JEPA ViT-H/16 at 448 resolution achieves 81.1%, matching iBOT ViT-L/16 (81.0%) without any hand-crafted data augmentations. This opens a path to domain-agnostic self-supervised learning.
- **Compute efficiency compounds with scale**: I-JEPA's single-view processing and fast convergence mean that scaling up models (ViT-H, ViT-G) remains practical. A ViT-Huge I-JEPA is cheaper to train than a ViT-Small iBOT -- an order-of-magnitude efficiency gap.
- **I-JEPA uniquely preserves local spatial information while learning semantic features**: The strong Clevr/Dist results (72.4%, far exceeding DINO's 53.4% and iBOT's 62.8%) confirm that avoiding view augmentations prevents the destruction of spatial structure that plagues invariance-based methods.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
  note={arXiv:2301.08243}
}
```
{% endraw %}

---
title: "Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning"
type: paper
paper_id: P027
authors:
  - "Mo, Shentong"
  - "Tong, Shengbang"
year: 2024
venue: "NeurIPS 2024"
arxiv_id: "2410.19560"
url: "https://arxiv.org/abs/2410.19560"
pdf: "../../raw/mo-2024-neurips.pdf"
tags: [JEPA, contrastive-learning, VICReg, I-JEPA, collapse-prevention, self-supervised-learning]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
cited_by: []
---

# Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning

> **C-JEPA** (Contrastive-JEPA) integrates VICReg's variance-invariance-covariance regularization into the I-JEPA framework to prevent entire model collapse and improve the learning of mean patch representations, achieving faster convergence and +0.8/+1.0 gains over I-JEPA on ImageNet-1K linear probing and fine-tuning with ViT-B/16.

**Authors:** Shentong Mo (Carnegie Mellon University), Shengbang Tong (New York University) | **Venue:** NeurIPS 2024 | **arXiv:** [2410.19560](https://arxiv.org/abs/2410.19560)

---

## Problem & Motivation

[[assran-2023-cvpr]] ([I-JEPA](../papers/assran-2023-cvpr.md)) introduced a powerful masked image modeling approach that predicts features in embedding space rather than pixel space, using a context encoder and a target encoder updated via Exponential Moving Average (EMA). Despite its success, two critical limitations have been identified:

1. **EMA is insufficient to prevent entire collapse**: The EMA-based target encoder update can fail to prevent the model from producing constant or near-constant representations, especially during early training or under suboptimal hyperparameters.
2. **I-JEPA prediction struggles to accurately learn the mean of patch representations**: The predictor mechanism in I-JEPA inadequately captures the mean representation across masked patches, leading to suboptimal representation quality.

These limitations hinder both the performance and stability of JEPA-based learning. The authors argue that connecting JEPA with contrastive self-supervised learning principles -- specifically VICReg's regularization -- can address both issues simultaneously.

---

## Core Idea

C-JEPA draws a theoretical and empirical connection between I-JEPA and SimSiam, showing that both share the underlying principle of minimizing distances between representations of different views/patches. Building on this connection, C-JEPA integrates VICReg's variance and covariance regularization terms into the I-JEPA framework:

- **Variance regularization** prevents entire collapse by ensuring each dimension of the embedding space maintains meaningful variance.
- **Covariance regularization** decorrelates embedding dimensions to prevent redundancy.
- **Invariance regularization** (from VICReg) aligns the mean of representations across different mask blocks of the same image.

The combined system prevents collapse more robustly than EMA alone and produces better patch-level representations.

---

## How It Works

### Components

C-JEPA retains the I-JEPA architecture:
- **Context encoder** f_theta: processes unmasked image patches.
- **Target encoder** f'_theta: updated via EMA of the context encoder.
- **Predictor** g_theta: predicts target representations from context representations and mask tokens.

Additionally, C-JEPA introduces VICReg regularization terms applied to the learned representations.

### Training

The total loss combines the I-JEPA masking objective with VICReg's three regularization terms:

**I-JEPA loss** (Eq. 1):
```
L_I-JEPA = (1/|M|) * sum_i sum_{j in B_i} ||b_hat_{y_j} - b_{y_j}||^2
```
where b_hat are predicted patch representations and b are target patch representations.

**Variance regularization** (Eq. 3):
```
v(z) = (1/d) * sum_j max(0, gamma - sqrt(Var(z_j) + epsilon))
```
Hinge function on standard deviation per dimension -- prevents collapse by ensuring variance stays above threshold gamma=1.

**Covariance regularization** (Eq. 4):
```
c(z) = (1/d) * sum_{i!=j} [correlation_matrix(z)]^2_{i,j}
```
Minimizes squared off-diagonal correlation coefficients to encourage decorrelated features.

**Invariance term**: Minimizes mean-squared Euclidean distance between representations z and z_{r_j} from different augmented views for all random patches.

### Theoretical Connection to SimSiam (Section 3.2)

The paper establishes that I-JEPA and SimSiam share a structural parallel: both minimize the distance between predicted/transformed representations and target representations. Using Neural Tangent Kernel (NTK) analysis, the authors analyze the representation dynamics under a linear predictor W_P with eigenvalues lambda_k. The dynamics for each mode k follow:

```
dz_{y_j,k}/dt = eta * lambda_k * (1 - lambda_k) * z_hat_{y_j,k}
```

When lambda_k < 1, the dynamics converge (representations grow); when lambda_k > 1, they have opposite sign (representations decay). Without the predictor, representations converge to zero -- confirming that the predictor is essential for preventing collapse, but still insufficient on its own.

### Inference

Standard frozen or fine-tuned evaluation. The context encoder produces representations used for downstream tasks via linear probing or full fine-tuning.

---

## Results

### ImageNet-1K Image Classification (Table 1, ViT-B/16)

| Method | Pretrain Epochs | Linear Probe | Fine-tune | AP^box | AP^mask | mIoU (ADE20K) |
|---|---|---|---|---|---|---|
| DINO | 1600 | 78.2 | 82.8 | 50.1 | 43.4 | 46.8 |
| BEiT | 800 | 56.7 | 83.4 | 49.8 | 44.4 | 47.1 |
| MAE | 1600 | 68.0 | 83.6 | 50.3 | 44.9 | 48.1 |
| iBOT | 1600 | 79.5 | 84.0 | 51.2 | 44.2 | 50.0 |
| data2vec | 800 | 60.8 | 84.2 | -- | -- | 48.2 |
| I-JEPA | 600 | 72.9 | 83.5 | 49.9 | 44.5 | 47.6 |
| **C-JEPA (ours)** | **600** | **73.7** | **84.5** | **50.7** | **45.3** | **48.7** |

C-JEPA improves over I-JEPA by +0.8 on linear probe, +1.0 on fine-tune, +0.8 on AP^box, +0.8 on AP^mask, and +1.1 on mIoU, all with the same 600 pretrain epochs.

### Scaling to ViT-L/16 (Table 2)

| Method | Pretrain Epochs | Linear Probe | Fine-tune | (J&F)_m | Clevr/Count | Clevr/Dist |
|---|---|---|---|---|---|---|
| I-JEPA | 600 | 77.5 | 85.3 | 56.6 | 85.6 | 71.2 |
| **C-JEPA** | **600** | **78.1** | **86.2** | **58.3** | **86.8** | **71.6** |

Improvements persist at ViT-L scale: +0.6 linear probe, +0.9 fine-tune, +1.7 (J&F)_m on DAVIS-2017 video segmentation.

### Ablation: Component Analysis for Faster Convergence (Table 3, ViT-B/16, 100 epochs)

| I-JEPA Config | Var/Cov | Invariance | Linear Probe | Fine-tune | (J&F)_m |
|---|---|---|---|---|---|
| Baseline | none | none | 63.7 | 82.5 | 52.3 |
| + mean EMA | collapse | none | 68.3 | 83.2 | 54.6 |
| + Var/Cov + Inv | yes | yes | **69.5** | **83.6** | **55.2** |
| EMA for collapse | none | mean | 67.6 | 82.8 | 53.9 |

The combination of Variance/Covariance and Invariance terms yields the best results, confirming that both are needed.

### Ablation: Better Convergence (Table 4, ViT-B/16, 600 epochs)

| I-JEPA Config | Var/Cov | Invariance | Linear Probe | Fine-tune | (J&F)_m |
|---|---|---|---|---|---|
| Baseline | none | none | 72.9 | 83.9 | 56.2 |
| + Var/Cov + Inv | yes | yes | **73.7** | **84.5** | **57.5** |

Benefits persist to full training length, showing VICReg's contribution is not just faster convergence but also better final quality.

---

## Comparison to Prior Work

| | **C-JEPA** | **I-JEPA** ([[assran-2023-cvpr]] ([Assran 2023](../papers/assran-2023-cvpr.md))) | **VICReg** | **MAE** |
|---|---|---|---|---|
| Prediction target | Embedding space | Embedding space | None (contrastive) | Pixel space |
| Anti-collapse | EMA + VICReg regularization | EMA + stop-gradient | Variance + Covariance | Reconstruction loss |
| Masking | Block masking (same as I-JEPA) | Block masking | None | Random masking |
| ImageNet lin. probe (ViT-B/16) | 73.7 | 72.9 | -- | 68.0 |
| Dense tasks (mIoU) | 48.7 | 47.6 | -- | 48.1 |

**vs [[assran-2023-cvpr]] ([I-JEPA](../papers/assran-2023-cvpr.md)):** C-JEPA is a direct extension of I-JEPA that adds VICReg regularization to address the identified weaknesses of EMA-based collapse prevention and inaccurate mean prediction. The improvements are consistent across tasks and scales.

**vs [[lecun-2022-openreview]] ([JEPA concept](../papers/lecun-2022-openreview.md)):** LeCun 2022 proposed prediction in representation space as the key principle. C-JEPA demonstrates that this principle benefits from explicit regularization to prevent collapse, aligning with the position paper's intuition that architectural and training safeguards are needed.

**vs SimSiam/BYOL:** The paper draws theoretical connections between I-JEPA's masked prediction and SimSiam's view-invariance objectives, showing they share the same underlying dynamics when analyzed through NTK.

---

## Strengths

- **Identifies specific failure modes of I-JEPA**: The EMA collapse problem and inaccurate mean prediction are well-diagnosed with both theoretical analysis and empirical evidence.
- **Simple and effective modification**: Adding VICReg terms to I-JEPA is straightforward to implement and consistently improves performance across all evaluated tasks and scales.
- **Theoretical connection between JEPA and contrastive SSL**: The NTK-based analysis connecting I-JEPA with SimSiam provides useful insight into why both architectures work.
- **Comprehensive evaluation**: Benchmarks span classification (ImageNet-1K), detection/segmentation (COCO, ADE20K), video segmentation (DAVIS-2017), and low-level tasks (Clevr).
- **Qualitative attention visualizations**: Attention maps show C-JEPA produces more focused and contextually relevant representations than baseline I-JEPA.

---

## Weaknesses & Limitations

- **Incremental contribution**: The core modification is adding VICReg loss to I-JEPA -- a straightforward combination of two existing methods. The theoretical analysis, while interesting, does not produce novel guarantees.
- **Limited novelty in the theoretical connection**: The SimSiam-JEPA connection is drawn through structural parallels rather than a deep unification. The NTK analysis assumes a linear predictor, which may not hold for the deep predictors used in practice.
- **Same 600-epoch regime**: All experiments use 600 pretrain epochs, matching I-JEPA but fewer than DINO (1600) or MAE (1600). It is unclear how C-JEPA performs with longer training.
- **No comparison with other JEPA improvements**: The paper does not compare against other concurrent works addressing I-JEPA's limitations.
- **Moderate improvements**: The gains (+0.8 linear probe, +1.0 fine-tune on ViT-B/16) are consistent but modest.

---

## Key Takeaways

- **VICReg regularization effectively addresses two key I-JEPA failure modes**: Variance/covariance terms prevent entire collapse, while the invariance term improves mean patch representation learning.
- **I-JEPA and SimSiam share underlying dynamics**: Both minimize distances between predicted and target representations, differing primarily in masking strategy (block masks vs. random augmented views).
- **C-JEPA achieves consistent improvements across tasks and scales**: +0.8 ImageNet linear probe, +1.0 fine-tune, +1.1 ADE20K mIoU, +1.7 DAVIS (J&F)_m over I-JEPA, all at ViT-B/16 with 600 epochs.
- **Explicit regularization is complementary to architectural anti-collapse mechanisms**: EMA alone is insufficient; combining it with VICReg's distributional constraints produces more stable and richer representations.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{mo2024cjepa,
  title={Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning},
  author={Mo, Shentong and Tong, Shengbang},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  note={arXiv:2410.19560}
}
```
{% endraw %}

---
title: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
type: paper
paper_id: P039
authors:
  - "Balestriero, Randall"
  - "LeCun, Yann"
year: 2025
venue: ICLR 2026
arxiv_id: "2511.08544"
url: "https://arxiv.org/abs/2511.08544"
pdf: "../../raw/balestriero-2025-iclr.pdf"
tags: [JEPA, self-supervised-learning]
created: 2026-04-10
updated: 2026-04-10
cites:
  - lecun-2022-openreview
cited_by:
  - maes-2026-arxiv
---

# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

> **Latent-Euclidean JEPA (LeJEPA)** proves that JEPA embeddings should follow an isotropic Gaussian distribution to minimize downstream risk, enforces this via SIGReg (Sketched Isotropic Gaussian Regularization using the Epps-Pulley characteristic-function test), and achieves 79% ImageNet-1k frozen linear accuracy with ViT-H/14 — all without stop-gradients, teacher-student networks, negative samples, or hyperparameter schedulers.

**Authors:** Randall Balestriero (Brown University, Meta-FAIR), Yann LeCun (NYU, Meta-FAIR) — equal contribution | **Venue:** ICLR 2026 | **arXiv:** [2511.08544](https://arxiv.org/abs/2511.08544)

---

## Problem & Motivation

Joint Embedding Predictive Architectures (JEPAs) have emerged as a powerful paradigm for self-supervised pre-training, but their practical adoption has been blocked by a fundamental problem: without explicit constraints, the encoder can trivially minimize the prediction loss by mapping all inputs to the same constant embedding (*complete collapse*) or to a low-dimensional subspace (*dimensional collapse*). Current solutions are entirely heuristic — stop-gradients, teacher-student networks with carefully tuned EMA schedules, negative sample contrasting, feature whitening, and explicit normalization layers. These anti-collapse mechanisms suffer from four compounding failure modes:

1. **Under-specification**: the regularization criteria can be minimized while embeddings remain degenerate.
2. **Quadratic time and memory complexity** with mini-batch size, preventing large-batch scaling.
3. **Hyperparameter sensitivity**: small changes in EMA momentum or learning rate schedules cause training instability.
4. **No theory**: the community has no principled answer to *why* any particular heuristic works, making architecture innovation an ad-hoc game of Whac-A-Mole.

As a practical consequence, recent JEPA progress has come largely from dataset scaling and recipe tuning rather than principled insight. LeJEPA aims to replace this empirical scaffolding with a single theoretically grounded design.

---

## Core Idea

The authors ask a question that nobody had answered before: *what distribution should JEPA embeddings follow in order to minimize expected risk on arbitrary downstream tasks?* They prove rigorously that the answer is the **isotropic Gaussian** — not because of aesthetic preference, but because anisotropic embeddings demonstrably amplify both bias (Lemma 1) and variance (Lemma 2) of any downstream estimator, and the isotropic Gaussian is the unique distribution that minimizes integrated square bias for both linear probes (OLS/Tikhonov) and nonlinear probes (radius-based k-NN, kernel methods). Once the target distribution is known, building a loss that enforces it becomes a statistics problem: sketch the embedding distribution via random 1D projections and run a characteristic-function test against a standard Gaussian. The result is SIGReg — a loss with O(N) complexity, bounded gradients, and no hyperparameters beyond one trade-off weight λ. The combined system, LeJEPA, is the first JEPA that is **provably collapse-free by construction** rather than heuristically stabilized.

---

## How It Works

### Overview

LeJEPA has two components trained jointly: (1) the standard JEPA **prediction loss**, which enforces that embeddings of different views of the same sample are mutually predictable; and (2) **SIGReg**, which regularizes the embedding distribution to be isotropic Gaussian. The total loss has one hyperparameter λ balancing the two terms. The entire implementation is ~50 lines of PyTorch and runs on any backbone.

### Theoretical Foundation: Why Isotropic Gaussian? (Section 3)

Let Z ∈ ℝ^{N×K} be the N embeddings from encoder f_θ. For **linear probing** (OLS with Tikhonov regularization), the optimal probe β̂ minimizes ||y - Zβ||² + λ||β||². Lemma 1 proves that whenever the covariance of Z has unequal eigenvalues (anisotropy), there always exists a downstream task y for which the anisotropic distribution produces higher bias than the isotropic one. Lemma 2 proves that the OLS estimator variance is strictly lower under isotropic covariance than under any anisotropic covariance with the same energy. Theorem 1 formalizes this as: the isotropic Gaussian uniquely minimizes the integrated square bias (ISB) over query points, for both k-NN and kernel prediction. The same result extends to **nonlinear probing**: analyzing radius-based k-NN and kernel regression confirms the isotropic Gaussian as the unique optimum (Section 3.2). This establishes the target distribution from first principles, not convention.

### SIGReg: Sketched Isotropic Gaussian Regularization (Section 4)

Having identified the target, the challenge is enforcing it efficiently in high dimension. Existing distribution-matching approaches (moment matching: Jarque-Bera, Extended JB, VICReg; CDF-based: Cramér-von Mises, Anderson-Darling) fail for different reasons:

- **Moment-based**: Theorem 3 proves that matching any finite set of moments does NOT guarantee P_θ = Q (the isotropic Gaussian), because the Gaussian is determined by infinitely many moments.
- **CDF-based**: require sorting (O(N log N), non-differentiable, incompatible with DDP's parallel reduction).

SIGReg uses **characteristic-function (CF)-based** tests, specifically the **Epps-Pulley (EP) test**:

```
EP = N ∫_{-∞}^{∞} |φ̂_X(t) - φ(t)|² w(t)dt
```

where φ̂_X(t) = (1/N) ∑ e^{itX_j} is the empirical CF of the data and φ(t) = e^{-t²/2} is the standard Gaussian CF. The ECF is computed as simple complex exponentials, naturally differentiable and parallelizable via `all_reduce`. Theorem 4 proves bounded gradients: |∂EP/∂z_i| ≤ 4σ²/N, |∂²EP/∂z_i²| ≤ C√π σ³/(2N). This contrasts with moment-based tests where gradient norms can grow as O(k²m_{2(k-1)}) for the k-th moment.

**Sketching beats the curse of dimensionality** (Section 4.3): Rather than testing the full K-dimensional distribution (intractable), SIGReg projects embeddings onto M random unit-norm directions:

```
SIGReg_T(𝔸, {f_θ(x_n)}) = (1/|𝔸|) ∑_{a∈𝔸} T({a^⊤ f_θ(x_n)})
```

Theorem 5 proves that by the Hyperspherical Cramér-Wold theorem, matching all 1D projections implies full distributional equality. The number of directions |𝔸| needed scales as O(K) (linear in embedding dimension), and because directions are resampled each step, the compounding effect of SGD means even |𝔸| = 16 directions can approximate full distributional matching. Theorem 6 proves that the gradient bias from mini-batch estimation vanishes at rate O(1/N).

The full SIGReg implementation (Algorithm 1) is ~15 lines of PyTorch with DDP support:
```python
def SIGReg(x, global_step, num_slices=256):
    A = torch.randn(proj_shape, generator=g)  # random directions
    A /= torch.norm(A, p=2, dim=0)
    x_t = (x @ A).unsqueeze(2) * t           # projected samples × freq grid
    ecf = (1j * x_t).exp().mean(0)           # empirical CF
    ecf = all_reduce(ecf, op="AVG")          # sync across GPUs
    err = (ecf - exp_f).abs().square().mul(exp_f)
    return torch.trapz(err, t, dim=1) * N
```

### LeJEPA Loss (Section 5)

The **prediction loss** follows DINO's view setup: V_g global views and V_l local views. The global views' embeddings are averaged to form a "mean center" μ_n, and the prediction loss is:

```
L_pred = (1/V) ∑_v ||μ_n - z_{n,v}||²
```

This is equivalent to predicting the global-view mean embedding from each view's local embedding — note that unlike I-JEPA or V-JEPA 2, no separate predictor network is required; the loss is a direct L2 distance in embedding space.

The **total LeJEPA loss** is:

```
L_LeJEPA = λ · (1/V) ∑_v SIGReg(z from view v) + (1-λ)/B · ∑_n L_pred(z_{n,v})
```

Single hyperparameter λ controls the trade-off. The recommended defaults are **λ=0.05, V_g=2, V_l=8, batch size ≥ 128**. No stop-gradients, no teacher network, no EMA, no whitening layers.

### Training Setup

- **Backbone**: Any (ViT, ResNet, ConvNeXt, Swin, MaxViT) — architecture-agnostic
- **Loss**: L_LeJEPA as above; λ=0.05 default
- **Implementation**: ~50 lines PyTorch (Algorithms 1 and 2); DDP-compatible
- **Scale**: Experiments up to ViT-g (1.8B parameters) on ImageNet-1K (1.28M images)

### Inference

Standard encoder evaluation. No planning, no rollout. For representation quality: frozen backbone + linear probe (top-1 accuracy). For in-domain datasets: also reported 1-shot and full fine-tuning accuracies.

---

## Results

### Stability Across Hyperparameters (Table 1, ViT-Large/14, ImageNet-1K, 100 epochs, frozen linear probe top-1%)

A key claim of LeJEPA is that *none* of its hyperparameters cause catastrophic collapse:

| Hyperparameter | Range Tested | Top-1 Range |
|---|---|---|
| Epps-Pulley integration domain | [-1,1] → [-5,5] | 71.8 → 74.8 |
| num_slices (|𝔸|) | 512 → 2048 | ~74% stable |
| quadrature points | 5 → 41 | <0.5% variation |
| Number of views (V_g=2) | 4 → 10 total | 72.2 → 75.1 |
| Batch size | 128 → 1024 | 72.2 → 74.2 |
| Embedding dim | 64 → 1024 | 75.3 → 73.9 |
| Projector dim | 512 → 2048 | ~74-75% |
| Register tokens | 0 → 8 | 75.1 → 75.8 |

No setting causes collapse. This contrasts sharply with DINOv2/I-JEPA where improper EMA momentum or batch size can destabilize training entirely.

### Scale: ViT-H/14, ImageNet-1K

LeJEPA reaches **79% top-1 (frozen linear probe)** with a ViT-H/14, establishing competitive performance at large scale. Training loss has **94.52% Spearman correlation** with downstream linear probe accuracy on ViT-base — the first practical model-selection signal for JEPA training that does not require supervised probing.

### In-Domain Pretraining vs. Transfer Learning (Galaxy10, Figure 1)

On Galaxy10 (astronomy image classification), LeJEPA pretrained in-domain consistently outperforms DINOv2 and DINOv3 transfer learning from natural images:

| Method | 1-shot (FT) | Full (FT) | 1-shot (Frozen) | Full (Frozen) |
|---|---|---|---|---|
| **LeJEPA ConvNeXt-V2 Nano (in-domain)** | **29.42** | **82.72** | **28.74** | **76.52** |
| LeJEPA ResNet-34 (in-domain) | 24.27 | **83.28** | **31.08** | **78.17** |
| DINOv2 ViT-S/16 (transfer) | 21.05 | 78.34 | 27.68 | 67.62 |
| DINOv3 ViT-S/16 (transfer) | 24.71 | 81.60 | 30.17 | 71.38 |

This demonstrates that *domain-specific SSL beats generic transfer learning* even against massive frontier models — and LeJEPA makes domain-specific SSL practical at any scale.

### Training Stability

ViT-g/14 (1.8B parameters) trains stably on ImageNet with LeJEPA, while DINOv2-style heuristics require careful tuning at this scale. The training curve (Figure 1 top-right) shows monotonic convergence without instability over 72 epochs.

### Ablations

**Number of views** (Figure 8, ResNet50, ImageNet-100, 400 epochs): peak performance obtained by adjusting λ proportionally to number of views; 2-view peak ≈ 8-view peak when λ is scaled. More views improve data efficiency with a fixed λ.

**λ sensitivity**: performance is unimodal in λ (peak around 0.05) but stable across ±1 order of magnitude, in contrast to methods with hard hyperparameter cliffs. No λ causes training to diverge.

---

## Comparison to Prior Work

| | **LeJEPA** | I-JEPA / V-JEPA | VICReg | DINO/DINOv2 |
|---|---|---|---|---|
| Core approach | JEPA prediction + isotropic Gaussian regularization | JEPA prediction + asymmetric predictor | Variance-invariance-covariance regularization | Distillation with EMA teacher + stop-gradient |
| Anti-collapse mechanism | SIGReg (provably sufficient by Cramér-Wold) | Stop-gradient + architecture asymmetry | Variance term + whitening | EMA teacher + centering |
| Time complexity | O(N) | O(N) | O(N²) for covariance | O(N) |
| Theoretical guarantee | Provably eliminates collapse; optimal embedding distribution proved | None | Partial (VICReg can still collapse) | None |
| Hyperparameters | 1 (λ) | Multiple (EMA, crop scale, projector dims) | Multiple (μ, ν, coeff weights) | Multiple (τ, EMA momentum, centering rate) |
| ImageNet-1K (frozen ViT-H/14) | 79% | — | — | ~82% DINOv2 ViT-H/14 |
| In-domain small datasets | Consistently beats frontier transfer | Not shown | Not shown | Worse than in-domain LeJEPA |

**vs [[lecun-2022-openreview]] ([JEPA position paper](lecun-2022-openreview.md)):** LeCun 2022 proposed JEPA as a conceptual framework and argued for prediction in representation space as the key design principle, but provided no training recipe or solution to the collapse problem. LeJEPA is the first instantiation that makes that framework rigorous: it answers LeCun's implicit question of *which* representation space (isotropic Gaussian) and how to enforce it provably.

**vs I-JEPA (Assran et al., 2023, P030):** I-JEPA uses a target encoder (EMA of online encoder) and a separate context-conditioned predictor network to predict masked patch embeddings. This asymmetric architecture prevents collapse empirically but has no theoretical guarantee and introduces multiple sensitive hyperparameters (EMA momentum, mask ratio, predictor depth). LeJEPA replaces this entire anti-collapse apparatus with SIGReg and proves collapse cannot occur.

**vs V-JEPA / V-JEPA 2 (P029, P024):** V-JEPA applies the I-JEPA masked-prediction paradigm to video, again relying on teacher-student asymmetry. LeJEPA's architecture-agnostic design applies identically to video without modification.

**vs VICReg (Bardes et al., 2021):** VICReg prevents collapse via a variance regularization term (requiring variance > threshold) and a covariance regularization term (penalizing off-diagonal covariance entries). Section 5.2 shows that VICReg is a special case of SIGReg when T is the moment-based EJB test — but Theorem 3 proves this is provably insufficient to guarantee P_θ = Q. LeJEPA's SIGReg uses the Epps-Pulley CF test, which Lemma 3 (Hyperspherical Cramér-Wold) proves is sufficient.

**vs SimCLR / MoCo (contrastive methods):** Contrastive methods prevent collapse via explicit negative samples, requiring large batch sizes (SimCLR needs >4096) or memory banks. LeJEPA works with batch sizes as small as 128.

**vs DINO / DINOv2:** DINO and DINOv2 rely on a teacher network (exponential moving average of student), centering, and stop-gradients — a system with many interacting hyperparameters. LeJEPA eliminates all these mechanisms. On large-scale ImageNet, DINOv2 ViT-H/14 achieves ~82% frozen linear; LeJEPA ViT-H/14 achieves 79% — a small gap considering LeJEPA has no architectural complexity and runs on small datasets without transfer.

---

## Strengths

- **Theoretical grounding**: First JEPA with provable collapse-free guarantees; optimal embedding distribution derived from first principles, not convention.
- **Simplicity**: ~50 lines PyTorch, one hyperparameter λ, no stop-gradients, no teacher network, no negative samples.
- **Linear complexity**: SIGReg is O(N) in time and memory, DDP-compatible, scales to 1.8B parameter models.
- **Architecture-agnostic**: Same loss works on ViTs, ResNets, ConvNeXts, Swin, MaxViT without modification.
- **Informative training loss**: 94.52% Spearman correlation between training loss and downstream accuracy — enables model selection without supervised validation sets.
- **Enables in-domain SSL**: Shows domain-specific pretraining on small datasets beats frontier transfer learning, making SSL practical beyond the large-scale pre-training paradigm.

---

## Weaknesses & Limitations

- **Results are partially shown in the main paper**: Section 6 is cut off at the stability ablations; full benchmark comparisons (ImageNet-1K at multiple scales, Food101, etc.) are presumably in an appendix not included in the 13-page main text.
- **79% vs ~82% for DINOv2**: On ImageNet-1K frozen linear with ViT-H/14, LeJEPA trails DINOv2 by ~3%. This may close with more epochs or larger scale, but is not yet resolved.
- **Gradient bias O(1/N)**: The Epps-Pulley estimator introduces a gradient bias of order 1/N from mini-batch approximation. The paper argues this is negligible in practice but notes unbiased alternatives exist (U-statistic debiasing).
- **Single-modal image experiments**: All experiments are image classification. The paper argues the framework generalizes (video, robotics) but does not provide experiments beyond image SSL.

---

## Key Takeaways

- **The isotropic Gaussian is provably optimal**: Not a heuristic choice — Theorem 1 proves it uniquely minimizes downstream bias for both linear and nonlinear probes. This transforms JEPA design from empirical search to targeted optimization.
- **SIGReg eliminates all collapse heuristics**: Stop-gradients, teacher networks, negative samples, and whitening layers are replaced by a single ~15-line regularizer with linear complexity, bounded gradients, and one hyperparameter.
- **LeJEPA is the principled instantiation of LeCun's JEPA vision**: The 2022 position paper proposed the idea; LeJEPA provides the first implementation with provable guarantees.
- **Training loss is now a meaningful signal**: The 94.52% Spearman correlation between training loss and downstream accuracy means practitioners can select model checkpoints without supervised probing — a capability missing from all prior JEPAs.
- **Domain-specific SSL beats generic frontier transfer**: LeJEPA ConvNeXt-V2 Nano pretrained on Galaxy10 outperforms DINOv2 ViT-S/16 transfer in 1-shot settings, demonstrating principled SSL is practical even for small, specialized datasets.

---

## BibTeX

```bibtex
@inproceedings{balestriero2025lejepa,
  title={LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author={Balestriero, Randall and LeCun, Yann},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  note={arXiv:2511.08544}
}
```

---
title: "KerJEPA: Kernel Discrepancies for Euclidean Self-Supervised Learning"
type: paper
paper_id: P058
authors:
  - "Zimmermann, Eric"
  - "Wiltzer, Harley"
  - "Szeto, Justin"
  - "Alvarez-Melis, David"
  - "Mackey, Lester"
year: 2025
venue: arXiv
arxiv_id: "2512.19605"
url: "https://arxiv.org/abs/2512.19605"
pdf: "../../raw/zimmermann-2025-arxiv.pdf"
tags: [JEPA, kernel-methods, self-supervised, regularization, sigreg, mmd, ksd, slicing]
created: 2026-04-15
updated: 2026-04-15
cites:
  - balestriero-2025-iclr
  - lecun-2022-openreview
cited_by: []
---

# KerJEPA: Kernel Discrepancies for Euclidean Self-Supervised Learning

> **KerJEPA** introduces a unified family of kernel-based regularizers for Euclidean self-supervised learning that generalizes the SIGReg regularizer from LeJEPA. The paper proves that SIGReg is equivalent to a sliced Maximum Mean Discrepancy (MMD) with a specific dimension-dependent kernel, and then expands the design space to include alternative kernels (Gaussian, IMQ), alternative discrepancies (Kernel Stein Discrepancy), alternative priors (Gaussian, Laplace, Student-t), and both sliced and unsliced variants -- providing practitioners with a principled toolbox of regularizers with explicit computational and statistical tradeoffs.

**Authors:** Eric Zimmermann, David Alvarez-Melis, Lester Mackey (Microsoft Research); Harley Wiltzer, Justin Szeto (Mila / McGill University) | **Venue:** arXiv preprint, December 2025 | **arXiv:** [2512.19605](https://arxiv.org/abs/2512.19605)

---

## Problem & Motivation

LeJEPA ([balestriero-2025-iclr](balestriero-2025-iclr.md)) established that regularizing JEPA embeddings toward an isotropic Gaussian distribution via the SIGReg regularizer (based on the Epps-Pulley characteristic-function test) yields provable collapse prevention and strong downstream performance. However, this approach has two structural limitations:

1. **Rigidity of the regularizer**: SIGReg is locked to the Epps-Pulley test with a Gaussian kernel and a Gaussian prior. This precludes exploration of alternative regularizers, non-Gaussian priors, and heavy-tailed kernels that may offer superior geometric properties for downstream transfer.

2. **Fundamental tension in slicing**: SIGReg relies on random 1D projections (slicing) to lift a univariate test to high dimensions. While slicing offers linear computational scaling, the finite approximation of the integral over the sphere introduces gradient noise that scales with embedding dimension. Maintaining training stability in high-dimensional regimes requires increasing the number of projections, which eventually undermines the linear scaling advantage.

The paper asks: *Can we identify a general algorithmic structure for Euclidean SSL regularization that encompasses SIGReg and reveals a broader family of viable alternatives -- including methods that avoid slicing entirely?*

---

## Core Idea

KerJEPA reframes the LeJEPA regularizer as a special case within a much larger family of kernel-based discrepancy measures. The key insight is a chain of equivalences:

1. **Epps-Pulley is an MMD**: The paper proves (Proposition 1) that the Epps-Pulley test statistic used by SIGReg is exactly the squared MMD between the empirical embedding distribution and an isotropic Gaussian, computed with a Gaussian kernel.

2. **SIGReg is a sliced MMD with a dimension-dependent kernel**: Theorem 7 proves that SIGReg equals the squared MMD with a specific kernel kappa_d that is *not* the Gaussian kernel but a Kummer confluent hypergeometric function with polynomially decaying (heavy) tails. This means the slicing operation itself implicitly changes the kernel -- a previously unrecognized effect.

3. **Slicing is neither necessary nor sufficient for breaking the curse of dimensionality**: Since the MMD admits unbiased sample-based estimators with dimension-free sample complexity, one can compute equivalent regularizers directly in high-dimensional space without slicing, trading variance reduction for quadratic compute.

With these equivalences established, the paper systematically expands two axes of the design space:

- **Discrepancy type**: Maximum Mean Discrepancy (MMD) and Kernel Stein Discrepancy (KSD)
- **Implementation**: sliced (finite or analytic) vs. unsliced (direct high-dimensional computation)

Each combination can be paired with different kernels (Gaussian, IMQ) and priors (Gaussian, Laplace, Student-t), yielding a rich design space of regularizers termed **KerJEPAs**.

---

## How It Works

### Background: Kernel Mean Embeddings and MMD

A positive definite kernel k maps probability distributions into a Reproducing Kernel Hilbert Space (RKHS) via kernel mean embeddings. The Maximum Mean Discrepancy (MMD) measures the RKHS distance between two distributions:

```
MMD_k(P, Q) = ||mu_P - mu_Q||_H
```

When the kernel is *characteristic* (e.g., Gaussian, IMQ), MMD = 0 if and only if P = Q, meaning it captures full distributional differences, not just moment differences. For shift-invariant kernels, Bochner's theorem provides an equivalent spectral representation in terms of characteristic functions, which is the bridge to the Epps-Pulley test.

### Connecting Epps-Pulley to MMD (Proposition 1, Corollary 3)

The Epps-Pulley test compares the empirical characteristic function of the data to that of a Gaussian, weighted by the spectral density of a Gaussian kernel. The paper proves:

```
EP(P) = MMD^2_{k_gsn}(P, N(0, sigma^2))
```

This is a direct identification: the Epps-Pulley statistic *is* the squared Gaussian-kernel MMD to a Gaussian prior. The scaled estimator used by LeJEPA approximates this MMD with O(1/n) bias (Corollary 3).

### The Hidden Kernel of SIGReg (Theorem 7)

SIGReg averages the Epps-Pulley test over random 1D projections. Computing the closed-form limit of infinitely many slices reveals:

```
SIGReg(P) = MMD^2_{kappa_d}(P, N(0, sigma^2 I_d))
```

where kappa_d(x, y) = 1F1(1/2; d/2; -gamma ||x - y||^2) is the Kummer confluent hypergeometric function. Critically, this kernel has **polynomially decaying tails** -- as opposed to the exponentially decaying Gaussian kernel. This means slicing implicitly imposes a heavy-tailed similarity measure on latent embeddings. Additionally, the effective kernel bandwidth is **dimension-dependent**: as d grows, slicing effectively decreases the bandwidth of the resulting kernel, which has direct implications for training stability and convergence.

### Kernel Stein Discrepancy (Section 3.3)

As an alternative to MMD, the paper introduces the Kernel Stein Discrepancy (KSD) for SSL regularization. KSD uses the score function s_Q(x) = nabla_x log Q(x) of the target prior, eliminating the need to sample from or compute expectations over the target distribution:

```
KSD^2(P, Q) = E_{X,X' ~ P}[k_stein(X, X')]
```

where k_stein is the Stein kernel derived from a base kernel k via the Stein operator. Key advantages of KSD over MMD:

- **No target sampling required**: Only the score function of Q is needed, not samples. For isotropic priors (Gaussian, Laplace, Student-t), scores have simple closed forms.
- **Broader prior support**: Any distribution with a tractable score can serve as the regularization target, including non-Gaussian priors.
- **Metrizes weak convergence**: For specific kernel/dimension classes, minimizing KSD guarantees distributional convergence -- preventing collapse.

The paper derives spectral representations (Proposition 4, Corollary 5) and closed-form sliced KSD expressions (Theorem 9) for the Gaussian prior case.

### Sliced vs. Unsliced Implementations (Sections 4.1-4.2)

**Sliced methods (MMDReg, KSDReg):** Follow the LeJEPA approach -- project n embeddings of dimension d onto r random directions, compute 1D discrepancies via quadrature with u knots. Complexity: Theta(nr(d + u)), linear in batch size. The paper provides PyTorch pseudo-code (Figure 1) for both sliced MMDReg and sliced KSDReg.

**Unsliced (generalized) methods:** Compute pairwise kernel evaluations directly in R^d. Complexity: Theta(n^2 d), quadratic in batch size. Key advantage: exact computation with no approximation variance from slicing. For KSD, the Gaussian and IMQ kernels with Gaussian, Laplace, and Student-t priors all admit fully tractable closed-form Stein kernels (Appendices A-D). The unsliced MMD with a Gaussian kernel and Gaussian prior (the BHEP estimator) has closed form (Theorem 8, Equation 12).

### The KerJEPA Loss Framework (Section 4)

The general SSL objective is:

```
L(theta) = (1/n) sum_l sum_{(i,j) in P_l} L_align(f_theta(v^i), f_theta(v^j)) + lambda * Omega(Z; Q)
```

where L_align is MSE between positive pairs, and Omega is any kernel discrepancy regularizer from the KerJEPA family (MMDReg or KSDReg, sliced or unsliced, with chosen kernel and prior). This directly generalizes the LeJEPA loss where Omega = SIGReg.

---

## Results

### Ablation on ImageNette (Table 1, ViT-S/8, 800 epochs)

The paper benchmarks the full KerJEPA family against the LeJEPA baseline on ImageNette with identical training setups (single 80GB A100, batch size 256, 4 views, projector dim 128, 1024 slices, 21 quadrature knots):

| Algorithm | Type | Kernel | Prior | Acc. (%) |
|---|---|---|---|---|
| **LeJEPA** (baseline) | Sliced (Finite) | Gaussian | Gaussian | 91.13 +/- 0.45 |
| MMD | Unsliced | Gaussian | Gaussian | 91.29 +/- 0.45 |
| MMD | Sliced (Analytic) | Gaussian | Gaussian | 91.13 +/- 0.45 |
| MMD | Sliced (Finite) | Gaussian | Laplace | 90.25 +/- 0.47 |
| KSD | Unsliced, Gaussian | Gaussian | -- | 91.31 +/- 0.45 |
| KSD | Unsliced, Gaussian | Laplace | -- | 91.18 +/- 0.45 |
| **KSD** | **Unsliced, IMQ** | **Gaussian** | -- | **91.90 +/- 0.44** |
| KSD | Unsliced, IMQ | Laplace | -- | 91.12 +/- 0.45 |
| KSD | Sliced (Analytic) | Gaussian | Gaussian | 91.11 +/- 0.45 |
| KSD | Sliced (Finite) | Gaussian | Gaussian | 91.36 +/- 0.45 |
| KSD | Sliced (Finite) | Gaussian | Laplace | 90.70 +/- 0.46 |

Key findings:

- **All KerJEPA variants are viable**: The entire family performs comparably, with accuracies spanning 90.25-91.90%.
- **Best performer: unsliced KSD with IMQ kernel and Gaussian prior** (91.90%), marginally but consistently outperforming LeJEPA's 91.13%.
- **Heavy-tailed kernels are not required**: Despite the theoretical appeal of IMQ's polynomial tail decay for capturing long-range dependencies, the Gaussian kernel performs nearly as well.
- **Gaussian vs. Laplace prior**: Choice of prior has limited impact on downstream classification, suggesting that the favorable learning dynamics of Euclidean gradients may matter more than the specific target geometry.

### Impact of Finite Slicing (Figure 2)

The paper systematically varies output dimension (16, 128, 1024) and training epochs (100, 300, 800) for the MMD regularizer, comparing LeJEPA (finite slicing with 16/128/1024 slices) against the analytically sliced KerJEPA variant (infinite slices):

- **Projector dimension has negligible impact** on the analytically sliced variant, confirming Theorem 7's closed-form result.
- **Finite slicing variance increases with output dimension**: With 1024-dimensional projectors, LeJEPA with only 16 slices shows significant instability and slower convergence.
- **Insufficient slicing delays convergence**: Across all settings, the analytically sliced (infinite-slice) variant converges faster and more stably, with the gap widening for larger embedding dimensions and shorter training schedules.
- **The analytically sliced variation was the top performer in all settings**, demonstrating that eliminating Monte Carlo variance from slicing yields consistent practical benefits.

---

## Comparison to Prior Work

| | **KerJEPA** | **LeJEPA** ([balestriero-2025-iclr](balestriero-2025-iclr.md)) | **VICReg** | **HSIC-SSL** |
|---|---|---|---|---|
| Regularizer | Family of kernel discrepancies (MMD, KSD) | SIGReg (Epps-Pulley, a specific MMD) | Variance + covariance matching | Kernel dependence maximization |
| Kernel choice | Gaussian, IMQ, or any PD kernel | Implicitly Gaussian (via EP test) | N/A (moment-based) | Gaussian or polynomial |
| Prior choice | Gaussian, Laplace, Student-t | Gaussian only | N/A (no explicit prior) | N/A |
| Slicing | Optional (sliced or unsliced) | Required (sliced EP) | Not applicable | Not applicable |
| Complexity | O(n^2 d) unsliced or O(nr(d+u)) sliced | O(nr(d+u)) sliced only | O(n^2 d) for covariance | O(n^2 d) |
| Theoretical contribution | Identifies SIGReg as sliced MMD; derives closed-form limits; introduces KSD for SSL | Proves isotropic Gaussian optimal; derives SIGReg | Empirical regularizer | Kernel independence criterion |

**vs [[balestriero-2025-iclr]] (LeJEPA/SIGReg):** KerJEPA is a direct theoretical extension. It reveals that SIGReg is a sliced MMD with a dimension-dependent heavy-tailed kernel (not the Gaussian kernel, as one might assume), derives the closed-form infinite-slice limit, and shows that equivalent or better regularizers can be constructed without slicing. This paper does *not* challenge LeJEPA's core result (isotropic Gaussian optimality) but substantially enriches the space of regularizers that enforce it.

**vs [[lecun-2022-openreview]] (JEPA position paper):** LeCun's 2022 position paper proposed the JEPA framework conceptually. LeJEPA provided the first principled instantiation via SIGReg. KerJEPA further strengthens the theoretical foundations by showing that SIGReg is one point in a rich family of regularizers -- the JEPA paradigm is not dependent on any single anti-collapse mechanism but admits a toolkit of kernel-based alternatives.

**vs [[maes-2026-arxiv]] (LeWorldModel):** LeWorldModel adopts SIGReg from LeJEPA for its end-to-end world model training. KerJEPA's results suggest that LeWorldModel could potentially benefit from unsliced MMD or KSD regularizers, particularly the IMQ-kernel KSD variant, which achieved the highest accuracy in KerJEPA's ablation and would eliminate the slicing variance that may compound across the multi-step prediction losses in world models.

---

## Strengths

- **Unifying theoretical framework**: The chain of equivalences (Epps-Pulley = MMD; SIGReg = sliced MMD with kappa_d kernel) elegantly reveals the hidden structure of existing methods and opens a systematic design space.
- **Theorem 7 is a genuine insight**: Showing that slicing implicitly changes the kernel to a heavy-tailed, dimension-dependent one is a non-obvious result with practical implications for understanding training dynamics.
- **Key finding that slicing is not necessary**: Proving that equivalent regularizers exist without slicing (via direct MMD or KSD computation) challenges the assumption that slicing is essential for high-dimensional distributional regularization.
- **KSD introduction to SSL**: Bringing Kernel Stein Discrepancy into the SSL framework is novel and practically valuable -- it requires only the target's score function, enabling regularization toward arbitrary differentiable priors without target sampling.
- **Honest computational analysis**: The paper clearly delineates the tradeoffs (sliced: linear but noisy vs. unsliced: quadratic but exact) rather than advocating a single "best" method.
- **Concrete implementation**: PyTorch pseudo-code for both sliced MMDReg and sliced KSDReg (Figure 1) enables direct reproduction.

---

## Weaknesses & Limitations

- **Small-scale evaluation only**: All experiments use ImageNette (a 10-class subset of ImageNet with ~9,500 training images) and ViT-S/8. The paper explicitly acknowledges this: "We acknowledge that the claims are fundamentally driven by the interpretation of results on ImageNette, which is a small dataset." No results on full ImageNet-1K, larger architectures, or other benchmarks are provided.
- **Marginal empirical differences**: The accuracy spread across all KerJEPA variants is only ~1.65% (90.25-91.90%), making it difficult to draw strong conclusions about which regularizer is truly superior. Error bars overlap for most methods.
- **Quadratic cost of unsliced methods**: The best-performing unsliced KSD-IMQ variant requires O(n^2 d) computation per batch. For the batch sizes (256) and projector dimensions (128) used, this is manageable, but scaling to large batch sizes (4096+) or high projector dimensions would be expensive. The paper does not benchmark wall-clock time.
- **No downstream task diversity**: Only linear probe top-1 accuracy is reported. No k-NN evaluation, fine-tuning, detection, segmentation, or transfer tasks are tested. The paper's theoretical analysis suggests kernel choice affects representation geometry, but this is not validated through diverse probing.
- **Laplace prior underperforms**: The Laplace prior consistently performs worse than Gaussian across all methods, yet no analysis explains why, despite the theoretical suggestion that heavy-tailed priors might better capture long-range structure.
- **Limited to image SSL**: No experiments on video, audio, or multimodal settings. The framework is general, but its practical benefits beyond image classification remain unvalidated.

---

## Key Takeaways

- **SIGReg is a sliced MMD, not just an Epps-Pulley test**: Theorem 7 reveals that SIGReg equals the MMD with a Kummer hypergeometric kernel kappa_d that has polynomially decaying tails -- the slicing operation implicitly changes the effective similarity measure from exponential to polynomial decay.
- **Slicing divergences are neither necessary nor sufficient for breaking the curse of dimensionality**: Unsliced kernel discrepancies (MMD, KSD) achieve dimension-free sample complexity and can match or exceed sliced methods, at the cost of quadratic computation.
- **KSD is a natural fit for SSL regularization**: By depending only on the target's score function, KSD regularizers avoid target sampling entirely and extend naturally to non-Gaussian priors (Laplace, Student-t).
- **The design space is broad but forgiving**: Across kernels (Gaussian, IMQ), discrepancies (MMD, KSD), priors (Gaussian, Laplace), and implementations (sliced, unsliced), all variants perform comparably on downstream classification. This suggests that the primary benefit of Euclidean SSL regularization comes from enforcing *some* isotropic structure, with the specific choice of regularizer being a second-order effect.
- **Practical recommendation**: For stability and simplicity, the analytically sliced MMD (closed-form infinite-slice limit) eliminates Monte Carlo variance while maintaining tractability. For maximum flexibility, unsliced KSD with an IMQ kernel achieved the best results and supports arbitrary differentiable priors.
- **Direct relevance to world models**: Since [[maes-2026-arxiv]] (LeWorldModel) builds on SIGReg, KerJEPA's findings about slicing variance and alternative regularizers are directly applicable to improving the stability and performance of JEPA-based world models.

---

## BibTeX

{% raw %}
```bibtex
@article{zimmermann2025kerjepa,
  title={{KerJEPA}: Kernel Discrepancies for Euclidean Self-Supervised Learning},
  author={Zimmermann, Eric and Wiltzer, Harley and Szeto, Justin and Alvarez-Melis, David and Mackey, Lester},
  journal={arXiv preprint arXiv:2512.19605},
  year={2025}
}
```
{% endraw %}

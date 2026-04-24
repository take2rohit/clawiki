---
title: "Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations"
type: paper
paper_id: P043
authors:
  - "Kuang, Yilun"
  - "Dagade, Yash"
  - "Rudner, Tim G. J."
  - "Balestriero, Randall"
  - "LeCun, Yann"
year: 2026
venue: "arXiv"
arxiv_id: "2602.01456"
url: "https://arxiv.org/abs/2602.01456"
pdf: "../../raw/kuang-2026-arxiv.pdf"
tags: [JEPA, self-supervised-learning, sparsity, distribution-matching, maximum-entropy, rectified-Gaussian, non-negative, regularization]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - balestriero-2025-iclr
cited_by: []
---

# Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations

> **Rectified LpJEPA** introduces the Rectified Generalized Gaussian (RGG) distribution as a principled target for JEPA feature regularization, enforced via Rectified Distribution Matching Regularization (RDMReg) -- a sliced two-sample distribution-matching loss that generalizes LeJEPA's SIGReg from isotropic Gaussian to sparse, non-negative, maximum-entropy representations with controllable sparsity via parameters $\{\mu, \sigma, p\}$, achieving competitive downstream accuracy on ImageNet-100 while enabling a continuous spectrum from fully dense ($p=2$, recovering LeJEPA) to highly sparse ($p \to 0$) representations.

**Authors:** Yilun Kuang, Yash Dagade, Tim G. J. Rudner, Randall Balestriero, Yann LeCun (NYU, Duke University, University of Toronto, Brown University) | **Venue:** arXiv 2026 | **arXiv:** [2602.01456](https://arxiv.org/abs/2602.01456)

---

## Problem & Motivation

JEPA methods regularize feature distributions via projection-based distribution matching to prevent collapse. [[balestriero-2025-iclr]] (LeJEPA) introduced SIGReg, which aligns one-dimensional projected feature marginals toward a univariate Gaussian, enforcing isotropic Gaussian embeddings. While provably optimal for downstream probing, this approach has a fundamental limitation: it **inherently favors dense representations** and cannot capture **sparsity** -- a key property observed in efficient neural codes across neuroscience, signal processing, and deep learning.

Sparsity matters for several reasons:
- **Biological plausibility:** Neurons in sensory cortex produce sparse, non-negative activations under metabolic constraints.
- **Interpretability:** Sparse representations yield parts-based decompositions (NMF) rather than holistic ones.
- **Efficiency:** Sparse features enable efficient retrieval, storage, and computation.
- **Out-of-distribution detection:** Sparsity statistics can distinguish in- vs. out-of-distribution inputs.

However, restricting feature distributions to isotropic Gaussian severely limits the representational structures that can be expressed. The question is: **can we design a JEPA regularizer that achieves controllable sparsity while preserving maximum entropy and preventing collapse?**

---

## Core Idea

Replace the isotropic Gaussian target distribution in LeJEPA with the **Rectified Generalized Gaussian (RGG)** distribution $\mathcal{RGN}_p(\mu, \sigma)$ -- a novel family that arises from applying ReLU to the Generalized Gaussian distribution. RGG distributions are maximum-entropy under expected $\ell_p$ norm constraints with non-negativity, naturally induce sparsity (expected $\ell_0$ norm determined by $\{\mu, \sigma, p\}$), and strictly generalize the isotropic Gaussian (recovered when $p=2$, $\mu=0$, no rectification). Enforce this target via **RDMReg** -- a sliced two-sample distribution-matching loss using the Cramer-Wold theorem applied to randomly projected 1D marginals -- which is necessary because RGG is not closed under linear combinations (unlike Gaussians).

---

## How It Works

### Rectified Generalized Gaussian (RGG) Distribution

**Generalized Gaussian** $\mathcal{GN}_p(\mu, \sigma)$: A family parameterized by shape $p > 0$, location $\mu$, and scale $\sigma$:
$$f_{\mathcal{GN}_p(\mu,\sigma)}(x) = \frac{p^{1-1/p}}{2\sigma\Gamma(1/p)} \exp\left(-\frac{|x-\mu|^p}{p\sigma^p}\right)$$
Reduces to Laplace ($p=1$) and Gaussian ($p=2$).

**Rectified Generalized Gaussian** $\mathcal{RGN}_p(\mu, \sigma) = \text{ReLU}(\mathcal{GN}_p(\mu, \sigma))$: A mixture of a Dirac measure at zero (probability mass $\Phi_{\mathcal{GN}_p(0,1)}(-\mu/\sigma)$) and a Truncated Generalized Gaussian on $(0, \infty)$.

**Key properties:**
- **Maximum entropy** under expected $\ell_p$ norm constraint with non-negative support (Proposition 3.3).
- **Controllable sparsity:** Expected $\ell_0$ norm is $d \cdot \Phi_{\mathcal{GN}_p(0,1)}(\mu/\sigma)$ (Proposition 3.5). Decreasing $\mu$ (more negative) increases sparsity.
- **Entropy characterization** via Renyi information dimension $d(\xi)$ (Theorem 3.6), enabling entropy quantification even though standard differential entropy is ill-defined for the mixed measure.
- **Strict generalization of LeJEPA:** When $p=2$, $\mu=0$, and no rectification, $\mathcal{GN}_2(0, \sigma) = \mathcal{N}(0, \sigma^2)$.

### RDMReg: Rectified Distribution Matching Regularization

Since RGG is **not closed under linear combinations** (projecting a Rectified Gaussian along different directions yields distributions outside the RGG family -- see Figure 1b), the 1D projected marginals cannot be compared against a known RGG reference. Instead, RDMReg uses a **two-sample** test:

1. Sample $\mathbf{y} \sim \prod_{i=1}^D \mathcal{RGN}_p(\mu, \sigma)$ from the RGG target.
2. For random unit-norm projections $\mathbf{c}_i \in \mathbb{R}^D$, compare the projected empirical feature distribution $\mathbb{P}_{\mathbf{c}^\top \mathbf{z}}$ with the projected target distribution $\mathbb{P}_{\mathbf{c}^\top \mathbf{y}}$.
3. Use **sliced 2-Wasserstein distance** as the divergence: $\mathcal{L}(\mathbb{P}_{\mathbf{c}^\top \mathbf{z}} \| \mathbb{P}_{\mathbf{c}^\top \mathbf{y}}) = \frac{1}{B}\|(\mathbf{Zc}_i)^\dagger - (\mathbf{Yc}_i)^\dagger\|_2^2$ where $\dagger$ denotes sorting in ascending order.

### Full Loss

$$\min_\theta \mathbb{E}_{\mathbf{x}, \mathbf{x}'}[\|\mathbf{z} - \mathbf{z}'\|_2^2] + \mathbb{E}_{\mathbf{c}}[\mathcal{L}(\mathbb{P}_{\mathbf{c}^\top \mathbf{z}} \| \mathbb{P}_{\mathbf{c}^\top \mathbf{y}}) + \mathcal{L}(\mathbb{P}_{\mathbf{c}^\top \mathbf{z}'} \| \mathbb{P}_{\mathbf{c}^\top \mathbf{y}})]$$

- **First term:** Invariance -- enforces consistency between representations of two views.
- **Second/third terms:** RDMReg -- aligns each view's feature distribution toward the RGG target.

### Architecture

Following standard JEPA/SSL practice:
- **Encoder** $f_{\theta_1}$: ResNet-50 (or ViT, ConvNeXt) backbone.
- **Projector** $f_{\theta_2}$: 3-layer MLP with hidden and output dimension 2048.
- **Rectification:** $\mathbf{z} = \text{ReLU}(f_{\theta_2}(f_{\theta_1}(\mathbf{x})))$ -- explicit ReLU at the projector output enforces non-negativity.
- **RDMReg loss** applied to $\mathbf{z}$ and $\mathbf{z}'$ from two augmented views.
- Linear probe evaluations on both $\mathbf{z}$ (projector features) and $f_{\theta_1}(\mathbf{x})$ (encoder features).

### Hyperparameters of Target Distribution

The set $\{\mu, \sigma, p\}$ collectively controls the feature distribution:
- **$\mu$:** Mean shift. More negative $\mu$ = sparser representations (more mass at zero after ReLU).
- **$p$:** Shape parameter. $p=1$ = Rectified Laplace; $p=2$ = Rectified Gaussian. Smaller $p$ further increases sparsity.
- **$\sigma$:** Scale. Default choice: $\sigma_\text{GN}$ (fixes pre-rectification variance to 1). Alternative: $\sigma_\text{RGN}$ (fixes post-rectification variance to 1).

### Non-Negative VCReg Recovery

The paper proves (Proposition I.1) that minimizing RDMReg loss recovers a form of **Non-Negative VCReg** -- explicitly controlling second-order dependencies through covariance regularization over non-negative features, using only linear projections.

### Inference

Standard frozen-backbone linear probe evaluation on encoder features $f_{\theta_1}(\mathbf{x})$ or projector features $\mathbf{z}$.

---

## Results

### ImageNet-100 Linear Probe (Table 1)

| Method | Target | Encoder Acc1$\uparrow$ | Projector Acc1$\uparrow$ | L1 Sparsity$\downarrow$ | L0 Sparsity$\downarrow$ |
|---|---|---|---|---|---|
| **Rectified LpJEPA** | $\mathcal{RGN}_{2.0}(0, \sigma_\text{GN})$ | **85.08** | 80.00 | 0.3412 | 0.7298 |
| **Rectified LpJEPA** | $\mathcal{RGN}_{1.0}(0.25, \sigma_\text{GN})$ | 84.98 | **80.76** | 0.3745 | 0.7437 |
| **Rectified LpJEPA** | $\mathcal{RGN}_{2.0}(1.0, \sigma_\text{GN})$ | **85.08** | 80.54 | 0.6278 | 0.8668 |
| NCL-ReLU | -- | 82.58 | 76.88 | 0.0037 | 0.0085 |
| NVICReg-ReLU | -- | 84.48 | 77.74 | 0.5207 | 0.7117 |
| VICReg (dense) | -- | 84.18 | 78.88 | 0.7954 | 1.0000 |
| SimCLR (dense) | -- | 83.44 | 77.90 | 0.6338 | 1.0000 |
| LeJEPA (dense) | -- | 84.80 | 79.52 | 0.6365 | 1.0000 |

Rectified LpJEPA achieves the **best encoder accuracy** (85.08%) while maintaining controllable sparsity. Dense baselines (VICReg, SimCLR, LeJEPA) always have L0 sparsity = 1.0 (no zeros). Sparse baselines (NCL-ReLU) achieve extreme sparsity but lower accuracy.

### Controllable Sparsity (Figure 3b)

Empirical $\ell_0$ norms closely track theoretical predictions from Proposition 3.5 across different $\mu$ and $p$ values, confirming that the target distribution parameters provide reliable control over representation sparsity.

### Sparsity-Performance Pareto Frontier (Figure 3c)

Performance drops smoothly as sparsity increases, with a cliff-like drop only when ~95% of entries are zero. This demonstrates significant exploitable sparsity before accuracy degrades -- a favorable trade-off for efficient downstream use.

### Statistical Independence (Figure 4b)

Rectified LpJEPA achieves **lower normalized HSIC** (Hilbert-Schmidt Independence Criterion) than VICReg, NVICReg, and contrastive baselines, indicating that RDMReg encourages not just sparsity but also reduced higher-order statistical dependencies between feature dimensions.

### Maximum Entropy (Figure 4a)

Across a range of sparsity levels, Rectified LpJEPA features lie on the Pareto frontier of entropy vs. sparsity, confirming that the RGG target achieves maximum-entropy representations at each sparsity level.

### Transfer Learning (Tables 3-8)

Rectified LpJEPA demonstrates competitive accuracy across 1-shot and full-shot transfer to DTD, CIFAR-10/100, Flowers, Food, and Pets datasets. Pretrained models exhibit **dataset-adaptive sparsity** (Figure 4c) -- different downstream datasets elicit different sparsity patterns, suggesting sparsity statistics could serve as a proxy for distribution shift detection.

### CIFAR-100 Linear Probe (Table 2)

Encoder accuracies of 65.97-66.29% with varying RGG parameters, competitive with NCL-RepReLU (66.32%) and LeJEPA (65.65%), while achieving controlled sparsity levels.

---

## Comparison to Prior Work

**vs [[balestriero-2025-iclr]] (LeJEPA):** Rectified LpJEPA is a strict generalization. LeJEPA uses SIGReg to match embeddings to $\mathcal{N}(0, I)$ -- equivalent to $\mathcal{GN}_2(0, \sigma)$ without rectification. Rectified LpJEPA generalizes in two ways: (1) ReLU rectification enforces non-negativity and sparsity, (2) Generalized Gaussian with $p \neq 2$ provides additional control over tail behavior and sparsity. When $p=2$, $\mu=0$, and no rectification, LpJEPA reduces to LeJEPA.

**vs [[lecun-2022-openreview]] (JEPA position paper):** LeCun proposed representation space prediction without reconstruction. Rectified LpJEPA contributes to the regularization design space for JEPAs, showing that the target distribution can be extended beyond Gaussians to incorporate sparsity as a principled inductive bias.

**vs [[assran-2023-cvpr]] (I-JEPA):** I-JEPA prevents collapse via EMA + stop-gradient. Rectified LpJEPA replaces these heuristics with explicit distribution matching, following the LeJEPA lineage but extending to sparse targets.

**vs VICReg (Bardes et al., 2022):** VICReg regularizes second-order statistics (variance and covariance). The paper proves (Appendix I) that RDMReg on rectified features recovers a form of Non-Negative VCReg, but goes beyond second-order by matching the full distribution.

**vs Non-Negative Contrastive Learning (NCL, Wang et al., 2024):** NCL applies contrastive losses over ReLU features. Rectified LpJEPA achieves better sparsity-accuracy trade-offs and higher statistical independence (lower HSIC).

---

## Strengths

- **Principled theoretical foundation:** The RGG distribution is derived as the maximum-entropy distribution under $\ell_p$ norm constraints with non-negative support -- not an ad-hoc choice.
- **Strict generalization of LeJEPA:** Recovers LeJEPA as a special case, ensuring no loss of capability while gaining sparsity control.
- **Controllable sparsity:** The parameters $\{\mu, \sigma, p\}$ provide continuous, predictable control over the fraction of zero entries, validated both theoretically and empirically.
- **Reduced higher-order dependencies:** RDMReg achieves lower HSIC than methods that only constrain second-order statistics, indicating more independent feature dimensions.
- **Dataset-adaptive sparsity:** Different downstream tasks elicit different sparsity levels from the same pretrained model, suggesting sparsity as a useful signal for distribution shift detection.
- **Thorough mathematical treatment:** Extensive appendices (40+ pages) covering distribution theory, proofs, entropy characterization, and detailed experimental analysis.

---

## Weaknesses & Limitations

- **Only ImageNet-100 scale:** All experiments use ImageNet-100 (100 classes) with ResNet-50. No experiments at ImageNet-1K scale or with larger models (ViT-L/H).
- **No video experiments:** Despite the "JEPA" framing, all experiments are image-only with two-view augmentation, not video prediction.
- **Two-sample test adds overhead:** RDMReg requires sampling from the RGG target distribution and computing sorted projections, adding computational cost relative to SIGReg's one-sample characteristic-function test.
- **Limited downstream task diversity:** Evaluated only on classification (linear probe). No evaluation on dense tasks (segmentation, detection), retrieval, or generation.
- **The practical benefit of sparsity is not fully demonstrated:** While sparsity is motivated by efficiency and interpretability, no experiments measure actual speedups, memory savings, or improved interpretability.
- **Sensitivity to $\{\mu, \sigma, p\}$ not fully characterized:** While the paper shows controllable sparsity, the optimal choice for downstream accuracy depends on the specific task and dataset.

---

## Key Takeaways

- **Sparse, non-negative representations are achievable within the JEPA framework via principled distribution matching:** Rectified LpJEPA shows that the target distribution for JEPA regularization can be generalized beyond Gaussians to include sparsity as a first-class design parameter.
- **RGG distributions are maximum-entropy under $\ell_p$ norm constraints with non-negativity:** This is not an arbitrary distributional choice -- it is the information-theoretically optimal distribution class for sparse, non-negative features.
- **Sparsity and performance trade off favorably up to ~95% zero entries:** A large fraction of feature dimensions can be zeroed out before accuracy degrades significantly, suggesting substantial room for efficient representation use.
- **RDMReg generalizes SIGReg to non-Gaussian target distributions:** The Cramer-Wold theorem guarantees distributional matching via 1D projections regardless of the target family, enabling principled distribution matching beyond Gaussians.
- **Sparsity statistics vary across downstream tasks:** This opens avenues for using representation sparsity as a lightweight proxy for distribution shift or task difficulty.

---

## BibTeX

{% raw %}
```bibtex
@article{kuang2026rectified,
  title={Rectified {LpJEPA}: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations},
  author={Kuang, Yilun and Dagade, Yash and Rudner, Tim G. J. and Balestriero, Randall and LeCun, Yann},
  journal={arXiv preprint arXiv:2602.01456},
  year={2026}
}
```
{% endraw %}

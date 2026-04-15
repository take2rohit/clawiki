---
title: "Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture"
type: paper
paper_id: P037
authors:
  - "Gogl, Moritz"
  - "Yau, Christopher"
year: 2026
venue: "arXiv"
arxiv_id: "2603.20111"
url: "https://arxiv.org/abs/2603.20111"
pdf: "../../raw/gogl-2026-arxiv.pdf"
tags: [JEPA, variational-inference, VAE, ELBO, tabular-data, collapse-prevention, uncertainty, generative-model]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
  - balestriero-2025-iclr
cited_by: []
---

# Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture

> **Var-JEPA** reverse-engineers a probabilistic latent-variable model from the standard JEPA architecture, showing that JEPA's deterministic encoders and predictor map directly onto variational posteriors and a learned conditional prior in a coupled VAE -- deriving a unified ELBO objective that naturally prevents representational collapse without ad-hoc regularizers (EMA, stop-gradients), enables principled uncertainty quantification via posterior covariances, and is validated through Var-T-JEPA, a tabular-data implementation that achieves competitive downstream performance with selective evaluation via learned uncertainty across 7 datasets.

**Authors:** Moritz Gogl (University of Oxford), Christopher Yau (University of Oxford, Health Data Research UK) | **Venue:** arXiv 2026 | **arXiv:** [2603.20111](https://arxiv.org/abs/2603.20111)

---

## Problem & Motivation

JEPA ([lecun-2022-openreview](../papers/lecun-2022-openreview.md)) is typically framed as a non-generative alternative to likelihood-based self-supervised learning. This framing has two practical consequences:

1. **Representational collapse:** JEPA's deterministic encoders can trivially minimize the prediction loss by mapping all inputs to constant vectors. Existing solutions -- EMA updates ([bardes-2024-tmlr](../papers/bardes-2024-tmlr.md), [assran-2023-cvpr](../papers/assran-2023-cvpr.md)), variance-covariance regularization (VICReg), SIGReg ([balestriero-2025-iclr](../papers/balestriero-2025-iclr.md)) -- are ad-hoc heuristics without a unified probabilistic justification.

2. **No principled uncertainty:** Deterministic JEPA provides no mechanism for estimating per-sample predictive confidence, limiting applicability in safety-critical or noisy settings.

This paper argues that the separation between JEPA and generative modeling is **rhetorical rather than structural**: the general JEPA design (coupled encoders with a context-to-target predictor) aligns closely with a coupled variational autoencoder (VAE). Making this connection explicit yields a principled objective that simultaneously prevents collapse and enables uncertainty quantification.

---

## Core Idea

Reinterpret JEPA within a probabilistic latent-variable framework: replace deterministic encoders with variational posteriors, treat the predictor as a learned latent-space conditional prior $p_\theta(s_y \mid s_x, z)$, add generative decoders, and derive a single Evidence Lower Bound (ELBO) that unifies predictive representation learning with conditional generation. Standard JEPA emerges as a deterministic specialization where regularization is enforced by architectural heuristics rather than an explicit likelihood.

---

## How It Works

### Structure

The model operates on context observations $x$ and target observations $y$, learning latent representations $s_x$ (context), $s_y$ (target), and an auxiliary predictive variable $z$.

**Directed acyclic graph (DAG):**
$$p_\theta(x, y, s_x, z, s_y) = p(s_x) \cdot p(z) \cdot p_\theta(x \mid s_x) \cdot p_\theta(s_y \mid s_x, z) \cdot p_\theta(y \mid s_y)$$

- $p(s_x) = \mathcal{N}(0, I)$ and $p(z) = \mathcal{N}(0, I)$: fixed standard normal priors
- $p_\theta(x \mid s_x)$: context decoder (reconstructs $x$ from $s_x$)
- $p_\theta(s_y \mid s_x, z)$: **predictive network** (JEPA predictor as conditional prior)
- $p_\theta(y \mid s_y)$: target decoder (reconstructs $y$ from $s_y$)

### Variational Posterior

$$q_\phi(s_x, z, s_y \mid x, y) = q_\phi(s_x \mid x) \cdot q_\phi(z \mid s_x) \cdot q_\phi(s_y \mid s_x, z, y)$$

All components are Gaussians with learned means and covariances parameterized by neural networks.

### Training: ELBO Objective

The ELBO decomposes into 5 interpretable loss terms:

$$\mathcal{L}_\text{ELBO} = \alpha^\text{rec} \mathcal{L}^\text{rec} + \alpha^\text{gen} \mathcal{L}^\text{gen} + \alpha_{s_x}^\text{KL} \mathcal{L}_{s_x}^\text{KL} + \alpha_z^\text{KL} \mathcal{L}_z^\text{KL} + \alpha_{s_y}^\text{KL} \mathcal{L}_{s_y}^\text{KL}$$

- $\mathcal{L}^\text{rec}$ **(Context Reconstruction):** How well $x$ is recovered from $s_x$.
- $\mathcal{L}^\text{gen}$ **(Target Generation):** How well $y$ is recovered from $s_y$.
- $\mathcal{L}_{s_x}^\text{KL}$ **(KL on $s_x$):** Regularizes context latents toward $\mathcal{N}(0, I)$.
- $\mathcal{L}_z^\text{KL}$ **(KL on $z$):** Regularizes auxiliary latents toward $\mathcal{N}(0, I)$.
- $\mathcal{L}_{s_y}^\text{KL}$ **(Prediction/Entropy):** $\text{KL}(q_\phi(s_y \mid s_x, z, y) \| p_\theta(s_y \mid s_x, z))$ -- maintains diversity in $s_y$ and serves as the predictive accuracy term.

**Key insight:** The KL terms on $s_x$ and $z$ (with fixed $\mathcal{N}(0, I)$ priors) admit the standard "ELBO surgery" decomposition into aggregated-posterior KL terms, connecting directly to the SIGReg regularization in [balestriero-2025-iclr](../papers/balestriero-2025-iclr.md). The KL on $s_y$ regularizes toward a **learned conditional prior** $p_\theta(s_y \mid s_x, z)$ rather than a fixed distribution.

### Relationship to Standard JEPA

Standard JEPA is recovered as a special case: set $q_\phi(s_x \mid x) = \delta(s_x - f_\text{ctx}(x))$ (deterministic encoder), $q_\phi(s_y \mid s_x, z, y) = \delta(s_y - f_\text{trg}(y))$ (deterministic target encoder), remove decoders ($\alpha^\text{rec} = \alpha^\text{gen} = 0$), and remove KL terms -- leaving only the prediction loss $\|g_\theta(s_x, z) - \text{sg}(s_y)\|^2$ plus heuristic regularizers.

### Var-T-JEPA: Tabular Data Implementation

The framework is instantiated for heterogeneous tabular data:
- Feature-level masking creates context/target splits (following T-JEPA)
- Context encoder, auxiliary encoder, target posterior, and predictor are all transformer-based
- Feature-specific decoders handle numerical (Gaussian) and categorical (softmax) features
- KL annealing schedules prevent posterior collapse early in training
- Per-sample uncertainty estimates computed from posterior covariances

### Inference

For downstream representation learning: use posterior means $s_x = \mu_\phi^{s_x}$, $z = \mu_\phi^z$, $s_w = \mu_\phi^{s_w}$ as deterministic embeddings. For uncertainty estimation: aggregate standard deviations of the target posterior $q_\phi(s_y \mid s_x, z, w)$. Selective evaluation discards the most uncertain samples for improved accuracy.

---

## Results

### Simulation Study (Table 1)

A controlled synthetic setting with known ground-truth latent structure tests 10 objective variants (A-J):

- **(A) Full ELBO (Var-JEPA):** Best overall -- achieves near-perfect probe accuracy (0.996 for $s_x$, 0.993 for $s_y$) with well-behaved distributional metrics.
- **(E) No KL on $s_x$:** Severe distributional collapse (aggregated KL divergence 8.374, Frobenius norm 7.197).
- **(G) No reconstruction/generation:** Representational collapse -- probe accuracy drops to 0.571.
- **(I) No KL terms at all:** Complete distributional failure (SIGReg-MSE > 10, Frobenius > 8.8).
- **SIGReg partially compensates** for removed KL terms but the full ELBO remains optimal.

### Downstream Tabular Performance (Table 2)

Across 7 datasets (Adult, Covertype, Electricity, Credit Card, Bank Marketing, MNIST, Simulated) with 6 downstream predictors (MLP, DCNv2, ResNet, AutoInt, FT-Transformer, XGBoost):

- **Var-T-JEPA** consistently matches or slightly outperforms deterministic T-JEPA embeddings on standard evaluation.
- **Selective evaluation** (discarding 10-50% most uncertain samples) systematically improves accuracy across all datasets and predictors, demonstrating meaningful uncertainty estimates.
- The coverage-accuracy trade-off shows clear risk-coverage curves, particularly on MNIST and SIM datasets with known ground-truth uncertainty.

### Uncertainty Quantification (Figure 3)

- Latent uncertainty correlates with simulated uncertainty (Spearman 0.64 on SIM).
- ROC curves for detecting high-ambiguity samples show AUC of 0.865-0.985 on semi-synthetic datasets.

---

## Comparison to Prior Work

**vs [lecun-2022-openreview](../papers/lecun-2022-openreview.md) (JEPA position paper):** LeCun framed JEPA as explicitly non-generative. Var-JEPA argues this distinction is rhetorical: the JEPA predictor is naturally a learned conditional prior, and the full architecture maps onto a coupled VAE. This reinterpretation does not contradict JEPA's design but provides a probabilistic lens for understanding it.

**vs [assran-2023-cvpr](../papers/assran-2023-cvpr.md) (I-JEPA) and [bardes-2024-tmlr](../papers/bardes-2024-tmlr.md) (V-JEPA):** These instantiate deterministic JEPA with EMA-based collapse prevention. Var-JEPA shows these are deterministic specializations of a broader variational framework, where collapse prevention emerges naturally from the ELBO.

**vs [balestriero-2025-iclr](../papers/balestriero-2025-iclr.md) (LeJEPA):** LeJEPA motivates isotropic Gaussian embeddings and enforces them via SIGReg. Var-JEPA provides a complementary perspective: per-sample KL terms in the ELBO achieve comparable aggregated-distribution behavior (confirmed in Table 1), while the target latent $s_y$ is regularized toward a learned conditional prior rather than a fixed distribution. The paper explicitly studies SIGReg as an augmentation to the ELBO (variants B-D) and finds it compatible but not necessary when full KL terms are present.

**vs Huang 2026 (VJEPA, [huang-2026-arxiv](../papers/huang-2026-arxiv.md)):** Acknowledged as concurrent work. Huang's VJEPA explores a probabilistic formulation for uncertainty-aware latent prediction. Var-JEPA instead formulates JEPA as a coupled latent-variable generative model with a unified ELBO, thereby bridging predictive joint-embedding learning and generative modeling in a single objective.

**vs standard VAEs (Kingma & Welling 2013):** Var-JEPA can be viewed as a coupled VAE where two latent spaces ($s_x$, $s_y$) are linked by a conditional prior. The JEPA predictor is the conditional prior, and the ELBO provides a principled training objective.

---

## Strengths

- **Elegant theoretical bridge:** Demonstrates that JEPA and VAEs are points on the same design spectrum, with the ELBO providing a principled unifying objective.
- **Natural collapse prevention:** The KL divergence terms prevent collapse without ad-hoc mechanisms, and the simulation study (Table 1) validates this rigorously with 10 ablation variants.
- **Practical uncertainty quantification:** Per-sample uncertainty from posterior covariances enables selective evaluation, with clear risk-coverage trade-offs demonstrated across 7 datasets.
- **Connection to LeJEPA/SIGReg:** Shows that per-sample KL regularization toward fixed priors achieves comparable aggregated distributional properties to explicit SIGReg regularization, unifying two independent lines of work.
- **Detailed implementation for tabular data:** Var-T-JEPA is a complete, practical system with transformer encoders, feature-specific decoders, KL annealing, and 7 benchmark datasets.

---

## Weaknesses & Limitations

- **No vision or video experiments:** Validated only on tabular data. The paper outlines how to extend to vision (Section A.4) but provides no image or video results, limiting assessment of scalability.
- **Modest performance gains over deterministic T-JEPA:** On standard (non-selective) evaluation, Var-T-JEPA shows minimal improvement over T-JEPA -- the primary benefit is uncertainty estimation rather than representation quality.
- **Computational overhead:** The variational framework adds decoders, KL computation, and reparameterization, increasing training cost relative to deterministic JEPA.
- **Gaussian assumptions:** All posteriors and priors are diagonal Gaussians, which may be too restrictive for multi-modal or heavy-tailed latent distributions.
- **No comparison with other probabilistic world models:** No empirical comparison with PlaNet, Dreamer, or other latent dynamics models that also use variational objectives.

---

## Key Takeaways

- **JEPA is a deterministic specialization of a variational latent-variable model:** The predictor is a conditional prior, the encoders are amortized posteriors, and ad-hoc anti-collapse costs are indirect forms of distributional regularization. Making this explicit via the ELBO provides a principled foundation.
- **The ELBO naturally prevents collapse:** Removing KL terms causes distributional and representational collapse (simulation study variants E, G, I); the full ELBO maintains well-behaved representations without additional heuristics.
- **Per-sample KL regularization achieves effects comparable to SIGReg:** The standard "ELBO surgery" decomposition connects per-sample KL terms to aggregated distributional regularization, linking Var-JEPA to [balestriero-2025-iclr](../papers/balestriero-2025-iclr.md).
- **Uncertainty quantification enables selective evaluation:** Discarding uncertain samples improves downstream accuracy, a capability absent from deterministic JEPA.
- **The vision extension is promising but unvalidated:** The paper sketches how to apply Var-JEPA to images/video (tokenize, define masked context/target, use ViT encoders, add decoders) but this remains future work.

---

## BibTeX

{% raw %}
```bibtex
@article{gogl2026varjepa,
  title={{Var-JEPA}: A Variational Formulation of the Joint-Embedding Predictive Architecture -- Bridging Predictive and Generative Self-Supervised Learning},
  author={Gogl, Moritz and Yau, Christopher},
  journal={arXiv preprint arXiv:2603.20111},
  year={2026}
}
```
{% endraw %}

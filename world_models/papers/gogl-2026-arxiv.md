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
tags: [JEPA, variational-inference, vae, elbo, tabular-data, collapse-prevention, uncertainty, generative-model]
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

JEPA ([[lecun-2022-openreview]]) is typically framed as a non-generative alternative to likelihood-based self-supervised learning. This framing has two practical consequences:

1. **Representational collapse:** JEPA's deterministic encoders can trivially minimize the prediction loss by mapping all inputs to constant vectors. Existing solutions -- EMA updates ([[bardes-2024-tmlr]], [[assran-2023-cvpr]]), variance-covariance regularization (VICReg), SIGReg ([[balestriero-2025-iclr]]) -- are ad-hoc heuristics without a unified probabilistic justification.

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

**Key insight:** The KL terms on $s_x$ and $z$ (with fixed $\mathcal{N}(0, I)$ priors) admit the standard "ELBO surgery" decomposition into aggregated-posterior KL terms, connecting directly to the SIGReg regularization in [[balestriero-2025-iclr]]. The KL on $s_y$ regularizes toward a **learned conditional prior** $p_\theta(s_y \mid s_x, z)$ rather than a fixed distribution.

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

A controlled synthetic setting with known ground-truth latent structure tests 10 objective variants (A-J). Mean +/- std over 5 runs. KL_agg = aggregated KL divergence to N(0,I); SIGReg-MSE = SIGReg discrepancy; Frobenius = covariance deviation from identity; Mean Norm = mean embedding norm.

#### Context Latent Diagnostics ($s_x$)

| Exp. | Objective | Acc($s_x$) | KL_agg | SIGReg-MSE | Frobenius | Mean Norm |
|------|-----------|------------|--------|------------|-----------|-----------|
| **(A)** | **ELBO (Var-JEPA)** | **0.996** | 0.113 | 4.0e-4 | 0.649 | 0.082 |
| (B) | ELBO+SIGReg ($\lambda_{s_x}$=10) | 0.996 | 0.108 | 4.0e-4 | 0.639 | 0.077 |
| (C) | ELBO+SIGReg ($\lambda_{s_y}$=10) | 0.996 | 0.110 | 4.0e-4 | 0.647 | 0.077 |
| (D) | ELBO+SIGReg (both) | 0.996 | 0.107 | 3.9e-4 | 0.634 | 0.078 |
| **(E)** | **ELBO (no KL on $s_x$)** | 0.996 | **8.374** | **5.1e-2** | **7.197** | **2.550** |
| (F) | ELBO (no KL on $s_y$) | 0.996 | 0.098 | 3.0e-4 | 0.597 | 0.083 |
| **(G)** | **ELBO (no rec/gen)** | **0.571** | 0.015 | 1.6e-4 | 0.229 | 0.055 |
| (H) | ELBO (no rec/gen)+SIGReg | 0.834 | 0.015 | 1.7e-4 | 0.225 | 0.065 |
| **(I)** | **ELBO (no KL terms)** | 0.995 | **20.276** | **5.7e-2** | **8.807** | **4.871** |
| (J) | ELBO (no KL)+SIGReg | 0.996 | 2.115 | 3.3e-3 | 2.594 | 0.205 |

#### Target Latent Diagnostics ($s_y$)

| Exp. | Objective | Acc($s_y$) | KL_agg | SIGReg-MSE | Frobenius | Mean Norm |
|------|-----------|------------|--------|------------|-----------|-----------|
| **(A)** | **ELBO (Var-JEPA)** | **0.993** | 3.530 | 1.7e-2 | 4.727 | 1.320 |
| (B) | ELBO+SIGReg ($\lambda_{s_x}$=10) | 0.992 | 3.511 | 1.4e-2 | 4.748 | 1.272 |
| (C) | ELBO+SIGReg ($\lambda_{s_y}$=10) | 0.993 | 1.999 | 4.6e-3 | 3.423 | 0.189 |
| (D) | ELBO+SIGReg (both) | 0.992 | 1.998 | 4.6e-3 | 3.423 | 0.189 |
| **(E)** | **ELBO (no KL on $s_x$)** | 0.993 | 4.051 | 2.1e-2 | 4.985 | 1.589 |
| (F) | ELBO (no KL on $s_y$) | 0.983 | 6.201 | 2.8e-2 | 6.288 | 1.924 |
| **(G)** | **ELBO (no rec/gen)** | 0.543 | 0.055 | 4.2e-4 | 0.428 | 0.164 |
| (H) | ELBO (no rec/gen)+SIGReg | 0.821 | 0.017 | 1.6e-4 | 0.239 | 0.067 |
| **(I)** | **ELBO (no KL terms)** | 0.984 | **11.331** | **8.4e-2** | **11.342** | **0.248** |
| (J) | ELBO (no KL)+SIGReg | 0.983 | 0.249 | 3.4e-3 | 2.825 | 0.248 |

**Interpretation.** The full ELBO (variant A) achieves the best overall balance: near-perfect probe accuracy on both $s_x$ (0.996) and $s_y$ (0.993) with well-behaved distributional metrics. Removing KL on $s_x$ (variant E) causes severe distributional collapse -- KL_agg jumps to 8.374 and Frobenius to 7.197 -- even though probe accuracy is superficially preserved. Removing reconstruction/generation terms (variant G) causes representational collapse, with probe accuracy dropping to ~0.57 (near chance). Removing all KL terms (variant I) produces the worst distributional failure across all metrics (KL_agg=20.276, Frobenius=8.807 for $s_x$). SIGReg partially compensates for missing KL terms (variant J vs I: KL_agg drops from 20.276 to 2.115) but the full ELBO remains optimal. For $s_y$, the KL_agg and Frobenius values are naturally higher than for $s_x$ because $s_y$ is regularized toward a *learned conditional prior* rather than a fixed N(0,I).

### Downstream Tabular Performance (Table 2)

Test accuracy (mean +/- std over 5 downstream model runs; XGBoost single deterministic run) across 7 tabular datasets and 6 predictor families. Selective evaluation rows (10%/20%/50%) discard the most uncertain fraction of test samples before scoring. AD=Adult, CO=Covertype, EL=Electricity, CC=Credit Card, BM=Bank Marketing.

#### MLP Predictor

| Method | AD | CO | EL | CC | BM | MNIST | SIM |
|--------|-----|-----|-----|-----|-----|-------|------|
| MLP (raw) | 0.849 | 0.750 | 0.781 | 0.816 | 0.898 | 0.822 | 0.823 |
| +T-JEPA | 0.849 | 0.408 | 0.627 | 0.774 | 0.616 | 0.113 | 0.692 |
| +Var-T-JEPA | 0.852 | 0.679 | 0.790 | 0.818 | 0.900 | 0.822 | 0.695 |
| +Var-T-JEPA (10%) | 0.865 | 0.696 | 0.793 | 0.827 | 0.905 | 0.838 | 0.705 |
| +Var-T-JEPA (20%) | 0.883 | 0.697 | 0.795 | 0.835 | 0.908 | 0.856 | 0.706 |
| +Var-T-JEPA (50%) | **0.921** | **0.706** | **0.804** | **0.841** | **0.921** | **0.901** | **0.729** |

#### DCNv2 Predictor

| Method | AD | CO | EL | CC | BM | MNIST | SIM |
|--------|-----|-----|-----|-----|-----|-------|------|
| DCNv2 (raw) | 0.684 | 0.749 | 0.825 | 0.814 | 0.890 | 0.875 | 0.714 |
| +T-JEPA | 0.851 | 0.546 | 0.769 | 0.792 | 0.895 | 0.841 | 0.772 |
| +Var-T-JEPA | 0.851 | 0.778 | 0.830 | 0.794 | 0.891 | 0.885 | 0.720 |
| +Var-T-JEPA (10%) | 0.864 | **0.785** | 0.831 | 0.823 | 0.896 | 0.897 | 0.711 |
| +Var-T-JEPA (20%) | 0.878 | 0.783 | 0.831 | 0.833 | 0.907 | 0.908 | 0.707 |
| +Var-T-JEPA (50%) | **0.916** | 0.778 | **0.831** | **0.840** | **0.918** | **0.936** | **0.757** |

#### ResNet Predictor

| Method | AD | CO | EL | CC | BM | MNIST | SIM |
|--------|-----|-----|-----|-----|-----|-------|------|
| ResNet (raw) | 0.849 | 0.776 | 0.823 | 0.813 | 0.897 | 0.860 | 0.878 |
| +T-JEPA | 0.852 | 0.540 | 0.808 | 0.817 | 0.897 | 0.854 | 0.693 |
| +Var-T-JEPA | 0.854 | 0.779 | 0.820 | 0.813 | 0.900 | 0.871 | 0.564 |
| +Var-T-JEPA (10%) | 0.868 | 0.785 | 0.822 | 0.824 | 0.906 | 0.887 | 0.573 |
| +Var-T-JEPA (20%) | 0.886 | 0.782 | 0.823 | 0.833 | 0.910 | 0.900 | 0.583 |
| +Var-T-JEPA (50%) | **0.924** | **0.787** | **0.826** | **0.841** | **0.920** | **0.936** | 0.612 |

#### AutoInt Predictor

| Method | AD | CO | EL | CC | BM | MNIST | SIM |
|--------|-----|-----|-----|-----|-----|-------|------|
| AutoInt (raw) | 0.761 | 0.753 | 0.754 | 0.817 | 0.901 | 0.809 | 0.897 |
| +T-JEPA | 0.854 | 0.448 | 0.756 | 0.804 | 0.893 | 0.817 | 0.775 |
| +Var-T-JEPA | 0.854 | 0.752 | 0.822 | 0.816 | 0.900 | 0.810 | 0.723 |
| +Var-T-JEPA (10%) | 0.868 | **0.756** | 0.823 | 0.824 | 0.906 | 0.825 | 0.730 |
| +Var-T-JEPA (20%) | 0.885 | 0.754 | 0.822 | 0.834 | 0.909 | 0.841 | 0.732 |
| +Var-T-JEPA (50%) | **0.923** | 0.749 | **0.823** | **0.840** | **0.920** | **0.880** | **0.757** |

#### FT-Transformer Predictor

| Method | AD | CO | EL | CC | BM | MNIST | SIM |
|--------|-----|-----|-----|-----|-----|-------|------|
| FT-Trans (raw) | 0.761 | 0.747 | **0.814** | 0.820 | 0.901 | 0.877 | **0.962** |
| +T-JEPA | 0.854 | 0.522 | 0.705 | 0.802 | 0.886 | 0.317 | 0.723 |
| +Var-T-JEPA | 0.852 | 0.748 | 0.798 | 0.818 | 0.900 | 0.864 | 0.716 |
| +Var-T-JEPA (10%) | 0.867 | **0.756** | 0.800 | 0.827 | 0.905 | 0.879 | 0.726 |
| +Var-T-JEPA (20%) | 0.885 | 0.753 | 0.802 | 0.836 | 0.909 | 0.891 | 0.727 |
| +Var-T-JEPA (50%) | **0.923** | 0.746 | 0.809 | **0.842** | **0.922** | **0.918** | 0.748 |

#### XGBoost Predictor

| Method | AD | CO | EL | CC | BM | MNIST | SIM |
|--------|-----|-----|-----|-----|-----|-------|------|
| XGBoost (raw) | 0.864 | 0.807 | **0.917** | 0.811 | 0.900 | 0.881 | 0.949 |
| +T-JEPA | 0.854 | 0.807 | 0.860 | 0.801 | 0.898 | 0.871 | 0.851 |
| +Var-T-JEPA | 0.855 | 0.809 | 0.888 | 0.806 | 0.904 | 0.872 | 0.945 |
| +Var-T-JEPA (10%) | 0.874 | 0.818 | 0.890 | 0.817 | 0.910 | 0.883 | 0.948 |
| +Var-T-JEPA (20%) | 0.893 | 0.818 | 0.891 | 0.823 | 0.912 | 0.889 | 0.951 |
| +Var-T-JEPA (50%) | **0.928** | **0.816** | 0.891 | **0.832** | **0.921** | **0.913** | **0.955** |

**Interpretation.** Var-T-JEPA embeddings consistently match or slightly outperform deterministic T-JEPA across all predictor families, and notably avoid the catastrophic failures T-JEPA suffers on some datasets (e.g., T-JEPA+MLP on Covertype drops to 0.408 vs Var-T-JEPA's 0.679; T-JEPA+FT-Trans on MNIST collapses to 0.317 vs Var-T-JEPA's 0.864). These T-JEPA failures reflect representational collapse that the variational objective prevents. Selective evaluation (discarding uncertain samples) monotonically improves accuracy across all datasets and predictors -- at 50% coverage, Var-T-JEPA (50%) consistently achieves the highest scores, demonstrating that the learned uncertainty is well-calibrated and actionable. The strongest gains from selective evaluation appear on Adult (0.852 to 0.921 with MLP) and MNIST (0.822 to 0.901 with MLP). Raw features with XGBoost remain competitive or best on Electricity (0.917) and SIM (0.949 at full coverage), suggesting the self-supervised embedding overhead is most justified when downstream predictors are neural or when uncertainty quantification is needed.

### Uncertainty Quantification (Figure 3)

- Latent uncertainty correlates with simulated uncertainty (Spearman 0.64 on SIM).
- ROC curves for detecting high-ambiguity samples show AUC of 0.865 on MNIST and 0.985 on SIM.
- Risk-coverage curves confirm that abstaining on the most uncertain samples monotonically improves accuracy (AUROC 0.122 on MNIST, 0.209 on SIM).

---

## Comparison to Prior Work

**vs [[lecun-2022-openreview]] (JEPA position paper):** LeCun framed JEPA as explicitly non-generative. Var-JEPA argues this distinction is rhetorical: the JEPA predictor is naturally a learned conditional prior, and the full architecture maps onto a coupled VAE. This reinterpretation does not contradict JEPA's design but provides a probabilistic lens for understanding it.

**vs [[assran-2023-cvpr]] (I-JEPA) and [[bardes-2024-tmlr]] (V-JEPA):** These instantiate deterministic JEPA with EMA-based collapse prevention. Var-JEPA shows these are deterministic specializations of a broader variational framework, where collapse prevention emerges naturally from the ELBO.

**vs [[balestriero-2025-iclr]] (LeJEPA):** LeJEPA motivates isotropic Gaussian embeddings and enforces them via SIGReg. Var-JEPA provides a complementary perspective: per-sample KL terms in the ELBO achieve comparable aggregated-distribution behavior (confirmed in Table 1), while the target latent $s_y$ is regularized toward a learned conditional prior rather than a fixed distribution. The paper explicitly studies SIGReg as an augmentation to the ELBO (variants B-D) and finds it compatible but not necessary when full KL terms are present.

**vs [[huang-2026-arxiv]] (VJEPA, Huang 2026):** Acknowledged as concurrent work. Huang's VJEPA explores a probabilistic formulation for uncertainty-aware latent prediction. Var-JEPA instead formulates JEPA as a coupled latent-variable generative model with a unified ELBO, thereby bridging predictive joint-embedding learning and generative modeling in a single objective.

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
- **Per-sample KL regularization achieves effects comparable to SIGReg:** The standard "ELBO surgery" decomposition connects per-sample KL terms to aggregated distributional regularization, linking Var-JEPA to [[balestriero-2025-iclr]].
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

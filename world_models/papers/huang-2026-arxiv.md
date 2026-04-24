---
title: "VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models"
type: paper
paper_id: P035
authors:
  - "Huang, Yongchao"
year: 2026
venue: "arXiv"
arxiv_id: "2601.14354"
url: "https://arxiv.org/abs/2601.14354"
pdf: "../../raw/huang-2026-arxiv.pdf"
tags: [JEPA, variational-inference, world-model, probabilistic, Bayesian, state-space-model, planning, control, uncertainty]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
  - assran-2025-arxiv
cited_by: []
---

# VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models

> **Variational JEPA (VJEPA)** replaces deterministic JEPA's point-estimate predictions with explicit predictive distributions over future latent states via a variational objective (negative log-likelihood + KL regularization), unifying JEPA with Predictive State Representations and Bayesian filtering, proving that VJEPA representations serve as sufficient information states for optimal control, providing formal collapse avoidance guarantees, and extending to Bayesian JEPA (BJEPA) for modular constraint injection -- validated on a "Noisy TV" linear system where VJEPA filters out nuisance distractors that cause generative baselines to collapse.

**Authors:** Yongchao Huang (University of Aberdeen) | **Venue:** arXiv 2026 | **arXiv:** [2601.14354](https://arxiv.org/abs/2601.14354)

---

## Problem & Motivation

Existing JEPA formulations ([[lecun-2022-openreview]], [[assran-2023-cvpr]], [[bardes-2024-tmlr]]) rely on **deterministic** regression objectives (MSE or L1 loss), producing point-estimate predictions in representation space. This has four fundamental limitations:

1. **No explicit uncertainty quantification:** Deterministic predictors output a single embedding for each target, providing no information about predictive confidence or the spread of possible futures.
2. **No multi-modal futures:** Real-world dynamics are often stochastic -- a single point prediction cannot represent multiple plausible outcomes.
3. **Unclear probabilistic semantics:** The conditions under which JEPA representations constitute a *sufficient information state* for optimal control remain unformalized.
4. **No connection to classical filtering theory:** Despite JEPA's structural similarity to state-space models, existing work has not formalized the link to Bayesian filtering or Predictive State Representations (PSRs).

Several recent works have deployed JEPA-based world models for planning and control, but they uniformly assume the learned representation is predictively sufficient without providing formal justification.

---

## Core Idea

Reformulate JEPA as a **probabilistic predictive model** by replacing the deterministic predictor with a learned predictive distribution $p_\phi(Z_T \mid Z_C, \xi_T)$ over future latent states. Train via a variational objective that combines negative log-likelihood of the predictive distribution with KL regularization against a prior. This minimal extension preserves JEPA's core design (no observation reconstruction, no autoregressive likelihood) while enabling explicit uncertainty modeling, formal collapse avoidance through the objective structure, and principled connections to state-space models, Bayesian filtering, and optimal control.

---

## How It Works

### Components

**Context encoder** $f_\theta(x_C)$: Maps context observations to a deterministic latent state $Z_C$.

**Target encoder** $f_{\theta'}(x_T)$: EMA copy that parameterizes an amortized inference distribution $q_{\theta'}(Z_T \mid x_T)$ -- typically a diagonal Gaussian whose mean and variance are output by $f_{\theta'}$.

**Probabilistic predictor** $p_\phi(Z_T \mid Z_C, \xi_T)$: A learned predictive distribution (e.g., Gaussian with learned mean and covariance, mixture model, or normalizing flow) conditioned on the context embedding and structural target specification $\xi_T$.

### Training: Variational Objective

$$\mathcal{L}_\text{VJEPA} = \mathbb{E}_x \mathbb{E}_{Z_T \sim q_{\theta'}(\cdot|x_T)} \left[-\log p_\phi(Z_T \mid Z_C, \xi_T)\right] + \beta \, \mathbb{E}_x \text{KL}(q_{\theta'}(Z_T \mid x_T) \| p(Z_T))$$

- **First term:** Negative log-likelihood of the sampled target under the predictive distribution -- trains the predictor to match the target encoder's distribution in representation space.
- **Second term:** KL regularization against a fixed prior $p(Z_T) = \mathcal{N}(0, I)$, preventing trivial or degenerate target encodings.
- **Special case:** When $q_{\theta'}$ is a Dirac delta and $p_\phi$ is Gaussian with fixed variance, the objective reduces to standard deterministic JEPA (MSE loss).

Training uses the reparameterization trick for backpropagation through $q_{\theta'}$. Target encoder parameters are updated via EMA: $\theta' \leftarrow \tau\theta' + (1-\tau)\theta$.

### Collapse Avoidance (Theorem 1)

Under two mild assumptions -- (i) target diversity (different target inputs produce different inference distributions) and (ii) nontrivial conditioning (the predictive family can distinguish different contexts) -- any globally optimal solution of the VJEPA objective is non-collapsed. Collapse avoidance arises from the objective structure itself (KL regularization penalizes degenerate encodings, likelihood term requires context-dependent predictions), rather than relying solely on architectural heuristics like EMA or stop-gradients.

### Time-Indexed VJEPA as a Latent Dynamical System

When context and target are indexed by time ($x_C \equiv x_{\leq t}$, $x_T \equiv x_{t+\Delta}$), VJEPA defines a latent state-space model:
- **Latent state:** $Z_t := f_\theta(x_{\leq t})$
- **Stochastic transition:** $Z_{t+\Delta} \sim p_\phi(Z_{t+\Delta} \mid Z_t, \xi_{t+\Delta})$
- **No observation likelihood required** -- learning proceeds without $p(x_t \mid Z_t)$

This is sequential but **not autoregressive** over observations: JEPA predicts latent representations directly without factorizing an observation-level likelihood.

**Belief propagation** in latent space:
$$p(Z_{t+\Delta}) = \int p_\phi(Z_{t+\Delta} \mid Z_t, \xi_{t+\Delta}) \, p(Z_t) \, dZ_t$$

approximated via Monte Carlo rollouts for multi-step prediction and planning.

### VJEPA for Control (Section 6)

The paper proves that VJEPA latent states serve as **predictive information states** sufficient for optimal control in POMDPs:
- **Predictive sufficiency:** If $Z_t$ satisfies $p(Z_{t+1:t+H} \mid h_t, u_{t:t+H-1}) = p(Z_{t+1:t+H} \mid Z_t, u_{t:t+H-1})$, then there exists an optimal policy depending only on $Z_t$.
- **Sampling-based latent MPC:** Sample action sequences, roll out latent trajectories via $p_\phi$, evaluate expected cost, select the best action. This requires no observation reconstruction.

### Bayesian JEPA (BJEPA, Section 8)

An extension that factorizes the predictive belief into:
- A **dynamics expert** (learned from data): $p_\phi(Z_{t+\Delta} \mid Z_t, \xi_{t+\Delta})$
- A **modular prior expert** (encoding constraints, physics, goals): $p_\text{prior}(Z_{t+\Delta})$

Combined via **Product of Experts**: $p_\text{BJEPA}(Z_{t+\Delta}) \propto p_\phi(\cdot) \cdot p_\text{prior}(\cdot)$

This enables zero-shot task transfer by swapping prior experts and injecting physical constraints without retraining the dynamics model. Planning uses gradient-based optimization in latent space.

### Inference

For point predictions: use the predictive mean $\hat{Z}_T^\text{mean} = \mathbb{E}_{p_\phi}[Z_T]$ or MAP estimate. For distributional predictions: sample $Z_T^{(m)} \sim p_\phi(Z_T \mid Z_C, \xi_T)$ to approximate expectations of downstream quantities.

---

## Results

The paper presents primarily theoretical contributions, with one empirical validation:

### Noisy TV Linear System (Section 9)

A controlled toy environment with two observation channels: (1) a **task-relevant signal** evolving via linear dynamics $s_{t+1} = A s_t + B u_t + \epsilon$, and (2) a **nuisance "Noisy TV" channel** with high-variance i.i.d. noise uncorrelated with the task.

| Method | Task Signal Recovery | Nuisance Invariance |
|---|---|---|
| Generative (pixel reconstruction) | Fails -- learns to reconstruct nuisance | No |
| Deterministic JEPA | Partial | Partial |
| **VJEPA** | **Full** | **Yes** |
| **BJEPA** | **Full** | **Yes** |

VJEPA and BJEPA successfully filter out the high-variance nuisance distractors because the JEPA/VJEPA objective has no incentive to model observation-level noise -- it only needs to predict task-relevant latent dynamics. Generative baselines that reconstruct observations are forced to model the nuisance channel, wasting capacity and degrading task performance.

---

## Comparison to Prior Work

**vs [[lecun-2022-openreview]] (JEPA position paper):** LeCun proposed Hierarchical JEPA (H-JEPA) as a conceptual blueprint for world models with multi-level, multi-timescale predictive representations. VJEPA formalizes one aspect of this vision: making the predictive distribution explicit. However, VJEPA does not address the hierarchical or multi-timescale aspects.

**vs [[bardes-2024-tmlr]] (V-JEPA) and [[assran-2025-arxiv]] (V-JEPA 2):** These are deterministic instantiations of clip-level JEPA for video. VJEPA provides the probabilistic generalization, replacing MSE/L1 loss with a variational objective. The paper clarifies that clip-level V-JEPA captures intra-clip correlations but does not define a compositional temporal transition model -- VJEPA's time-indexed formulation fills this gap.

**vs [[balestriero-2025-iclr]] (LeJEPA):** LeJEPA proves that isotropic Gaussian embeddings are optimal for downstream probing and enforces this via SIGReg. VJEPA takes a complementary approach: it derives collapse avoidance from the variational objective structure (Theorem 1) rather than from distributional regularization. The two are compatible -- SIGReg could be used within VJEPA's framework.

**vs PlaNet/Dreamer (latent world models):** These learn latent dynamics conditioned on actions but require observation reconstruction (ELBO over pixels) and reward supervision. VJEPA eliminates observation-level likelihood, learning purely through representation prediction. This makes VJEPA invariant to observation noise (demonstrated in the Noisy TV experiment).

**vs MuZero:** MuZero learns latent dynamics without observation reconstruction but relies on value and policy heads, requiring reward supervision. VJEPA is purely self-supervised.

**vs [[maes-2026-arxiv]] (LeWorldModel):** LeWorldModel combines LeJEPA representations with action-conditioned prediction for end-to-end world modeling. VJEPA provides the theoretical foundation for why such JEPA-based world models work (predictive sufficiency for control) and extends them with explicit uncertainty modeling.

---

## Strengths

- **Rigorous theoretical framework:** Provides the first formal probabilistic semantics for JEPA, connecting it to PSRs, Bayesian filtering, and optimal control theory with formal proofs.
- **Collapse avoidance from objective structure:** Theorem 1 shows collapse is fundamentally incompatible with optimality of the variational objective, not merely discouraged by heuristics.
- **Principled uncertainty quantification:** Enables distributional predictions, credible intervals, and sampling-based planning without ensemble methods.
- **Nuisance invariance:** The JEPA prediction objective naturally filters out high-variance observation noise that degrades generative baselines.
- **Modular extension (BJEPA):** The Product of Experts formulation enables zero-shot task transfer and constraint injection without retraining.
- **Clarifies JEPA's sequential vs. autoregressive distinction:** The paper precisely formalizes why JEPA can be sequential (multi-step prediction via latent transitions) without being autoregressive over observations.

---

## Weaknesses & Limitations

- **Empirical validation is minimal:** Only a single toy linear system is tested. No experiments on realistic video, images, or complex control tasks.
- **Scalability untested:** The theoretical framework is general, but practical instantiation with large ViTs, video data, and real robotics tasks is not demonstrated.
- **BJEPA is entirely theoretical:** The Product of Experts extension and gradient-based planning are presented without implementation or experiments.
- **Single author, no code release:** Limits reproducibility and community validation.
- **Gaussian assumptions may be limiting:** The concrete instantiations assume diagonal Gaussian distributions; richer families (mixtures, flows) are discussed but not evaluated.
- **Comparison to concurrent work is theoretical only:** The paper discusses Var-JEPA (gogl-2026-arxiv) as concurrent work but provides no empirical comparison.

---

## Key Takeaways

- **Deterministic JEPA implicitly optimizes a Gaussian likelihood:** Minimizing MSE is equivalent to maximum likelihood under a fixed-variance Gaussian predictive model. VJEPA makes this explicit and generalizes it.
- **JEPA representations can serve as sufficient information states for optimal control:** This is the first formal proof that JEPA's learned latent states are predictively sufficient for planning and control without observation reconstruction.
- **JEPA is sequential but not autoregressive:** A critical distinction -- JEPA predicts latent representations directly, avoiding the need to factorize an observation-level likelihood, which is what gives it invariance to observation noise.
- **The variational objective provides principled collapse avoidance:** Unlike EMA/stop-gradient heuristics, collapse avoidance is a mathematical consequence of the VJEPA objective under mild conditions.
- **BJEPA enables modular world models:** Factorizing predictive belief into dynamics experts and prior experts allows zero-shot transfer and constraint injection -- a step toward the modular cognitive architecture envisioned in [[lecun-2022-openreview]].

---

## BibTeX

{% raw %}
```bibtex
@article{huang2026vjepa,
  title={{VJEPA}: Variational Joint Embedding Predictive Architectures as Probabilistic World Models},
  author={Huang, Yongchao},
  journal={arXiv preprint arXiv:2601.14354},
  year={2026}
}
```
{% endraw %}

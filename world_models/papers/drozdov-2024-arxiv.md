---
title: "Video Representation Learning with Joint-Embedding Predictive Architectures"
type: paper
paper_id: P038
authors:
  - "Drozdov, Katrina"
  - "Shwartz-Ziv, Ravid"
  - "LeCun, Yann"
year: 2024
venue: "arXiv"
arxiv_id: "2412.10925"
url: "https://arxiv.org/abs/2412.10925"
pdf: "../../raw/drozdov-2024-arxiv.pdf"
tags: [JEPA, video-representation, variance-covariance-regularization, VICReg, latent-variable, self-supervised-learning, collapse-prevention]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
cited_by: []
---

# Video Representation Learning with Joint-Embedding Predictive Architectures

> **VJ-VCR** (Video JEPA with Variance-Covariance Regularization) is a JEPA model for self-supervised video representation learning that prevents collapse via variance-covariance regularization (VICReg-style) rather than asymmetric architectures (EMA/stop-gradient), demonstrating that JEPA-style abstract representation prediction captures high-level dynamics (object speeds, actions) better than pixel-level generative baselines on MovingMNIST, CLEVRER, and CATER, while also exploring discrete and sparse latent variables for encoding stochastic future uncertainty.

**Authors:** Katrina Drozdov, Ravid Shwartz-Ziv, Yann LeCun (NYU Center for Data Science, Meta FAIR) | **Venue:** arXiv 2024 | **arXiv:** [2412.10925](https://arxiv.org/abs/2412.10925)

---

## Problem & Motivation

Video representation learning faces two fundamental challenges:

1. **Pixel-level prediction wastes capacity on low-level detail.** Generative models that predict future frames in pixel space must reconstruct textures, lighting, and background dynamics -- details that may be irrelevant for understanding high-level dynamics like object motion, interactions, and actions.

2. **Collapse in JEPAs without architectural asymmetry.** V-JEPA ([[bardes-2024-tmlr]]) prevents collapse through an asymmetric EMA teacher-student architecture with stop-gradients. This introduces hyperparameter sensitivity (EMA momentum, masking strategy) and lacks theoretical grounding for why it prevents collapse.

Additionally, real-world video dynamics are often **stochastic** -- the future is not fully determined by the past. Standard deterministic JEPA predictors cannot represent this uncertainty.

This work asks: (1) Can variance-covariance regularization (VICReg-style) replace architectural asymmetry for collapse prevention in video JEPA? (2) Do JEPA-style abstract representations capture high-level dynamics better than generative models? (3) Can latent variables effectively encode stochastic uncertainty in the JEPA framework?

---

## Core Idea

Build a Video JEPA that prevents collapse through **explicit regularization** (variance-covariance regularization applied directly to the hidden representations) rather than architectural heuristics (EMA teacher, stop-gradients). Train the model to predict future frame representations from past frame representations in the abstract embedding space, and compare against generative baselines that predict in pixel space. Introduce latent variables to handle stochastic video dynamics.

---

## How It Works

### Components

- **Encoder** $f_\theta$: Maps input frames $x$ (past) and target frames $y$ (future) to hidden representations $h_x$ and $h_y$ respectively. Architecture varies by dataset: 5-layer CNN for MovingMNIST, SimVP encoder for CLEVRER/CATER.
- **Predictor** $\text{Pred}(h_x, z)$: Takes the hidden representation of past frames and an optional latent variable $z$, and predicts the hidden representation of future frames $\tilde{h}_y$. Architecture: MLP (2 hidden layers) for MovingMNIST, Swin Transformer for CLEVRER/CATER.
- **Decoder** $\text{Dec}(\tilde{h}_y)$ (optional): Reconstructs target frames $\hat{y}$ from predicted hidden representations. Mirrors the encoder architecture.
- **Latent variable** $z$ (optional): Captures stochastic information about the future not determinable from the past.

### Variance-Covariance Regularization (VCR)

Adapted from VICReg (Bardes et al., 2022) to the video setting. Applied to the set of hidden representations $\{h_x, h_y\}$ across the batch:

**Variance term** (prevents complete collapse):
$$l_\text{var}(\mathbf{H}) = \frac{1}{Td} \sum_{t=1}^T \sum_{k=1}^d \max\left(0, \tau - \sqrt{\text{Var}(\mathbf{H}_{t,k}) + \varepsilon}\right)$$

Encourages each feature dimension to have standard deviation above threshold $\tau = 1$.

**Covariance term** (prevents dimensional collapse):
$$l_\text{cov}(\mathbf{H}) = \frac{1}{Td} \sum_{t=1}^T \sum_{i \neq j} \left[\text{Cov}(\mathbf{H}_{t,:})\right]^2_{i,j}$$

Penalizes off-diagonal entries of the covariance matrix, encouraging de-correlated features.

**Combined:** $l_\text{vcr} = \alpha \cdot l_\text{var} + \beta \cdot l_\text{cov}$

### Training Objective

The energy function for VJ-VCR:
$$E_{\theta_\text{VJ-VCR}}(x, y, z) = \|\text{Pred}(h_x, z) - h_y\|_2^2 + \alpha \cdot l_\text{var}([h_x, h_y]) + \beta \cdot l_\text{cov}([h_x, h_y]) + \gamma \cdot \|\text{Dec}(\text{Pred}(h_x, z)) - y\|_2^2$$

The reconstruction term ($\gamma$) is optional. In the default VJ-VCR setting, $\gamma = 0$ (no pixel-level prediction).

### Latent Variables for Stochastic Dynamics

Two approaches are explored for incorporating latent variables in non-deterministic settings:

1. **Discrete latent:** One-hot vector of dimension 5 (matching the number of possible trajectory switches in stochastic MovingMNIST). The active component selects the top linear layer of the predictor.

2. **Sparse latent:** A sparse vector of dimension 20, inferred at test time via FISTA optimization to minimize the energy function. Different sparsity levels (20% to 80% zeros) are explored.

### Inference

For deterministic tasks: the frozen encoder and predictor are used directly; downstream probes (linear regression for speed prediction, linear classifier for action recognition) are trained on the predicted hidden representations.

For stochastic tasks: the optimal latent $z^*$ is inferred by gradient-based minimization of the energy function with respect to $z$:
$$z^* = \arg\min_z \left(\|\text{Pred}(h_x, z) - h_y\|_2^2 + \gamma\|\text{Dec}(\text{Pred}(h_x, z)) - y\|_2^2\right)$$

---

## Results

### Deterministic Setting: Speed Probing (Table 1)

| Model | Loss($h_y$) | Loss($y$) | VCR | MovingMNIST MSE$\downarrow$ | PSNR$\uparrow$ | CLEVRER MSE$\downarrow$ | RankMe$\uparrow$ |
|---|---|---|---|---|---|---|---|
| VJ-VCR w/o Decoder | Yes | No | Yes | **0.04** | 19.5 | **0.19** | 423.7 |
| VJ-VCR with Decoder | Yes | Yes | Yes | **0.04** | 21.2 | **0.19** | 359.8 |
| Generative with VCR | No | Yes | Yes | 0.10 | **22.9** | 0.22 | 427.4 |
| Generative w/o VCR | No | Yes | No | 0.15 | 22.8 | 0.23 | 160.2 |

JEPA-based models (top two rows) achieve **2.5-3.75x lower speed prediction MSE** than generative baselines, confirming that prediction in abstract representation space captures high-level dynamics (object speed) better than pixel-level prediction. The generative models achieve better pixel reconstruction (higher PSNR) but worse dynamic understanding.

### Non-Deterministic: Stochastic MovingMNIST (Table 2)

| Latent Type | $z^* \to \psi$ (switch prediction) | $z^* \to$ digit identity |
|---|---|---|
| Discrete (one-hot, dim 5) | 79.7% | 11.4% |
| Sparse (80% sparsity) | 94.7% | 31.6% |
| Sparse (20% sparsity) | 99.5% | 57.6% |

Latent variables successfully capture stochastic trajectory information. Higher sparsity constrains the latent to encode only stochastic information (trajectory switch), while lower sparsity allows "leakage" of static information (digit identity). Discrete latents predict switches at 79.7% but are nearly random for digit identity (11.4%), achieving better separation.

### Non-Deterministic: CATER Action Recognition (Figure 2)

VJ-VCR latent variables achieve **67.4% mAP** on multi-label action prediction from inferred $z^*$, vs. 53.8% for generative baselines and 39.6% for random baselines -- a 13.6% improvement over generative models, confirming that JEPA-style abstract prediction produces more informative latent variables.

### Information-Theoretic Analysis (Figure 4)

SVD analysis of the hidden representation matrix shows VJ-VCR has **more uniformly distributed singular values** than generative models (both at training start and end), indicating less dimensional collapse. The cumulative explained variance curve rises more gradually for VJ-VCR, confirming higher effective dimensionality.

---

## Comparison to Prior Work

**vs [[lecun-2022-openreview]] (JEPA position paper):** LeCun 2022 proposed JEPA and argued for latent variables to handle uncertainty in predictions. VJ-VCR provides one of the first empirical explorations of discrete and sparse latent variables within the JEPA framework for video.

**vs [[bardes-2024-tmlr]] (V-JEPA):** V-JEPA uses masking + EMA teacher for collapse prevention and evaluates on large-scale action recognition. VJ-VCR takes a fundamentally different approach: VICReg-style regularization without EMA/stop-gradients, and focuses on small synthetic datasets (MovingMNIST, CLEVRER, CATER) to isolate the question of whether abstract prediction captures dynamics better than pixel prediction.

**vs [[assran-2023-cvpr]] (I-JEPA):** I-JEPA is image-only. VJ-VCR extends JEPA to temporal video prediction (past frames -> future frame representations).

**vs VICReg (Bardes et al., 2022):** VJ-VCR adapts VICReg's variance-covariance regularization from the image domain to video JEPA, applying it to both input and target frame representations across time.

**vs [[balestriero-2025-iclr]] (LeJEPA):** LeJEPA provides a theoretically grounded alternative to heuristic collapse prevention via SIGReg. VJ-VCR uses VICReg-style regularization, which [[balestriero-2025-iclr]] shows is provably insufficient for guaranteeing collapse avoidance (only matches finite moments). However, VJ-VCR demonstrates it works empirically on synthetic video tasks.

---

## Strengths

- **Clean experimental design isolating JEPA vs. generative:** By comparing architecturally identical models (same encoder, predictor, decoder) with only the loss function varying (abstract vs. pixel prediction), the paper provides strong evidence that abstract prediction captures high-level dynamics better.
- **First systematic study of latent variables in video JEPA:** Explores both discrete and sparse formulations, demonstrating that latent variables can encode stochastic information while the hidden representations encode deterministic dynamics.
- **VCR as an alternative to EMA for collapse prevention:** Shows that explicit regularization can replace architectural asymmetry in the video JEPA setting.
- **Information-theoretic analysis:** SVD analysis of hidden representations provides evidence that VCR effectively prevents dimensional collapse.
- **Computationally accessible:** All experiments run on a single NVIDIA RTX 8000 in under 48 hours.

---

## Weaknesses & Limitations

- **Only synthetic datasets:** MovingMNIST (28x28 digits), CLEVRER (64x64 objects), and CATER (128x128 objects) are far from realistic video. Scalability to real-world video (Kinetics, SSv2) is untested.
- **Small model scale:** 5-layer CNNs and Swin Transformers on 64-128px resolution -- orders of magnitude smaller than V-JEPA's ViT-H on 224-384px.
- **No comparison with V-JEPA:** The paper does not compare against V-JEPA's EMA-based approach, making it unclear whether VCR matches EMA for collapse prevention at scale.
- **Latent variable inference requires gradient descent at test time:** For non-deterministic settings, computing $z^*$ requires iterative optimization, which is slow and may not scale.
- **VCR requires batch-level statistics:** Unlike EMA which operates per-sample, VCR depends on batch-level variance and covariance, introducing sensitivity to batch size and composition.
- **Reconstruction quality not primary goal but still relevant:** The paper acknowledges that adding a reconstruction loss improves visualization without hurting dynamics understanding, but does not fully explore this trade-off.

---

## Key Takeaways

- **JEPA-style abstract prediction captures high-level dynamics better than pixel prediction:** On controlled synthetic benchmarks, predicting in hidden space yields 2.5-3.75x lower speed prediction error than predicting pixels, even though pixel predictors achieve better reconstruction quality.
- **Variance-covariance regularization is a viable alternative to EMA for video JEPA collapse prevention:** VCR applied to the encoder's representations prevents both complete and dimensional collapse without requiring stop-gradients or teacher networks.
- **Latent variables can separate deterministic from stochastic information:** Discrete latents cleanly encode trajectory switches (79.7% accuracy) without digit identity (11.4%), while sparse latents offer a controllable trade-off between stochastic and static information via sparsity level.
- **VJ-VCR latent variables are more informative than generative model latents:** On CATER action recognition, VJ-VCR achieves 67.4% mAP vs. 53.8% for generative baselines from inferred latent variables.

---

## BibTeX

{% raw %}
```bibtex
@article{drozdov2024video,
  title={Video Representation Learning with Joint-Embedding Predictive Architectures},
  author={Drozdov, Katrina and Shwartz-Ziv, Ravid and LeCun, Yann},
  journal={arXiv preprint arXiv:2412.10925},
  year={2024}
}
```
{% endraw %}

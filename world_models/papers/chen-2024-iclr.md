---
title: "Denoising with a Joint-Embedding Predictive Architecture (D-JEPA)"
type: paper
paper_id: P033
authors:
  - "Chen, Dengsheng"
  - "Hu, Jie"
  - "Wei, Xiaoming"
  - "Wu, Enhua"
year: 2024
venue: "ICLR 2025"
arxiv_id: "2410.03755"
url: "https://arxiv.org/abs/2410.03755"
pdf: "../../raw/chen-2024-iclr.pdf"
tags: [JEPA, diffusion, generative-modeling, image-generation, next-token-prediction, flow-matching, representation-learning]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
  - bardes-2024-tmlr
cited_by: []
---

# Denoising with a Joint-Embedding Predictive Architecture (D-JEPA)

> **D-JEPA** pioneers the integration of JEPA into generative modeling by reinterpreting masked image modeling as generalized next-token prediction and incorporating a per-token diffusion loss alongside the JEPA prediction loss, achieving state-of-the-art class-conditional ImageNet 256x256 generation (FID=2.04 for D-JEPA-H without classifier-free guidance) with excellent scalability and training efficiency.

**Authors:** Dengsheng Chen, Xiaoming Wei (Meituan), Jie Hu, Enhua Wu (Key Laboratory of System Software, Chinese Academy of Sciences; Institute of Software, CAS) | **Venue:** ICLR 2025 | **arXiv:** [2410.03755](https://arxiv.org/abs/2410.03755)

---

## Problem & Motivation

Joint-embedding predictive architectures (JEPAs) have demonstrated substantial promise in self-supervised representation learning ([[assran-2023-cvpr]] ([I-JEPA](../papers/assran-2023-cvpr.md)), [[bardes-2024-tmlr]] ([V-JEPA](../papers/bardes-2024-tmlr.md))), yet their application to **generative modeling** remains underexplored. JEPA predicts feature embeddings, which cannot be directly decoded into high-quality images. Conversely, diffusion models excel at modeling arbitrary probability distributions but have not directly benefited from the representation learning advances in JEPAs.

The key insight is that JEPA's masked image modeling can be reframed as a **generalized next-token prediction** strategy: instead of predicting one token at a time, the model predicts a set of masked tokens from context tokens. By adding a per-token diffusion (or flow matching) loss to model p(x_i|z_i) -- the distribution of each token conditioned on its predicted latent variable -- D-JEPA bridges representation learning and generative modeling in a single architecture.

---

## Core Idea

D-JEPA consists of three identical visual transformers serving as context encoder, target encoder, and feature predictor. The model is trained with two complementary losses:

1. **Prediction loss** (L_p): Standard JEPA loss using smoothed L1 distance between predicted and target embeddings -- captures high-level semantic information.
2. **Diffusion loss** (L_d): Per-token denoising loss applied to each masked token individually -- models the conditional distribution p(x_i|z_i) for data generation.

The diffusion loss serves a dual purpose: it enables generation *and* prevents representation collapse (since the diffusion loss does not reside in the same energy landscape as JEPA, it avoids the collapse attractor). This eliminates the need for carefully tuned EMA schedules -- D-JEPA uses a constant EMA rate of 0.9999.

Generation uses an iterative **next set-of-tokens prediction** strategy: starting from all-masked tokens, the model progressively unmasks and denoises tokens over T auto-regressive steps following a cosine schedule.

---

## How It Works

### Architecture

Three **identical ViT backbones** (no architectural asymmetry needed):
- **Context encoder** phi: processes unmasked tokens Y (visible context).
- **Target encoder** phi_bar: processes all tokens U; updated via EMA of phi with constant rate alpha=0.9999.
- **Feature predictor** gamma: takes context tokens C from phi and predicts features Z for masked tokens.

Additionally:
- **Projection head** u_theta: two-layer MLP projecting predicted features z_i for the prediction loss.
- **Denoising MLP** epsilon_theta: compact MLP with residual blocks (LayerNorm -> Linear -> SiLU -> Linear + residual). Conditioned on noise schedule timestep t via AdaLN. Size: 6/8/12 residual blocks for Base/Large/Huge variants.

### Tokenization and Masking (Section 3.1)

Images are encoded into non-overlapping semantic tokens U using a VAE (following Stable Diffusion). For a 256x256 image, the VAE produces a 32x32 latent, patchified with p=1 into N=256 tokens. Random masking ratios r_mask are sampled from a truncated normal (mean=1.0, std=0.25, lower bound=0.7), so >70% of tokens are typically masked.

Note: D-JEPA can also work directly on raw pixel patches (Appendix E.2), though latent-space training is preferred for generative quality.

### Training Losses

**Prediction loss** (Eq. 2):
```
L_p = E_{z_i = psi(c_i), g_i} [D(u_theta(z_i), g_i)]
```
where D is smoothed L1 distance, g_i = sg(phi_bar(U)) are stop-gradiented target embeddings. This loss is optimized only for masked tokens.

**Diffusion loss** (Eq. 4):
```
L_d = E_{epsilon, t} [||epsilon - epsilon_theta(x^t_i | t, z_i)||^2]
```
where x^t_i = sqrt(alpha_t) * x_i + sqrt(1-alpha_t) * epsilon is the noise-corrupted token, and epsilon_theta is the small denoising MLP conditioned on both timestep t and predicted latent z_i. Each token is denoised independently.

**Total loss**:
```
L = L_d + L_p
```
No balancing hyperparameter needed -- the two losses are complementary rather than conflicting. The diffusion loss prevents collapse and enables generation; the prediction loss provides high-level semantic features to the denoising MLP.

**Flow matching** alternative (Eq. 7): The paper also supports substituting diffusion loss with flow matching loss, which achieves faster convergence and sometimes higher-quality generation.

### Sampling: Generalized Next Set-of-Tokens Prediction (Algorithm 1)

1. Initialize: all tokens masked (X = empty).
2. For T auto-regressive steps (cosine schedule from mask ratio 1.0 to 0.0):
   a. Encode sampled tokens: C = phi(X).
   b. Predict features for unsampled tokens: Z = gamma(C).
   c. Randomly select n tokens from unsampled set Z.
   d. Denoise selected tokens: {x_0,...,x_n} = denoise(epsilon_theta, {z_0,...,z_n}, tau).
   e. Add denoised tokens to X.
3. Return complete X.

Temperature tau controls sampling diversity (tau=0.98 with CFG, tau=0.94 without). Typically T=64 auto-regressive steps with 100 DDPM denoising steps per token.

Key finding: optimal performance is NOT achieved at T=N (one token at a time). As model scale increases, fewer auto-regressive steps are needed -- D-JEPA-H achieves best results with just 64 steps.

---

## Results

### ImageNet 256x256 Conditional Generation WITHOUT Classifier-Free Guidance (Table 1)

| Model | #Params | #Epochs | FID | IS | Pre. | Rec. |
|---|---|---|---|---|---|---|
| **Base scale (<300M)** | | | | | | |
| MAR-B | 208M | 800 | 3.48 | 192.4 | 0.78 | 0.58 |
| **D-JEPA-B** | **212M** | **1400** | **3.40** | **197.1** | **0.77** | **0.61** |
| MaskGIT | 227M | 300 | 6.18 | 182.1 | 0.80 | 0.51 |
| **Large scale (300-700M)** | | | | | | |
| MAR-L | 479M | 800 | 2.60 | 221.4 | 0.79 | 0.60 |
| DiT-XL | 675M | 1400 | 9.62 | 121.5 | 0.67 | -- |
| SiT-XL | 675M | 1400 | 8.60 | -- | -- | -- |
| **D-JEPA-L** | **687M** | **480** | **2.32** | **233.5** | **0.79** | **0.62** |
| **Huge scale (900M+)** | | | | | | |
| MAR-H | 943M | 800 | 2.35 | 227.8 | 0.79 | 0.62 |
| VDM++ | 2.0B | 1120 | 2.40 | 225.3 | -- | -- |
| **D-JEPA-H** | **1.4B** | **320** | **2.04** | **239.3** | **0.79** | **0.62** |

D-JEPA-L (687M, 480 epochs) surpasses MAR-H (943M, 800 epochs) in FID. D-JEPA-H achieves FID=2.04 with only 320 epochs -- state-of-the-art across all scales without classifier-free guidance.

### With Classifier-Free Guidance (Table 2)

| Model | #Params | FID | IS |
|---|---|---|---|
| MAGVIT-v2 | 307M | 1.78 | 319.4 |
| MAR-L | 479M | 1.78 | 296.0 |
| MDTv2-XL | 676M | **1.58** | 314.7 |
| D-JEPA-L (cfg=3.0) | 687M | **1.58** | 303.1 |
| D-JEPA-L (cfg=3.9) | 687M | 1.65 | **327.2** |
| MAR-H | 943M | 1.55 | 303.7 |
| D-JEPA-H (cfg=3.9) | 1.4B | **1.54** | 324.2 |
| D-JEPA-H (cfg=4.3) | 1.4B | 1.68 | **341.0** |

D-JEPA-H achieves FID=1.54 with cfg=3.9, approaching the VAE reconstruction limit (FID=1.199). IS of 341.0 is the highest reported.

### Scaling Law

As model scale increases:
- Training epochs needed for convergence *decrease* (D-JEPA-B: 1400, D-JEPA-L: 480, D-JEPA-H: 320).
- Optimal number of auto-regressive steps *decreases* (D-JEPA-H achieves best results with 64 steps).
- FID consistently improves: 3.40 -> 2.32 -> 2.04 (without CFG).

### Inference Speed

D-JEPA achieves FID ~4.0 in just **43 milliseconds** and FID ~2.0 in 120 milliseconds on a single H800 GPU (batch size 256). This is significantly faster than standard diffusion models.

---

## Comparison to Prior Work

| | **D-JEPA** | **MAR** | **DiT** | **I-JEPA** |
|---|---|---|---|---|
| Task | Generation + Representation | Generation | Generation | Representation only |
| Architecture | 3 ViTs (context enc, target enc, predictor) + denoising MLP | MAE encoder-decoder + diffusion | Diffusion transformer | Context enc + target enc + predictor |
| Token processing | Continuous (latent space) | Continuous (latent space) | Continuous | Continuous (embedding space) |
| Prediction strategy | Next set-of-tokens | Next set-of-tokens | Full image denoising | Masked prediction |
| Collapse prevention | Diffusion loss + constant EMA | N/A | N/A | Carefully tuned EMA |
| FID (Huge, no CFG) | **2.04** | 2.35 | 9.62 | N/A |
| Training epochs (Large) | **480** | 800 | 1400 | -- |

**vs [[lecun-2022-openreview]] ([JEPA position paper](../papers/lecun-2022-openreview.md)):** LeCun 2022 envisioned JEPA as a path toward world models that predict in representation space. D-JEPA shows that adding a per-token generative loss to JEPA enables high-quality data generation while maintaining the representational benefits, taking a step toward unified understanding-and-generation models.

**vs [[assran-2023-cvpr]] ([I-JEPA](../papers/assran-2023-cvpr.md)):** I-JEPA focuses exclusively on representation learning and cannot generate images. D-JEPA extends the same masked-prediction paradigm with a diffusion loss, enabling generation. The diffusion loss also solves I-JEPA's collapse problem: D-JEPA uses identical architectures for context and target encoders with a simple constant EMA (0.9999), whereas I-JEPA requires careful EMA scheduling.

**vs MAR (Li et al., 2024):** MAR is the closest competitor -- also uses masked prediction with diffusion for generation. However, MAR uses an MAE-style encoder-decoder (predicting from unmasked to masked), while D-JEPA uses the JEPA framework with a separate target encoder providing training targets. D-JEPA achieves better FID with fewer epochs and parameters.

**vs DiT (Peebles & Xie, 2023):** DiT applies diffusion to the entire image jointly. D-JEPA applies per-token diffusion conditioned on JEPA-predicted features, which is more efficient and leverages the structured prediction from JEPA.

---

## Strengths

- **Unifies representation learning and generative modeling**: First work to successfully integrate JEPA with diffusion for high-quality image generation, demonstrating that the two objectives are complementary.
- **State-of-the-art generation quality**: D-JEPA-H achieves FID=2.04 without CFG and FID=1.54 with CFG on ImageNet 256x256, surpassing all prior models at comparable or larger scale.
- **Excellent scalability**: Performance improves monotonically with scale, and larger models need *fewer* training epochs -- a unique and desirable scaling property.
- **Fast inference**: 43ms for FID~4.0 images; the next-set-of-tokens strategy with only 64 auto-regressive steps is far faster than standard 256+ step diffusion.
- **Eliminates careful EMA tuning**: The diffusion loss prevents collapse, allowing a constant EMA rate (0.9999) instead of the complex warmup schedules needed by I-JEPA and V-JEPA.
- **Flexible generation**: Supports both diffusion and flow matching losses; extends to text-conditioned generation, video, and audio (Appendix F, G).

---

## Weaknesses & Limitations

- **Scaling remains a challenge**: The authors acknowledge in the conclusion that scaling up D-JEPA further is a "significant challenge that warrants further investigation."
- **Depends on VAE tokenizer**: Latent-space training requires a pretrained VAE, and the FID lower bound is limited by VAE reconstruction quality (FID=1.199 for the used VAE). Raw-pixel training is possible but produces lower-quality results.
- **No representation learning evaluation**: Despite being built on JEPA, the paper does not evaluate D-JEPA's representations on standard SSL benchmarks (linear probing, fine-tuning on classification). It is unclear whether the generative objective helps or hurts representation quality.
- **High computational cost at largest scale**: D-JEPA-H has 1.4B parameters and trains on 4 workers with 8 H800 GPUs each. The computational requirements are substantial.
- **Generalized next-token prediction complexity**: The iterative sampling with T=64 steps, each involving DDPM denoising, adds complexity compared to single-pass generation.

---

## Key Takeaways

- **JEPA and diffusion are complementary, not competing**: The prediction loss provides high-level semantic features; the diffusion loss models per-token distributions for generation and prevents collapse. Their combination (L = L_d + L_p) needs no balancing hyperparameter.
- **Masked image modeling is generalized next-token prediction**: D-JEPA reframes JEPA's masked prediction as iterative autoregressive generation, where each step predicts and denoises a subset of tokens.
- **The diffusion loss solves JEPA's collapse problem**: By introducing a loss outside JEPA's energy landscape, D-JEPA can use identical encoder architectures with constant EMA, eliminating the fragile scheduling required by I-JEPA/V-JEPA.
- **Larger models need fewer epochs and fewer auto-regressive steps**: D-JEPA exhibits a favorable scaling law where both training and inference become more efficient at larger scale.
- **State-of-the-art FID with fast inference**: FID=2.04 (no CFG) in 120ms and FID~4.0 in 43ms, demonstrating practical viability for real-time generation.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{chen2025djepa,
  title={Denoising with a Joint-Embedding Predictive Architecture},
  author={Chen, Dengsheng and Hu, Jie and Wei, Xiaoming and Wu, Enhua},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  note={arXiv:2410.03755}
}
```
{% endraw %}

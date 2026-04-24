---
title: "Comparison: LeJEPA vs LeWorldModel"
type: query
created: 2026-04-10
papers: [balestriero-2025-iclr, maes-2026-arxiv]
---

# LeJEPA vs LeWorldModel

> **One-line framing**: LeJEPA asks *what distribution should embeddings follow?* — deriving the isotropic Gaussian and SIGReg from theory. LeWorldModel asks *given that answer, how do you build a world model for control?* — applying SIGReg to action-conditioned sequential dynamics, achieving 48× faster planning than DINO-WM with a 15M-param model on a single GPU.

---

## The Relationship in One Paragraph

LeWM is a direct downstream application of LeJEPA. LeJEPA (Balestriero & LeCun, ICLR 2026) proves that JEPA embeddings should follow an isotropic Gaussian distribution and introduces SIGReg as the practical mechanism to enforce this. LeWM (Maes et al., 2026) lifts exactly that regularizer into the sequential world modeling setting: replace the cross-view consistency prediction task of LeJEPA with action-conditioned next-state prediction, and replace the downstream task of image classification with latent planning via CEM. SIGReg is the shared DNA; the problem contexts are entirely different.

---

## Side-by-Side Comparison

| Dimension | **LeJEPA** (P039) | **LeWorldModel** (P031) |
|---|---|---|
| **Goal** | Prove optimal embedding distribution; provide heuristic-free SSL | Build a stable latent world model for robot control from raw pixels |
| **Domain** | Static image SSL (ImageNet, Galaxy10, Food101) | Sequential control (Push-T, Reacher, OGBench-Cube, Two-Room) |
| **Architecture** | Encoder only (ViT-B/H/g) | Encoder (ViT-tiny, 5M) + action-conditioned Transformer predictor (10M) |
| **Prediction task** | Predict global-view embedding from local view | Predict next frame embedding conditioned on current embedding + action |
| **Anti-collapse** | SIGReg (Epps-Pulley CF test on random projections) | Same SIGReg — directly imported from LeJEPA |
| **Loss terms** | 2: prediction loss + SIGReg | 2: next-embedding MSE + SIGReg |
| **Hyperparameters** | 1 (λ=0.05 default) | 1 (λ=0.1 default) |
| **Training data** | Image pairs/views (augmentation-based) | Offline observation-action trajectories (no reward) |
| **Scale** | ViT-H/14 (600M+); large-batch multi-GPU | 15M total; **single GPU, hours** |
| **Inference** | Frozen encoder → linear probe or fine-tuning | Latent planning via CEM (MPC), 0.98s per plan |
| **Downstream metric** | Top-1 accuracy (ImageNet-1k: 79% frozen linear) | Task success rate (Push-T: 90%, Reacher: 84%) |
| **Key baseline beaten** | DINOv2/DINOv3 transfer learning on in-domain datasets | DINO-WM (48× faster) and PLDM (+77pp on Push-T) |
| **Theoretical contribution** | Proves isotropic Gaussian minimizes downstream risk (Theorem 1) | Validates SIGReg as anti-collapse in sequential setting; emergent physics |
| **Training stability** | Stable at 1.8B params, monotonic loss | Monotonic 2-term loss vs. PLDM's noisy 7-term |

---

## What They Share

1. **SIGReg as the sole anti-collapse mechanism**: Both papers rely entirely on Sketched Isotropic Gaussian Regularization — random projections + Epps-Pulley test — to prevent representation collapse. No stop-gradients, no EMA teachers, no negative samples, no whitening.

2. **Single hyperparameter design**: Both converge to λ ≈ 0.05–0.1 as the recommended setting, with performance stable across ±1 order of magnitude.

3. **Architecture agnosticism**: Both work on ViTs and CNNs (ResNet) without modification.

4. **The same authors**: Balestriero and LeCun are co-authors on both; LeWM is partly the applied validation of LeJEPA's theoretical claims in a new domain.

5. **Rejection of generative/pixel-space objectives**: Neither paper reconstructs pixels. Both operate purely in abstract representation space — the core JEPA philosophy.

---

## Where They Diverge

### 1. Scope: SSL Theory vs. World Model Engineering

LeJEPA's primary contribution is **theoretical**: it derives from first principles that the isotropic Gaussian is the unique optimal embedding distribution (Theorems 1–6), then builds a practical regularizer. The empirical results (79% ImageNet-1k) are validation, not the main point.

LeWM's primary contribution is **engineering**: given that SIGReg works (imported directly from LeJEPA with a citation), how do you build a useful world model? The theory is LeJEPA's; the contribution is the architecture, the MPC planning loop, the physical probing, and the demonstration that the two-term loss enables 48× faster planning than alternatives.

### 2. What "prediction" means

In LeJEPA, prediction is across views of the *same* image: the encoder's embedding of one crop should predict the embedding of another crop (via a mean predictor). There is no temporal or causal structure.

In LeWM, prediction is across *time* with an *action*: given the current frame embedding z_t and the action a_t, the predictor Transformer must predict z_{t+1}. This requires learning causal dynamics — the sequential structure of the world. This is the step from SSL to world modeling.

### 3. Scale and accessibility

LeJEPA scales to ViT-g (1.8B parameters) and reports results at that scale. This requires significant multi-GPU infrastructure. LeWM deliberately targets the opposite end: 15M parameters, single GPU, hours of training. The design philosophy is about **democratizing** JEPA world model research, not scaling it.

### 4. Failure modes differ

LeJEPA's failure mode: downstream performance trails DINOv2 by ~3% at the ViT-H scale (79% vs. ~82%). This is a representation quality gap.

LeWM's failure mode: Two-Room (87% vs. 100% for simpler baselines). The isotropic Gaussian prior is counterproductive for environments with very low intrinsic dimensionality — SIGReg spreads features into all latent dimensions even when the task only needs a few.

---

## When to Use Which

| If you want... | Use |
|---|---|
| Strong frozen representations for image classification or transfer | **LeJEPA** |
| A world model for goal-conditioned robot control from pixels | **LeWM** |
| Prove/understand why SSL works without heuristics | **LeJEPA** (has the theory) |
| Fast inference (< 1s planning) on a single GPU | **LeWM** |
| Scale to billion-parameter backbones | **LeJEPA** |
| Detect physically implausible events in video | **LeWM** |
| In-domain pretraining on small specialized datasets | **LeJEPA** |
| Train a world model on offline robot trajectories | **LeWM** |

---

## The Bigger Picture: A Two-Paper Recipe

Together, LeJEPA and LeWM define a complete pipeline from first principles to a working robot controller:

1. **LeJEPA**: *What should latent representations look like?* Answer: isotropic Gaussian. *How to achieve it?* SIGReg.
2. **LeWM**: *Given good representations, how do you model temporal dynamics?* Answer: action-conditioned Transformer predictor. *How do you act?* CEM-based latent MPC.

Neither paper alone closes the loop from theory to embodied agent. Together, they do.

A natural next step — not yet published as of this comparison — is combining LeJEPA-scale pretraining (large ViTs on diverse video data) with LeWM's dynamics predictor, analogous to how DINOv2 pretraining enables DINO-WM but without freezing the encoder.

---

## Open Questions

1. **Does LeJEPA-scale pretraining + LeWM-style finetuning close the OGBench-Cube gap?** DINO-WM wins there because DINOv2's pretraining gives richer 3D features. Would a LeJEPA-pretrained ViT-H/14 followed by LeWM dynamics training close the 12-point gap?

2. **Does LeWM's isotropic Gaussian latent space accelerate RL?** The latent space is structurally well-suited for downstream policy learning — isotropic features minimize downstream bias. Would pairing LeWM with an RL algorithm produce stronger policies than Dreamer/TD-MPC starting from scratch?

3. **How does the SIGReg failure mode in low-dimensionality environments generalize?** LeWM authors note this is a limitation for Two-Room. Is there a principled way to adapt the target distribution to the environment's intrinsic dimensionality, or does hierarchical world modeling (mentioned as future work) naturally resolve this?

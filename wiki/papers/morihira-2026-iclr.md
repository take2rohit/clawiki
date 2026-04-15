---
title: "R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation"
type: paper
paper_id: P046
authors:
  - "Morihira, Naoki"
  - "Nahar, Amal"
  - "Bharadwaj, Kartik"
  - "Kato, Yasuhiro"
  - "Hayashi, Akinobu"
  - "Harada, Tatsuya"
year: 2026
venue: ICLR 2026
arxiv_id: "2603.18202"
url: "https://arxiv.org/abs/2603.18202"
pdf: "../../raw/morihira-2026-iclr.pdf"
tags: [world-model, decoder-free, self-supervised, Dreamer, Barlow-Twins, model-based-rl]
created: 2026-04-15
updated: 2026-04-15
cites: []
cited_by: []
---

# R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

> **R2-Dreamer** replaces the pixel decoder in DreamerV3 with a Barlow-Twins-inspired redundancy-reduction loss between image embeddings and projected latent states, achieving competitive performance on DMC / Meta-World and superior results on tasks with tiny, task-critical objects -- all without a decoder or data augmentation.

## Author Affiliations

| Author | Affiliations |
|---|---|
| Naoki Morihira | Honda R&D Co., Ltd.; The University of Tokyo |
| Amal Nahar | Honda R&D Co., Ltd. |
| Kartik Bharadwaj | Honda R&D Co., Ltd. |
| Yasuhiro Kato | The University of Tokyo |
| Akinobu Hayashi | Honda R&D Co., Ltd.; The University of Tokyo |
| Tatsuya Harada | The University of Tokyo; RIKEN AIP |

## Problem & Motivation

Image-based Model-Based Reinforcement Learning (MBRL) requires representations that capture task-essential information while discarding irrelevant visual details. The dominant Dreamer family ([[hafner-2023-arxiv]] DreamerV3) learns representations via pixel-level reconstruction, which wastes capacity on task-irrelevant regions (e.g., backgrounds) and is computationally expensive. Existing decoder-free alternatives (e.g., DreamerPro, [[hansen-2024-iclr]] TD-MPC2) avoid reconstruction but depend critically on Data Augmentation (DA) -- typically random image shifts -- to prevent representation collapse. DA is a fragile, task-dependent heuristic: random shifting can discard small but crucial objects, and color jittering is harmful when color itself is a key feature. The paper asks whether a principled internal regularizer can fully replace both the decoder and DA, yielding a more versatile and robust framework.

## Core Idea (Plain Language)

Instead of teaching the world model by making it reconstruct pixels (expensive, wastes capacity on backgrounds) or by feeding it artificially distorted images (DA -- risky for small objects), R2-Dreamer uses a statistical trick from self-supervised learning: it computes the cross-correlation matrix between the image encoder's output and the projected latent state, then penalizes (a) deviations of diagonal entries from 1 (ensuring the two representations carry the same information) and (b) large off-diagonal entries (reducing redundancy across feature dimensions). This single, lightweight objective -- borrowed from Barlow Twins -- acts as an internal regularizer that prevents collapse without any external augmentation or pixel decoder.

## How It Works

### Architecture Components

R2-Dreamer modifies [[hafner-2023-arxiv]] DreamerV3 in exactly two ways: (1) remove the image decoder, and (2) add a lightweight linear projector head from latent state to embedding space. Everything else -- the RSSM dynamics, actor-critic, KL balancing -- is identical to DreamerV3.

| Component | Definition |
|---|---|
| Image Encoder | e_t = f_theta(o_t) |
| Sequence Model (GRU) | h_t = f_phi(s_{t-1}, a_{t-1}) |
| Dynamics Predictor | z-hat_t ~ p_phi(z-hat_t \| h_t) |
| Representation Model | z_t ~ q_phi(z_t \| h_t, e_t) |
| Reward Predictor | r-hat_t ~ p_phi(r-hat_t \| s_t) |
| Continue Predictor | c-hat_t ~ p_phi(c-hat_t \| s_t) |
| Projector (new) | k_t = f_phi(s_t) -- linear map from latent state to encoder feature space |

The composite latent state is s_t = (h_t, z_t), combining a deterministic recurrent state h_t with a stochastic categorical state z_t (32 categories x 32 classes, as in DreamerV3).

### The Barlow Twins Objective (L_BT)

The core novelty. Given a mini-batch of B x T samples, compute the cross-correlation matrix **C** between the standardized projector output k_t and the (stop-gradient) image embedding e_t:

```
L_BT = sum_i (1 - C_ii)^2  +  alpha * sum_{i != j} C_ij^2
       [invariance term]         [redundancy term]
```

- **Invariance term**: diagonal entries of C should be 1, meaning each feature dimension of k_t and e_t are correlated -- they carry the same information.
- **Redundancy term**: off-diagonal entries should be 0, decorrelating different feature dimensions and preventing collapse to a low-dimensional subspace.
- **alpha**: single hyperparameter weighting the redundancy term (set to 1/d where d is the feature dimension, following the original Barlow Twins paper).

The target e_t is detached (stop-gradient) to enhance training stability, similar to the strategy in [[hansen-2024-iclr]] TD-MPC2. No augmented views are created; the "two views" are the image embedding e_t and the projected latent state k_t -- a natural pair of views from the model's own internal signals.

### World Model Loss

The full world model objective replaces DreamerV3's reconstruction term with L_BT:

```
L_world = E[ beta_BT * L_BT + sum_t ( L_pred(t) + beta_dyn * L_dyn(t) + beta_rep * L_rep(t) ) ]
```

where L_pred combines reward and continuation prediction losses, L_dyn and L_rep are the two KL-divergence terms with free-bits and KL-balancing (beta_dyn = 1, beta_rep = 0.1), all inherited from DreamerV3.

### Actor-Critic Learning

Unchanged from [[hafner-2023-arxiv]] DreamerV3. The critic predicts lambda-return distributions; the actor is trained with REINFORCE, entropy regularization, and robust return normalization (5th-95th percentile EMA scaling). The critic trains on both imagined rollouts and replay trajectories; the actor trains on imagined trajectories only.

### Training

- RSSM + projector + Barlow Twins loss replaces RSSM + decoder + reconstruction loss.
- All other training details (replay buffer, batch construction, symlog transforms, etc.) match DreamerV3.
- A single hyperparameter configuration is used across all benchmarks (DMC, Meta-World, DMC-Subtle).

### Inference

Identical to DreamerV3: the learned dynamics model imagines trajectories in latent space; the actor selects actions based on the current latent state. No decoder is needed at inference time (which is also true for DreamerV3 during acting, but R2-Dreamer also removes it during training).

## Results

### DeepMind Control Suite (20 tasks, 1M steps)

R2-Dreamer is **competitive** with DreamerV3, TD-MPC2, DreamerPro, and DrQ-v2 on standard DMC in both mean and median task scores. Performance curves (Figure 3) show comparable final scores and learning speed across all baselines.

### Meta-World MT1 (50 tasks, 1M steps)

R2-Dreamer achieves **competitive mean and median success rates** across 50 robotic manipulation tasks, including contact-rich tasks with small objects. Performance is on par with DreamerV3 and TD-MPC2 (Figure 4).

### DMC-Subtle (5 tasks, 1M steps) -- New Benchmark

This is R2-Dreamer's standout result. DMC-Subtle scales down task-critical objects (e.g., Reacher target reduced to 1/3 original size), stress-testing representation quality. R2-Dreamer **substantially outperforms all baselines** on all five tasks (Ball In Cup Catch, Cartpole Swingup, Finger Turn, Point Mass, Reacher), demonstrating superior focus on small, task-relevant visual cues.

### Computational Efficiency

| Method | Training Time (hours) |
|---|---|
| **R2-Dreamer** | **4.4** |
| Dreamer (PyTorch) | 7.0 |
| DreamerPro (PyTorch) | 10.4 |
| Dreamer (JAX, official) | 6.6 |

Measured on DMC Walker Walk, 1M steps, single NVIDIA RTX 3080 Ti. R2-Dreamer is **1.59x faster** than its own DreamerV3 PyTorch reproduction and **2.36x faster** than DreamerPro, due to eliminating the decoder's image generation and DreamerPro's augmentation processing.

### Ablation Studies

**On standard DMC (20 tasks)**:
- **R2-Dreamer + DA**: adding random shifts yields only marginal gains, confirming L_BT alone is sufficient.
- **DreamerPro without DA**: collapses, confirming DA is critical for contrastive/prototypical methods.
- **Dreamer without Decoder**: removing the decoder with no replacement degrades close to "no visual objective" -- the representation learning signal matters.
- **R2-Dreamer (half batch)**: halving the batch size (B=8 to B=16 vs default B=16 to B=32 -- i.e., from 16 to 8 for batch dim) does not significantly degrade performance, indicating robustness to batch size -- consistent with Barlow Twins' known batch-size robustness.

**On DMC-Subtle**:
- **R2-Dreamer + DA**: adding DA **hurts** performance, confirming that random shifts can discard subtle task-critical features. This is the key argument for DA-free internal regularization.

### Saliency Analysis

Occlusion-based saliency maps (Figure 6) on DMC-Subtle Reacher show R2-Dreamer's policy is **sharply focused on the target**, while Dreamer, Dreamer-InfoNCE, and DreamerPro exhibit diffuse, unfocused saliency. This provides qualitative evidence that redundancy reduction encourages task-relevant representations.

## Comparison to Prior Work

| Method | Decoder | DA | Representation Objective | Key Limitation Addressed |
|---|---|---|---|---|
| [[hafner-2023-arxiv]] DreamerV3 | Yes | No | Pixel reconstruction | Wastes capacity on irrelevant pixels |
| [[hafner-2021-iclr]] DreamerV2 | Yes | No | Pixel reconstruction | Same as DreamerV3 |
| DreamerPro (Deng et al., 2022) | No | **Yes** | Prototypical (SwAV-style) | Collapses without DA |
| [[hansen-2024-iclr]] TD-MPC2 | No | **Yes** | Temporal difference + joint embedding | Needs DA for stability |
| Dreamer-InfoNCE | No | No | Contrastive (InfoNCE) | Weaker than R2-Dreamer across benchmarks |
| [[alonso-2024-neurips]] DIAMOND | Yes (diffusion) | No | Full observation generation | Heavier decoder (diffusion model) |
| [[maes-2026-arxiv]] LeWorldModel | No | No | JEPA-style latent prediction | Different JEPA paradigm; also reconstruction-free |
| **R2-Dreamer** | **No** | **No** | **Barlow Twins redundancy reduction** | **No decoder, no DA, competitive performance** |

R2-Dreamer is most directly comparable to DreamerPro (both are decoder-free Dreamer variants), but replaces DA-dependent contrastive/prototypical losses with DA-free redundancy reduction. Relative to the [[lecun-2022-openreview]] JEPA paradigm explored by [[maes-2026-arxiv]] LeWorldModel, R2-Dreamer takes a complementary approach: rather than predicting latent targets in embedding space (JEPA-style), it aligns image embeddings with projected latent states via cross-correlation statistics. Both validate the broader thesis that reconstruction-free world models are viable.

## Strengths

- **Surgical experimental design**: the only change from DreamerV3 is swapping the decoder for a projector + L_BT, cleanly isolating the contribution of the representation learning objective.
- **No data augmentation**: eliminates a fragile, task-dependent heuristic that can distort task-critical information (demonstrated concretely on DMC-Subtle).
- **Computational efficiency**: 1.59x faster than DreamerV3 (PyTorch) and 2.36x faster than DreamerPro, since there is no decoder to generate images and no augmented views to process.
- **Simplicity and minimal hyperparameters**: the Barlow Twins objective adds only a single hyperparameter (alpha), and the method uses one configuration across all benchmarks.
- **DMC-Subtle benchmark**: a useful new testbed that exposes a real failure mode of both reconstruction-based and DA-based methods on tasks with tiny, task-relevant objects.
- **Theoretical grounding**: the paper provides an information-theoretic motivation (Appendix A) connecting L_BT to a Sequential Information Bottleneck objective.
- **Open source**: unified PyTorch codebase released with all baselines re-implemented.

## Weaknesses & Limitations

- **Standard benchmark gains are marginal**: on DMC and Meta-World, R2-Dreamer matches but does not clearly exceed DreamerV3 or TD-MPC2. The main empirical advantage is confined to DMC-Subtle, a new (author-proposed) benchmark.
- **DMC-Subtle is narrow**: only 5 tasks, all involving scaled-down objects. It is unclear how well the advantage generalizes to other challenging scenarios (dynamic distractors, partial observability, complex 3D scenes).
- **No evaluation on Atari or discrete-action domains**: all experiments are continuous control. Generality to discrete settings is untested.
- **No Humanoid or high-dimensional action spaces**: acknowledged by the authors as future work.
- **Comparison to JAX DreamerV3 is indirect**: the primary comparisons use the authors' own PyTorch DreamerV3 reproduction. The official JAX DreamerV3 training time (6.6h) is faster than the PyTorch reproduction (7.0h), narrowing the efficiency gap with R2-Dreamer (4.4h).
- **No dynamic distractor environments**: the Distracting Control Suite is mentioned as future work but not evaluated.
- **Single SSL objective tested**: only Barlow Twins is used; the paper does not systematically compare other information-theoretic objectives (e.g., VICReg) as the internal regularizer, though this is a deliberate simplicity choice.

## Key Takeaways

- A single self-supervised redundancy-reduction loss (Barlow Twins) can fully replace both the pixel decoder and data augmentation in RSSM-based world models, achieving competitive performance with significantly less computation.
- Data augmentation is not just unnecessary but actively harmful when task-relevant information is spatially subtle -- random shifts can destroy the very features the agent needs to perceive.
- The "two views" for Barlow Twins do not need to be augmented copies of the same image; the image embedding and the projected latent state serve as a natural pair of views from the model's internal representations.
- DMC-Subtle reveals a genuine blind spot in existing MBRL methods: reconstruction-based methods waste capacity on backgrounds while DA-based methods risk discarding small objects, and R2-Dreamer avoids both failure modes.
- The decoder-free, DA-free design yields a 1.59x training speedup over DreamerV3, making it practical for scaling to larger problems.

## BibTeX

{% raw %}
```bibtex
@inproceedings{morihira2026r2dreamer,
  title     = {{R2-Dreamer}: Redundancy-Reduced World Models without Decoders or Augmentation},
  author    = {Morihira, Naoki and Nahar, Amal and Bharadwaj, Kartik and Kato, Yasuhiro and Hayashi, Akinobu and Harada, Tatsuya},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.18202}
}
```
{% endraw %}

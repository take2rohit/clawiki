---
title: "Next Embedding Prediction Makes World Models Stronger"
type: paper
paper_id: P053
authors:
  - "Bredis, George"
  - "Balagansky, Nikita"
  - "Gavrilov, Daniil"
  - "Rakhimov, Ruslan"
year: 2026
venue: arXiv
arxiv_id: "2603.02765"
url: "https://arxiv.org/abs/2603.02765"
pdf: "../../raw/bredis-2026-arxiv.pdf"
tags: [world-model, decoder-free, embedding-prediction, Dreamer, model-based-rl]
created: 2026-04-15
updated: 2026-04-15
cites:
  - hafner-2023-arxiv
cited_by: []
---

# Next Embedding Prediction Makes World Models Stronger (NE-Dreamer)

> **One sentence** — NE-Dreamer replaces pixel reconstruction in the Dreamer pipeline with next-step encoder embedding prediction via a causal temporal transformer, achieving substantial gains over DreamerV3 and decoder-free baselines on memory-intensive DMLab Rooms tasks while matching performance on DeepMind Control Suite.

**Authors:** George Bredis, Nikita Balagansky, Daniil Gavrilov, Ruslan Rakhimov (T-Tech) | **Venue:** arXiv (March 2026) | **arXiv:** [2603.02765](https://arxiv.org/abs/2603.02765)

---

## Problem & Motivation

Model-based reinforcement learning (MBRL) from pixels relies on learning compact latent states that capture enough information for long-horizon prediction and control. The dominant approach, exemplified by [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)), trains a world model with a pixel decoder that reconstructs observations. While reconstruction provides dense supervision, it introduces a heavy generative objective that allocates model capacity to visually detailed but task-irrelevant features (textures, backgrounds) and complicates optimization.

Decoder-free alternatives remove pixel reconstruction entirely, training representations with self-supervised objectives. However, most decoder-free methods enforce only *same-timestep* (instantaneous) agreement between encoder and latent representations. Under partial observability — where an agent must integrate information across time rather than react to a single frame — instantaneous alignment is insufficient. The representation must be *predictive across time*: it needs to anticipate what comes next, not merely describe what is observed now. Without an explicit temporal constraint, training can drift or collapse, leading to weak long-horizon structure that surfaces as failure on memory- and navigation-heavy tasks.

---

## Core Idea

NE-Dreamer makes one targeted change to the Dreamer pipeline: it replaces the pixel decoder with a **next-embedding prediction** objective. At each timestep, a causal temporal transformer uses only information available up to time *t* to predict the encoder embedding of the *next* observation at *t+1*. This prediction is aligned to a stop-gradient copy of the actual next-step embedding using a Barlow Twins redundancy-reduction loss. The shift from same-timestep matching to next-step prediction turns representation learning into **causal next-step prediction** — the latent state is explicitly trained to be predictive of future observations in embedding space, without ever needing to reconstruct pixels.

---

## How It Works

### Overview

NE-Dreamer retains the full Dreamer RSSM dynamics backbone, actor-critic imagination loop, and reward/continuation heads. The only modification is the representation learning objective: the pixel decoder is removed and replaced with next-embedding predictive alignment using a causal transformer.

### Latent World Model (RSSM)

The backbone is a standard recurrent state-space model with:

- **Encoder:** Maps observations to embeddings: e_t = f_enc(x_t)
- **Recurrence:** Deterministic state update: h_t = f_rec(h_{t-1}, z_{t-1}, a_{t-1})
- **Stochastic latent:** Prior p_phi(z_t | h_t) and posterior q_phi(z_t | h_t, e_t)
- **Reward and continuation heads:** p_phi(r_t | h_t, z_t) and p_phi(c_t | h_t, z_t)

During training, z_t is sampled from the posterior; during imagination, from the prior.

### Causal Next-Embedding Predictor

A lightweight causal temporal transformer T_theta processes the history of RSSM states, stochastic latents, and actions up to time *t* (using a causal mask) and outputs a predicted next-step embedding:

> e-hat_{t+1} = T_theta(h_{<=t}, z_{<=t}, a_{<=t})

The target is the stop-gradient encoder embedding of the actual next observation:

> e*_{t+1} = sg(f_enc(x_{t+1}))

Gradients flow through e-hat_{t+1} into the transformer T_theta and the RSSM, but not through the target e*_{t+1}. This asymmetry (predicting a stop-gradient target) prevents representational collapse.

**Transformer configuration:** hidden dim = 256, 2 layers, 4 attention heads — a small, lightweight module.

### Alignment Loss (Barlow Twins)

The next-embedding loss L_NE is instantiated as a Barlow Twins redundancy-reduction objective. Let tilde-e_{t+1} and tilde-e*_{t+1} be embeddings normalized per dimension (zero mean, unit variance) within each minibatch. The cross-correlation matrix is:

> C_ij = (1/N) * sum_{(b,t) in I} tilde-e^(b)_{t+1,i} * tilde-e*^(b)_{t+1,j}

where I is the set of valid (non-terminal) transitions and N = |I|. The loss is:

> L_NE = sum_i (1 - C_ii)^2 + lambda_BT * sum_{i != j} C_ij^2

The first term encourages invariance (diagonal correlations close to 1), the second discourages redundancy (off-diagonal correlations close to 0). The key difference from standard Barlow Twins (as used in R2-Dreamer) is that this is applied to **next-step prediction** targets rather than same-timestep paired views.

**BT redundancy lambda:** 5 x 10^{-4}.

### World Model Objective

The complete world model loss replaces the standard decoder reconstruction with the next-embedding loss:

> L_wm = L_rew + L_cont + beta_kl * L_kl + beta_ne * L_NE

where L_rew and L_cont are negative log-likelihoods for reward and continuation prediction, L_kl is the KL regularizer between posterior and prior (with standard Dreamer stabilizers like KL balancing and free-nats), and L_NE is the next-embedding alignment loss. The loss weights beta_ne = 1.0 and beta_kl uses standard Dreamer scaling.

### Actor-Critic Learning

Identical to [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)). Policy and value function are trained on imagined trajectories of horizon H = 15 steps in latent space. The critic predicts the distribution of lambda-returns; the actor maximizes normalized advantages with entropy regularization. All hyperparameters match DreamerV3 (discount gamma = 0.85, lambda = 0.95, entropy scale eta = 3 x 10^{-4}).

### Training Details

- **Architecture:** DreamerV3 Small (12M parameters), Dreamer-S
- **Image resolution:** 64 x 64 RGB
- **Benchmarks:** DeepMind Control Suite (DMC, 20 tasks, 1M steps) and DeepMind Lab (DMLab Rooms, 4 tasks, 50M steps)
- **Seeds:** 5 random seeds per task
- **Environment instances:** 16 (DMC) / 16 (DMLab)
- **Action repeat:** 2 (DMC) / 4 (DMLab)
- **Batch size:** 16, sequence length 64
- **Replay buffer:** 5 x 10^6 capacity
- **All methods use the same unified PyTorch R2-Dreamer codebase** with matched capacity (12M parameters)

---

## Results

### DMLab Rooms: Long-Horizon Memory/Navigation (C1)

The headline result. On four challenging DMLab Rooms tasks requiring memory, spatial reasoning, and long-horizon credit assignment, NE-Dreamer dramatically outperforms all baselines under matched compute and model size (50M environment steps, 12M parameters, 5 seeds):

| Task | NE-Dreamer | DreamerV3 | R2-Dreamer | DreamerPro |
|------|-----------|-----------|------------|------------|
| Rooms Collect Good Objects Train | ~9 | ~3 | ~4 | ~2 |
| Rooms Exploit Deferred Effects Train | ~45 | ~5 | ~10 | ~3 |
| Rooms Select Nonmatching Object | ~55 | ~10 | ~15 | ~5 |
| Rooms Watermaze | ~13 | ~3 | ~5 | ~2 |

(Approximate values read from Figure 3 learning curves at 50M steps.)

NE-Dreamer learns reliably and achieves substantially higher final returns across all four tasks. The largest gains occur where success depends on maintaining state over long horizons — remembering key scene elements, planning multi-step behaviors — rather than reacting to short-lived visual cues.

### DMC: No Regression Without Reconstruction (C3)

On the 20-task DeepMind Control Suite (1M environment steps), NE-Dreamer matches or slightly exceeds DreamerV3 and competitive decoder-free baselines (R2-Dreamer, DreamerPro) in both task mean and task median return. This confirms that replacing pixel reconstruction with next-embedding prediction does **not** sacrifice performance on standard continuous-control benchmarks — the gains on harder DMLab tasks come without regression on easier domains.

### Ablations: Isolating the Mechanism (C2)

Three ablations on DMLab Rooms isolate what drives NE-Dreamer's performance:

| Variant | Effect |
|---------|--------|
| **No transformer** (remove causal temporal transformer, use feedforward/shallow) | Performance collapses on all tasks — causal sequence modeling is indispensable for partial observability |
| **No next-step shift** (predict current-step embedding instead of next-step) | Nearly complete loss of the gains — temporal prediction, not just the transformer, is critical |
| **No projector** (remove lightweight projection head before alignment) | Minor reduction in asymptotic performance — projector helps optimization but is not fundamental |

These ablations demonstrate that the core mechanism is the *combination* of a causal temporal transformer and a next-step prediction target. Neither alone is sufficient.

### Representation Diagnostics

Post-hoc pixel decoders (trained on frozen latents, not used during agent training) reveal qualitative differences:

- **NE-Dreamer:** Reconstructions preserve task-relevant objects and spatial layout consistently across time. Object identity and position are stable over long sequences.
- **Dreamer / R2-Dreamer:** Task-specific attributes (e.g., relevant objects in a room) appear transiently in one timestep but disappear or degrade in subsequent latents, even when the underlying scene has not changed.

This temporal consistency directly explains NE-Dreamer's advantage on memory tasks: same-timestep objectives allow latent drift toward transient visual details, while next-step prediction forces the latent state to retain information that is predictive of future structure.

---

## Comparison to Prior Work

| Method | Decoder | Representation Objective | Temporal Modeling | DMLab Rooms | DMC |
|--------|---------|-------------------------|-------------------|-------------|-----|
| **NE-Dreamer** | None | Next-step embedding (Barlow Twins) | Causal transformer | Best | Matches baselines |
| DreamerV3 | Pixel | Reconstruction | RSSM (GRU) | Weak | Strong |
| R2-Dreamer | None | Same-step Barlow Twins + projector | RSSM (GRU) | Moderate | Strong |
| DreamerPro | None | Same-step SwAV + augmentations | RSSM (GRU) | Weak | Strong |
| Dreamer (no recon) | None | Reward + continuation + KL only | RSSM (GRU) | Very weak | Moderate |

**[[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)):** The primary decoder-based baseline. NE-Dreamer keeps the same RSSM backbone and actor-critic but removes the pixel decoder. On DMLab Rooms, DreamerV3 fails to learn effectively because reconstruction allocates capacity to task-irrelevant visual details. NE-Dreamer dramatically outperforms it on all four Rooms tasks while matching performance on DMC.

**R2-Dreamer:** The closest decoder-free baseline — it also uses Barlow Twins but applies it to *same-timestep* encoder-latent pairs via a lightweight projector. NE-Dreamer's critical innovation is shifting the alignment target to the *next* timestep and using a causal transformer, which yields substantially better long-horizon representations. R2-Dreamer is outperformed on all DMLab Rooms tasks.

**DreamerPro:** Uses SwAV with strong data augmentations (random image shifts) to enforce invariance. Despite augmentations, it struggles on DMLab Rooms because per-timestep invariance does not guarantee temporal predictiveness. NE-Dreamer outperforms it without any data augmentation.

**[[maes-2026-arxiv]] ([LeWorldModel](../papers/maes-2026-arxiv.md)):** Also adopts decoder-free world modeling with embedding prediction, but approaches it from a different angle — using a JEPA-style architecture. NE-Dreamer and LeWorldModel converge on the same high-level insight that predicting future embeddings (rather than reconstructing pixels or matching current-step representations) produces stronger world models. NE-Dreamer validates this within the Dreamer/RSSM framework specifically.

**[[lecun-2022-openreview]] ([LeCun, 2022](../papers/lecun-2022-openreview.md)):** The JEPA position paper argues that prediction in embedding space (rather than pixel space) is the right objective for world models. NE-Dreamer provides empirical evidence supporting this thesis within model-based RL, demonstrating that next-step embedding prediction outperforms reconstruction on tasks requiring temporal coherence.

**[[micheli-2023-iclr]] ([IRIS](../papers/micheli-2023-iclr.md)):** Uses discrete VQ-VAE tokens with an autoregressive transformer for world modeling. IRIS operates in token space rather than continuous embedding space. NE-Dreamer operates in continuous embedding space with a redundancy-reduction loss, avoiding the information bottleneck of discrete tokenization.

**[[alonso-2024-neurips]] ([DIAMOND](../papers/alonso-2024-neurips.md)):** Uses a diffusion model operating directly in pixel space for world modeling. DIAMOND and NE-Dreamer represent opposite strategies: DIAMOND invests more compute into pixel-level fidelity, while NE-Dreamer abandons pixel modeling entirely in favor of embedding-space prediction. Both outperform standard DreamerV3 on their respective target benchmarks.

---

## Strengths

- Achieves dramatic improvements on memory- and navigation-heavy DMLab Rooms tasks — the settings where temporal coherence matters most — without any domain-specific tuning, augmentations, or auxiliary losses.
- Minimal architectural change: only replaces the pixel decoder with a small causal transformer (256-dim, 2 layers, 4 heads) and a Barlow Twins loss on next-step embeddings. All other components (RSSM, actor-critic, hyperparameters) remain identical to DreamerV3.
- Clean ablation study isolates the two essential ingredients (causal transformer + next-step target shift) and shows each is necessary — the gains are not from any single architectural trick.
- No regression on DMC continuous control, confirming the approach is broadly applicable and does not sacrifice performance on easier domains.
- Representation diagnostics (post-hoc decoder reconstructions) provide compelling qualitative evidence that next-step prediction yields temporally stable, task-relevant representations where same-timestep methods produce temporally inconsistent ones.

## Weaknesses & Limitations

- Evaluated only on DMLab Rooms (4 tasks) and DMC (20 tasks) under a single model size (12M parameters). No evaluation on Atari, Minecraft, ProcGen, or other standard benchmarks where DreamerV3 has been validated.
- DMLab results are reported at 50M environment steps; whether NE-Dreamer's advantages hold at larger training budgets (100M+) is unknown.
- The paper acknowledges that experiments focus on environments where long-term structure is the primary challenge. Whether decoder-free, prediction-based objectives can match reconstruction in domains requiring fine visual detail remains open.
- All baselines use the same 12M-parameter Dreamer-S architecture. Scaling behavior (whether the gap persists or closes at 200M+ parameters) is not explored.
- Only one alignment loss (Barlow Twins) is tested. The paper notes that any loss encouraging expressiveness and non-degeneracy could be substituted, but no alternatives (VICReg, SimSiam, BYOL, contrastive) are evaluated.
- Single affiliation (T-Tech), no code release mentioned in the paper.

## Key Takeaways

- **Next-step embedding prediction is a stronger representation objective than same-timestep alignment for partially observable MBRL.** The shift from matching current observations to predicting future embeddings forces the latent state to be temporally coherent, directly addressing the failure mode of decoder-free methods on memory tasks.
- **The core mechanism is predictive sequence modeling** (causal transformer + next-step target), not reconstruction or auxiliary tricks. Ablations show that removing either the transformer or the temporal shift individually collapses most of the gains.
- **Decoder-free world models can outperform reconstruction-based ones** on challenging partially observable tasks — but only when the representation objective explicitly enforces temporal predictiveness, not just instantaneous agreement.
- **No regression on standard benchmarks.** Replacing reconstruction with next-embedding prediction matches DreamerV3 on DMC, establishing that the approach is a strict improvement under matched compute and model size.
- **Convergence with the JEPA thesis.** NE-Dreamer provides RL-specific evidence for the broader hypothesis ([[lecun-2022-openreview]]) that prediction in embedding space, not pixel space, is the right objective for learning world models. The parallel with [[maes-2026-arxiv]] (LeWorldModel) reinforces this direction.

---

## BibTeX
{% raw %}
```bibtex
@article{bredis2026nedreamer,
  title     = {Next Embedding Prediction Makes World Models Stronger},
  author    = {Bredis, George and Balagansky, Nikita and Gavrilov, Daniil and Rakhimov, Ruslan},
  journal   = {arXiv preprint arXiv:2603.02765},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.02765},
  eprint    = {2603.02765},
  archivePrefix = {arXiv}
}
```
{% endraw %}

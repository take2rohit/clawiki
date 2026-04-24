---
title: "What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?"
type: paper
paper_id: P021
authors:
  - "Terver, Basile"
  - "Yang, Tsung-Yen"
  - "Ponce, Jean"
  - "Bardes, Adrien"
  - "LeCun, Yann"
year: 2025
venue: ICLR 2026
arxiv_id: "2512.24497"
url: "https://arxiv.org/abs/2512.24497"
pdf: "../../raw/terver-2025-iclr.pdf"
tags: [JEPA, world-model, planning, model-predictive-control]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - hafner-2019-icml
  - hafner-2023-arxiv
  - hansen-2022-icml
  - hansen-2024-iclr
  - bar-2024-cvpr
cited_by:
  - maes-2026-arxiv
---

# What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?

> A systematic ablation study of JEPA-based world models (JEPA-WMs) for physical planning, investigating planner choice, multistep rollout training, proprioception, context length, encoder type, predictor architecture, and model scale -- culminating in a unified recipe that outperforms both DINO-WM and V-JEPA-2-AC on navigation and manipulation tasks across simulated and real-world robotic environments.

**Authors:** Basile Terver (Meta FAIR / INRIA Paris), Tsung-Yen Yang (Meta FAIR), Jean Ponce (ENS / NYU), Adrien Bardes (Meta FAIR), Yann LeCun (Meta FAIR) | **Venue:** ICLR 2026 | **arXiv:** [2512.24497](https://arxiv.org/abs/2512.24497)

---

## Problem & Motivation

A growing family of methods trains action-conditioned world models in the learned representation space of a pretrained visual encoder, then plans by optimizing action sequences directly in that latent space. These Joint-Embedding Predictive World Models (JEPA-WMs) -- including DINO-WM (Zhou et al., 2024a), V-JEPA-2-AC (Assran et al., 2025), and PLDM (Sobal et al., 2025) -- promise more efficient planning than pixel-space models by abstracting away irrelevant visual detail. However, no prior work has systematically studied the design decisions that determine whether a JEPA-WM actually succeeds at planning: the choice of planning optimizer, whether to use multistep rollout losses, the role of proprioception, the effect of training context length, which pretrained encoder to use, how to condition the predictor on actions, and how model scale interacts with task complexity.

Each of these decisions was previously made ad hoc within individual papers (DINO-WM uses feature conditioning with sincos embeddings; V-JEPA-2-AC uses sequence conditioning with RoPE). Without controlled comparisons, the field cannot determine which choices genuinely matter and which are incidental. This paper fills that gap.

---

## Core Idea

The authors formalize JEPA-WMs as a unified family (Equations 1-4 in the paper) sharing a common structure: a frozen visual encoder, an optional proprioceptive encoder, an action encoder, and a learned predictor. They then vary one design dimension at a time across seven axes -- planner, multistep rollout, proprioception, context length, encoder type, predictor architecture, and model scale -- while holding all else constant. The base configuration is DINO-WM with a ViT-S encoder and depth-6 predictor. The study spans six simulated environments (Metaworld Reach/Reach-Wall, Push-T, PointMaze, Wall) and two real-world robotic settings (DROID manipulation, Robocasa zero-shot transfer), yielding a comprehensive recipe for optimal JEPA-WM design.

---

## How It Works

### Training

A JEPA-WM trains a predictor $P_\theta$ on top of a **frozen** pretrained visual encoder $E_\phi^{vis}$. For each training sample, the encoder embeds observations into a latent space. An action encoder $A_\theta$ embeds the robotic actions. The predictor takes a context window of past state embeddings and actions (up to $W$ timesteps) and predicts the next state embedding. The training loss is MSE between the predictor's output and the encoder's embedding of the actual next observation (teacher forcing):

$$\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} L\bigl[P_\theta\bigl(E_{\phi,\theta}(o_{t-w:t}^b), A_\theta(a_{t-w:t}^b)\bigr),\; E_{\phi,\theta}(o_{t+1}^b)\bigr]$$

The predictor is trained with a **causal frame-level attention mask** and simultaneously learns to predict from all context lengths $w = 0$ to $w = W-1$, where $W$ defaults to 3.

Key design variants studied:

- **Multistep rollout loss**: Beyond 1-step teacher forcing ($\mathcal{L}_1$), additional $k$-step rollout losses $\mathcal{L}_k$ are computed by auto-regressively unrolling the predictor $k$ steps using its own predictions, with truncated backpropagation through time (TBPTT). Models are trained with the sum $\mathcal{L}_1 + \cdots + \mathcal{L}_k$ for up to $k=6$ steps.
- **Proprioception**: An optional shallow proprioceptive encoder $E_\theta^{prop}$ is jointly trained alongside the predictor. Both visual and proprioceptive loss terms are used.
- **Action conditioning**: Two paradigms -- **feature conditioning** (DINO-WM style: action embeddings concatenated to visual features along the embedding dimension) vs. **sequence conditioning** (V-JEPA-2-AC style: actions encoded as separate tokens concatenated along the sequence dimension with RoPE). The paper also tests AdaLN conditioning (action information injected at every transformer layer via adaptive layer normalization) and AdaLN-zero.
- **Encoder types**: DINOv2 (ViT-S, ViT-B, ViT-L), DINOv3 (ViT-L), V-JEPA (ViT-L), V-JEPA-2 (ViT-L).

### Inference (Planning)

At planning time, given a start observation $o_t$ and a goal observation $o_g$:

1. **Encode** both observations: $z_t = E_{\phi,\theta}(o_t)$, $z_g = E_{\phi,\theta}(o_g)$
2. **Sample** candidate action trajectories $a_{t:t+H-1}$ of horizon $H$
3. **Roll out** the predictor auto-regressively on each trajectory to produce predicted state embeddings
4. **Evaluate** each trajectory with a planning cost $L^p_\alpha$ -- a dissimilarity metric (L1, L2, or negative cosine similarity) between predicted and goal embeddings, applied pairwise on all intermediate and final predictions, with an optional proprioceptive term weighted by $\alpha$
5. **Optimize** the action sequence using one of four planners:
   - **CEM** (Cross-Entropy Method): Population-based; iteratively refines a Gaussian proposal distribution over actions using elite samples
   - **NG** (Nevergrad): A meta-optimizer from the Nevergrad black-box optimization library, using the NGOpt wizard which selects diagonal CMA-ES; requires no hyperparameter tuning
   - **Adam / GD**: Gradient-based planners that directly backpropagate through the differentiable cost function
6. **Execute** the first $m$ actions (MPC receding horizon), observe the new state, and replan

The planning uses a sliding context window $W^p$ over past predictions fed to the predictor, fixed at $W^p = 2$ for all experiments.

---

## Results

### Planning Optimizer Comparison (Figure 3a)

| Optimizer | Metaworld (Avg) | 2D Nav (Avg) | DROID | Robocasa (Avg) | Overall Avg |
|---|---|---|---|---|---|
| CEM $L_2$ | Strong | **Best** | **Best** | Competitive | **Best overall** |
| CEM $L_1$ | Good | Good | Good | Good | Slightly below $L_2$ |
| NG $L_2$ | Good | Moderate | Competitive with CEM | Competitive | Good, fewer hyperparams |
| Adam $L_2$ | **Best** | Poor | Poor | Good | Task-dependent |
| GD $L_2$ | Good | Very poor | Poor | Good | Fails on hard tasks |

**Interpretation:** CEM with $L_2$ distance is the best overall planner. Gradient-based methods (Adam, GD) excel on Metaworld where the cost landscape is smooth and goals are greedily reachable, but catastrophically fail on 2D navigation (multi-modal cost landscapes cause them to get stuck in local minima) and on DROID (complex real-world manipulation). NG matches CEM on real-world manipulation data (DROID, Robocasa) with zero hyperparameter tuning, making it a practical alternative when transitioning to new tasks. $L_2$ consistently outperforms $L_1$ across all settings and models.

### Multistep Rollout Training (Figure 3b)

Adding a 2-step rollout loss to teacher forcing improves performance across all environments. However, going beyond 2 steps (k > 2) hurts performance on simulated environments -- with the default planning context $W^p = 2$, higher-step rollout terms make the predictor less specialized for the actual test-time prediction regime. On DROID, the optimal is 6-step rollout, likely because complex real-world dynamics benefit from longer training horizons.

### Impact of Proprioception (Figure 4a)

| Setting | Metaworld (Avg) | 2D Navigation (Avg) | DROID |
|---|---|---|---|
| With proprioception | **Higher** (~5-10% gain) | **Higher** (~5-10% gain) | **Higher** |
| Without proprioception | Lower | Lower | Lower |

Proprioception consistently helps. On Metaworld, failures are commonly due to the arm oscillating around the goal; proprioception provides precise distance-to-goal information. On 2D navigation, it allows more precise planning. Proprioception is excluded for Robocasa (zero-shot transfer with misaligned proprioceptive spaces).

### Encoder Type (Figure 4b)

| Encoder | Metaworld | 2D Nav | DROID | Robocasa |
|---|---|---|---|---|
| DINOv2 ViT-L | Strong | **Best** | Good | Good |
| DINOv3 ViT-L | Slightly lower on Maze/Wall | Slower to converge | **Best** | **Best** |
| V-JEPA ViT-L | Lower | Lower | Lower | Lower |
| V-JEPA-2 ViT-L | Lower | Lower | Lower | Lower |

**Interpretation:** DINO (image) encoders clearly outperform V-JEPA (video) encoders. The authors attribute this to DINO's superior fine-grained object segmentation capabilities, which are crucial for tasks requiring precise spatial perception of agent and object locations. DINOv3 outperforms DINOv2 specifically on photorealistic environments (DROID, Robocasa), likely because its pretraining data distribution better matches real-world images. On simpler 2D tasks (Maze, Wall), DINOv3 converges slower.

### Predictor Architecture (Figure 5a)

| Architecture | Average Performance | Notes |
|---|---|---|
| AdaLN + feature cond. | **Best average** | Action info at every transformer layer; avoids vanishing of action signal |
| AdaLN-zero + feature cond. | Higher peak but less consistent | Outperforms AdaLN on Metaworld, worse on reliable-signal environments |
| RoPE + feature cond. | Competitive | No clear gain from RoPE over sincos |
| RoPE + sequence cond. | Slightly lower | V-JEPA-2-AC's approach |
| Sincos + feature cond. | Baseline (DINO-WM) | Solid but not optimal |

**Interpretation:** AdaLN is the best conditioning technique on average because it injects action information at every transformer layer, preventing the action signal from vanishing in deeper layers. It is also more compute-efficient. Results are task-dependent: sincos + feature conditioning wins on Metaworld. AdaLN-zero shows higher average but with noisier results; the non-zero AdaLN variant gives more consistent performance on the environments with the most reliable signal (DROID, Push-T, Maze).

### Maximum Context Size (Figure 5b)

Training with $W = 1$ (single-frame context, no velocity information) performs dramatically worse than $W = 2$. The predictor needs at least 2 frames to infer velocity. Beyond $W = 2$, performance is task-dependent: simulated environments plateau at $W = 3$, while DROID benefits from longer context ($W = 5$) due to more complex real-world dynamics. Critically, the planning context $W^p$ must satisfy $W^p \leq W$ -- the model degrades rapidly when asked to predict from contexts longer than those seen during training.

### Model Scaling (Figures 6a, 6b)

| Setting | Effect of Scaling Up |
|---|---|
| Simulated environments | **No improvement** -- performance saturates at ViT-S / depth-6 |
| DROID (real-world) | **Clear improvement** -- both larger encoders and deeper predictors help |

On simulated tasks, scaling encoder size (ViT-S to ViT-L) or predictor depth (3 to 12) yields no gains and can even hurt. Larger embedding spaces make it harder for the planner to distinguish nearby states. On DROID, there is a clear positive correlation with both encoder size and predictor depth, indicating real-world dynamics genuinely benefit from higher-capacity models. Optimal predictor depth for simulated environments is ~6; for the simplest 2D navigation tasks, depth 3 suffices.

### Final Optimized Model vs. Baselines (Table 1)

| Model | Maze | Wall | Push-T | MW-R | MW-RW | Rc-R | Rc-Pl | DROID |
|---|---|---|---|---|---|---|---|---|
| DINO-WM | 81.6 (3.4) | 64.1 (4.6) | 66.0 (4.7) | 44.8 (8.9) | 35.1 (9.4) | 19.1 (13.4) | 21.7 (7.2) | 39.4 (2.1) |
| V-JEPA-2-AC | -- | -- | -- | -- | -- | 16.2 (8.3) | **33.1 (7.2)** | 42.9 (2.5) |
| **Ours** | **83.9 (2.3)** | **78.8 (3.9)** | **70.2 (2.8)** | **58.2 (9.3)** | **41.6 (10.0)** | **25.4 (16.6)** | 30.7 (8.0) | **48.2 (1.8)** |

Standard deviations in parentheses. Bold indicates best. MW-R = Metaworld Reach; MW-RW = Metaworld Reach-Wall; Rc-R = Robocasa Reach; Rc-Pl = Robocasa Place.

**Interpretation:** The proposed optimized model outperforms both DINO-WM and V-JEPA-2-AC on 7 of 8 tasks. The improvements are particularly large on Wall (+14.7 points), Metaworld Reach (+13.4 points), and DROID (+8.8 points over DINO-WM, +5.3 over V-JEPA-2-AC). The only task where V-JEPA-2-AC wins is Robocasa Place. The optimized recipe uses:
- **Simulated environments**: ViT-S encoder, depth-6 predictor, AdaLN conditioning, RoPE embeddings, proprioception, 2-step rollout loss, $W = 3$, DINOv2, CEM $L_2$ planner
- **DROID/Robocasa**: DINOv3 ViT-L encoder, depth-12 ViT-L predictor, no proprioception, CEM $L_2$ planner, DINOv2 for all environments except photorealistic ones (DINOv3)

---

## Comparison to Prior Work

**vs [[lecun-2022-openreview]] ([A Path Towards Autonomous Machine Intelligence](../papers/lecun-2022-openreview.md)):** This paper operationalizes the JEPA vision laid out in LeCun's position paper. LeCun proposed that world models should predict in representation space using joint-embedding architectures; the present work systematically studies how to make that proposal work for physical planning in robotic environments, identifying which specific design choices (encoder, predictor architecture, training regime, planner) matter most.

**vs [[balestriero-2025-iclr]] ([LeJEPA/SIGReg](../papers/balestriero-2025-iclr.md)):** LeJEPA provides the theoretical foundation (SIGReg regularizer) for JEPA training. This paper operates in a different regime: it keeps the visual encoder frozen (no collapse risk) and focuses on the predictor/dynamics model learned on top. The design choices studied here are complementary to LeJEPA's contributions on representation learning.

**vs [[maes-2026-arxiv]] ([LeWorldModel](../papers/maes-2026-arxiv.md)):** LeWorldModel directly builds on the findings of this paper. Where Terver et al. study the frozen-encoder regime and optimize the predictor, LeWM trains the encoder end-to-end using SIGReg, eliminating the frozen encoder entirely. LeWM adopts several insights validated here: AdaLN-zero conditioning, CEM planning, and MSE prediction loss. The present paper's finding that DINO encoders outperform V-JEPA encoders motivates LeWM's goal of training a comparably good encoder from scratch (avoiding dependence on DINOv2's 124M-image pretraining).

**vs [[hafner-2019-icml]] ([PlaNet](../papers/hafner-2019-icml.md)) and [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)):** PlaNet and Dreamer use reconstruction-based latent dynamics (RSSM) with reward-driven RL. JEPA-WMs differ fundamentally: they use pretrained frozen encoders (no reconstruction), MSE in embedding space (no pixel decoder), and reward-free goal-conditioned planning. This paper shows the JEPA-WM approach can match or exceed Dreamer/TD-MPC baselines when reward annotation is removed.

**vs [[hansen-2022-icml]] ([TD-MPC](../papers/hansen-2022-icml.md)) and [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)):** TD-MPC uses latent dynamics with MPPI planning and reward-based Q-learning. DINO-WM was originally shown to outperform DreamerV3 and TD-MPC2 when reward annotation is removed. The present paper's optimized JEPA-WM further improves upon DINO-WM, widening this gap. Interestingly, the paper introduces Nevergrad as an alternative to CEM/MPPI that matches CEM on real-world data with less hyperparameter tuning.

**vs [[bar-2024-cvpr]] ([Navigation World Models](../papers/bar-2024-cvpr.md)):** Bar et al. focus on navigation-specific world models using web-scale video pretraining. The present paper's JEPA-WMs are general-purpose (navigation and manipulation) and operate in a different regime: offline robotic trajectories with action labels rather than internet-scale egocentric video. Both validate the principle of planning in learned representation spaces.

---

## Strengths

- **Comprehensive and controlled ablation**: Seven design axes studied independently across eight diverse tasks (2D navigation, simulated manipulation, real-world manipulation, zero-shot transfer), yielding actionable design guidelines rather than a single model claim.
- **Practical engineering contributions**: Introduction of Nevergrad (NG) as a hyperparameter-free planning optimizer is a genuine practical advance -- it matches CEM on real-world data without any planner tuning, lowering the barrier for applying JEPA-WMs to new tasks and datasets.
- **Clear finding on encoder choice**: The demonstration that image-based DINO encoders outperform video-based V-JEPA encoders for manipulation/navigation is a valuable empirical finding that informs encoder selection for the broader JEPA world model community.
- **Reproducibility**: All code, data, and checkpoints are publicly released at [github.com/facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms), with a unified training and evaluation framework that generalizes over DINO-WM and V-JEPA-2-AC.
- **Strong final model**: The optimized recipe outperforms both established baselines (DINO-WM and V-JEPA-2-AC) on most tasks, validating that the ablation insights compose effectively.

---

## Weaknesses & Limitations

- **Frozen encoder assumption**: The entire study operates with frozen pretrained encoders, which limits findings to the "predictor learning" regime. End-to-end training (as in [[maes-2026-arxiv]]) may alter which design choices matter most -- e.g., AdaLN's advantage may change if the encoder co-adapts with the predictor.
- **No reward-based baselines in final comparison**: Table 1 compares only against DINO-WM and V-JEPA-2-AC; Dreamer and TD-MPC (with reward access) are not included in the main results, making it unclear how the optimized JEPA-WM compares to the best reward-based methods.
- **Task diversity**: Despite eight tasks, all are short-horizon robotic manipulation or 2D navigation. Long-horizon tasks, multi-step reasoning, and tasks requiring semantic understanding are not covered.
- **Scaling findings are partially negative**: The result that scaling does not help on simulated environments is useful but raises questions about whether the simulated benchmarks are too simple to meaningfully differentiate models, limiting the generalizability of the scaling conclusions.
- **Variance remains high**: Several results (e.g., Robocasa Reach at 25.4 with std 16.6) show substantial variance across seeds and evaluation episodes, making some comparisons statistically uncertain.
- **Data augmentation not studied**: The paper mentions data augmentation as a design axis in the abstract/contributions but does not present systematic results on its effect.

---

## Key Takeaways

- **CEM with $L_2$ distance is the best overall planner** for JEPA-WMs, but Nevergrad is a strong zero-tuning alternative for real-world data.
- **2-step rollout training helps universally**; more steps only help on complex real-world dynamics (DROID).
- **Proprioception consistently improves planning** by providing precise distance-to-goal information.
- **DINO (image) encoders beat V-JEPA (video) encoders** for manipulation and navigation, due to superior fine-grained spatial features. DINOv3 surpasses DINOv2 on photorealistic environments.
- **AdaLN conditioning is the best predictor architecture on average**, injecting action information at every transformer layer to prevent signal vanishing.
- **Minimum context of $W = 2$ is critical** (velocity inference); beyond that, optimal context is task-dependent ($W = 3$ for simulation, $W = 5$ for real-world).
- **Scaling only helps on real-world data** -- simulated environments saturate at small model sizes, while DROID benefits from both larger encoders and deeper predictors.
- **The combined recipe outperforms DINO-WM and V-JEPA-2-AC** on 7 of 8 benchmarks, demonstrating that systematic design choice optimization within the JEPA-WM family yields substantial gains.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{terver2025drives,
  title={What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?},
  author={Terver, Basile and Yang, Tsung-Yen and Ponce, Jean and Bardes, Adrien and LeCun, Yann},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://arxiv.org/abs/2512.24497}
}
```
{% endraw %}

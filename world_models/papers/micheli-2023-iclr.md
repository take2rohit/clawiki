---
title: Transformers are Sample-Efficient World Models
type: paper
paper_id: P010
authors:
- Micheli, Vincent
- Alonso, Eloi
- Fleuret, François
year: 2023
venue: ICLR 2023 (Notable — Top 5%)
arxiv_id: '2209.00588'
url: https://arxiv.org/abs/2209.00588
pdf: ../../raw/micheli-2023-iclr.pdf
tags:
- world-model
- transformer
- actor-critic
- autoregressive
- model-based-rl
created: 2026-04-10
updated: 2026-04-10
cites:
- ha-2018-neurips
- hafner-2019-icml
- hafner-2021-iclr
cited_by:
- alonso-2024-neurips
- maes-2026-arxiv
- robine-2023-iclr
- wang-2025-iclr

---

# Transformers are Sample-Efficient World Models (IRIS)

> **One sentence** — IRIS combines a discrete autoencoder and an autoregressive GPT-like Transformer as a world model, achieving a mean human normalized score of 1.046 on Atari 100k (26 games, ~2 hours of gameplay), outperforming humans on 10 out of 26 games and setting a new state of the art for methods without lookahead search.

**Authors:** Vincent Micheli, Eloi Alonso, François Fleuret | **Venue:** ICLR 2023 (Notable — Top 5%) | **arXiv:** [2209.00588](https://arxiv.org/abs/2209.00588)

---

## Problem & Motivation

Deep RL agents are notoriously sample inefficient, requiring months of gameplay for methods like DreamerV2 or billions of frames for MuZero — far beyond what is feasible in real-world deployments. World models address this by training policies in imagination, decoupling the policy from direct environment interaction, but this approach relies critically on having an accurate world model that remains faithful over extended rollouts. Prior imagination-based methods used RSSMs (recurrent state-space models) as the world model backbone, which struggle to capture complex multi-modal visual dynamics at the token level. The success of Transformers on sequence modeling tasks — particularly their ability to model long-range dependencies over discrete token vocabularies — suggests they are well-suited for this role, but their quadratic attention cost makes applying them directly to raw pixels impractical. The question IRIS addresses is whether a discrete autoencoder (converting frames to a small number of tokens) combined with a causal Transformer can serve as a sufficiently accurate and efficient world model for sample-efficient RL in complex visual domains.

---

## Core Idea

IRIS (Imagination with auto-Regression over an Inner Speech) recasts environment dynamics learning as a sequence modeling problem over a symbolic language invented by the agent itself. A discrete autoencoder first compresses each 64×64 frame into K=16 discrete tokens from a vocabulary of N=512 — a compact "inner speech" representation. A GPT-like Transformer then models sequences of these token sequences interleaved with actions, learning to autoregressively predict future frames, rewards, and episode terminations. Policies trained purely in this imagined token space achieve strong performance with only 100K real environment actions (~2 hours of gameplay), and the discrete token bottleneck makes the Transformer's self-supervised objective directly analogous to language modeling.

---

## How It Works

### Overview

Real frame x_t → discrete autoencoder E → K=16 tokens z_t ∈ {1,...,512}^16 → Transformer G models sequence (z_0, a_0, z_1, a_1, ..., z_t, a_t) → actor-critic π/V trained on imagined trajectories starting from real observations, using reconstructed frames x̂_t = D(z_t) as observations. Policy collected in real environment uses reconstructed frames (not raw observations) to keep input distribution consistent.

### Discrete Autoencoder (E, D) — Tokenizing Frames

The encoder E: ℝ^{h×w×3} → {1,...,N}^K converts an input frame into K discrete tokens via nearest-neighbor lookup in a learned embedding table:

- Architecture: CNN with 4 layers, 2 residual blocks per layer, 64 channels, self-attention at resolution 8/16 (based on VQGAN with discriminator removed)
- Vocabulary: N=512 embeddings of dimension d=512
- Token assignment: z^k = argmin_i ||y^k - e_i||_2 where y^k is the k-th CNN output vector
- Decoder D: {1,...,N}^K → ℝ^{h×w×3} — same architecture reversed (transposed CNN + residual + self-attention)
- Straight-through estimator (Bengio et al. 2013) allows backpropagation through the discrete argmin

**Training loss** (Appendix A, Table 2):

L(E, D, E) = ||x - D(z)||_1 + ||sg(E(x)) - E(z)||_2^2 + ||sg(E(z)) - E(x)||_2^2 + L_perceptual(x, D(z))

The first term is L1 reconstruction; the next two terms are the VQ commitment loss (stop-gradient sg prevents gradient flow into the codebook on one side, encoder on the other); the last term is a perceptual loss (feature matching). After 120 games of training, the autoencoder achieves pixel-perfect predictions in games like Pong.

**Default hyperparameters** (Table 2-3):
- Frame input: 64×64×3
- Tokens per frame: K=16
- Vocabulary size: N=512
- Token embedding dimension: d=512
- CNN: 4 layers, 2 residual blocks per layer, 64 channels

### Transformer G — Autoregressive Dynamics Model

G is a GPT-like causal Transformer (based on minGPT, Karpathy 2020) that operates over sequences of L=20 interleaved frame-token and action-token sequences. The full input sequence has L(K+1) = 20×17 = 340 tokens. G predicts three quantities for each timestep (Equations 1-3):

- **Transition**: z̃^k_{t+1} ~ p_G(z̃^k_{t+1} | z_{≤t}, a_{≤t}, z^{<k}_{t+1}) — next frame tokens, autoregressive within each frame (each of the K tokens conditions on previously predicted tokens at the same timestep)
- **Reward**: r̂_t ~ p_G(r̂_t | z_{≤t}, a_{≤t}) — scalar reward prediction
- **Termination**: d̂_t ~ p_G(d̂_t | z_{≤t}, a_{≤t}) — episode end binary prediction

**Training losses**: Cross-entropy for transition and termination predictions; MSE for reward prediction.

**Architecture hyperparameters** (Table 4):
- Timesteps L: 20
- Embedding dimension D: 256
- Transformer layers M: 10
- Attention heads: 4
- Weight decay: 0.01
- Embedding / attention / residual dropout: 0.1

### Actor-Critic in Imagination (Section 2.3, Appendix B)

The policy π and value function V learn entirely inside the imagination MDP. At each imagined step, π observes the decoded frame x̂_t = D(z_t) and samples action a_t ~ π(a_t | x̂_{≤t}). The Transformer then predicts z_{t+1}, r̂_t, d̂_t autoregressively, producing x̂_{t+1} = D(z_{t+1}).

**Actor-critic architecture**: Shared CNN + LSTM backbone (weights shared except last layer). CNN: 3×3 conv, stride 1, padding 1, ReLU, 2×2 max-pool stride 2, same layer repeated 4×. LSTM hidden size: 512. Burn-in: 20 previous frames initialize LSTM hidden state before each imagination rollout.

**λ-return** (Equation 4):

Λ_t = r̂_t + γ(1 − d̂_t)[(1−λ)V(x̂_{t+1}) + λΛ_{t+1}]  if t < H
Λ_t = V(x̂_H)                                               if t = H

**Critic loss** (Equation 5): L_V = E_π[Σ_{t=0}^{H-1} (V(x̂_t) − sg(Λ_t))^2] — squared error against stop-gradient λ-returns.

**Actor loss** (Equation 6): Pure REINFORCE with value baseline and entropy regularization:

L_π = −E_π[Σ_{t=0}^{H-1} log π(a_t | x̂_{≤t}) sg(Λ_t − V(x̂_t)) + η H(π(a_t | x̂_{≤t}))]

**RL hyperparameters** (Table 6):
- Imagination horizon H: 20
- Discount γ: 0.995
- λ: 0.95
- Entropy coefficient η: 0.001

### Training Loop

Three interleaved procedures (Algorithm 1):

1. **collect_experience**: 200 env steps per epoch (ε=0.01 greedy), frames decoded by autoencoder for consistent input distribution; dataset D grows with (x_t, a_t, r_t, d_t) tuples
2. **update_world_model**: Sample sequences of length L from D, update autoencoder (E, D) and Transformer (G); autoencoder starts at epoch 5, Transformer at epoch 25
3. **update_behavior**: Sample x_0 from D, encode with E, roll out H=20 steps in imagination, update π and V; actor-critic starts at epoch 50

**Training hyperparameters** (Table 5):
- Total epochs: 600 (collection epochs: 500)
- Training steps per epoch: 200 for each component
- Batch sizes: autoencoder 256, Transformer 64, actor-critic 64
- Optimizer: Adam (β_1=0.9, β_2=0.999), lr=1×10^{-4}, max grad norm=10.0

### Inference

At test time, each frame x_t is encoded by E to tokens z_t; the decoded x̂_t = D(z_t) is passed to π to sample action a_t. The Transformer G is not called during real environment interaction — only E and D are used for the encode-decode cycle. The world model (G) runs only during imagination phases for behavior learning.

---

## Results

### Atari 100k Benchmark (26 games, 100K actions ≈ 2 hours)

Table 1 — Full results across 26 games:

| Game | Random | Human | SimPLe | CURL | DrQ | SPR | IRIS |
|------|--------|-------|--------|------|-----|-----|------|
| Alien | 227.8 | 7127.7 | 616.9 | 711.0 | 865.2 | 841.9 | 420.0 |
| Amidar | 5.8 | 1719.5 | 88.0 | 113.7 | 137.8 | 179.7 | 143.0 |
| Assault | 222.4 | 742.0 | 527.2 | 500.9 | 579.6 | 565.6 | 1524.4 |
| Asterix | 210.0 | 8503.3 | 1128.3 | 567.2 | 763.6 | 962.5 | 853.6 |
| BankHeist | 14.2 | 753.1 | 34.2 | 65.3 | 232.9 | 345.4 | 53.1 |
| BattleZone | 2360.0 | 37187.5 | 4031.2 | 8997.8 | 10165.3 | 14834.1 | 13074.0 |
| Boxing | 0.1 | 12.1 | 7.8 | 1.2 | 6.0 | 35.7 | 70.1 |
| Breakout | 1.7 | 30.5 | 16.4 | 4.9 | 16.1 | 17.1 | 83.7 |
| ChopperCommand | 811.0 | 7387.8 | 1246.9 | 2500.2 | 780.3 | 974.8 | 1565.0 |
| CrazyClimber | 10780.5 | 35829.4 | 62583.6 | 9154.4 | 21539.0 | 36700.5 | 59324.2 |
| DemonAttack | 152.1 | 1971.0 | 3527.0 | 817.6 | 1113.4 | 1113.4 | 2034.4 |
| Freeway | 0.0 | 29.6 | 20.3 | 26.7 | 9.8 | 24.4 | 31.1 |
| Frostbite | 65.2 | 4334.7 | 254.7 | 1181.3 | 331.1 | 1821.5 | 259.1 |
| Gopher | 257.6 | 2412.5 | 771.0 | 669.3 | 636.3 | 715.2 | 2236.1 |
| Hero | 1027.0 | 30826.4 | 2656.6 | 6279.3 | 3736.3 | 6582.0 | 7037.4 |
| Jamesbond | 29.0 | 302.8 | 125.3 | 471.0 | 236.0 | 365.4 | 462.7 |
| Kangaroo | 52.0 | 3035.0 | 323.1 | 872.5 | 940.6 | 3276.4 | 838.2 |
| Krull | 1598.0 | 2665.5 | 4539.9 | 4229.6 | 4018.1 | 3688.9 | 6616.4 |
| KungFuMaster | 258.5 | 22736.3 | 17257.2 | 14307.8 | 9111.0 | 13192.7 | 21759.8 |
| MsPacman | 307.3 | 6951.6 | 1480.0 | 1465.5 | 960.5 | 1204.1 | 999.1 |
| Pong | -20.7 | 14.6 | 12.8 | -16.5 | -19.3 | 20.0 | 14.6 |
| PrivateEye | 24.9 | 69571.3 | 35.0 | 218.4 | -13.6 | 43.9 | 100.0 |
| Qbert | 163.9 | 13455.0 | 1288.8 | 1042.4 | 854.4 | 1152.9 | 745.7 |
| RoadRunner | 11.5 | 7845.0 | 5640.6 | 5661.0 | 8566.4 | 14220.5 | 9614.6 |
| Seaquest | 68.4 | 42054.7 | 683.3 | 384.5 | 301.2 | 583.1 | 661.3 |
| UpNDown | 533.4 | 11693.2 | 3350.3 | 2955.2 | 3180.8 | 17251.0 | 3546.2 |

**Aggregate metrics** (Table 1, bottom):

| Metric | Random | Human | SimPLe | CURL | DrQ | SPR | IRIS |
|--------|--------|-------|--------|------|-----|-----|------|
| #Superhuman | 0 | N/A | 1 | 2 | 3 | 6 | **10** |
| Mean HNS | 0.000 | 1.000 | 0.261 | 0.340 | 0.465 | 0.616 | **1.046** |
| Median HNS | 0.000 | 1.000 | 0.134 | 0.092 | 0.313 | 0.337 | **0.289** |
| IQM | 0.000 | 1.000 | 0.130 | 0.113 | 0.280 | 0.465 | **0.501** |
| Optimality Gap | 1.000 | 0.000 | 0.729 | 0.768 | 0.631 | 0.577 | **0.512** |

IRIS achieves Mean HNS=1.046, IQM=0.501, and 10 superhuman games — new SOTA for methods without lookahead search, outperforming SPR (+70% mean, +49% IQM, +11% optimality gap). The probability of IRIS outperforming any individual baseline on a randomly selected game exceeds 0.5 for all baselines (Figure 6b).

Note: MuZero (5 superhuman, Mean=0.562) and EfficientZero (14 superhuman, Mean=1.943) use MCTS lookahead search; IRIS outperforms MuZero without search.

### Double Exploration Problem (Section 3.2, Figure 7)

IRIS identifies a fundamental failure mode: the world model must first encounter a new game mechanic (level transition, new enemy type) in the real environment before the policy can learn to exploit it in imagination. This creates a compounding exploration challenge:

- **Frostbite** (IRIS=259, Human=4335): To reach level 2, the agent must execute a long, low-probability sequence of actions (build igloo, then travel back to bottom). The world model never internalizes level 2 mechanics; the policy trained in imagination cannot learn them.
- **Krull** (IRIS=6616, new SOTA): Multi-level but transitions are frequent, giving the world model diverse level coverage. IRIS achieves SOTA here precisely because the double exploration problem does not apply.

Games without significant distributional shift (Pong, Breakout, Boxing) are strong suits; games requiring discovery of rare events are weak spots.

### Token Count Ablation (Appendix E, Table 7)

| Game | IRIS 16 tokens | IRIS 64 tokens | Improvement |
|------|---------------|----------------|-------------|
| Alien | 420.0 | 570.0 | +36% |
| Asterix | 853.6 | 1890.4 | +121% |
| BankHeist | 53.1 | 282.5 | +432% |

Visually complex games with many possible configurations (BankHeist, Asterix) benefit substantially from 64 tokens per frame. Alien's modest gain (+36%) suggests its difficulty is an exploration problem rather than a reconstruction fidelity problem.

### Scaling to 10M Steps (Appendix F, Table 8)

| Metric | IRIS (100k) | IRIS (10M) |
|--------|-------------|------------|
| #Superhuman | 10 | **15** |
| Mean HNS | 1.046 | **7.488** |
| Median HNS | 0.289 | **1.207** |
| IQM | 0.501 | **2.239** |
| Optimality Gap | 0.512 | **0.282** |

Increasing environment steps from 100k to 10M (with reduced optimization ratio 1:50) dramatically improves performance, providing evidence that IRIS scales beyond the sample-efficient regime. Notable 10M scores: DemonAttack 96218.6, CrazyClimber 111068.8, UpNDown 114690.1.

---

## Comparison to Prior Work

| Method | World model type | Policy training | Lookahead search | Atari 100k Mean HNS |
|--------|-----------------|-----------------|-----------------|---------------------|
| **IRIS** | Discrete AE + Transformer | Actor-critic in imagination | No | **1.046** |
| SPR | Contrastive (no generation) | Augmented model-free | No | 0.616 |
| DrQ | Augmentation only | Model-free | No | 0.465 |
| CURL | Contrastive representation | Model-free | No | 0.340 |
| SimPLe | RNN (pixel space) | PPO in pixel imagination | No | 0.261 |
| DreamerV2 | RSSM + Gaussian/categorical | Actor-critic in imagination | No | ~0.2 (200M steps) |
| MuZero | Latent (task-specific) | MCTS | Yes | 0.562 |
| EfficientZero | MuZero + consistency + reset | MCTS | Yes | 1.943 |

**SimPLe (Kaiser et al. 2020):** Learns a video prediction model in pixel space and trains PPO on imagined pixel frames, similar in spirit to [[ha-2018-neurips]] ([Ha & Schmidhuber 2018](../papers/ha-2018-neurips.md)) but in pixel space. IRIS outperforms SimPLe by 4× on mean HNS while training in a compact token space — pixel-space imagination is inefficient and computationally expensive, whereas IRIS's 16-token bottleneck makes Transformer sequence modeling tractable.

**[[hafner-2021-iclr]] ([DreamerV2](../papers/hafner-2021-iclr.md)):** Uses RSSM (GRU + categorical latents) as the world model, trained with image reconstruction and KL objectives. DreamerV2 was developed and evaluated with 200M frames — far beyond the 100k sample-efficient regime — and achieves ~0.2 mean HNS at 100k steps. IRIS surpasses DreamerV2's architecture by replacing the recurrent backbone with a Transformer that captures longer-range token dependencies.

**SPR (Schwarzer et al. 2021):** Learns consistent self-predictive representations with data augmentation; no generative world model and no imagination. IRIS outperforms SPR on mean HNS (+70%), IQM (+49%), and superhuman games (10 vs 6), demonstrating that accurate generative imagination provides value beyond representation learning.

**MuZero / EfficientZero:** Use MCTS lookahead search at decision time, providing substantial per-step planning at the cost of computational complexity. IRIS outperforms MuZero on mean HNS (1.046 vs 0.562) without any search, and trails EfficientZero (1.943) which additionally uses prioritized replay, episode resets, and a self-supervised consistency loss.

---

## Strengths
- Discrete tokenization makes Transformer world modeling tractable: 16 tokens per frame reduces the quadratic attention cost to a manageable sequence length of 340 tokens over 20 timesteps
- Autoregressive Transformer captures complex multi-modal visual dynamics (enemy spawning patterns, ball trajectories, level mechanics) that RSSM-style recurrent models miss
- New SOTA on Atari 100k for methods without lookahead search: Mean HNS=1.046, 10 superhuman games, IQM=0.501, outperforming MuZero despite using no search
- Token count is a direct compute-quality knob: increasing from 16 to 64 tokens/frame yields up to 432% improvement on visually complex games (BankHeist)
- Clean modular architecture with explicit components (autoencoder + Transformer + actor-critic) that can be individually analyzed and improved
- Strong scaling behavior: 10M steps gives IQM=2.239 (vs 0.501 at 100k), 15 superhuman games
- Open-source code at github.com/eloialonso/iris; ICLR 2023 Notable Top 5%

## Weaknesses & Limitations
- Double exploration problem: the world model can only simulate game mechanics it has encountered in the real environment; rare level transitions (Frostbite) create a compounding failure where the world model is ignorant of content the policy needs to master
- Training computational cost scales with Transformer sequence length L(K+1); 64 tokens/frame quadruples the sequence length and makes training significantly more expensive
- Policy operates on decoded reconstructed frames x̂_t rather than true observations x_t, introducing reconstruction artifacts as a consistent bias throughout training and evaluation
- No explicit mechanism to handle distributional shift between world model predictions and real environment dynamics as training progresses — model exploitation remains a risk for longer imagination horizons
- Atari 100k optimization ratio reduced from 1:1 to 1:50 at 10M steps to maintain training time, meaning the 10M results use less computation per env step than 100k results
- The approach requires careful warm-up scheduling (autoencoder first at epoch 5, Transformer at 25, actor-critic at 50) to prevent the Transformer from overfitting to poor early representations

## Key Takeaways
- IRIS establishes that an autoregressive Transformer over discrete image tokens is a superior world model backbone to recurrent state-space models for Atari 100k, achieving Mean HNS=1.046 (vs SPR's 0.616) with only ~2 hours of real gameplay
- The discrete autoencoder bottleneck (K=16 tokens, N=512 vocabulary per frame) is load-bearing: it makes Transformer sequence modeling tractable while providing sufficient fidelity for behaviors to transfer from imagination to reality; increasing to K=64 tokens yields +432% on BankHeist at the cost of 4× longer sequences
- The "double exploration problem" is a fundamental limitation of any pure imagination-based agent: the world model cannot simulate game content it has not observed, meaning rare transitions (Frostbite's igloo-building sequence) create a ceiling on imagination-based learning even with infinite imagined trajectories
- IRIS outperforms MuZero (Mean HNS 1.046 vs 0.562) without any lookahead search, demonstrating that accurate generative world models can close the gap with search-based methods in the sample-efficient regime
- Scaling from 100k to 10M environment steps increases superhuman games from 10 to 15 and IQM from 0.501 to 2.239, confirming that IRIS's architecture scales predictably beyond the sample-efficient setting

---

## BibTeX
```bibtex
@inproceedings{micheli2023iris,
  title     = {Transformers are Sample-Efficient World Models},
  author    = {Micheli, Vincent and Alonso, Eloi and Fleuret, Fran{\c{c}}ois},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2209.00588},
  eprint    = {2209.00588},
  archivePrefix = {arXiv},
  note      = {Notable -- Top 5\%}
}
```

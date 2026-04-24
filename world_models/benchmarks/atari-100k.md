---
title: "Atari 100k"
type: benchmark
tags: [benchmark, atari, sample-efficiency, model-based-rl, discrete-control]
created: 2026-04-10
updated: 2026-04-10
---

# Atari 100k

> A sample-efficiency benchmark measuring the performance of RL agents on 26 Atari games after exactly 100,000 environment steps (~2 hours of real gameplay), used to evaluate world model agents trained in imagination.

## Description

The Atari 100k benchmark (also called "Atari data-efficient") uses the Arcade Learning Environment (ALE) with the standard 26-game subset from Kaiser et al. (SimPLe, 2019). Each agent is limited to 100,000 real environment interactions across training — equivalent to approximately 2 hours of gameplay at 60 fps. The benchmark tests sample efficiency and the quality of learned world models, since model-free agents typically require 50–200× more data to achieve comparable performance. Sticky actions (probability 0.25 of repeating the previous action) are used in some evaluations (e.g., DreamerV3) to prevent deterministic exploitation. Scores are reported on a held-out evaluation of the trained policy in the real environment, not inside the world model.

The 26 games: Alien, Amidar, Assault, Asterix, BankHeist, BattleZone, Boxing, Breakout, ChopperCommand, CrazyClimber, DemonAttack, Freeway, Frostbite, Gopher, Hero, Jamesbond, Kangaroo, Krull, KungFuMaster, MsPacman, Pong, PrivateEye, Qbert, RoadRunner, Seaquest, UpNDown.

## Metrics

- **Primary metric: Mean Human Normalized Score (HNS)** — computed per game as (agent\_score − random\_score) / (human\_score − random\_score), then averaged across all 26 games. Scores above 1.0 mean the agent exceeds human-level performance on average.
- **IQM (Interquartile Mean HNS)** — the mean HNS over the middle 50% of games, more robust to outliers and score clipping. Recommended by Agarwal et al. (2021) as the primary metric for statistical reliability.
- **Optimality Gap** — 1 − IQM over the distribution of normalized scores; measures how far the agent is from optimal performance.
- **Superhuman games** — count of individual games where the agent exceeds the human baseline score.
- **Median HNS** — less common; sensitive to individual game failures.

For all metrics, higher is better (except Optimality Gap, where lower is better).

## Leaderboard

| Rank | Method | Paper | Year | Mean HNS | IQM | Superhuman | Notes |
|------|--------|-------|------|----------|-----|------------|-------|
| 1 | **DIAMOND** | [[alonso-2024-neurips]] ([DIAMOND](../papers/alonso-2024-neurips.md)) | 2024 | **1.46** | 0.641 | 11 | Diffusion (EDM) world model; pixel-space; 3 NFE/frame |
| 2 | STORM | (Zhang et al., 2023) | 2023 | 1.27 | 0.497 | 9 | DreamerV3 + transformer dynamics; not in this corpus |
| 3 | DreamerV3 | [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)) | 2023 | ~1.10 | 0.501 | 10 | Fixed hyperparams; sticky actions; 200M params |
| 4 | IRIS | [[micheli-2023-iclr]] ([IRIS](../papers/micheli-2023-iclr.md)) | 2023 | 1.046 | 0.459 | 10 | VQ-AE + GPT transformer; Notable Top 5% at ICLR |
| 5 | TWM | [[robine-2023-iclr]] ([TWM](../papers/robine-2023-iclr.md)) | 2023 | 0.956 | 0.130 | 8 | Transformer world model with RSSM; ICLR 2023 |
| — | DreamerV2 | [[hafner-2021-iclr]] ([DreamerV2](../papers/hafner-2021-iclr.md)) | 2021 | ~0.20 | — | — | Evaluated at 200M steps; not tuned for 100k regime |
| — | World Models | [[ha-2018-neurips]] ([World Models](../papers/ha-2018-neurips.md)) | 2018 | — | — | — | Evaluated on CarRacing/Doom, not Atari 100k |
| Ref | SimPLe (baseline) | (Kaiser et al., 2019) | 2019 | 0.332 | 0.130 | 1 | First world model agent on Atari 100k |
| Ref | SPR | (Schwarzer et al., 2021) | 2021 | 0.616 | — | 6 | Best model-free approach before world model surge |
| Ref | EfficientZero | (Ye et al., 2021) | 2021 | 1.943 | — | 14 | Uses MCTS + prioritized replay + episode resets |

## Notes

**Comparability caveats:**

- Sticky actions (p=0.25) are used in DreamerV3 results but not in all other entries; this makes comparisons between those rows approximate.
- EfficientZero and MuZero use lookahead search (MCTS) at decision time — a fundamentally different computational budget that makes direct comparison misleading. All other entries in this table do not use lookahead search.
- Mean HNS is sensitive to outliers: a single game where the agent scores dramatically above human (e.g., CrazyClimber) can dominate the mean. IQM is the more reliable primary metric per Agarwal et al. (2021) statistical evaluation guidelines.
- Compute and training time vary substantially: DIAMOND uses ~2.9 GPU-days/game on RTX 4090; IRIS uses similar compute; DreamerV3 uses significantly more.
- World model evaluation is always in the real environment (not inside imagination) for the reported scores.
- The benchmark was originally proposed by Kaiser et al. as SimPLe in 2019; the 26-game subset and 100k budget are now a standard protocol but not all papers use exactly the same 26-game subset or random seeds.

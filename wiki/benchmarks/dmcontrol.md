---
title: "DMControl Suite"
type: benchmark
tags: [benchmark, continuous-control, dmcontrol, model-based-rl, locomotion, manipulation]
created: 2026-04-10
updated: 2026-04-10
---

# DMControl Suite

> DeepMind Control Suite (DMControl) is a standard benchmark for continuous-action model-based RL, providing a set of physics-based locomotion and manipulation tasks from pixels or proprioceptive state with standardized evaluation protocols and dense rewards.

## Description

DMControl (Tassa et al., 2018) provides a set of continuous-control tasks implemented in MuJoCo, organized around different morphologies: Cartpole, Reacher, Cheetah, Walker, Hopper, Finger, Pendulum, Fish, Ball-in-Cup, Humanoid, Dog, and others. Tasks come in "Easy" and "Hard" (or named difficulty) variants. The standard evaluation protocol reports the average episode return (normalized 0–1000) over 10 evaluation episodes after a fixed number of environment steps (commonly 500k steps for proprioceptive state, 1M for visual/pixel-based variants).

Two sub-protocols are common in this corpus:

- **State-based DMControl** (proprioceptive input): The agent receives ground-truth joint positions, velocities, and forces. This removes the visual representation learning problem and isolates dynamics modeling. TD-MPC and TD-MPC2 primarily target this setting.
- **Visual DMControl** (pixel input): The agent receives 84×84 pixel observations, requiring the world model to jointly learn visual representations and dynamics. DreamerV3 is evaluated here.

TD-MPC2 covers 39 DMControl tasks in its single-task evaluation, spanning easy locomotion (Cartpole Balance, Cheetah Run) through challenging high-dimensional tasks (Dog tasks with A ∈ ℝ³⁸, Humanoid Walk/Run with A ∈ ℝ²¹). DreamerV3 covers visual DMControl tasks alongside other benchmarks.

## Metrics

- **Primary metric: Episode return** — sum of rewards per episode, normalized 0–1000 by the task's practical upper bound. Higher is better.
- **Normalized score across tasks** — arithmetic mean of per-task episode returns, normalized so that 0 = random policy and 100 = expert/reference performance. Used by TD-MPC2 for aggregation across 80 multi-task experiments.
- **Sample efficiency** — performance at a fixed step budget (e.g., 500k, 1M steps). World model agents typically outperform model-free methods at equivalent sample budgets.
- **Multi-task normalized score** — used by TD-MPC2 for the 80-task multi-task setting; same normalization but averaged across all 80 tasks.

## Leaderboard

### Representative Single-Task Results (selected tasks, proprioceptive state unless noted)

| Method | Paper | Year | Cheetah Run | Walker Walk | Humanoid Walk | Dog Walk | Notes |
|--------|-------|------|-------------|-------------|---------------|---------|-------|
| **TD-MPC2 (5M)** | [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)) | 2024 | ~950 | ~980 | ~700 | ~850 | Single set of hyperparams; 104 tasks |
| TD-MPC | [[hansen-2022-icml]] ([TD-MPC](../papers/hansen-2022-icml.md)) | 2022 | ~950 | ~970 | ~600 | ~700 | Per-task hyperparams; DMControl focus |
| DreamerV3 (visual) | [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md)) | 2023 | ~820 | ~970 | ~500 | unstable | Visual input; fixed hyperparams |
| SAC | (Haarnoja et al., 2018) | 2018 | ~900 | ~960 | ~550 | unstable | Model-free; strong single-task baseline |
| DrQ-v2 | (Yarats et al., 2021) | 2021 | ~870 | ~975 | — | — | Data-augmentation model-free; visual |

### Multi-Task Scaling (TD-MPC2, 80 tasks)

| Model Size | Paper | Year | Multi-Task Normalized Score |
|-----------|-------|------|-----------------------------|
| **317M params** | [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)) | 2024 | **70.6** |
| 48M params | [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)) | 2024 | 68.0 |
| 19M params | [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)) | 2024 | 57.1 |
| 5M params | [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)) | 2024 | 49.5 |
| 1M params | [[hansen-2024-iclr]] ([TD-MPC2](../papers/hansen-2024-iclr.md)) | 2024 | 16.0 |

## Notes

**Comparability caveats:**

- State-based and visual (pixel) results are not directly comparable; always note which input modality is used.
- DreamerV3 reports failures on Dog tasks under the state-based setting due to numerical instability; TD-MPC2 with SimNorm solves all Dog variants.
- Different papers use different task subsets, step budgets, and numbers of seeds. TD-MPC2 uses 3 seeds per task and 500k steps (single-task); DreamerV3 uses 5 seeds and 1M steps (visual).
- The 80-task multi-task evaluation in TD-MPC2 includes tasks from DMControl, Meta-World, ManiSkill2, and MyoSuite — not DMControl alone. Normalized scores across this heterogeneous set should not be compared directly to DMControl-only numbers.
- "Unstable" in the table indicates tasks where a method fails to converge on multiple seeds; not a reported score.
- Performance scales log-linearly with parameter count in TD-MPC2 from 1M to 317M — scaling has not saturated, suggesting further improvement is possible.

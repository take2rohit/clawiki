---
title: "Conceptual Verbal Reinforcement (CVRF)"
type: method
tags: [verbal-reinforcement, prompt-optimization, financial-trading, episodic-learning]
created: 2026-04-10
updated: 2026-04-10
papers: [yu-2024-fincon]
---

# Conceptual Verbal Reinforcement (CVRF)

An over-episode learning mechanism introduced in [[yu-2024-fincon]] ([FinCon](../papers/yu-2024-fincon.md)) that updates an LLM agent's **investment beliefs** between trading episodes by comparing profitable and losing trajectories — without gradient descent on model weights.

## Core Idea

CVRF treats an LLM agent's system prompt as a policy parameter (analogous to weights in a neural network). At the end of each training episode k, it:

1. Compares the current episode's trading trajectory H_k against the prior episode H_{k-1}
2. Extracts "conceptualized insights" — what decisions led to profit in H_k that were absent or opposite in H_{k-1}
3. Computes an update step size using the overlapping percentage of trading decisions between episodes (high overlap → small update, like a small learning rate)
4. Issues a textual update to the manager agent's prompt: θ ← M_r(θ, τ, meta_prompt)

This is **textual gradient descent**: the "gradient" is the natural-language contrast between good and bad episodes; the "step size" is overlap-based; the "parameter" is the manager's investment belief prompt.

## Formal Analogy to Actor-Critic RL

| RL Concept | CVRF Analog |
|---|---|
| Policy parameters (θ) | Manager agent's investment belief prompt |
| Gradient | Conceptualized contrast between profitable/losing episodes |
| Learning rate | 1 − (overlap % of decisions between H_k and H_{k-1}) |
| Critic | Meta-prompt M_r that evaluates episode performance |
| Policy update | Textual rewrite of manager prompt |

## Empirical Importance

Ablation from FinCon: removing CVRF drops Portfolio 1 (TSLA, MSFT, PFE) from **113.84% → 28.43% cumulative return** — a 85-point degradation. On individual stocks: GOOG drops from 25.08% → -11.94%. This makes CVRF the single most impactful component in FinCon aside from the CVaR within-episode risk control.

## Within-Episode Complement: CVaR

CVRF operates *between* episodes (training only). Its complement, CVaR monitoring, operates *within* episodes (both training and test). Together they form FinCon's dual-level risk-control component:
- **CVaR** (real-time): triggers risk-averse prompt when daily PnL drops below threshold
- **CVRF** (episodic): updates long-term investment beliefs based on episode comparisons

## Limitations

- Training-only: CVRF updates are frozen at test time, so beliefs do not adapt to new market conditions during deployment
- Requires multiple training episodes to accumulate contrast signal; only 4 episodes used in FinCon experiments
- Textual gradient is inherently coarser than numerical gradient — cannot capture fine-grained parameter adjustments

## Related Methods

- [Manager-Analyst Hierarchy](manager-analyst-hierarchy.md) — the architecture CVRF operates on top of
- Related concept: Reflexion (Shinn et al.) — verbal reinforcement applied to general task agents

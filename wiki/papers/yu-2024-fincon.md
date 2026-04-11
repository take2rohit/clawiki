---
title: "FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making"
type: paper
paper_id: P003
authors:
  - "Yu, Yangyang"
  - "Yao, Zhiyuan"
  - "Li, Haohang"
  - "Deng, Zhiyang"
  - "Jiang, Yuechen"
  - "Cao, Yupeng"
  - "Chen, Zhi"
  - "Suchow, Jordan W."
  - "Xie, Qianqian"
year: 2024
venue: NeurIPS 2024
arxiv_id: "2407.06567"
url: "https://arxiv.org/abs/2407.06567"
pdf: "../../raw/yu-2024-fincon.pdf"
tags: [multi-agent, financial-trading, portfolio-management, verbal-reinforcement, risk-control, NeurIPS]
created: 2026-04-10
updated: 2026-04-10
cites: []
cited_by: [xiao-2024-tradingagents, li-2025-orchestration-financial]
---

# FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making

> **One sentence.** FinCon introduces a manager-analyst hierarchical multi-agent framework with a dual-level risk-control mechanism (within-episode CVaR monitoring + over-episode Conceptual Verbal Reinforcement) that achieves the highest cumulative returns and Sharpe ratios across 8 stocks and 2 portfolios, including a 113.8% cumulative return on a 3-stock portfolio — significantly outperforming all LLM and DRL baselines.

**Authors:** Yangyang Yu, Zhiyuan Yao, Haohang Li, Zhiyang Deng, Yuechen Jiang, Yupeng Cao, Zhi Chen, Jordan W. Suchow, et al. | **Venue:** NeurIPS 2024 | **arXiv:** [2407.06567](https://arxiv.org/abs/2407.06567)

*Affiliations: ¹Stevens Institute of Technology; ²Harvard University; ³The Fin AI. Corresponding: Qianqian Xie (Yale/The Fin AI).*

---

## Problem & Motivation

LLM-based financial agents (FinGPT, FinMem, FinAgent) achieved impressive single-stock trading results, but they had three compounding failure modes:
1. **Short-term risk blindness**: agents based their risk preferences on short-term market fluctuations, ignoring long-term exposure — they could run up large losses before reacting.
2. **Single-asset limitation**: prior systems were designed for trading one stock at a time, making portfolio management (multi-stock, correlated positions) infeasible.
3. **Context overload on a single agent**: putting all information into one agent's context window degrades reasoning quality and creates high communication costs when using multi-agent peer discussions (e.g., StockAgent's expensive all-to-all debate).

These failures mattered because real investment firms don't rely on a single analyst and don't ignore quantitative risk measures — they use hierarchical teams and formal risk management.

---

## Core Idea

Real investment firms work because analysts specialize and communicate up a hierarchy to a manager who synthesizes their inputs and makes decisions. FinCon replicates this: specialized analyst agents each process one data source and report concise investment insights to a single manager agent, which is dramatically more efficient than peer-to-peer debate. Crucially, the manager doesn't just react to today's market — it *updates its investment beliefs* between episodes using a verbal reinforcement mechanism that compares profitable and losing episodes to distill what worked and why. This episodic self-critique is analogous to gradient descent but applied to natural-language investment beliefs rather than model weights.

---

## How It Works

### Overview

FinCon has two top-level components operating in sequence:

1. **Manager-Analyst Agent Group** — handles daily trading within an episode
2. **Risk-Control Component** — operates both within an episode (real-time CVaR alerts) and between episodes (belief updates via CVRF)

The system is formalized as a Partially Observable Markov Decision Process (POMDP) where textual prompts serve as the policy parameterization, and optimization proceeds via textual gradient descent rather than backpropagation.

### Manager-Analyst Agent Group

**Analyst Agents (7 types)**, each processing a single modality:
- *Data Agent*: daily stock prices, financial metrics, momentum indicators
- *News Agent*: daily financial news, sentiment extraction
- *10-Q Filing Agent*: quarterly SEC filings, forward-looking disclosures
- *10-K Filing Agent*: annual SEC filings, business overview, risk factors
- *Analyst Reports Agent*: SeekingAlpha expert guidance, price targets
- *ECC Audio Agent*: earnings call conference recordings via Whisper API
- *Stock Selection Agent*: portfolio selection using classic risk diversification (cross-stock correlation analysis + CVaR)

Each analyst uses four cognitive modules: (i) General Configuration/Profiling (role, trading target, sector), (ii) Perception (data format conversion for LLM input), (iii) Memory (working/procedural/episodic memory with decay rates tuned per modality timeliness), (iv) Action (send investment insight to manager).

**Manager Agent** — sole decision-maker. Receives distilled insights from all analysts. Outputs:
- For single-stock trading: buy/sell/hold decision + reasoning text
- For portfolio management: directional decisions per stock → portfolio weights via mean-variance optimization:

  max⟨**w**, μ⟩ − ⟨**w**, Σ**w**⟩ subject to w_n ∈ {[0,1] buy, [−1,0] sell, 0 hold}

Manager also receives within-episode risk alerts and over-episode belief updates from the Risk-Control Component.

### Risk-Control Component

**Within-Episode Risk Control (real-time, both training and testing):**
- Monitors daily CVaR (Conditional Value at Risk = average of worst 1% daily PnL)
- If CVaR drops or daily return goes negative → triggers risk-averse prompt for manager agent
- Provides short-term downside protection regardless of long-term strategy

**Over-Episode Risk Control (training only):**
- After each episode, compares trading trajectory H_k vs H_{k-1}
- Computes Conceptual Verbal Reinforcement (CVRF): extracts "conceptualized insights" from profitable vs. losing episodes
- Uses overlapping percentage of trading decisions as the learning rate analog (more overlap = smaller update)
- Updates manager and relevant analyst prompts via textual gradient descent:
  θ ← M_r(θ, τ, meta_prompt)

This is explicitly analogous to Actor-Critic RL with verbal rather than gradient-based updates.

### Training Setup

- **Backbone LLM**: GPT-4-Turbo (temperature 0.3 for consistency)
- **Training data**: Jan 3, 2022 – Oct 4, 2022 (9 months)
- **Test data**: Oct 5, 2022 – Jun 10, 2023 (8 months)
- **DRL agent training**: Jan 1, 2018 – Oct 4, 2022 (5 years — to ensure convergence)
- **Episodes**: 4 training episodes sufficient; substantially fewer than DRL-based approaches

### Inference

At test time, within-episode CVaR risk control remains active. Over-episode CVRF updates are disabled (frozen beliefs from training). Manager receives analyst insights daily, makes trading decisions, manager self-reflects when triggered by CVaR alert.

---

## Results

### Single Asset Trading (8 stocks, Jan–Oct 2022 train, Oct 2022–Jun 2023 test)

| Stock | B&H CR% | FinCon CR% | FinCon SR | Best LLM Competitor | Best DRL |
|---|---|---|---|---|---|
| TSLA | 6.43 | **82.87** | 1.97 | FinMem 34.62 | PPO 1.41 |
| AMZN | 2.03 | **24.85** | 0.90 | FinAgent 11.96 | DQN 11.17 |
| NIO | −77.21 | **17.46** | 0.34 | FinAgent 0.93 | — (DRL fails) |
| MSFT | −31.82 | **31.63** | 1.54 | FinGPT 21.34 | A2C 21.40 |
| AAPL | 22.32 | **27.35** | 1.60 | FinAgent 20.76 | PPO 14.04 |
| GOOG | 22.42 | **25.08** | 1.05 | FinAgent 20.76 | PPO 8.56 |
| NFLX | 57.34 | **69.24** | 2.37 | FinAgent 61.30 | — |
| COIN | −21.76 | **57.05** | 0.83 | FinAgent −5.97 | — (DRL fails) |

FinCon achieves highest or 2nd-highest CR and SR across all 8 stocks. The gap over competitors is especially large for volatile/bearish stocks (NIO, COIN) where DRL agents fail entirely.

### Portfolio Management

| Portfolio | FinCon CR% | FinCon SR | Markowitz CR% | FinRL-A2C CR% | EW ETF CR% |
|---|---|---|---|---|---|
| P1 (TSLA, MSFT, PFE) | **113.84** | **3.27** | 12.64 | 19.46 | 9.34 |
| P2 (AMZN, GM, LLY) | **32.92** | **1.37** | 10.29 | 11.59 | 15.06 |

FinCon dominates both portfolio management tasks. No prior LLM-based agent had tackled portfolio management; FinCon is the first.

### Ablation Studies

**Within-episode CVaR ablation (does real-time risk control matter?):**

| Setting | GOOG (Bullish) CR% | NIO (Bearish) CR% |
|---|---|---|
| w/ CVaR | 25.08 | 17.46 |
| w/o CVaR | −1.46 | −52.89 |

Removing CVaR causes catastrophic loss in bearish conditions. CVaR is the critical component for downside protection.

**Over-episode belief update ablation (does CVRF matter?):**

| Setting | GOOG CR% | NIO CR% | P1 Portfolio CR% |
|---|---|---|---|
| w/ belief | 25.08 | 17.46 | 113.84 |
| w/o belief | −11.94 | 8.20 | 28.43 |

Removing CVRF degrades performance substantially (GOOG goes negative, portfolio drops from 114% to 28%). Both risk-control mechanisms are independently critical.

---

## Comparison to Prior Work

| | FinCon | FinAgent | FinMem | StockAgent |
|---|---|---|---|---|
| Multi-agent | Manager-Analyst hierarchy | Single agent + tools | Single agent | All-to-all debate |
| Risk control | Dual-level (CVaR + CVRF) | None | None | None |
| Portfolio management | Yes | No | No | No |
| Data modalities | Multi (text, audio, tabular) | Multi (text, charts, tables) | Text + numeric | Text |
| Performance (vs B&H) | Strongly outperforms | Outperforms | Mixed | Not directly compared |

**vs [[xiao-2024-tradingagents]] ([TradingAgents](../papers/xiao-2024-tradingagents.md)):** Both use hierarchical multi-agent structures inspired by trading firms. FinCon uses a strict manager-analyst hierarchy (one decision-maker); TradingAgents uses a broader five-team structure (analysts → researchers → trader → risk → fund manager) with debate between Bull and Bear researchers. FinCon adds explicit quantitative risk control (CVaR, CVRF); TradingAgents focuses more on organizational modeling and communication quality.

**vs [[li-2025-orchestration-financial]] ([Li et al. 2025](../papers/li-2025-orchestration-financial.md)):** Li et al. focus on mapping the full algorithmic trading pipeline to agents (including backtesting, execution, audit agents); FinCon focuses on the decision-making quality within episodes and episodic learning. Different scope: FinCon is about agent intelligence; Li et al. is about orchestration infrastructure.

---

## Strengths

- **First to tackle portfolio management with LLM agents**: previous work was single-stock only. Extending to multi-asset portfolios with mean-variance optimization is a meaningful step.
- **Dual-level risk control**: CVaR for real-time downside protection and CVRF for episodic belief updating are both ablated and shown independently critical.
- **Sample efficiency**: achieves strong results in only 4 training episodes, vs. 5 years of data needed for DRL baselines. Practical for real deployment where historical data is limited.

---

## Weaknesses & Limitations

- **Scaling to large portfolios**: context window limits mean that handling 42-stock pools (the paper's full universe) is already near the edge; portfolios of 50–100 stocks remain out of reach.
- **Hallucination risk in portfolio management**: multi-asset decision-making increases input length and complexity, leading to occasional hallucinated non-existent memory indices.
- **Backtesting only**: no live trading evidence; test period is Oct 2022–Jun 2023 (8 months), which includes a volatile but ultimately recoverable market.
- **Closed-source dependency**: all LLM-based experiments use GPT-4-Turbo — no open-source alternative evaluated.

---

## Key Takeaways

- The manager-analyst hierarchy reduces communication overhead vs. peer-to-peer multi-agent debate, while maintaining the specialization benefits of multi-agent systems.
- CVaR-based within-episode risk control is critical: without it, FinCon loses 52% in bearish conditions (NIO) — a 70-point swing.
- CVRF (verbal reinforcement via episode comparison) provides the learning signal that makes FinCon improve over training, analogous to how gradient descent works for neural networks but applied to prompts.
- FinCon is the first LLM agent to demonstrate portfolio management capability (not just single-stock trading) with strong results: 113.8% cumulative return vs. 12.6% for Markowitz baseline.
- GPT-4-Turbo at 4 training episodes achieves what DRL needs 5 years to learn — a strong argument for LLM-based agents in data-scarce trading environments.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{yu2024fincon,
  title={{FinCon}: A Synthesized {LLM} Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making},
  author={Yu, Yangyang and Yao, Zhiyuan and Li, Haohang and Deng, Zhiyang and Jiang, Yuechen and Cao, Yupeng and Chen, Zhi and Suchow, Jordan W. and Xie, Qianqian},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
{% endraw %}

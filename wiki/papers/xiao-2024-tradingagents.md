---
title: "TradingAgents: Multi-Agents LLM Financial Trading Framework"
type: paper
paper_id: P004
authors:
  - "Xiao, Yijia"
  - "Sun, Edward"
  - "Luo, Di"
  - "Wang, Wei"
year: 2024
venue: ICML 2025
arxiv_id: "2412.20138"
url: "https://arxiv.org/abs/2412.20138"
pdf: "../../raw/xiao-2024-tradingagents.pdf"
tags: [multi-agent, financial-trading, role-specialization, organizational-modeling, ICML]
created: 2026-04-10
updated: 2026-04-10
cites: [yu-2024-fincon]
cited_by: [li-2025-orchestration-financial]
---

# TradingAgents: Multi-Agents LLM Financial Trading Framework

> **One sentence.** TradingAgents proposes a five-team multi-agent LLM framework modeled on real trading firm organizational structures — with specialized analysts, bull/bear researchers, a trader, a risk management team, and a fund manager — achieving notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown over baselines on historical financial data (accepted ICML 2025, 24,800+ GitHub stars).

**Authors:** Yijia Xiao, Edward Sun, Di Luo, Wei Wang | **Venue:** ICML 2025 | **arXiv:** [2412.20138](https://arxiv.org/abs/2412.20138)

*Affiliations: ¹UCLA; ²MIT; ³Tauric Research. Open source: github.com/TauricResearch/TradingAgents*

---

## Problem & Motivation

Most existing LLM financial agents suffer from two structural problems:

1. **Lack of realistic organizational modeling**: frameworks treat agents as generic specialists that independently gather and aggregate data, disconnected from the workflows that real trading firms have proven effective over decades (analyst team → research/debate → risk management → execution approval chain).

2. **Inefficient communication interfaces**: using natural language as the primary communication medium for all agent interactions causes a "telephone effect" where information degrades as conversations lengthen, states become corrupted, and agents lose context. Unstructured information pools force agents to rely solely on retrieval, disrupting relational data integrity.

These problems mean that existing multi-agent systems don't actually benefit from the organizational intelligence that makes real trading firms effective.

---

## Core Idea

A trading firm's power comes not from any individual analyst but from how their insights flow through a structured organizational hierarchy before becoming a trade. TradingAgents replicates this: four analysts gather specialized data concurrently, pass it to researchers who debate bullish vs. bearish interpretations, which inform a trader's decision, which then passes through a risk management filter before a fund manager approves final execution. By combining structured outputs (for precision and control) with natural language dialogue (for debate and flexibility), the framework preserves both data integrity and the deliberative process that makes human trading teams effective.

---

## How It Works

### Architecture: Five Teams

**Team I — Analysts Team** (concurrent, parallel execution)
Four specialized agents gather market information simultaneously:
- *Fundamentals Analyst*: company financial metrics, earnings, valuation ratios
- *News Analyst*: real-time news from Bloomberg, Reuters, social media
- *Sentiment Analyst*: social media sentiment (Twitter/Reddit/StockTwits)
- *Technical Analyst*: price patterns, technical indicators (MACD, RSI, moving averages)

Each agent has a defined name, role, goal, constraints, context, skills, and tools. Outputs are structured (JSON/schema) for downstream processing clarity.

**Team II — Research Team** (debate)
Two researcher agents receive analyst outputs and debate market conditions:
- *Bull Researcher*: constructs bullish case from analyst evidence, challenges bear arguments
- *Bear Researcher*: constructs bearish case, challenges bull arguments

The debate produces a balanced research brief synthesizing both perspectives, with explicit buy and sell evidence surfaced.

**Team III — Trader**
Receives research team debate output and historical trading data. Makes trading decisions (buy/sell/hold, position sizing) based on synthesis of bullish and bearish evidence. Uses chain-of-thought reasoning for explainability.

**Team IV — Risk Management Team**
Risk guardians assess the trader's proposed transaction against current market conditions:
- Three risk profiles available: *Aggressive*, *Neutral*, *Conservative*
- Evaluates position size, portfolio exposure, drawdown risk
- Can veto or modify the trader's proposal

**Team V — Fund Manager**
Final authority. Reviews risk-adjusted transaction proposal. Approves or rejects execution. Provides oversight analogous to a principal-agent boundary in real trading operations.

### Communication Design

Two communication modes used strategically:
- **Structured outputs**: used by analyst agents for data reporting (preserves relational integrity, avoids telephone effect)
- **Natural language dialogue**: used in research team debate and risk management discussions (enables nuanced argumentation and flexible reasoning)

This hybrid approach ensures precision where accuracy matters and flexibility where deliberation matters.

### Supported LLM Backends

Framework is model-agnostic: OpenAI (GPT-4o, GPT-3.5), Google (Gemini), Anthropic (Claude), xAI (Grok), OpenRouter, Ollama (local models).

### Evaluation

Tested on historical financial data using:
- Cumulative return (CR%)
- Sharpe ratio (SR)
- Maximum drawdown (MDD%)

Compared against multiple baselines. Demonstrates improvements across all three metrics over single-agent and simpler multi-agent baselines.

---

## Results

The paper (4-page arXiv extended abstract version) reports qualitative improvements in CR, Sharpe, and MDD over baseline models. Full quantitative tables are in the ICML 2025 proceedings version.

Community validation: TradingAgents has 24,800 GitHub stars and 4,600 forks (as of late 2025) — the highest of any open-source financial agent framework surveyed in Li et al. (2025), indicating significant practical adoption.

---

## Comparison to Prior Work

| | TradingAgents | FinCon | StockAgent |
|---|---|---|---|
| Hierarchy | 5 teams | Manager-analyst | All agents equal |
| Debate mechanism | Bull vs Bear researchers | None (single manager decides) | All-to-all discussion |
| Risk management | Dedicated risk team + fund manager | CVaR + CVRF risk control | None |
| Communication | Hybrid (structured + NL) | Hierarchical NL prompts | NL conversation pool |
| Open source | Yes (24,800 GitHub stars) | Yes | Yes |
| Venue | ICML 2025 | NeurIPS 2024 | arXiv |

**vs [[yu-2024-fincon]] ([FinCon](../papers/yu-2024-fincon.md)):** FinCon uses a strict manager-analyst hierarchy where the manager is the sole decision-maker; TradingAgents uses a deeper chain (analysts → researchers debate → trader → risk team → fund manager). FinCon's innovation is its dual-level *risk control mechanism* (CVaR + CVRF); TradingAgents' innovation is its *organizational modeling fidelity* and hybrid communication. FinCon provides rigorous ablation studies and quantitative portfolio management results; TradingAgents is a shorter paper with broader community impact.

**vs [[li-2025-orchestration-financial]] ([Li et al. 2025](../papers/li-2025-orchestration-financial.md)):** TradingAgents focuses on the decision-making layer (analysts through fund manager). Li et al. map the *entire* algorithmic trading pipeline including data infrastructure, backtesting, execution, and audit. Li et al. use TradingAgents as one of the open-source reference systems when comparing the ecosystem.

**vs FinAgent (arXiv:2402.18485):** FinAgent uses a single multimodal agent with tool augmentation; TradingAgents distributes that role across four specialized analysts with richer organizational structure. Both support multiple data modalities but via different architectures.

---

## Strengths

- **Organizational fidelity**: the five-team structure maps directly onto how real trading firms operate, providing a principled justification for the architecture rather than arbitrary agent assignment.
- **Hybrid communication**: solving the telephone effect by using structured outputs for data and natural language for debate is a practical engineering contribution.
- **Open-source impact**: 24,800+ GitHub stars demonstrates real-world adoption and influence beyond academic publication.
- **Model agnosticism**: supports multiple LLM providers, making deployment accessible without vendor lock-in.

---

## Weaknesses & Limitations

- **Short paper (4 pages arXiv version)**: limited quantitative depth in the available preprint — full results are in ICML proceedings.
- **No explicit risk control mechanism**: unlike FinCon's CVaR/CVRF, TradingAgents' risk management relies on a risk team's LLM judgment, which may be less reliable than quantitative measures.
- **No portfolio management demonstrated**: single-stock trading only in the evaluated version.
- **Communication overhead**: five-team sequential pipeline is more complex than FinCon's manager-analyst hierarchy, potentially slower for high-frequency decisions.

---

## Key Takeaways

- Replicating real trading firm structure (not just assigning LLMs to tasks) is a productive architectural principle — organizational wisdom built up over decades is worth encoding.
- The Bull/Bear researcher debate is the key differentiator: it provides structured adversarial scrutiny before a trade is made, improving robustness vs. single-analyst systems.
- Hybrid communication (structured for data integrity, natural language for deliberation) solves the "telephone effect" that degrades pure-NL multi-agent systems.
- Community adoption (24,800+ GitHub stars) confirms that accessible, well-documented open-source frameworks drive field adoption more than pure performance gains.
- ICML 2025 acceptance validates the organizational modeling approach as a research contribution distinct from pure performance benchmarking.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{xiao2024tradingagents,
  title={{TradingAgents}: Multi-Agents {LLM} Financial Trading Framework},
  author={Xiao, Yijia and Sun, Edward and Luo, Di and Wang, Wei},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
{% endraw %}

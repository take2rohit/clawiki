---
title: "Large Language Model Agent in Financial Trading: A Survey"
type: paper
paper_id: P002
authors:
  - "Ding, Han"
  - "Li, Yinheng"
  - "Wang, Junhao"
  - "Chen, Hang"
  - "Guo, Doudou"
  - "Zhang, Yunbai"
year: 2024
venue: ICCMB 2026
arxiv_id: "2408.06361"
url: "https://arxiv.org/abs/2408.06361"
pdf: "../../raw/ding-2024-llm-trading-survey.pdf"
tags: [survey, llm-agents, financial-trading, architecture-taxonomy]
created: 2026-04-10
updated: 2026-04-10
cites: [nie-2024-llm-finance-survey]
cited_by: []
---

# Large Language Model Agent in Financial Trading: A Survey

> **One sentence.** The first dedicated survey of LLM agents in financial trading, reviewing 27 papers and organizing them into a two-category taxonomy (LLM as Trader vs. LLM as Alpha Miner) while finding that agents achieve 15–30% annualized returns above strong baselines in backtesting but face critical unsolved challenges around closed-source dependence and short backtesting windows.

**Authors:** Han Ding, Yinheng Li, Junhao Wang, Hang Chen, Doudou Guo, Yunbai Zhang | **Venue:** ICCMB 2026 | **arXiv:** [2408.06361](https://arxiv.org/abs/2408.06361)

---

## Problem & Motivation

Financial trading is a demanding task that requires synthesizing diverse signals, managing risk under uncertainty, and making rapid decisions. LLM-based agents are a natural fit because they can process vast unstructured text (news, filings, analyst reports) and produce investment signals. But the literature had grown rapidly without any systematic taxonomy of architectures, data types, evaluation methods, or failure modes. This survey is the first paper specifically focused on LLM *agents* (rather than LLMs generally in finance) for trading, filling that gap by reviewing all relevant work published through July 2024.

---

## Core Idea

LLM trading agents can be classified along one axis — whether the LLM *makes the trade* directly (LLM as a Trader) or *generates a quantitative signal* that downstream systems then trade (LLM as an Alpha Miner). Within each branch, distinct architectural families emerge based on how agents handle memory, feedback, and multi-agent coordination. Understanding this taxonomy reveals where the field is mature and where it is still nascent.

---

## How It Works

### Architecture Taxonomy

The survey identifies two top-level categories:

**LLM as a Trader** — the LLM directly outputs buy/hold/sell decisions by analyzing market data. Four sub-types:

1. **News-Driven**: individual stock news and macro updates injected into the prompt. LLM predicts next-period price movements. Examples: LLMFactor (uses reasoning to identify key factors from news vs. stock history), MarketSenseAI (daily news/macro summaries distilled into a memory module).
2. **Reflection-Driven**: agents build hierarchical memories from raw data and use reflection to refine trading decisions over time. FinMem and FinAgent exemplify this. FinAgent extends it to multimodal data (charts, tables, text) via GPT-4V.
3. **Debate-Driven**: multiple LLM agents with different roles (mood agent, rhetoric agent, dependency agent) debate a topic to improve sentiment classification and factual validity. TradingGPT and HAD use this pattern.
4. **RL-Driven**: LLM outputs are aligned with profit/loss feedback via reinforcement learning. SEP uses RL with memorization + reflection modules, training on a series of correct/incorrect decisions from financial market history.

**LLM as an Alpha Miner** — the LLM generates alpha factors (quantitative signals) rather than making final trades. Two examples:
- **QuantAgent**: inner loop where a writer agent produces trading scripts from trader ideas; a judge agent evaluates them; outer loop commits to real-market testing.
- **AlphaGPT**: human-in-the-loop alpha mining framework. Human interprets trading ideas; agent translates them into testable strategies.

### Data Types

Four categories of inputs used across surveyed papers:
- **Numerical**: stock prices, trading volumes, technical indicators
- **Textual — Fundamental**: Form 10-Q, Form 10-K filings, analyst reports, earnings call transcripts
- **Textual — Alternative**: news (Bloomberg, WSJ, CNBC), social media (Twitter, StockTwits, Reddit)
- **Visual**: Kline charts, volume charts, trading charts
- **Simulated**: synthetic market environments for agent behavior analysis and safety testing

Most common base LLMs: GPT-3.5 (most frequent, due to cost/performance), GPT-4, FinBERT, Qwen, Baichuan.

### Evaluation Framework

**Portfolio Performance Metrics:**
- Cumulative Return = (P_t − P_0) / P_0 × 100%
- Sharpe Ratio = (R_p − R_f) / σ_p (portfolio excess return vs. volatility)
- Maximum Drawdown (MDD) = max drop from peak to trough

**Signal Metrics:** F1 score, accuracy (for news sentiment prediction), win rate, Information Coefficient (IC)

**Backtesting Setup:**
| Period Length | Papers |
|---|---|
| 0–2 years | 8 |
| 2–5 years | 2 |
| ≥5 years | 4 |

Median backtesting period is only 1.3 years. Most papers use 2020–2024 data. 9 of 14 real-data papers use US stocks; 5 use Chinese markets.

### Training

No unified training procedure — surveyed papers vary widely. Most use in-context learning without fine-tuning (only SEP fine-tunes the LLM). Token cost is rarely reported; QuantAgent is the lone exception tracking computational complexity.

---

## Results

### Backtesting Performance

LLM-powered trading agents have achieved annualized returns 15–30% above the strongest traditional baseline (Buy-and-Hold or momentum rules) in backtesting. Sharpe ratios and MDD are also improved in most papers. However:

- Backtesting periods are short (median 1.3 years), often coinciding with the training period — credibility risk.
- Results are not standardized: different tickers, periods, and baselines make cross-paper comparison unreliable.
- Only FinAgent extends evaluation to cryptocurrency markets (ETH).

### Common Baselines

Rule-based: Buy-and-Hold, Mean Reversion, Short-Term Reversal
ML/DL: Random Forest, LightGBM, LSTM, BERT
RL: PPO, DQN

---

## Comparison to Prior Work

| | This Survey | Nie et al. (2406.11903) |
|---|---|---|
| Scope | LLM agents for trading only | All LLM applications in finance |
| Papers reviewed | 27 | 300+ |
| Agent focus | Deep (taxonomy + eval analysis) | Light (one subsection on ABM) |
| Limitations analysis | Specific to trading agents | Broader challenges |

**vs [[nie-2024-llm-finance-survey]] ([Nie et al. 2024](../papers/nie-2024-llm-finance-survey.md)):** Nie et al. cover all LLM-in-finance applications across 300+ papers in 28 pages; this survey focuses exclusively on trading agents, going deeper on architecture sub-types, data inputs, and backtesting methodology for 27 papers in 8 pages. Complementary in scope.

---

## Strengths

- **First dedicated taxonomy of LLM trading agents**: the two-category split (Trader vs. Alpha Miner) with four sub-architectures within Trader is a clean conceptual contribution.
- **Honest evaluation critique**: explicitly calls out that short backtesting periods and lack of standardization undermine result credibility.
- **Practical data survey**: useful summary of which LLM models dominate (GPT-3.5 >> GPT-4 for cost reasons) and which data sources are used.

---

## Weaknesses & Limitations

- **Only 27 papers, 8 pages**: scope is narrow; the survey is more a preliminary categorization than a comprehensive meta-analysis.
- **No quantitative meta-analysis**: does not aggregate results across papers with controlled comparisons.
- **Short backtesting window problem**: the survey identifies this but cannot fix it — 9 of 14 papers have under 2 years of backtesting.
- **No live trading evidence**: all results are from backtesting, which is susceptible to overfitting and lookahead bias.

---

## Key Takeaways

- LLM trading agents split into two fundamentally different roles: making direct trading decisions (Trader) vs. generating quantitative signals (Alpha Miner). Most research is on the Trader side.
- Within Trader architectures, reflection-driven and debate-driven agents are more sophisticated than simple news-driven ones, incorporating memory, self-critique, and multi-agent debate.
- Reported performance is promising (15–30% above baseline) but the evidence base is thin: short backtest periods (median 1.3 years), US/China stocks only, and no standard benchmarks.
- Most agents use closed-source models (GPT-3.5/4), rely on in-context learning without fine-tuning, and ignore trading costs — all critical gaps for real deployment.
- The field is pre-paradigm: no standard benchmark, no standard backtesting period, no standard set of tickers. The survey's main contribution is providing the first map of the territory.

---

## BibTeX

{% raw %}
```bibtex
@article{ding2024llmtradingsurvey,
  title={Large Language Model Agent in Financial Trading: A Survey},
  author={Ding, Han and Li, Yinheng and Wang, Junhao and Chen, Hang and Guo, Doudou and Zhang, Yunbai},
  journal={arXiv preprint arXiv:2408.06361},
  year={2024}
}
```
{% endraw %}

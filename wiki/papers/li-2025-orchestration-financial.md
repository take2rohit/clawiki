---
title: "Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading"
type: paper
paper_id: P005
authors:
  - "Li, Jifeng"
  - "Grover, Arnav"
  - "Alpuerto, Abraham"
  - "Cao, Yupeng"
  - "Liu, Xiao-Yang"
year: 2025
venue: NeurIPS 2025 Workshop (Generative AI in Finance)
arxiv_id: "2512.02227"
url: "https://arxiv.org/abs/2512.02227"
pdf: "../../raw/li-2025-orchestration-financial.pdf"
tags: [orchestration, financial-agents, algorithmic-trading, MCP, multi-agent, NeurIPS]
created: 2026-04-10
updated: 2026-04-10
cites: [yu-2024-fincon, xiao-2024-tradingagents]
cited_by: []
---

# Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading

> **One sentence.** Proposes a systematic mapping of traditional algorithmic trading (AT) system components to AI agent roles — using MCP for control messages and A2A for peer communication — achieving a 20.42% total return (Sharpe 2.63) on a 7-stock portfolio vs. S&P 500's 15.97%, and 8.39% return on BTC/USDT vs. Buy-and-Hold's 3.80%.

**Authors:** Jifeng Li, Arnav Grover, Abraham Alpuerto, Yupeng Cao, Xiao-Yang Liu | **Venue:** NeurIPS 2025 Workshop on Generative AI in Finance | **arXiv:** [2512.02227](https://arxiv.org/abs/2512.02227)

*Affiliations: ¹SecureFinAI Lab, Columbia University; ²Purdue University; ³Rensselaer Polytechnic Institute; ⁴Stevens Institute of Technology.*

---

## Problem & Motivation

Building an effective algorithmic trading system has historically required a professional team of engineers, quants, and domain experts working for years. The AT pipeline — from raw data → feature engineering → alpha signal generation → risk control → portfolio construction → execution → post-trade audit — involves highly specialized components that are difficult to assemble and maintain. Recent LLM agents (TradingAgents, FinCon) tackle the *decision-making* layer but leave the surrounding infrastructure (data pipelines, backtesting with proper walk-forward validation, execution with slippage modeling, audit trails) underspecified. The result is systems that are excellent at deciding *what* to trade but poorly equipped to reliably *trade it* end-to-end. This paper proposes democratizing the full AT pipeline by mapping each component to an AI agent pool, with standard protocols for inter-agent communication.

---

## Core Idea

Every component of a traditional algorithmic trading system has a natural agent analog. The key insight is that this mapping isn't superficial — the data flow relationships, dependency ordering, and isolation requirements between AT components correspond precisely to what agent orchestration frameworks (MCP, A2A) are designed to handle. By using MCP for control messages (structured task dispatch, health monitoring) and A2A for peer communication between agent pools, the full AT pipeline becomes orchestratable end-to-end without custom glue code. A memory agent provides shared context across the pipeline, allowing later stages to reason about what earlier stages did.

---

## How It Works

### Architecture: Component-to-Agent Mapping

The paper provides a direct correspondence between AT components and agent roles:

| AT Component | Agent Role |
|---|---|
| Strategy Research | Planner |
| Strategy Development | Orchestrator |
| Data Pre-processing | Data Agents |
| Alpha Model + Risk Model | Alpha Agents + Risk Agents |
| Portfolio Construction | Portfolio Agents |
| Pre-trade analysis | (gap in AT → covered by Backtest Agents) |
| Backtesting | Backtest Agents |
| Trade Execution | Execution Agents |
| Post-trade Analysis | Audit Agents |
| — | Memory Agent (shared context) |

**Planner**: high-level strategy definition — which markets, horizons, information sources to use. Sets the overall plan graph (DAG) for the pipeline.

**Orchestrator**: manages agent pools via MCP. Sends task control messages (node type, task ID, input schema, policy flags, timeout, retry budget). Monitors health via heartbeats. Receives status/logs/artifact IDs back from agents.

**Data Agents**: fetch, clean, align, and normalize data from multiple sources (Polygon, yfinance for equities; Polygon, Binance for BTC). Deduplicate, align calendars, compute baseline features (returns, momentum, volatility ratios). Output cleaned data references + quality diagnostics.

**Alpha Agents**: design signal/factor structures based on published literature. For equities: momentum, mean-reversion, breakout, trend signals. For BTC: order flow imbalance, bid-ask spread, volume spikes. Critically, Alpha Agents *do not* compute numerical signals — they *specify factor structures* that tool-based ML modules then compute. This design choice prevents LLMs from hallucinating returns.

**Risk Agents**: compute exposure limits (concentration, volatility, drawdown constraints). For stocks: enforce sector/name concentration limits. For BTC: stricter position-size caps, leverage limits, intraday drift/volatility gates.

**Portfolio Agents**: translate risk-adjusted signals into target positions. Apply long-only or long-short rules, turnover constraints, minimum position thresholds.

**Backtest Agents**: run walk-forward backtests with proper train/validation/test windows. Evaluation aggregates Sharpe, realized returns, drawdown. Realized returns are never exposed to LLM agents during design — only after the test window closes.

**Execution Agents**: simulate order placement with slippage, transaction cost, and partial fill modeling. Orders only execute when all Alpha, Risk, and Portfolio checks pass.

**Audit Agents**: verify equity curves, compute attribution, write full logs for planner/orchestrator to use in the next planning cycle.

**Memory Agent**: stores structural summaries (not evaluation labels), prompts, tool calls, and decisions. Provides shared context to all agent pools via A2A. Avoids storing evaluation-window outcomes to prevent lookahead.

### Communication Protocols

**MCP (Model Context Protocol)**: orchestrator → agent pools. Small structured control messages. Each agent pool is exposed as a tool-like endpoint with request-response schema. Inside a pool, a manager agent breaks tasks into subtasks and coordinates subordinates.

**A2A (Agent-to-Agent)**: peer-to-peer agent communication. Agents use simple message types: ask, tell, propose, confirm, with role tags and context IDs. Time-stamped and stored in memory for replay and audit.

### Trading Pipelines

**Equities Pipeline:**
- Universe: 7 stocks (AAPL, MSFT, GOOGL, JPM, TSLA, NVDA, META)
- Data: hourly bars, 09/2022–01/2025
- Evaluation window: 04/2024–12/2024
- Scrolling training window: 3 months

**BTC Pipeline:**
- Same agent structure, minute-level bars
- BTC/USDT, 05/2025–08/2025
- Prediction horizon: 1 minute, retrained every 24 hours
- ML model: XGBoost regression (300 trees, depth 6, LR 0.08)
- 100+ features: price momentum (1/3/5/10/15/30/60/240 min), volatility (GARCH-style), RSI with windows 14/30, MACD, Bollinger Bands

---

## Results

### Equities Trading (04/2024–12/2024)

| Metric | Ours | SPY | QQQ | IWM | VTI | EW Portfolio |
|---|---|---|---|---|---|---|
| Total Return ↑ | **20.42%** | 16.60% | 21.59% | 11.45% | 16.29% | **47.46%** |
| Annual Return ↑ | **31.08%** | 25.07% | 32.94% | 17.10% | 24.59% | **76.07%** |
| Volatility ↓ | **11.83%** | 13.49% | 18.38% | 21.61% | 13.72% | 22.54% |
| Sharpe Ratio ↑ | **2.63** | 1.86 | 1.79 | 1.79 | 1.79 | 3.37 |
| MDD ↑ | **−3.59%** | −8.89% | −14.13% | −11.60% | −9.06% | −16.21% |

The agentic strategy beats all ETF benchmarks in total return and significantly outperforms them on risk-adjusted metrics (Sharpe 2.63 vs. 1.79–1.86). However, it does not beat the equally-weighted portfolio (47.46% total return) — a high bar given the bull run of the selected universe.

The key advantage is risk management: lowest volatility (11.83%) and smallest drawdown (−3.59%) among all compared strategies.

### BTC Trading (07/2025–08/2025, 17 days)

| Metric | Ours | Buy & Hold |
|---|---|---|
| Cumulative Return | **8.39%** | 3.80% |
| Annualized Volatility | 24.23% | 25.82% |
| Sharpe Ratio | **0.378** | 0.170 |
| MDD | **−2.80%** | −5.26% |

+4.59 percentage points excess return vs. Buy-and-Hold, with lower volatility and smaller drawdown. Win rate 64.7% (B&H: 58.8%), 17 trades total, average holding time 16 hours.

### Ecosystem Survey

The paper also benchmarks open-source agentic trading projects:

| Project | GitHub Stars | Forks | Markets | Agents |
|---|---|---|---|---|
| TradingAgents | 24,800 | 4,600 | Equities | 6 |
| AI Hedge Fund | 42,300 | 7,500 | Equities | 18 |
| ContestTrade | 465 | 124 | Equities, CN, US | 2+ |
| StockAgent | 402 | 89 | Equities | 1 |

Higher-adoption projects (TradingAgents, AI Hedge Fund) all use multi-agent designs with persistent memory.

---

## Comparison to Prior Work

| | This Paper | TradingAgents | FinCon |
|---|---|---|---|
| Scope | Full AT pipeline (research → audit) | Decision layer (analyst → fund manager) | Decision layer (analyst → manager) |
| Infrastructure focus | High (MCP, A2A, walk-forward BT) | Low | Low |
| Risk control | Portfolio-level constraints via Risk Agents | Risk management team (LLM-based) | CVaR + CVRF (quantitative) |
| Asset classes | Equities + BTC | Equities | Equities |
| Backtesting rigor | Walk-forward, data leakage prevention | Standard | Standard |
| Communication | MCP (control) + A2A (peer) | Hybrid structured + NL | Hierarchical NL prompts |

**vs [[xiao-2024-tradingagents]] ([TradingAgents](../papers/xiao-2024-tradingagents.md)):** TradingAgents focuses on the decision-making organizational structure; this paper maps the *full* AT pipeline including infrastructure components (data agents, backtest agents, execution agents, audit agents) that TradingAgents leaves out. Li et al. cite TradingAgents as an open-source reference system in their ecosystem survey.

**vs [[yu-2024-fincon]] ([FinCon](../papers/yu-2024-fincon.md)):** FinCon demonstrates superior decision-making performance through its dual-level risk control; Li et al. provide the surrounding infrastructure (walk-forward backtesting, execution simulation, audit trails) that would be needed to deploy FinCon-like systems reliably in production. Complementary contributions.

---

## Strengths

- **End-to-end pipeline**: the only paper in this corpus that covers data ingestion through post-trade audit — much closer to a production system than decision-layer-only approaches.
- **Principled data leakage prevention**: Alpha Agents specify factor structures (from literature) but never see evaluation-window returns; all numerical computations done by tool modules. This is the most rigorous lookahead prevention of any paper reviewed.
- **Multi-asset coverage**: equities hourly + BTC minute-level in a single unified framework.
- **MCP/A2A standardization**: using emerging industry protocols (MCP, A2A) for agent communication enables future interoperability and ecosystem integration.

---

## Weaknesses & Limitations

- **Short evaluation windows**: equity evaluation is 9 months; BTC evaluation is only 17 days — too short to draw definitive conclusions.
- **EW portfolio beats it on return**: the equally-weighted portfolio (47.46%) outperforms the agentic strategy (20.42%) in total return during the test period — the risk-controlled profile is better but the benchmark is a high bar.
- **Workshop paper**: 11 pages, NeurIPS 2025 workshop rather than full conference. Limited peer review compared to FinCon (NeurIPS main) or TradingAgents (ICML).
- **No comparison to FinCon or TradingAgents directly**: the framework is positioned as complementary infrastructure rather than competing with decision-layer agents.

---

## Key Takeaways

- Mapping the full algorithmic trading pipeline to AI agents is tractable using standard protocols (MCP for control, A2A for peer communication) — no custom integration required.
- Separating *signal design* (LLM Alpha Agents from literature) from *signal computation* (tool-based ML modules) is the key innovation for preventing lookahead bias and hallucination in trading contexts.
- The agentic approach achieves superior risk-adjusted performance (Sharpe 2.63, MDD −3.59%) vs. ETF benchmarks, even if total return lags the simple EW portfolio in the test window.
- Walk-forward backtesting with strict data isolation between training, validation, and test windows is the most important methodological advance over prior agent trading papers that use simple single-period backtests.
- The ecosystem is bifurcating: decision-layer systems (FinCon, TradingAgents) optimize *what* to trade; infrastructure systems (this paper) optimize *how* to execute it reliably.

---

## BibTeX

{% raw %}
```bibtex
@article{li2025orchestration,
  title={Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading},
  author={Li, Jifeng and Grover, Arnav and Alpuerto, Abraham and Cao, Yupeng and Liu, Xiao-Yang},
  journal={Workshop on Generative AI in Finance, NeurIPS 2025},
  year={2025}
}
```
{% endraw %}

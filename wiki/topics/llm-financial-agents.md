---
title: "LLM Financial Agents"
type: topic
tags: [llm-agents, financial-trading, multi-agent]
created: 2026-04-10
updated: 2026-04-10
---

# LLM Financial Agents

LLM-based agents for financial applications — encompassing trading, portfolio management, risk control, and end-to-end algorithmic trading pipelines.

## Papers

- [[nie-2024-llm-finance-survey]] ([Nie et al. 2024](../papers/nie-2024-llm-finance-survey.md)) — comprehensive survey of all LLM applications in finance; agent-based modeling section covers StockAgent, FinAgent, FinMem, QuantAgent, TradingGPT
- [[ding-2024-llm-trading-survey]] ([Ding et al. 2024](../papers/ding-2024-llm-trading-survey.md)) — dedicated survey of LLM trading agents; taxonomy of Trader vs. Alpha Miner architectures
- [[yu-2024-fincon]] ([Yu et al. 2024](../papers/yu-2024-fincon.md)) — NeurIPS 2024; manager-analyst hierarchy + dual-level risk control; best results on 8 stocks and 2 portfolios
- [[xiao-2024-tradingagents]] ([Xiao et al. 2024](../papers/xiao-2024-tradingagents.md)) — ICML 2025; five-team organizational structure; 24,800+ GitHub stars
- [[li-2025-orchestration-financial]] ([Li et al. 2025](../papers/li-2025-orchestration-financial.md)) — NeurIPS 2025 Workshop; full AT pipeline mapping; MCP + A2A protocols

## Architecture Taxonomy

### Decision-Layer Systems

**LLM as a Trader** (direct buy/sell/hold decisions):
- *News-driven*: LLMFactor, MarketSenseAI
- *Reflection-driven*: FinMem, FinAgent
- *Debate-driven*: TradingGPT, HAD, TradingAgents (Bull/Bear researchers)
- *RL-driven*: SEP

**LLM as an Alpha Miner** (generates quantitative signals):
- QuantAgent (inner/outer loop code generation)
- AlphaGPT (human-in-the-loop)

### Full Pipeline Systems

**End-to-end orchestration** (data → alpha → risk → portfolio → execution → audit):
- Orchestration Framework (Li et al. 2025): Planner → Orchestrator → Agent Pools → Memory Agent

## Key Open Questions

1. Can LLM agents generalize across market regimes (bull → bear → sideways)?
2. How do short backtesting windows (median 1.3 years) understate overfitting risk?
3. Does the organizational hierarchy (manager-analyst, five-team) provide signal beyond the data each agent has access to?
4. Can walk-forward backtesting with data leakage prevention (Li et al.) be combined with FinCon's dual-level risk control?

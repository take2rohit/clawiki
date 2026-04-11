---
title: "MCP + A2A Orchestration for Financial Agents"
type: method
tags: [orchestration, multi-agent, MCP, A2A, financial-trading, protocols]
created: 2026-04-10
updated: 2026-04-10
papers: [li-2025-orchestration-financial]
---

# MCP + A2A Orchestration for Financial Agents

A two-protocol communication architecture introduced in [[li-2025-orchestration-financial]] ([Li et al. 2025](../papers/li-2025-orchestration-financial.md)) for coordinating multi-agent financial pipelines. Uses **MCP (Model Context Protocol)** for orchestrator-to-pool control messages and **A2A (Agent-to-Agent protocol)** for peer-to-peer communication between agent pools.

## Protocol Roles

### MCP — Vertical Control (Orchestrator → Agent Pools)

Structured, task-dispatch messages. Each agent pool is exposed as a tool-like endpoint with request/response schema. Control message fields:
- Node type, task ID
- Input schema, policy flags
- Timeout, retry budget

The Orchestrator also uses MCP for health monitoring (heartbeats from agent pools). Agents send status/logs/artifact IDs back via MCP response.

### A2A — Horizontal Peer Communication (Agent ↔ Agent)

Simple typed messages between agent pools. Message types: `ask`, `tell`, `propose`, `confirm` — each with role tags and context IDs. All A2A messages are time-stamped and stored in the Memory Agent for replay and audit.

## Agent Pool Mapping

| AT Pipeline Component | Agent Pool |
|---|---|
| Strategy Research | Planner |
| Strategy Development | Orchestrator |
| Data Pre-processing | Data Agents |
| Alpha Model | Alpha Agents |
| Risk Model | Risk Agents |
| Portfolio Construction | Portfolio Agents |
| Backtesting | Backtest Agents |
| Trade Execution | Execution Agents |
| Post-trade Analysis | Audit Agents |
| Shared context | Memory Agent |

## Key Design Decisions

**Separation of signal design and computation**: Alpha Agents specify factor *structures* (from literature); tool-based ML modules compute the actual signals. LLMs never see evaluation-window returns during design. This is the primary mechanism for preventing lookahead bias and hallucinated returns.

**Walk-forward backtesting isolation**: Backtest Agents use rolling train/validation/test windows. Realized returns are never exposed to LLM agents until after the test window closes.

**Memory Agent scope**: stores structural summaries, prompts, tool calls, and decisions — but *not* evaluation-window outcomes, to prevent lookahead leakage into future planning cycles.

## Why Standardized Protocols Matter

Prior agentic trading frameworks (FinCon, TradingAgents) use custom natural-language prompt chains for inter-agent communication. This limits composability — swapping one agent pool requires rewriting prompt interfaces throughout the system. MCP and A2A are emerging industry standards that make agent pools interchangeable and independently deployable.

## Empirical Results

Equities (7-stock portfolio, 04–12/2024): 20.42% total return, Sharpe 2.63, MDD −3.59% vs. S&P 500 15.97%.
BTC (07–08/2025, 17 days): 8.39% cumulative return, Sharpe 0.378 vs. Buy-and-Hold 3.80%, 3.80%.

## Related Methods

- [Manager-Analyst Hierarchy](manager-analyst-hierarchy.md) — decision-layer architecture that MCP/A2A could orchestrate
- [Conceptual Verbal Reinforcement](conceptual-verbal-reinforcement.md) — episodic learning mechanism; could be integrated as an Audit Agent → Planner feedback loop in this framework

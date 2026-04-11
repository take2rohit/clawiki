---
title: "Manager-Analyst Hierarchy"
type: method
tags: [multi-agent, financial-trading, hierarchy, llm-agents]
created: 2026-04-10
updated: 2026-04-10
papers: [yu-2024-fincon, xiao-2024-tradingagents]
---

# Manager-Analyst Hierarchy

An architectural pattern for multi-agent LLM systems in which specialized **analyst agents** each process a single data modality or task domain and report distilled insights upward to a **manager agent** that synthesizes inputs and makes final decisions. Inspired by real investment firm organizational structures.

## Core Properties

- **Vertical specialization**: each analyst owns exactly one data source or modality — this bounds context length per agent and ensures focused reasoning
- **Single decision point**: only the manager issues final buy/sell/hold signals; analysts are advisory
- **Low communication overhead**: O(N) messages (each analyst → manager) vs. O(N²) in peer-to-peer debate architectures

## Implementations

### FinCon ([Yu et al. 2024](../papers/yu-2024-fincon.md))
7 analyst agents (Data, News, 10-Q, 10-K, Analyst Reports, ECC Audio, Stock Selection) → 1 Manager agent. Manager also receives risk alerts from the CVaR monitor and episodic belief updates from CVRF. Communication is hierarchical NL prompts. The manager uses mean-variance optimization to translate directional signals into portfolio weights.

### TradingAgents ([Xiao et al. 2024](../papers/xiao-2024-tradingagents.md))
Extends the pattern with a 5-team chain: Analysts Team (4 concurrent agents) → Research Team (Bull + Bear debate) → Trader → Risk Management → Fund Manager. The debate layer between analysts and trader is an addition over FinCon's strict single-hop hierarchy.

## Comparison

| | FinCon | TradingAgents |
|---|---|---|
| Depth | 2-level (analyst → manager) | 5-level chain |
| Debate | None (manager synthesizes) | Bull vs. Bear researcher debate |
| Risk control | Quantitative (CVaR + CVRF) | LLM-based risk team |
| Portfolio management | Yes (mean-variance) | No (single stock) |

## Why It Works

Context window management: each analyst receives only its own modality, so reasoning quality is high without token overflow. The manager sees distilled summaries rather than raw data, keeping its context tractable. This compares favorably to StockAgent's all-to-all conversation pool, which scales quadratically and degrades under long contexts.

## Related Methods

- [Conceptual Verbal Reinforcement (CVRF)](conceptual-verbal-reinforcement.md) — FinCon's over-episode learning mechanism, operates on the manager's investment beliefs
- [MCP + A2A Orchestration](mcp-a2a-orchestration.md) — Li et al.'s infrastructure layer that could host manager-analyst hierarchies

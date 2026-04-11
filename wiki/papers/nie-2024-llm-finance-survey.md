---
title: "A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges"
type: paper
paper_id: P001
authors:
  - "Nie, Yuqi"
  - "Kong, Yaxuan"
  - "Dong, Xiaowen"
  - "Mulvey, John M."
  - "Poor, H. Vincent"
  - "Wen, Qingsong"
  - "Zohren, Stefan"
year: 2024
venue: arXiv
arxiv_id: "2406.11903"
url: "https://arxiv.org/abs/2406.11903"
pdf: "../../raw/nie-2024-llm-finance-survey.pdf"
tags: [survey, llm, finance, agent-based-modeling, benchmarks]
created: 2026-04-10
updated: 2026-04-10
cites: []
cited_by: [ding-2024-llm-trading-survey]
---

# A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges

> **One sentence.** A 28-page comprehensive survey from Princeton and Oxford covering all LLM applications in finance across six task categories — from sentiment analysis to agent-based modeling — including a thorough taxonomy of 40+ financial LLMs, 20+ benchmarks, and four classes of unsolved challenges.

**Authors:** Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M. Mulvey, H. Vincent Poor, Qingsong Wen, Stefan Zohren | **Venue:** arXiv 2024 | **arXiv:** [2406.11903](https://arxiv.org/abs/2406.11903)

*Affiliations: Yuqi Nie & H.Vincent Poor — Princeton ECE; John M. Mulvey — Princeton ORFE; Yaxuan Kong, Xiaowen Dong, Stefan Zohren — Oxford Engineering Science; Qingsong Wen — Squirrel AI.*

---

## Problem & Motivation

Prior surveys on LLMs in finance either focused only on model-level comparisons, or covered only a narrow slice of applications (e.g., sentiment analysis only). None provided a holistic view bridging academic research with practical applications, and none systematically catalogued the unique challenges of deploying LLMs in finance — particularly lookahead bias, legal responsibility, hallucinations in financial reports, and signal decay. This survey fills that gap with a structured treatment of models, applications, datasets/benchmarks, and challenges.

---

## Core Idea

LLMs are uniquely well-suited for finance because financial data is predominantly textual — earnings calls, analyst reports, regulatory filings, news — and because finance demands the same capabilities LLMs excel at: contextual understanding, sentiment detection, reasoning under uncertainty, and planning. The survey's organizing insight is to structure the field into six application areas, each building on prior capabilities, from basic textual extraction up to fully autonomous multi-agent trading systems.

---

## How It Works

### Model Taxonomy (Section 2)

The survey catalogs two tiers of LLMs used in finance:

**General-domain LLMs adapted to finance:** GPT-series (GPT-3.5, GPT-4), BERT/RoBERTa, T5, BLOOM, Llama 1/2/3.

**Finance-specific LLMs (fine-tuned or pre-trained from scratch):**

| Model | Base | Key Feature |
|---|---|---|
| FinBERT-19/20/21 | BERT | Continual pre-training on financial text |
| FLANG | ELECTRA | Financial-specific masking, span objectives |
| BBT-FinT5 | T5 | Chinese financial sector, BBT-FinCorpus |
| BloombergGPT | BLOOM | 50B params, Bloomberg proprietary data |
| XuanYuan 2.0 | BLOOM | Chinese financial chat, hybrid-tuning |
| FinGPT | LLaMA | Open-source, 50K instruction samples (LoRA) |
| FinMA (PIXIU) | LLaMA | 136K finance instructions |
| InvestLM | LLaMA-65B | Investment recommendations |

**Zero-shot vs. Fine-tuning:** Fine-tuning preferred when domain accuracy is critical; zero-shot when labeled data is scarce or modularity/privacy is needed. Instruction tuning and LoRA are common efficiency choices.

### Application Taxonomy (Section 3)

Six application categories, each progressively more complex:

**3.1 Linguistic Tasks**
- *Summarization & Extraction*: processing lengthy 10-K/10-Q filings, chunking strategies for RAG
- *Financial Relation Construction*: knowledge graphs for entity relationships (FinDKG — temporal knowledge graph for finance)
- *NER*: extracting company names, financial KPIs, events from filings; KPI-BERT uses BERT + RNN for German financial documents

**3.2 Sentiment Analysis**
- Pre-LLM methods: lexicon-based (LM word lists), ML (SVM, Random Forest), embedding (Word2Vec, GloVe, ELMo)
- LLM-era: ChatGPT/GPT-4 significantly outperform FinBERT on earnings calls, regulatory filings, and social media; adversarially robust due to contextual understanding
- Data sources: social media (Twitter/StockTwits/Reddit), news, earnings calls (ECC), FOMC minutes, ECB policy decisions

**3.3 Financial Time Series Analysis**
- *Forecasting*: direct LLM stock prediction (NASDAQ-100), GPT-4 zero-shot price movement from microblogging (beating BERT on Apple/Tesla 2017)
- *Anomaly Detection*: LLM-based multi-agent framework applied to S&P 500 anomaly identification
- *Classification*: classifying bull/bear/sideways regimes
- *Data Augmentation*: generative AI for synthetic financial data (order book modeling)
- *Imputation*: filling missing values in financial time series

**3.4 Financial Reasoning**
- *Planning*: GPT-4 for business strategy development using NER + ZSC, corporate financial planning
- *Recommendation*: LLM investment advisory (ChatGPT 3% monthly alpha on policy-related news), Cogniwealth (Llama 2 for investment recommendations)
- *Support Decision-Making*: ZeroShotALI (GPT-4 + SentenceBERT for financial audit compliance), fraud detection via FinChain-BERT
- *Real-time Reasoning*: WeaverBird (GPT-fine-tuned for financial Q&A with web search integration), GPTQuant (few-shot backtesting assistant)

**3.5 Agent-based Modeling**
- *Trading & Investments*: StockAgent (multi-agent GPT-3.5/Gemini for stock trading simulation), FinAgent (multimodal: charts + text + tables via GPT-4V), FinMem (layered memory + character design), QuantAgent (inner-outer loop alpha mining), Alpha-GPT (human-in-the-loop alpha mining)
- *Simulating Markets*: EconAgent (LLM macroeconomic simulation), Horton (Homo Silicus rational economic agents), CompeteAI (restaurant competition simulation)
- *Automated Processes*: FlowMind (automated financial workflows), AUCARENA (strategic LLM planning in auction environments)
- *Multi-agent Systems*: TradingGPT (three-tier layered memory, heterogeneous agent personalities), HAD (heterogeneous debate for financial sentiment), SocraPlan (multi-agent corporate planning)

### Datasets and Benchmarks (Section 4)

Key datasets: Financial PhraseBank (FPB), FiQA (aspect-based sentiment QA), FinQA (numerical reasoning over financial reports).

Key benchmarks:

| Benchmark | Year | Focus | Open Source |
|---|---|---|---|
| FLUE | 2022 | 5 financial NLP tasks | Yes |
| PIXIU/FLARE | 2023 | Multi-task + stock prediction | Yes |
| AlphaFin | 2024 | Stock trend prediction + QA | Yes |
| FinanceBench | 2023 | Financial QA (earnings reasoning) | Yes |
| DocMath-Eval | 2023 | Numerical reasoning over long docs | Yes |
| EconLogicQA | 2024 | Sequential economic reasoning | Yes |
| R-Judge | 2024 | Risk awareness in financial decisions | Yes |

### Challenges (Section 5)

**Data Issues:**
- *High-Dimensional Financial Data*: LLMs struggle with long time-series; hybrid models needed
- *Data Pollution*: LLM-generated financial text contaminating training data
- *Signal Decay*: as more traders use LLM strategies, edges erode

**Modeling Issues:**
- *Inference Speed/Cost*: high compute costs; hybrid small/large model routing can cut large-model calls by 40%
- *Lookahead Bias*: models inadvertently train on future data; TimeMachineGPT uses point-in-time LLMs as a remedy
- *Hallucinations*: LLMs generate factually incorrect financial statements; GenAudit (fact-checking LLM outputs) as mitigation

**Benchmarking Issues:**
- Existing benchmarks pre-date LLMs and may no longer be appropriate
- New benchmarks adaptable to LLM-generated signals needed

**Ethical Issues:**
- *Benign Alignment*: avoiding harmful investment recommendations
- *Legal Responsibility*: unclear accountability when LLMs cause financial harm
- *Safety & Privacy*: data breach risks with cloud-based LLMs; local deployment needed for confidential data
- *Understanding Incentives*: LLMs in high-stakes finance require transparent incentive structures

---

## Comparison to Prior Work

| Survey | Financial LLMs | Benchmarks | Applications | Challenges |
|---|---|---|---|---|
| Lee et al. | ✓ | ✓ | ○ | ○ |
| Li et al. | ✓ | ✗ | ○ | ○ |
| Dong et al. | ✗ | ✓ | ✓ | ○ |
| Zhao et al. | ✗ | ✓ | ✓ | ○ |
| **Nie et al. (Ours)** | **✓** | **✓** | **✓** | **✓** |

**vs [[ding-2024-llm-trading-survey]] ([Ding et al. 2024](../papers/ding-2024-llm-trading-survey.md)):** Ding et al. focus exclusively on 27 papers about LLM trading agents with deep architectural analysis; this survey covers all LLM-in-finance applications (300+ papers) at broader scope. Nie et al. is the right reference for the full landscape; Ding et al. for trading agent architecture specifics.

---

## Strengths

- **Comprehensive scope**: six application categories, 40+ models, 20+ benchmarks, four challenge classes in a single coherent reference.
- **Practical orientation**: emphasizes both academic state-of-the-art and real-world constraints (regulatory, cost, interpretability).
- **Honest challenges section**: lookahead bias, data pollution, and signal decay are often overlooked; this survey names them explicitly.

---

## Weaknesses & Limitations

- **Breadth over depth**: with 300+ papers in 28 pages, individual sections are thin. The agent-based modeling section (one subsection) covers less than Ding et al.'s entire paper.
- **No empirical meta-analysis**: no aggregated comparison of model performance across papers; just narrative summaries.
- **Rapidly dated**: the field moves fast. Papers from mid-2024 onward (TradingAgents, FinCon, Orchestration Framework) are not covered.

---

## Key Takeaways

- Finance is LLMs' most promising real-world application domain: the data is predominantly text, decisions are high-stakes, and interpretability is valued — all matching LLM strengths.
- Six application categories form a progression: Linguistic Tasks → Sentiment → Time Series → Reasoning → Agents → Other. Each builds on capabilities established below.
- Agent-based modeling is the frontier: multi-agent systems that replicate investment firm structures (manager-analyst hierarchies, debate among specialists) are emerging as the dominant architecture.
- Four key unsolved challenges: lookahead bias in backtesting, LLM hallucinations in regulated financial outputs, signal decay from LLM adoption, and unclear legal accountability.
- ~100 citations in 18 months indicates high uptake; should be the first reference when framing any LLM-in-finance project.

---

## BibTeX

{% raw %}
```bibtex
@article{nie2024llmfinancesurvey,
  title={A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges},
  author={Nie, Yuqi and Kong, Yaxuan and Dong, Xiaowen and Mulvey, John M. and Poor, H. Vincent and Wen, Qingsong and Zohren, Stefan},
  journal={arXiv preprint arXiv:2406.11903},
  year={2024}
}
```
{% endraw %}

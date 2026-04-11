---
name: literature-review
description: "Core schema for the literature review knowledge base. Defines directory structure, wiki index format, wiki templates, cross-referencing rules, and conventions. Activates when working with wiki pages, PDFs, or the paper index."
user-invocable: false
paths: "raw/**, wiki/**, bibtex/**"
---

# Literature Review Knowledge Base — Schema

A persistent, compounding knowledge base for academic literature review, following the [LLM Wiki](https://github.com/karpathy/llm-wiki) pattern. Raw sources are immutable, the wiki is LLM-maintained, and this schema governs all operations.

## Directory Structure

```
.
├── raw/                   # Downloaded PDFs (IMMUTABLE — never modify)
│   └── {author}-{year}-{venue}.pdf
├── wiki/
│   ├── index.md           # Master index — SINGLE SOURCE OF TRUTH for all papers
│   ├── overview.md        # High-level synthesis
│   ├── log.md             # Append-only activity log
│   ├── papers/            # One page per ingested paper
│   ├── topics/            # Concept and topic pages
│   ├── methods/           # Method description pages
│   ├── benchmarks/        # Benchmark comparison pages
│   └── queries/           # Archived synthesis results
└── bibtex/
    └── references.bib     # Exported BibTeX citations
```

## Naming Convention

**PDF files:** `{first-author-lastname}-{year}-{venue}.pdf` (all lowercase, hyphenated)
- Examples: `hafner-2023-jmlr.pdf`, `wu-2022-corl.pdf`, `chen-2025-arxiv.pdf`
- If no formal venue, use `arxiv`
- On collision (same author/year/venue), append a letter: `smith-2024-neurips-a.pdf`, `smith-2024-neurips-b.pdf`

**Wiki pages:** Same base name as the PDF: `wiki/papers/{author}-{year}-{venue}.md`

## Index as Single Source of Truth (`wiki/index.md`)

`wiki/index.md` is the **only** place that tracks all papers and their status. There is no CSV or database.

```markdown
---
layout: default
title: "Literature Review Index"
---

# Literature Review Index

> Last updated: YYYY-MM-DD | Papers: N ingested, M downloaded | Topics: K | Methods: J

## Overview & Log

- [Overview](overview.md) — high-level synthesis of the research landscape
- [Activity Log](log.md) — append-only record of every operation

## Papers
| ID | Title | Year | Venue | 1st Author (Inst.) | Last Author (Inst.) | Citations | Status | PDF | Wiki | Notes |
|----|-------|------|-------|-------------------|---------------------|-----------|--------|-----|------|-------|
| P001 | Paper Title | 2024 | NeurIPS | Hafner (Google DeepMind) | Lillicrap (Google DeepMind) | ~4.2k | ingested | [PDF](../raw/hafner-2023-jmlr.pdf) | [Notes](papers/hafner-2023-jmlr.md) | [seminal] |
| P002 | Another Paper | 2025 | arXiv | Chen (MIT) | — | — | downloaded | [PDF](../raw/chen-2025-arxiv.pdf) | — | |

## Topics
(bulleted list with dual-format links and one-line descriptions)

## Methods
(bulleted list)

## Benchmarks
(bulleted list)

## Queries / Syntheses
(bulleted list with dates)
```

### Index column reference

| Column | Format | Source | Use `—` when |
|--------|--------|--------|--------------|
| `1st Author (Inst.)` | `LastName (Institution)` | arXiv abstract HTML affiliations; fallback: Semantic Scholar web page | affiliation not found |
| `Last Author (Inst.)` | `LastName (Institution)` | same as above | affiliation not found or single-author paper |
| `Citations` | `~Nk` (rounded) or exact count | WebSearch snippet "cited by N" or Semantic Scholar web page for the paper | count not found |
| `Notes` | space-separated tags | inferred from search signals | no applicable tag |

**Notes tags:** `[seminal]` — foundational/must-read work · `[survey]` — survey or review paper

**How to fetch institution data:** WebFetch `https://arxiv.org/abs/{id}` — the HTML page includes author affiliations. Extract the first and last author names and their affiliated institutions. If affiliations are absent on arXiv (common for older papers), search `"{title}" site:semanticscholar.org` and WebFetch the result page.

**How to fetch citation counts:** WebSearch `"{title}" citations` — snippets often contain "X Citations" or "Cited by X". If not in snippet, WebFetch the Semantic Scholar page for the paper. Use `~Nk` notation for large counts (e.g. `~12k`, `~450`). Use `—` only after one reasonable attempt fails.

### Paper status in the index table

| Status | PDF column | Wiki column | Meaning |
|--------|-----------|-------------|---------|
| `discovered` | `[arXiv]({url})` or `—` | `—` | Found via `/discover` — index row only, no local PDF, no wiki page |
| `downloaded` | `[PDF](../raw/{name}.pdf)` | `—` | PDF saved locally, not yet read |
| `ingested` | `[PDF](../raw/{name}.pdf)` | `[Notes](papers/{name}.md)` | Read, extracted, full wiki page created |

**Link format is universal.** `../raw/` and `papers/{slug}.md` resolve correctly in Obsidian (vault at repo root), GitHub.com (viewing wiki/index.md), and GitHub Pages (Jekyll serves wiki/index.md at /wiki/). Never change this format — no rewriting needed for hosting.

**Status lifecycle:** `discovered` → `/ingest discovered` (downloads PDF + creates wiki page) → `ingested`

**Commands:**
- `/discover [query]` — web search, adds `discovered` rows to index only
- `/ingest discovered` — ingests all `discovered` papers (download + wiki page)
- `/ingest all` — ingests all `downloaded` and `discovered` papers
- `/ingest <name>` — ingests one specific paper by ID/arXiv/slug

### Paper IDs

Auto-increment: `P001`, `P002`, ... Assign the next available ID by reading the last row in the index table.

## Paper Page Frontmatter

Detailed metadata lives in each paper's wiki page frontmatter (not in the index):

```yaml
---
title: "Full Paper Title"
type: paper
paper_id: P001
authors:
  - "Last, First"
  - "Last2, First2"
year: 2024
venue: NeurIPS
arxiv_id: "2401.12345"
url: "https://arxiv.org/abs/2401.12345"
pdf: "../../raw/hafner-2023-jmlr.pdf"
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
cites: []
cited_by: []
---
```

## Cross-Referencing (Dual Format)

Use **both** formats everywhere:
```
[[hafner-2023-jmlr]] ([DreamerV3](../papers/hafner-2023-jmlr.md))
```

Rules:
1. **Bidirectional**: If A cites B, both pages link to each other
2. **Frontmatter arrays**: Paper pages have `cites: []` and `cited_by: []`
3. **Every `[[wikilink]]` target must exist** as a file
4. **After every ingest**: Grep wiki for mentions of the paper's authors/methods/terms

## Wiki Page Templates

Templates are in [templates/](templates/). Read them when creating new pages:
- [templates/paper.md](templates/paper.md) — Per-paper summary pages
- [templates/topic.md](templates/topic.md) — Concept/topic pages
- [templates/method.md](templates/method.md) — Method description pages
- [templates/benchmark.md](templates/benchmark.md) — Benchmark comparison pages

## Log Format (`wiki/log.md`)

Append-only. `wiki/log.md` has YAML frontmatter. Each entry is a bullet with the date inline. Parseable with `grep "^\- \[" wiki/log.md | tail -10`.

**Structure:**
```markdown
---
title: "Activity Log"
layout: default
---

# Activity Log

- [YYYY-MM-DD HH:MM] **verb** -	subject — details
- [YYYY-MM-DD HH:MM] **verb** -	subject — details
```

**Append** (one line per operation):
```bash
echo "- [$(date "+%Y-%m-%d %H:%M")] **verb** -	details" >> wiki/log.md
```

**Entry format:** `- [YYYY-MM-DD HH:MM] **verb** -	subject — details`

## GitHub Pages compatibility

All wiki content is hosted via `/host` on a `web` branch with Jekyll (`jekyll-theme-slate`). Two rules keep everything compatible:

1. **BibTeX blocks must use `{% raw %}...{% endraw %}`** around every ` ```bibtex ``` ` block. Jekyll processes Liquid tags before markdown, so `{{Title}}` patterns in BibTeX break the build. The paper template already includes this. Always use it.
2. **The repo must be public** for free GitHub Pages. Private repos require a paid GitHub plan.

The link format (`../raw/`, `papers/slug.md`) works in Obsidian, GitHub.com, and GitHub Pages without any changes.

## Rules

1. **Never modify files in `raw/`.** Immutable sources of truth.
2. **Every operation updates `wiki/index.md` and appends to `wiki/log.md`.**
3. **`wiki/index.md` is the single source of truth** for paper status and catalog.
4. **Bidirectional links.** If A→B exists, add B→A.
5. **Every `[[wikilink]]` target must exist.** No dangling links.
6. **Prefer Semantic Scholar for metadata.** Use arXiv for PDFs. Use WebSearch for broad discovery.
7. **Read PDFs with `pages` parameter:** `Read(file, pages="1-20")`. Multiple passes for long papers.
8. **Dual-format links everywhere.**
9. **Every ingestion should touch multiple existing pages.** If only one file was created, cross-referencing was missed.
10. **Log everything.**

## API Reference

For detailed API endpoints and tools, see [reference.md](reference.md).

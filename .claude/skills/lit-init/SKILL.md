---
name: lit-init
description: "Bootstrap a literature review workspace. Searches for papers on the given topic, discovers them into the wiki index (no PDF downloads), and creates the full workspace structure. The starting point for any new research area."
argument-hint: "<research_topic> [--top <N>] [--year <range>]"
---

# Initialize Literature Review

Bootstrap and populate workspace for: **$ARGUMENTS**

> **TOOL RULES — READ FIRST:**
> - **Search:** Use **WebSearch** (Claude's built-in tool). Spawn searches **in parallel**.
> - **No Bash/curl during init.** No PDF downloads. Discovery only.
> - **No Python scripts ever.** No `python`, `pip`, no `.py` script files.
> - **No direct API calls.** Do not call `api.semanticscholar.org`, `export.arxiv.org/api`, or any API endpoint.
> - **Abstracts:** Use **WebFetch** on arXiv HTML pages (`https://arxiv.org/abs/XXXX`) — regular web pages, not API endpoints.

---

## Phase 1 — Create workspace

Create directories: `raw/`, `wiki/papers/`, `wiki/topics/`, `wiki/methods/`, `wiki/benchmarks/`, `wiki/queries/`, `bibtex/`.

Create `wiki/overview.md` (placeholder) and empty `bibtex/references.bib`.

Create `wiki/log.md` with this exact content:
```markdown
---
title: "Activity Log"
layout: default
---

# Activity Log
```

**Files created:** `wiki/log.md`, `wiki/overview.md`, `bibtex/references.bib`

---

## Phase 2 — Find papers

Use **WebSearch** as the sole discovery tool. Spawn these searches **in parallel**:

**Seminal & foundational (highest priority):**
- `{topic} seminal paper highly cited foundational`
- `{topic} most cited paper site:semanticscholar.org`
- `{topic} most cited paper site:scholar.google.com`
- `{topic} original paper introduced`

**Survey papers (required — must include at least one):**
- `{topic} survey 2024 2025`
- `{topic} survey arxiv`
- `{topic} comprehensive review overview 2024 2025`

**Recent top-venue work:**
- `{topic} arxiv 2024 2025`
- `{topic} NeurIPS 2024 2025`
- `{topic} ICML 2024 2025`
- `{topic} site:openreview.net ICLR 2025`
- `{topic} site:openreview.net NeurIPS 2024`
- `{topic} recent state of the art 2025`

Run additional domain-specific searches as appropriate:
- For robotics: `{topic} CoRL 2024 RSS 2024`
- For vision: `{topic} CVPR 2024 2025`

From each result, extract: title, first author, year, venue, arXiv ID (from URL or snippet), and any citation count signals (e.g. "cited by N" in snippets).

**Fetch abstracts, affiliations, and citations:** For each candidate paper with an arXiv ID, use WebFetch on `https://arxiv.org/abs/{id}` to get the full abstract **and author affiliations**. Batch these WebFetch calls in parallel. While fetching abstracts, also run parallel WebSearches for citation counts: `"{title}" citations site:semanticscholar.org` — snippets often contain "X Citations". If a citation count is not in the snippet, WebFetch the Semantic Scholar result page for that paper. Record for each paper: first author name + institution, last author name + institution, citation count.

Deduplicate by title similarity and arXiv ID.

**Ranking priority (in order):**
1. **Seminal / foundational works** — papers that introduced the field or a core concept, or are widely recognized as must-reads (regardless of age). Identify these from search snippets mentioning high citation counts, "introduced", "first proposed", "seminal", or from their appearance in multiple search results.
2. **Highly cited** — prefer papers with explicit citation-count signals (e.g. "cited by 5000+") over unknown-citation papers.
3. **Survey papers** — guarantee at least one survey/review paper in the final set, preferring the most recent one. If no survey is found in search results, add a dedicated search `{topic} survey` before proceeding.
4. **Venue prestige + recency** — NeurIPS/ICML/ICLR/CVPR top-tier venues, then arXiv recency.

Select top N (default 20, override with `--top`). The final set must always include: all clearly seminal works found + at least one survey paper + remaining slots filled by recent high-quality papers.

---

## Phase 3 — Create the index

Create `wiki/index.md` with the following structure. See the [schema](../literature-review/SKILL.md) for the full column format.

Start the file with:
```markdown
---
layout: default
title: "Literature Review Index"
---

# Literature Review Index

> Last updated: {today} | Papers: N discovered | Topics: 0 | Methods: 0

## Overview & Log

- [Overview](overview.md) — high-level synthesis of the research landscape
- [Activity Log](log.md) — append-only record of every operation

## Papers
| ID | Title | Year | Venue | 1st Author (Inst.) | Last Author (Inst.) | Citations | Status | PDF | Wiki | Notes |
|----|-------|------|-------|-------------------|---------------------|-----------|--------|-----|------|-------|
```

Then add a row for every discovered paper (status `discovered`, arXiv URL as PDF link):

Each row:
```
| P0XX | {Title} | {Year} | {Venue} | {FirstAuthor} ({1stInst.}) | {LastAuthor} ({LastInst.}) | {Citations} | discovered | [arXiv]({url}) | — | {Notes} |
```

- `1st Author (Inst.)` / `Last Author (Inst.)`: from arXiv HTML affiliations. Use `—` if not found.
- `Citations`: from Semantic Scholar search snippet or page. Use `~Nk` notation. Use `—` if not found.
- `Notes`: `[seminal]` and/or `[survey]` tags as appropriate. Leave blank otherwise.

List seminal and survey papers first in the table, then remaining papers by descending citation count / recency.

**Files created/modified:** `wiki/index.md`

---

## Phase 4 — Report

Tell the user how many papers were discovered. Show the index table. Suggest `/ingest all` to download PDFs and build wiki pages.

Append to `wiki/log.md`:
```bash
echo "- [$(date "+%Y-%m-%d %H:%M")] **init** -	\"{topic}\" — discovered N papers, workspace ready (no PDFs downloaded)" >> wiki/log.md
```

**Files modified:** `wiki/log.md`

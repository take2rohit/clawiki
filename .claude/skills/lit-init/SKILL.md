---
name: lit-init
description: "Bootstrap a literature review workspace. Creates a new topic branch, searches for papers, discovers them into the wiki index (no PDF downloads), and sets up the full workspace structure. The starting point for any new research area."
argument-hint: "<research_topic> [--top <N>] [--year <range>]"
---

# Initialize Literature Review

Bootstrap and populate workspace for: **$ARGUMENTS**

> **TOOL RULES — READ FIRST:**
> - **Search:** Use **WebSearch** (Claude's built-in tool). Spawn searches **in parallel**.
> - **Bash/curl:** Allowed for branch setup and **PDF downloads**. Do **not** use Python scripts.
> - **No Python scripts ever.** No `python`, `pip`, no `.py` files.
> - **No direct API calls.** Do not call `api.semanticscholar.org`, `export.arxiv.org/api`, or any API endpoint.
> - **Abstracts:** Use **WebFetch** on arXiv HTML pages (`https://arxiv.org/abs/XXXX`) — regular web pages, not API endpoints.

---

## Phase 0 — Create topic branch

Generate a short branch name from the topic, create the branch, and ensure `wiki/` and `raw/` are tracked on it.

```bash
# Generate branch name: lowercase, underscores, max 30 chars
BRANCH=$(echo "{topic}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//' | cut -c1-30)

# Abort if branch already exists
if git show-ref --verify --quiet refs/heads/$BRANCH; then
  echo "Branch '$BRANCH' already exists. Delete it or use a different topic name."
  exit 1
fi

git checkout -b $BRANCH

# Remove wiki/ and raw/ from .gitignore if present (main ignores them; topic branches track them)
sed -i '' '/^wiki\/$/d; /^raw\/$/d' .gitignore
git add .gitignore
```

Tell the user: "Created branch `{branch}` — all work for this review will live here."

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

## Phase 2.5 — Download PDFs

For each discovered paper with an arXiv ID, download the PDF in parallel batches (up to 10 concurrent downloads):

```bash
curl -sL "https://arxiv.org/pdf/{arxiv_id}" -o "raw/{slug}.pdf"
```

**Slug format:** `{first_author_lastname}-{year}-{venue_short}.pdf` (lowercase, e.g. `hafner-2021-iclr.pdf`).

After each batch, verify: `file raw/{slug}.pdf` must report "PDF document" and size > 0.
- If download succeeds: mark as `downloaded`, PDF column = `[PDF](../raw/{slug}.pdf)`
- If download fails: mark as `discovered`, PDF column = `[arXiv]({arxiv_url})`, log and continue

Do **not** hold up the batch for a single failed download.

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

> Last updated: {today} | Papers: 0 ingested, D downloaded, F discovered | Topics: 0 | Methods: 0

## Overview & Log

- [Overview](overview.md) — high-level synthesis of the research landscape
- [Activity Log](log.md) — append-only record of every operation

## Papers
| ID | Title | Year | Venue | 1st Author (Inst.) | Last Author (Inst.) | Citations | Status | PDF | Wiki | Notes |
|----|-------|------|-------|-------------------|---------------------|-----------|--------|-----|------|-------|
```

Then add a row for every paper:

Each row:
```
| P0XX | {Title} | {Year} | {Venue} | {FirstAuthor} ({1stInst.}) | {LastAuthor} ({LastInst.}) | {Citations} | {status} | {pdf_link} | — | {Notes} |
```

- `Status`: `downloaded` if PDF was saved to `raw/`, `discovered` if download failed or no arXiv ID
- `PDF`: `[PDF](../raw/{slug}.pdf)` if downloaded, `[arXiv]({arxiv_url})` if discovered-only
- `1st Author (Inst.)` / `Last Author (Inst.)`: from arXiv HTML affiliations. Use `—` if not found.
- `Citations`: from Semantic Scholar search snippet or page. Use `~Nk` notation. Use `—` if not found.
- `Notes`: `[seminal]` and/or `[survey]` tags as appropriate. Leave blank otherwise.

List seminal and survey papers first in the table, then remaining papers by descending citation count / recency.

**Files created/modified:** `wiki/index.md`

---

## Phase 4 — Report

Tell the user how many papers were found and downloaded. Show the index table. Suggest `/ingest all` to build wiki pages from the downloaded PDFs.

Append to `wiki/log.md`:
```bash
echo "- [$(date "+%Y-%m-%d %H:%M")] **init** -	\"{topic}\" — found N papers ({D} downloaded, {F} discovered-only), workspace ready" >> wiki/log.md
```

Create `README.md` at the repo root on this branch:
```markdown
# {Topic} Literature Review

> Branch: `{branch_name}` | Papers discovered: N | Created: {today}
> Live: run `/host` to publish

A Clawiki knowledge base. See `main` branch for full documentation and quick-start guide.
```

**Files modified:** `wiki/log.md`, `README.md`

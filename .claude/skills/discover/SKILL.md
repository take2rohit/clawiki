---
name: discover
description: "Discover new papers via web search. Adds discovered rows to wiki/index.md only — no wiki pages created. Run /ingest discovered afterwards to fully ingest them."
argument-hint: "[query] [--n=<count>] [--since=<year>] [--venue=<v1,v2>] [--cite-expand]"
disable-model-invocation: true
---

# Discover New Papers

Query: **$ARGUMENTS**

> **ONLY FILE CHANGED: `wiki/index.md`** — adds rows with status `discovered`. No wiki pages, no stubs, no other files touched.

> **TOOL RULES:**
> - **Search:** Use **WebSearch** (Claude's built-in tool). Spawn searches **in parallel**.
> - **Bash/curl:** Allowed for non-search operations. Do **not** use Python scripts.
> - **No Python scripts ever.** No `python`, `pip`, no `.py` files.
> - **No direct API calls.** Do not call `api.semanticscholar.org`, `export.arxiv.org/api`, or any other API endpoint.
> - **Abstracts:** Use **WebFetch** on arXiv/OpenReview HTML pages (`https://arxiv.org/abs/XXXX`) — regular web pages, not APIs.

---

## Phase 1 — Load context

1. Read `wiki/index.md`:
   - Infer research topic from existing paper titles if no `query` given
   - Build **dedup set**: all existing arXiv IDs and normalized titles (including any already-`discovered` rows)
   - Note the next available P-ID

2. Read `wiki/overview.md` for topic keywords and themes. Used to score relevance.

3. Parse `$ARGUMENTS`:
   - `query`: primary search terms (defaults to workspace topic)
   - `--n=<N>`: max new rows to add to index (default **20**)
   - `--since=<YYYY>`: only papers from year ≥ YYYY (default: 2 years back)
   - `--venue=<v1,v2,...>`: restrict to specific venues
   - `--cite-expand`: also search for papers citing existing ingested papers

---

## Phase 2 — Search (parallel WebSearch)

Spawn all searches **simultaneously**:

1. `{topic} arxiv 2025`
2. `{topic} NeurIPS 2025`
3. `{topic} ICML 2025`
4. `{topic} site:openreview.net ICLR 2025`
5. `{topic} site:openreview.net ICLR 2026`
6. `{topic} site:openreview.net NeurIPS 2025`
7. `{topic} arxiv 2026`
8. `{most distinctive term from overview.md} 2025 2026`

Add domain searches if relevant (robotics: CoRL/RSS, vision: CVPR/ECCV).

If `--cite-expand`: also run `"{paper title}" cited 2025 2026` for each ingested paper.

From each result extract: title, first author, year, venue, arXiv ID.

---

## Phase 3 — Fetch abstracts, affiliations, and citations (parallel)

For each candidate with an arXiv ID, fetch the HTML abstract page in parallel:
```
https://arxiv.org/abs/{arxiv_id}
```
Extract: full abstract, full author list (first and last author names + affiliated institutions), confirmed year/venue. Drop any candidate where abstract cannot be retrieved.

In the same parallel batch, for each candidate run a WebSearch:
```
"{title}" citations site:semanticscholar.org
```
Snippets usually contain "X Citations". If the count is not in the snippet, WebFetch the Semantic Scholar result page. Record citation count as `~Nk` notation (e.g. `~4.2k`, `~450`). Use `—` if not found after one attempt — do not hold up the batch.

---

## Phase 4 — Deduplicate, filter, rank

**Deduplicate:** Skip if arXiv ID or normalized title matches anything already in index (any status).

**Filter:** Drop if year < `--since`, venue doesn't match `--venue`, or no abstract.

**Relevance score (0–100):**

| Signal | Max pts | How |
|--------|---------|-----|
| Venue prestige | 30 | NeurIPS/ICML/ICLR/JMLR=25, CVPR/ECCV/CoRL/RSS/TMLR=20, Nature/Science=30, arXiv=5 |
| Keyword overlap | 35 | Topic keywords from `overview.md` in title+abstract, normalized |
| Recency | 20 | current_year=20, −1=15, −2=8, older=0 |
| Multi-search hits | 15 | 3+ searches=15, 2=10, 1=5 |

Sort descending, take top N.

---

## Phase 5 — Update index

Add one row per discovered paper to the Papers table in `wiki/index.md`:

```
| P0XX | {Title} | {Year} | {Venue} | {FirstAuthor} ({1stInst.}) | {LastAuthor} ({LastInst.}) | {Citations} | discovered | [arXiv]({url}) | — | |
```

- `1st Author (Inst.)` / `Last Author (Inst.)`: from Phase 3 arXiv HTML. Use `—` if not found.
- `Citations`: from Phase 3 Semantic Scholar search. Use `~Nk` notation. Use `—` if not found.
- `Notes`: leave blank for discover (no seminal/survey tagging here)
- PDF column: `[arXiv]({arxiv_url})` if arXiv ID known, otherwise `—`
- Wiki column: `—` (no page yet)
- Status: `discovered`

Update the header stats: add `, K discovered` to the papers count and update `Last updated`.

Append to `wiki/log.md`:
```bash
DATE=$(date +%Y-%m-%d)
grep -q "^## $DATE" wiki/log.md || printf "\n## $DATE\n" >> wiki/log.md
echo "- **discover** | \"{query}\" — added N discovered rows, {dup} duplicates skipped" >> wiki/log.md
```

---

## Phase 6 — Report to user

```
## Discovery Results — "{query}" ({today})

Searched: {N} parallel queries | Candidates: {total} | Duplicates skipped: {dup} | Added: {N}

| Score | ID   | Title                          | Year | Venue        | 1st Author (Inst.)        | Last Author (Inst.)       | Citations | arXiv      |
|-------|------|--------------------------------|------|--------------|---------------------------|---------------------------|-----------|------------|
|    94 | P011 | What Drives Success in JEPA-WM | 2025 | ICLR 2026    | LeCun (Meta AI)           | Assran (Meta AI)          | ~1.2k     | 2512.24497 |
|    88 | P012 | seq-JEPA                       | 2025 | NeurIPS 2025 | Chen (MIT)                | Isola (MIT)               | —         | 2505.03176 |
...

To ingest all discovered papers:  /ingest discovered
To ingest one:                     /ingest {arxiv_id or P-ID}
To discover more:                  /discover --cite-expand
```

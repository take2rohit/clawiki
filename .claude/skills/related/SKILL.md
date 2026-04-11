---
name: related
description: "Find papers related to a specific paper by exploring its references, citations, and similar keywords. Adds discovered rows to wiki/index.md only — no PDF downloads. Run /ingest discovered afterwards to fully ingest them."
argument-hint: "<paper_name>"
---

# Find Related Works

Find papers related to: **$ARGUMENTS**

> **ONLY FILES CHANGED: `wiki/index.md`, `wiki/log.md`** — adds rows with status `discovered`. No PDF downloads, no wiki pages created.

Read [reference.md](../literature-review/reference.md) for tools reference.

## Workflow

1. **Read the paper's wiki page** from `wiki/papers/{name}.md`. Extract key topics, methods, cited papers, and tags from frontmatter.

2. **Find papers cited by this one** (outbound): check the `cites` frontmatter and "Key References" section. For any not yet in the index, fetch the arXiv abstract page via WebFetch to get metadata including **author affiliations** (first and last author name + institution).

3. **Search for similar papers** via WebSearch using the paper's key methods, topics, and venue as queries. Run searches in parallel:
   - `"{method name}" arxiv 2024 2025 2026`
   - `"{key topic}" site:openreview.net 2025`
   - `"{first author}" follow-up work 2025 2026`

4. **Fetch citation counts** for each new candidate in parallel via WebSearch: `"{title}" citations site:semanticscholar.org`. Use the snippet count or WebFetch the Semantic Scholar page. Use `—` if not found after one attempt.

5. **Deduplicate** against existing `wiki/index.md` entries (any status).

6. **Add rows to `wiki/index.md`** for each newly found paper with status `discovered`:
   ```
   | P0XX | {Title} | {Year} | {Venue} | {FirstAuthor} ({1stInst.}) | {LastAuthor} ({LastInst.}) | {Citations} | discovered | [arXiv]({url}) | — | |
   ```
   Update header stats and `Last updated`.

7. **Present results** as a scored table (same format as `/discover` output).

8. **Append to `wiki/log.md`:**
   ```bash
   echo "- [$(date +%Y-%m-%d)] **related** | {slug} — found N related papers, added as discovered" >> wiki/log.md
   ```

9. **Recommend next commands:**
   ```
   **What to do next:**
   - `/ingest discovered` — download PDFs and build wiki pages for all discovered papers
   - `/ingest <P-ID>` — cherry-pick the most relevant paper to ingest first
   - `/discover --cite-expand` — broaden the search to papers citing your full corpus
   ```

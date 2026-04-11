# API & Tools Reference

## Finding Papers

> **CRITICAL RULE:** Use **WebSearch ONLY** for all paper discovery. **Never** call any API endpoint (Semantic Scholar, arXiv API, etc.) for search. **Never** use Bash for search. Spawn multiple WebSearch calls in parallel for maximum speed.

**WebSearch** is the sole tool for finding papers. Run multiple searches simultaneously:

```
{topic} arxiv 2025 2026
{topic} NeurIPS 2025
{topic} ICML 2025
{topic} site:openreview.net ICLR 2025
{topic} site:openreview.net ICLR 2026
{topic} survey 2024 2025
"{first author}" "{paper title}" follow-up 2025 2026
```

**Fetching abstracts:** After finding paper arXiv IDs via WebSearch, use **WebFetch** on the HTML abstract page to get full abstract text. This is a regular web page fetch — it is allowed:
```
WebFetch: https://arxiv.org/abs/{arxiv_id}          ← arXiv abstract page (HTML, allowed)
WebFetch: https://openreview.net/forum?id={id}      ← OpenReview paper page (HTML, allowed)
```

**Bash/curl** is allowed for PDF downloads and file operations. Do **not** use Python scripts.

**Never use:**
- `api.semanticscholar.org/...` — API endpoint, not allowed
- `export.arxiv.org/api/query` — API endpoint, not allowed
- Python scripts (`python`, `pip`, `.py` files) — not allowed

## Downloading PDFs

Try sources in this order:

1. **arXiv** (most reliable): If the paper has an arXiv ID (check S2's `externalIds.ArXiv` field), download from `https://arxiv.org/pdf/{arxiv_id}` using Bash curl.
2. **S2 open access PDF**: Check `openAccessPdf.url` — if it's a non-empty string, try that URL. Note: this field is empty for most papers even when they're on arXiv, so always check arXiv ID first.
3. **Skip**: If neither source has a PDF, skip the paper and note it in the log. Don't fail the whole batch.

## Tools Mapping

| Tool | Used For |
|------|----------|
| `WebSearch` | Primary paper discovery — most reliable for finding relevant papers |
| `WebFetch` | arXiv/OpenReview HTML pages for metadata (not API endpoints) |
| `Bash` | Downloading PDFs (`curl`), creating directories |
| `Read` | Reading PDFs (with `pages` param), reading wiki files |
| `Write` | Creating new wiki pages, index, log |
| `Edit` | Updating existing wiki pages, index entries |
| `Grep` | Finding cross-references, checking mentions across wiki |
| `Glob` | Finding wiki files by pattern |

## How the Knowledge Base Compounds

1. **Init**: `/lit-init <topic>` searches, discovers papers (no downloads), creates workspace
2. **Ingest**: `/ingest all` downloads PDFs, reads every paper, builds wiki pages, cross-references
3. **Discover**: `/discover [query]` web-searches for new papers, adds `discovered` rows to index only — lightweight horizon-scan
4. **Expand**: `/related <name>` finds second-order papers, adds as `discovered` rows
5. **Query**: `/ask` to extract insights, `/compare` for head-to-head analysis
6. **Gap-fill**: `/gaps` identifies what's missing → `/ingest <name>` to fill holes
7. **Maintain**: `/lint` periodically for consistency

**Discover → Ingest flow:**
```
/discover                    # scan web, add N discovered rows to index
/related <slug>              # find papers related to a key paper, add as discovered
/ingest discovered           # batch: download PDFs + ingest all discovered papers
/ingest <slug>               # cherry-pick one paper to ingest first
```

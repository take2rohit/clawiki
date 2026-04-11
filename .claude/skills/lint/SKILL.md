---
name: lint
description: "Health-check the literature review wiki for consistency issues. Finds broken links, orphan pages, dead references, missing cross-references, and index/wiki mismatches. Auto-fixes deterministic issues."
disable-model-invocation: true
---

# Wiki Health Check

## Deterministic Checks (auto-fix)

1. **Index ↔ Wiki consistency:**
   - Every paper with status `ingested` in `wiki/index.md` must have a page in `wiki/papers/`
   - Every wiki paper page must have a corresponding row in the index
   - Fix: add missing index rows, report missing wiki pages

2. **Index ↔ PDF consistency:**
   - Every `[PDF]` link in the index must point to an existing file in `raw/`
   - Every PDF in `raw/` should have a corresponding row in the index
   - Fix: report missing PDFs, add index rows for orphan PDFs

3. **Index completeness:**
   - Every `.md` file in `wiki/` subdirectories must appear in `wiki/index.md` (in the appropriate section: Papers, Topics, Methods, Benchmarks, Queries)
   - Every index entry must point to an existing file
   - Fix: add missing entries, remove stale entries pointing to nonexistent files

4. **Dead wikilinks:**
   - Scan all `[[wikilinks]]` in wiki pages
   - Every target must exist as a `.md` file
   - Fix: remove or replace dead wikilinks. If the target paper exists under a different name, update the link. If it doesn't exist at all, remove the link.

5. **Dead markdown links:**
   - Scan all `[text](path)` links in wiki pages
   - Every relative path must resolve to an existing file
   - Fix: remove or update broken links. **No link should point to a file that doesn't exist.**

## Heuristic Checks (report only)

6. **Orphan pages:** Wiki pages with zero inbound links from any other page. These are invisible — they exist but nothing points to them.

7. **One-way links (bidirectional gap):** Paper A's `cites` includes B, but B's `cited_by` doesn't include A (or vice versa). Fix by adding the missing backlink.

8. **Incomplete paper pages:** Missing key sections (Method, Results, BibTeX). Report which sections are missing for each paper.

9. **Unreferenced topics/methods:** Topics or methods mentioned in 3+ paper pages but lacking a dedicated `wiki/topics/` or `wiki/methods/` page.

10. **Stale index stats:** The header line counts (Papers: N ingested, M downloaded) don't match the actual row counts. Fix by recounting.

11. **Singleton tags (auto-fix):** Read the `tags:` frontmatter field of every ingested paper page. Count how many papers use each tag. Any tag used by only **one** paper is a singleton — remove it from that paper's frontmatter. A tag must appear in **at least 2** paper pages to be kept. Report which tags were removed and from which pages.

## Output

Produce a summary table:

| Check | Issues | Auto-fixed |
|-------|--------|-----------|
| Index ↔ Wiki | ... | ... |
| Index ↔ PDF | ... | ... |
| Dead wikilinks | ... | ... |
| Dead markdown links | ... | ... |
| ... | ... | ... |

Auto-fix deterministic issues. Ask user before deleting any files. Report heuristic issues as a prioritized action list.

**Append to `wiki/log.md`:**
```
## [{today}] lint | N issues found, M auto-fixed
```

**Recommend next commands** at the end of the report:
```
**What to do next:**
- `/gaps` — if issues were found, gap analysis will surface deeper structural problems
- `/ask "<question>"` — the wiki is now clean; query it with confidence
- `/discover "<topic>"` — expand coverage on thin areas surfaced by the health check
```

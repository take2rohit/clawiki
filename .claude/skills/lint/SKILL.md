---
name: lint
description: "Health-check the literature review wiki for consistency issues and wiki page quality. Finds broken links, orphan pages, dead references, missing cross-references, index/wiki mismatches, and validates every ingested wiki page against the standard template. Auto-fixes deterministic issues."
disable-model-invocation: true
---

# Wiki Health Check

**Important:** The wiki directory is named after the current branch. Get it with:
```bash
BRANCH=$(git branch --show-current)
```
All paths below use `$BRANCH/` (e.g., `$BRANCH/papers/`, `$BRANCH/index.md`).

## Deterministic Checks (auto-fix)

1. **Index ↔ Wiki consistency:**
   - Every paper with status `ingested` in `$BRANCH/index.md` must have a page in `$BRANCH/papers/`
   - Every wiki paper page must have a corresponding row in the index
   - Fix: add missing index rows, report missing wiki pages

2. **Index ↔ PDF consistency:**
   - Every `[PDF]` link in the index must point to an existing file in `raw/`
   - Every PDF in `raw/` should have a corresponding row in the index
   - Fix: report missing PDFs, add index rows for orphan PDFs

3. **Index completeness:**
   - Every `.md` file in `$BRANCH/` subdirectories must appear in `$BRANCH/index.md` (in the appropriate section: Papers, Topics, Methods, Benchmarks, Queries)
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

8. **Unreferenced topics/methods:** Topics or methods mentioned in 3+ paper pages but lacking a dedicated `$BRANCH/topics/` or `$BRANCH/methods/` page.

9. **Stale index stats:** The header line counts (Papers: N ingested, M downloaded) don't match the actual row counts. Fix by recounting.

10. **Singleton tags (auto-fix):** Read the `tags:` frontmatter field of every ingested paper page. Count how many papers use each tag. Any tag used by only **one** paper is a singleton — remove it from that paper's frontmatter. A tag must appear in **at least 2** paper pages to be kept. Report which tags were removed and from which pages.

---

## Wiki Page Quality Audit

**This is the most important check.** Every ingested paper page must conform to the standard template. Read [the paper template](../literature-review/templates/paper.md) before running this check.

### Skip already-verified pages

Read `$BRANCH/index.md`. If a paper row has `Quality` = `pass` in the Notes column AND the `Last linted` date in the index Lint section is recent (within 7 days), **skip quality checking that page**. Only re-check pages that:
- Have never been linted (no `[pass]` tag)
- Were modified since last lint (compare file mtime vs last linted date)
- Previously failed (`[quality:fail]` tag)

### Required sections checklist

For each ingested wiki page that needs checking, read the file and verify ALL of the following sections exist with substantive content (not just a heading):

| # | Section | Check | Min content |
|---|---------|-------|-------------|
| 1 | **Frontmatter** | Has `---` delimited YAML with `title`, `type`, `paper_id`, `authors`, `year`, `venue`, `arxiv_id`, `pdf`, `tags`, `cites`, `cited_by` | All fields present and non-empty |
| 2 | **One-line summary** | First line after frontmatter is a `>` blockquote | At least 20 words, mentions method name |
| 3 | **Problem & Motivation** | `## Problem & Motivation` or `## Problem and Motivation` heading exists | At least 3 sentences (50+ words) |
| 4 | **Core Idea** | `## Core Idea` or `## Core idea` heading exists | At least 2 sentences, no jargon-heavy equations |
| 5 | **How It Works** | `## How It Works` or similar heading with subsections | At least 100 words, has subheadings (`###`) |
| 6 | **Training** | Subsection under How It Works mentioning loss, dataset, or compute | Mentions at least one of: loss function, dataset, training detail |
| 7 | **Results** | `## Results` or `## Experiments` heading with at least one table | Has `\|` table rows with numbers |
| 8 | **Comparison to Prior Work** | `## Comparison` or `## Prior Work` heading | At least one `[[slug]]` or `[Name](../papers/` cross-reference |
| 9 | **Key Takeaways** | `## Key Takeaways` or `## Takeaways` heading | 3-5 bullet points |
| 10 | **BibTeX** | `{% raw %}` block containing `@article` or `@inproceedings` or `@misc` | Valid BibTeX entry |

### Scoring

Each section scores 1 point if present and meeting the minimum bar. Total = 10.

- **pass** (8-10/10): Mark `[quality:pass]` in the Notes column of the index row
- **partial** (5-7/10): Mark `[quality:partial]` — report which sections are missing/thin
- **fail** (0-4/10): Mark `[quality:fail]` — page needs rewrite, report all failures

### Quality report format

For each checked page, output one line:

```
| slug | Score | Missing/Thin sections |
|------|-------|-----------------------|
| smith-2025-arxiv | 10/10 | — |
| jones-2026-iclr | 7/10 | No Training subsection, No BibTeX, Thin Core Idea |
```

---

## Update Index with Lint Results

After running all checks, update `$BRANCH/index.md`:

1. **Add a Lint section** at the bottom of the index (after Queries/Syntheses), or update if it already exists:

```markdown
## Lint

> Last linted: {today} | Quality: {N} pass, {M} partial, {K} fail | Issues: {total} found, {fixed} auto-fixed
```

2. **Update Notes column** for each checked paper: append `[quality:pass]`, `[quality:partial]`, or `[quality:fail]` tag. If a tag already exists, replace it with the new result. Do not duplicate tags.

3. **Fix stale header stats** if counts are wrong.

---

## Output

Produce a summary table:

| Check | Issues | Auto-fixed |
|-------|--------|-----------|
| Index ↔ Wiki | ... | ... |
| Index ↔ PDF | ... | ... |
| Dead wikilinks | ... | ... |
| Dead markdown links | ... | ... |
| One-way links | ... | ... |
| Stale stats | ... | ... |
| Singleton tags | ... | ... |
| **Wiki Quality** | **{N} fail, {M} partial** | — |

Then show the full quality audit table for any non-pass pages.

Auto-fix deterministic issues. Ask user before deleting any files. Report heuristic issues as a prioritized action list.

**Append to `$BRANCH/log.md`:**
```bash
BRANCH=$(git branch --show-current)
echo "- [$(date "+%Y-%m-%d %H:%M")] **lint** -	N issues found, M auto-fixed, quality: {P} pass / {Q} partial / {R} fail" >> $BRANCH/log.md
```

**Recommend next commands** at the end of the report:
```
**What to do next:**
- `/ingest <slug>` — re-ingest any [quality:fail] papers to regenerate their wiki pages
- `/gaps` — if issues were found, gap analysis will surface deeper structural problems
- `/ask "<question>"` — the wiki is now clean; query it with confidence
- `/discover "<topic>"` — expand coverage on thin areas surfaced by the health check
```

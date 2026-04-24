---
name: ingest
description: "Read a downloaded paper PDF, extract key information, create a wiki page, and cross-reference with existing knowledge. Use '/ingest <name>' for a specific paper or '/ingest all' to process every downloaded paper."
argument-hint: "<paper_name | all | discovered>"
disable-model-invocation: true
---

# Ingest Paper

Ingest: **$ARGUMENTS**

Read the templates from [literature-review/templates/](../literature-review/templates/) before creating wiki pages.
Read [reference.md](../literature-review/reference.md) for download details.

> **TOOL RULES — READ FIRST:**
> - **Search:** Use **WebSearch** (Claude's built-in tool) for finding papers not in the index.
> - **Bash/curl:** Allowed for PDF downloads and file operations. Do **not** use Python scripts.
> - **No Python scripts ever.** No `python`, `pip`, no `.py` script files.
> - **No direct API calls.** Do not call `api.semanticscholar.org` or any other API endpoint.
> - **HTML pages:** WebFetch on arXiv/OpenReview HTML pages is allowed for metadata — web pages, not API endpoints.

**Important:** The wiki directory is named after the current branch. Get it with:
```bash
BRANCH=$(git branch --show-current)
```
All paths below use `$BRANCH/` (e.g., `$BRANCH/papers/`, `$BRANCH/index.md`).

## Resolve what to ingest

- **`/ingest discovered`**: Read `$BRANCH/index.md`. Find every row with Status `discovered`. Ingest each one sequentially: download PDF first, then create wiki page. This is the primary way to process papers found by `/discover`.
- **`/ingest all`**: Find every row with Status `downloaded` or `discovered`. Process `downloaded` first, then `discovered`.
- **`/ingest <name>`**: Match `<name>` against the index by P-ID, arXiv ID, slug, or partial title. If not in the index, use **WebSearch** to find the paper, download the PDF via Bash `curl`, add a row, then proceed.

### Handling `downloaded` papers

Papers with status `downloaded` already have a PDF in `raw/` but no wiki page. These are ready for immediate ingestion — skip to the per-paper ingestion workflow below.

### Handling `discovered` papers

Papers with status `discovered` have an index row (with arXiv URL) but **no PDF and no wiki page**. Before ingesting:

1. Read the index row — get the arXiv URL from the PDF column (format: `[arXiv](url)`).
2. **Download the PDF using Bash `curl`**: `curl -L https://arxiv.org/pdf/{arxiv_id} -o raw/{slug}.pdf`
3. Verify the file exists, is > 0 bytes, and is not an HTML error page (check with `file raw/{slug}.pdf`).
4. If download fails: log the failure, leave status as `discovered`, skip. Don't fail the batch.
5. If download succeeds: update index row PDF column to `[PDF](../raw/{slug}.pdf)`, then proceed with the normal per-paper ingestion workflow below (which sets status → `ingested` and creates the wiki page).

## Per-paper ingestion workflow

### Step 1 — Read the PDF thoroughly

Use the Read tool with `pages` parameter. Read the **first 20 pages**, then read additional chunks as needed to cover:
- Abstract and introduction (problem statement, motivation, claimed contributions)
- Related work section (what prior work exists and how this paper differs)
- Method/architecture section (full technical description)
- Experiments section (all results tables, ablations, baselines)
- Conclusion

**Do not stop at 20 pages if the method or results sections are incomplete.** A typical paper needs 2–3 Read passes.

### Step 2 — Extract everything needed for a self-contained wiki page

Extract and record all of the following before writing anything:

**Metadata:** title, authors, year, venue, arXiv ID, BibTeX entry

**Problem framing:**
- The exact task or challenge this paper addresses
- What prior methods do and specifically why they fail or fall short
- Why this failure mode matters in practice (the real-world consequence)
- What gap this paper fills

**Core idea:**
- The single insight or conceptual shift that makes this paper work
- Stated in plain language, no jargon — as if explaining to a smart non-expert
- What the authors realized that prior work hadn't

**Technical method:**
- Every major component: name, input, output, architecture choices
- The training setup: loss function(s) with their terms and purpose, dataset (size, domain, collection), compute (GPUs, training time), important hyperparameters
- Inference procedure: how the model is actually used at test time

**Results:**
- Every results table: method names, metric names, all scores
- Which baselines are compared against and what each baseline represents
- Ablation tables: what was removed, what dropped, what that proves
- The interpretation: what does each result actually mean? Is the gap large or small?

**Prior work comparison:**
- Every method that appears in their experiments or related work section
- For each: what it does, how this paper differs, which benchmark each wins on

**Limitations:** what the authors acknowledge doesn't work or is out of scope

### Step 3 — Write the wiki page

Create `$BRANCH/papers/{name}.md` using the [paper template](../literature-review/templates/paper.md). The goal is a page so complete that a reader never needs to open the PDF.

**Section-by-section quality bar:**

**One-line summary:** Make it specific. Bad: "proposes a new world model." Good: "introduces DIAMOND, a diffusion-based world model that achieves 1.46 mean human-normalized score on Atari 100k by replacing deterministic transitions with a denoising diffusion process in latent space."

**Problem & Motivation:** Don't just describe — explain the failure mode of prior methods concretely. State what breaks and under what conditions. This section answers: why did this paper need to exist?

**Core Idea:** One short paragraph, zero jargon. What did the authors figure out? Not what they built — what they *realized*. If someone read only this section, what would they understand?

**How It Works:** Name every component. Describe the data flow. Include key equations only when they clarify something non-obvious. Describe the training objective completely — what is each loss term doing? For the inference section: is there planning? beam search? rollout? be explicit.

**Results:** Never copy a table without interpreting it. For every result: is the gap large or small in context? What does it prove about the method? Summarize ablations in prose: "Removing X drops performance by Y%, which shows Z is the critical component."

**Comparison to Prior Work:** This is the most important section for making the page self-contained. For every competing method that appears in the paper:
- Read its existing wiki page if it exists (grep `$BRANCH/papers/` for the method name)
- Write 2–4 sentences: what does that method do, what does this paper do differently, which wins where and why
- Use `[[slug]] ([Name](../papers/slug.md))` dual-format links
- Include a comparison table covering: core approach, key difference, benchmark scores, when to prefer each

**Key Takeaways:** 3–5 bullets. If someone reads only this section, what must they walk away knowing? Be concrete — include the headline number, the main insight, the main limitation.

### Step 4 — Cross-reference with existing wiki

- Grep `$BRANCH/` for this paper's authors, method names, benchmark names, key acronyms
- For every related paper found:
  - Add this paper's slug to that paper's `cited_by` frontmatter (or `cites`, as appropriate)
  - Add that paper's slug to this paper's `cites` or `cited_by`
  - Add a prose link in the Comparison to Prior Work section if not already there
- Update every existing page that mentions this paper's topic to link here

### Step 5 — Update topic, method, and benchmark pages

**Topic pages** (`$BRANCH/topics/`): add this paper under the relevant topic. If no topic page exists for this paper's subject, create one.

**Method pages** (`$BRANCH/methods/`): if this paper introduces a new method, create a method page. Update existing method pages that this paper extends or compares against.

**Benchmark pages** (`$BRANCH/benchmarks/`): add this paper's scores to every benchmark it evaluates on. Create a benchmark page if one doesn't exist.

### Step 6 — Update index and logs

**`$BRANCH/index.md`:** set Status → `ingested`, Wiki → `[Notes](papers/{name}.md)`, update header stats (ingested count, last updated date), add any new topic/method/benchmark pages to their index sections.

Also backfill any `—` values in the `1st Author (Inst.)`, `Last Author (Inst.)`, and `Citations` columns for this row — the PDF and wiki page now provide authoritative data for institutions, and the Semantic Scholar citation count can be looked up via WebSearch `"{title}" citations site:semanticscholar.org` if still missing.

**`$BRANCH/overview.md`:** update if this paper materially shifts the landscape — new SOTA, new paradigm, new benchmark, or contradicts existing understanding.

**`$BRANCH/log.md`:** append one line:
```bash
BRANCH=$(git branch --show-current)
echo "- [$(date "+%Y-%m-%d %H:%M")] **ingest** -	{slug} — {title}, {N} sections written, cross-referenced {M} existing pages" >> $BRANCH/log.md
```

## Quality checklist before finishing

Before marking a paper as done, verify:
- [ ] One-line summary is specific (includes method name + headline result)
- [ ] Problem & Motivation explains *why prior methods fail*, not just what they do
- [ ] Core Idea is in plain language with no jargon
- [ ] How It Works covers every major component with input/output
- [ ] Training section has: loss terms, dataset name, compute
- [ ] Results section interprets every table (not just copies it)
- [ ] Ablations are summarized in prose with what each component proves
- [ ] Comparison to Prior Work has at least one entry per baseline in their experiments
- [ ] Every comparison uses `[[slug]] ([Name](path))` dual-format links
- [ ] Key Takeaways has 3–5 specific, concrete bullets
- [ ] `cites` and `cited_by` frontmatter arrays are filled
- [ ] At least 2 existing wiki pages were updated with backlinks

## When ingesting all

After all papers are ingested, run a final pass to update `$BRANCH/overview.md` with a synthesis that reflects the full corpus. Report totals (papers ingested, pages updated, new topic/method/benchmark pages created).

## After ingestion — recommend next commands

Always end with a "What to do next:" block tailored to what was just ingested:

```
**What to do next:**
- `/ask "<question about the ingested paper(s)>"` — query the knowledge base while it's fresh
- `/compare <slug1> <slug2>` — if two methods in the same space were ingested, compare them
- `/related <slug>` — expand coverage around the most important paper just ingested
- `/gaps` — (after ingesting 5+ papers) identify what's still missing from the knowledge base
- `/lint` — run a health check to catch any broken links introduced by cross-referencing
```

Only suggest commands that are meaningful given what was ingested. Skip `/compare` if only one paper was ingested. Suggest `/gaps` only when the corpus is large enough to warrant it (5+ ingested papers).

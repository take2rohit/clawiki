---
name: gaps
description: "Identify research gaps, missing citations, contradictions, and open questions across the literature review wiki. Use when assessing completeness of the literature review."
---

# Research Gap Analysis

**Important:** The wiki directory is named after the current branch. Get it with:
```bash
BRANCH=$(git branch --show-current)
```

## Workflow

1. **Read `$BRANCH/overview.md`** and all topic pages in `$BRANCH/topics/`.

2. **Read all paper pages** with status `ingested` (check `$BRANCH/index.md` — rows where Wiki column has a `[Notes]` link).

3. **Identify gaps across these dimensions:**

   **Missing coverage:**
   - Topics discussed in papers but lacking a dedicated wiki page
   - Methods referenced frequently but not yet ingested
   - Benchmarks used by multiple papers where we lack comparison data
   - Highly-cited papers in the area that we haven't ingested

   **Contradictions:**
   - Conflicting claims between papers
   - Results that disagree across papers on the same benchmark

   **Open questions:**
   - Research questions raised across papers that no one has addressed
   - Limitations acknowledged by multiple papers pointing to the same gap

   **Under-explored areas:**
   - Topics with few papers relative to their importance
   - Recent methods that lack independent evaluation or replication

4. **Output a prioritized list** of gaps with runnable commands for each:
   - Missing topic coverage → `/discover "{topic keyword}"` — search for papers on this gap
   - Discovered paper not yet ingested → `/ingest <P-ID or slug>` — download and read it now
   - Topic/method page missing → `/ingest <slug>` — ingestion will trigger page creation
   - Referenced paper not in index → `/discover "<paper title>"` — find and add it

   Format each gap as:
   ```
   ### [Priority] Gap title
   Description of what's missing and why it matters.
   → `/command arg` — what this will fix
   ```

5. **Append to `$BRANCH/log.md`:**
   ```bash
   BRANCH=$(git branch --show-current)
   echo "- [$(date "+%Y-%m-%d %H:%M")] **gaps** -	N gaps identified across M dimensions" >> $BRANCH/log.md
   ```

6. **Recommend next commands** at the end:
   ```
   **What to do next:**
   - `/ingest <highest-priority slug>` — fill the most critical gap
   - `/discover "<topic>"` — expand coverage on the thinnest area
   - `/lint` — run a health check while you're auditing the wiki
   ```

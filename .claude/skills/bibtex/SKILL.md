---
name: bibtex
description: "Export BibTeX citations from ingested papers to bibtex/references.bib."
argument-hint: "[paper_id | all]"
disable-model-invocation: true
---

# Export BibTeX

Export: **$ARGUMENTS**

**Important:** The wiki directory is named after the current branch. Get it with:
```bash
BRANCH=$(git branch --show-current)
```

## Workflow

1. **Determine scope:**
   - If a specific paper_id (e.g., `P005`): export that paper only
   - If `all` or no argument: export all ingested papers (rows in `$BRANCH/index.md` where Wiki column has a `[Notes]` link)

2. **Collect BibTeX entries:**
   - For each paper in scope: read its wiki page from `$BRANCH/papers/`
   - Extract the BibTeX block from the `## BibTeX` section
   - If no BibTeX block exists, construct one from frontmatter metadata

3. **Write `bibtex/references.bib`:**
   - Sort entries alphabetically by citation key
   - One blank line between entries
   - Include a header comment: `% Generated from literature review wiki — {today}`

4. **Report** count of entries exported.

5. **Append to `$BRANCH/log.md`:**
   ```bash
   BRANCH=$(git branch --show-current)
   echo "- [$(date "+%Y-%m-%d %H:%M")] **bibtex** -	exported N entries to bibtex/references.bib" >> $BRANCH/log.md
   ```

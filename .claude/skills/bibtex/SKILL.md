---
name: bibtex
description: "Export BibTeX citations from ingested papers to bibtex/references.bib."
argument-hint: "[paper_id | all]"
disable-model-invocation: true
---

# Export BibTeX

Export: **$ARGUMENTS**

## Workflow

1. **Determine scope:**
   - If a specific paper_id (e.g., `P005`): export that paper only
   - If `all` or no argument: export all ingested papers (rows in `wiki/index.md` where Wiki column has a `[Notes]` link)

2. **Collect BibTeX entries:**
   - For each paper in scope: read its wiki page from `wiki/papers/`
   - Extract the BibTeX block from the `## BibTeX` section
   - If no BibTeX block exists, construct one from frontmatter metadata

3. **Write `bibtex/references.bib`:**
   - Sort entries alphabetically by citation key
   - One blank line between entries
   - Include a header comment: `% Generated from literature review wiki — {today}`

4. **Report** count of entries exported.

5. **Append to `wiki/log.md`:**
   ```bash
   echo "- [$(date +%Y-%m-%d)] **bibtex** | exported N entries to bibtex/references.bib" >> wiki/log.md
   ```

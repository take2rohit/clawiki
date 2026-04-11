---
name: compare
description: "Generate structured comparison tables between papers or methods. Compares architecture, training, results across shared benchmarks, and trade-offs."
argument-hint: "<paper1_slug> <paper2_slug> [paper3_slug ...]"
---

# Compare Papers/Methods

Compare: **$ARGUMENTS**

## Workflow

1. **Read the wiki pages** for each specified paper slug from `wiki/papers/`.

2. **Extract per paper:** method name, architecture type, key design choices, loss function, training details, parameter count, computational cost, results per benchmark.

3. **Generate comparison table** covering:
   - Architecture type and key innovation
   - Loss function / training objective
   - Key technique or trick (e.g., collapse prevention mechanism)
   - Parameter count / computational cost
   - Results across all shared benchmarks
   - Strengths and weaknesses of each approach

4. **Write to `wiki/queries/compare-{slug1}-vs-{slug2}-{date}.md`** with YAML frontmatter:
   ```yaml
   ---
   title: "Comparison: Method A vs Method B"
   type: query
   created: {today}
   papers: [slug1, slug2]
   ---
   ```

5. **Update `wiki/index.md`** — add to Queries/Syntheses section.

6. **Append to `wiki/log.md`:**
   ```bash
   echo "- [$(date +%Y-%m-%d)] **compare** | {slug1} vs {slug2} — written to wiki/queries/{filename}" >> wiki/log.md
   ```

7. **Recommend next commands:**
   ```
   **What to do next:**
   - `/ask "<question about the compared methods>"` — drill deeper into a specific dimension
   - `/related <slug-of-winner>` — find more papers building on the stronger method
   - `/gaps` — identify which methods or benchmarks are missing from the comparison
   ```

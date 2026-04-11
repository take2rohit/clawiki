---
name: lit-switch
description: "List all literature review branches and switch to one. With no argument, shows all topic branches with paper counts and last activity. With a branch name, switches directly."
user-invocable: true
argument-hint: "[branch_name]"
---

# Switch Literature Review

Switch between topic branches: **$ARGUMENTS**

## Workflow

### Step 1 — List all topic branches

```bash
# Get all branches except main, with last commit date
git branch --format='%(refname:short) %(committerdate:short)' | grep -v '^main '
```

For each branch found, read its state without switching:

```bash
# Paper count from wiki/index.md (ingested + discovered)
git show {branch}:wiki/index.md 2>/dev/null | grep -c "| P[0-9]"

# Last log entry
git show {branch}:wiki/log.md 2>/dev/null | grep "^\- \[" | tail -1

# README first line (topic name)
git show {branch}:README.md 2>/dev/null | head -1
```

Present as a table:

```
## Your Literature Reviews

| Branch           | Topic                        | Papers | Last Activity        |
|------------------|------------------------------|--------|----------------------|
| world_models     | World Models Literature...   | 43     | [2026-04-10 00:00]   |
| jepa_research    | JEPA Research Literature...  | 12     | [2026-04-10 14:30]   |

Current branch: main
```

### Step 2 — Switch

**If `$ARGUMENTS` is provided:** switch directly:
```bash
git checkout {branch}
```

**If no argument:** after showing the table, ask the user which branch to switch to. Then run `git checkout {branch}`.

If the user is on `main` with uncommitted changes in `.claude/` or other skill files — warn but proceed (those files exist on all branches).

### Step 3 — Report after switching

After switching, show a summary of the new active branch:

```
Switched to: world_models

  Topic:   World Models Literature Review
  Papers:  22 ingested, 21 discovered (43 total)
  Last:    [2026-04-10 00:00] host - pushed main — 22 wiki pages, 22 PDFs

Run /ask, /ingest, /discover, or /host to continue working.
```

Read this from:
- `README.md` first line → topic
- `wiki/index.md` header → paper counts
- `wiki/log.md` last line → last activity

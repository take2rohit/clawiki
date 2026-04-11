---
name: host
description: "Publish the knowledge base to GitHub Pages by syncing wiki/ and raw/ to the web branch. Always works on the web branch — never touches main."
user-invocable: true
---

# Host Knowledge Base to GitHub Pages

> **BRANCH RULE: This skill ALWAYS operates on the `web` branch. Never commit `wiki/`, `raw/`, or any knowledge base files to `main`. Only `.claude/skills/` content belongs on `main`.**

## Workflow

### Step 1 — Confirm starting state

Run `git branch --show-current` to confirm we are on `main`. If on any other branch, warn the user and stop.

Run `git status` to check for uncommitted changes. If there are staged changes on `main`, warn the user before proceeding (do not commit or stash them — just warn).

### Step 2 — Switch to `web` branch

```bash
git checkout web 2>/dev/null || git checkout -b web main
```

If `web` already exists, check it out. If not, create it from `main`.

Then bring in the latest skills from `main`:

```bash
git merge main --no-edit -X theirs 2>/dev/null || true
```

`-X theirs` resolves conflicts by preferring main's version — we will immediately overwrite `.gitignore` in the next step anyway.

### Step 3 — Write `.gitignore` for the `web` branch

Overwrite `.gitignore` with this exact content (removes the `wiki/` and `raw/` ignore lines so they get tracked):

```
.DS_Store
bibtex/
.claude/settings.local.json
wiki/.obsidian/
.obsidian/
```

### Step 4 — Write `_config.yml` at repo root

Create or overwrite `_config.yml`:

```yaml
title: "Clawiki"
description: "A structured, cross-referenced literature review knowledge base maintained with Claude Code."
theme: minima
markdown: kramdown
kramdown:
  input: GFM
exclude:
  - ".claude/"
  - "Gemfile"
  - "Gemfile.lock"
defaults:
  - scope:
      path: ""
    values:
      layout: default
```

### Step 5 — Build root `index.md` from `wiki/index.md`

Read `wiki/index.md`. Write a root `index.md` at the repo root with:

1. A Jekyll front matter header at the top:
   ```yaml
   ---
   layout: default
   title: "Literature Review Index"
   ---
   ```

2. The full content of `wiki/index.md` (append after front matter), with the following link substitutions so paths resolve correctly from the root:
   - `../raw/` → `raw/`
   - `](papers/` → `](wiki/papers/`
   - `](topics/` → `](wiki/topics/`
   - `](methods/` → `](wiki/methods/`
   - `](benchmarks/` → `](wiki/benchmarks/`
   - `](queries/` → `](wiki/queries/`

   Apply substitutions using `sed` on the content before writing.

### Step 6 — Stage and commit

```bash
git add -A
git diff --staged --quiet || git commit -m "chore: sync knowledge base [$(date +%Y-%m-%d)]"
```

Only commits if there are actual changes (idempotent).

### Step 7 — Push to origin

```bash
git push -u origin web
```

### Step 8 — Return to `main`

```bash
git checkout main
```

### Step 9 — Report to user

Tell the user:
- The `web` branch was pushed to origin
- GitHub Pages URL format: `https://<username>.github.io/<repo>/`
- **First-time setup:** Go to repo Settings → Pages → Source: Deploy from branch → Branch: `web` → Folder: `/ (root)` → Save
- After setup, the site is live and every future `/host` run auto-updates it
- Note: `[[wikilinks]]` in paper pages render as plain text on the web (not clickable) — this is a Jekyll limitation without plugins, not a bug

### Notes

- PDFs in `raw/` are included on the `web` branch and linked from paper pages as `/raw/<slug>.pdf`
- The `web` branch `.gitignore` intentionally differs from `main` — do not sync them
- The `web` branch always includes everything in `main` (skills, README, CLAUDE.md) plus the knowledge base
- Append to `wiki/log.md` before switching branches:
  ```
  ## [{today}] host | pushed web branch to origin — {N} wiki pages, {M} PDFs
  ```

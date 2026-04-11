---
name: host
description: "Publish the knowledge base to GitHub Pages by syncing wiki/ and raw/ to the web branch. Always works on the web branch — never touches main. Fully autonomous: enables GitHub Pages automatically on first run."
user-invocable: true
---

# Host Knowledge Base to GitHub Pages

> **BRANCH RULE: This skill ALWAYS operates on the `web` branch. Never commit `wiki/`, `raw/`, or any knowledge base files to `main`. Only `.claude/skills/` content belongs on `main`.**

The wiki link format (`../raw/`, `papers/slug.md`) is already compatible with Obsidian, GitHub.com, and GitHub Pages — no link rewriting is needed. `wiki/index.md` is served at `/wiki/` and all relative links resolve correctly from there.

## Workflow

### Step 1 — Confirm starting state

```bash
git branch --show-current
git status
```

Confirm we are on `main`. If there are uncommitted changes, warn the user but continue — do not stash or commit them.

### Step 2 — Switch to `web` branch

```bash
git checkout web 2>/dev/null || git checkout -b web main
```

Bring in any new skills from `main`:

```bash
git merge main --no-edit -X theirs 2>/dev/null || true
```

`-X theirs` resolves conflicts by preferring `main`'s files. We immediately overwrite `.gitignore` next anyway.

### Step 3 — Write `.gitignore` for `web` branch

Overwrite `.gitignore` (removes `wiki/` and `raw/` ignore lines so they get tracked):

```
.DS_Store
bibtex/
.claude/settings.local.json
wiki/.obsidian/
.obsidian/
```

### Step 4 — Write `_config.yml`

Create or overwrite `_config.yml` at the repo root:

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

### Step 5 — Write static root `index.md` (first run only)

Check if a root `index.md` already exists with the redirect. If not, write it:

```html
---
layout: default
title: "Literature Review"
---
<meta http-equiv="refresh" content="0; url=wiki/">
<p>→ <a href="wiki/">Open the Literature Review Index</a></p>
```

This file is written **once** and never needs updating. The actual content is `wiki/index.md`, served at `/wiki/` — no copy, no link rewriting.

### Step 6 — Append to `wiki/log.md` (before switching branches)

Count wiki pages and PDFs:

```bash
find wiki/papers -name "*.md" | wc -l
find raw -name "*.pdf" | wc -l
```

Append:
```
## [{today}] host | pushed web branch — {N} wiki pages, {M} PDFs
```

### Step 7 — Stage and commit

```bash
git add -A
git diff --staged --quiet || git commit -m "chore: sync knowledge base [$(date +%Y-%m-%d)]"
```

Only commits if there are actual changes (idempotent).

### Step 8 — Push to origin

```bash
git push -u origin web
```

### Step 9 — Enable GitHub Pages automatically

```bash
REPO=$(gh repo view --json owner,name --jq '"\(.owner.login)/\(.name)"')
```

Try to enable Pages (first run), or update branch if already enabled:

```bash
gh api repos/$REPO/pages --method POST -f source[branch]=web -f source[path]=/ 2>/dev/null \
  || gh api repos/$REPO/pages --method PUT  -f source[branch]=web -f source[path]=/ 2>/dev/null \
  || true
```

Get the live URL:

```bash
gh api repos/$REPO/pages --jq '.html_url' 2>/dev/null
```

### Step 10 — Return to `main`

```bash
git checkout main
```

### Step 11 — Report to user

Tell the user:
- The `web` branch was pushed
- The live URL (e.g. `https://take2rohit.github.io/clawiki/`)
- GitHub Pages may take 1–2 minutes to build on first deploy
- Note: `[[wikilinks]]` in paper pages appear as plain text on the web (not hyperlinks) — Jekyll limitation, not a bug

### Notes

- PDFs in `raw/` are served at `/raw/<slug>.pdf` — paper pages link to them with `../../raw/` which resolves correctly
- `wiki/index.md` is the real homepage, served at `/wiki/`; the root `index.md` just redirects there
- The `web` branch `.gitignore` intentionally differs from `main`
- Running `/host` again is safe and idempotent — only changed files get committed

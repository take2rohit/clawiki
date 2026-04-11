---
name: host
description: "Publish the knowledge base to GitHub Pages by syncing wiki/ and raw/ to the web branch. Always works on the web branch — never touches main. Fully autonomous: enables GitHub Pages, polls the build, and fixes errors."
user-invocable: true
---

# Host Knowledge Base to GitHub Pages

> **BRANCH RULE: This skill ALWAYS operates on the `web` branch via a git worktree. The `main` working directory is NEVER touched — wiki/ and raw/ files are safe.**

The wiki link format (`../raw/`, `papers/slug.md`) is compatible with Obsidian, GitHub.com, and GitHub Pages with no rewriting. `wiki/index.md` is served at `/wiki/` and all relative links resolve correctly from there.

## Workflow

### Step 1 — Confirm starting state

```bash
git branch --show-current
git status
```

Confirm we are on `main`. If there are uncommitted changes, warn the user but continue.

**IMPORTANT:** Do NOT use `git checkout` to switch to the web branch — this would delete wiki/ and raw/ from the filesystem. Use a worktree instead (Step 2).

### Step 2 — Set up a git worktree for `web`

```bash
WORKTREE=$(mktemp -d)
git worktree add "$WORKTREE" web 2>/dev/null || git worktree add "$WORKTREE" -b web main
```

All subsequent steps operate inside `$WORKTREE`, not in the main working directory. The main working directory (with wiki/ and raw/) is untouched throughout.

### Step 3 — Sync wiki/ and raw/ into the worktree

```bash
rsync -a --delete wiki/   "$WORKTREE/wiki/"
rsync -a --delete raw/    "$WORKTREE/raw/"
```

### Step 4 — Write `.gitignore` in the worktree

```bash
cat > "$WORKTREE/.gitignore" << 'EOF'
.DS_Store
bibtex/
.claude/settings.local.json
wiki/.obsidian/
.obsidian/
EOF
```

### Step 5 — Write `_config.yml` in the worktree

```bash
cat > "$WORKTREE/_config.yml" << 'EOF'
title: "Clawiki"
description: "A structured, cross-referenced literature review knowledge base maintained with Claude Code."
remote_theme: pages-themes/hacker@v0.2.0
plugins:
  - jekyll-remote-theme
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
EOF
```

### Step 6 — Write static root `index.md` (once only)

Only write if it doesn't already exist in the worktree:

```bash
if [ ! -f "$WORKTREE/index.md" ]; then
cat > "$WORKTREE/index.md" << 'EOF'
---
layout: default
title: "Literature Review"
---
<meta http-equiv="refresh" content="0; url=wiki/">
<p>→ <a href="wiki/">Open the Literature Review Index</a></p>
EOF
fi
```

### Step 7 — Pre-flight: fix Jekyll Liquid conflicts in wiki files

Jekyll 3 (used by GitHub Pages) processes Liquid `{{` and `{%` tags before markdown rendering, even inside code fences. BibTeX entries often contain `{{Title}}` patterns that break the build.

Run this scan-and-fix inside the worktree before committing:

```bash
# Find any BibTeX blocks in wiki/papers/*.md that contain {{ and wrap with {% raw %}...{% endraw %} if not already wrapped
for f in "$WORKTREE"/wiki/papers/*.md "$WORKTREE"/wiki/queries/*.md; do
  if grep -q '{{' "$f" 2>/dev/null; then
    # Check if already wrapped
    if ! grep -q '{% raw %}' "$f"; then
      # Wrap the BibTeX block: add {% raw %} after ```bibtex and {% endraw %} after closing ```
      python3 -c "
import re, sys
content = open('$f').read()
# Wrap any bibtex code fence that contains {{ with raw/endraw
def wrap_bibtex(m):
    block = m.group(0)
    if '{{' in block and '{% raw %}' not in block:
        return '\n{% raw %}\n' + block.strip() + '\n{% endraw %}\n'
    return block
fixed = re.sub(r'\`\`\`bibtex.*?\`\`\`', wrap_bibtex, content, flags=re.DOTALL)
open('$f', 'w').write(fixed)
print(f'Fixed: $f')
"
    fi
  fi
done
```

### Step 8 — Append to `wiki/log.md` in the worktree

```bash
N=$(find "$WORKTREE/wiki/papers" -name "*.md" | wc -l | tr -d ' ')
M=$(find "$WORKTREE/raw" -name "*.pdf" | wc -l | tr -d ' ')
echo "## [$(date +%Y-%m-%d)] host | pushed web branch — $N wiki pages, $M PDFs" >> "$WORKTREE/wiki/log.md"
```

### Step 9 — Stage and commit in the worktree

```bash
cd "$WORKTREE"
git add -A
git diff --staged --quiet || git commit -m "chore: sync knowledge base [$(date +%Y-%m-%d)]"
```

### Step 10 — Push to origin

```bash
cd "$WORKTREE"
git push -u origin web
```

### Step 11 — Remove the worktree (clean up)

```bash
cd -   # back to repo root
git worktree remove "$WORKTREE" --force
```

### Step 12 — Enable GitHub Pages automatically

```bash
REPO=$(gh repo view --json owner,name --jq '"\(.owner.login)/\(.name)"')
gh api repos/$REPO/pages --method POST --field "source[branch]=web" --field "source[path]=/" 2>/dev/null \
  || gh api repos/$REPO/pages --method PUT  --field "source[branch]=web" --field "source[path]=/" 2>/dev/null \
  || true
```

### Step 13 — Poll the build and fix errors

Poll the GitHub Pages build until it completes (up to 5 minutes):

```bash
for i in $(seq 1 30); do
  RESULT=$(gh api repos/$REPO/pages/builds/latest --jq '{status:.status,error:.error.message}')
  STATUS=$(echo "$RESULT" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['status'])")
  ERR=$(echo "$RESULT" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['error'] or '')")
  echo "Build status: $STATUS — $ERR"
  if [ "$STATUS" = "built" ]; then break; fi
  if [ "$STATUS" = "errored" ]; then
    echo "Build failed: $ERR"
    break
  fi
  sleep 10
done
```

**If the build errors:**

1. Check the error message for clues (`$ERR`).
2. Common causes and fixes:
   - **"Liquid syntax error"**: A wiki file has `{{` or `{%` not inside `{% raw %}` — re-run Step 7, commit, push, and re-poll.
   - **"YAML exception"**: A wiki file has malformed front matter — find it with `grep -r '---' wiki/ | head`, inspect and fix.
   - **"File not found"**: A broken relative link — run `/lint` to detect and fix.
3. After fixing, re-run from Step 9 (using the same worktree is gone; use git archive to push a fix directly).

### Step 14 — Report to user

Tell the user:
- Build result: built ✅ or errored ❌
- Live URL from `gh api repos/$REPO/pages --jq '.html_url'`
- Note: `[[wikilinks]]` in paper pages appear as plain text on the web — Jekyll 3 limitation, not a bug

## Notes

- **No branch switching**: `git worktree` keeps wiki/ and raw/ safe in the main working directory at all times
- **Idempotent**: Running `/host` again only commits and pushes changed files
- **The `web` branch `.gitignore` differs from `main`** — this is intentional and maintained by the skill
- **After ingest**: new BibTeX entries may have `{{` — Step 7 auto-fixes them before every push

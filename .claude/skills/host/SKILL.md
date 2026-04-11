---
name: host
description: "Commit any uncommitted wiki/raw changes, push the current topic branch to GitHub Pages, and verify the build. Must be on a topic branch — refuses if on main."
user-invocable: true
---

# Host Knowledge Base to GitHub Pages

Publishes the current topic branch to GitHub Pages. Never runs on `main`.

## Workflow

### Step 1 — Verify branch

```bash
BRANCH=$(git branch --show-current)
if [ "$BRANCH" = "main" ]; then
  echo "Error: /host cannot publish from main. main is a template."
  echo "Switch to a topic branch first: /lit-switch"
  exit 1
fi
echo "Publishing branch: $BRANCH"
```

### Step 2 — Pre-flight: fix Jekyll Liquid conflicts

Jekyll 3 processes `{{` as Liquid before markdown, even inside code fences. It also processes ALL `.md` files at the repo root unless excluded. Ensure `CLAUDE.md` is in the `exclude` list in `_config.yml` — it contains `{% raw %}` examples that Jekyll will try to parse.

Scan wiki/papers for any BibTeX blocks with `{{` not already wrapped in `{% raw %}`:

```python
import re, glob

for path in glob.glob('wiki/papers/*.md') + glob.glob('wiki/queries/*.md'):
    content = open(path).read()
    if '{{' in content and '{% raw %}' not in content:
        fixed = re.sub(
            r'```bibtex.*?```',
            lambda m: '\n{% raw %}\n' + m.group(0).strip() + '\n{% endraw %}\n',
            content, flags=re.DOTALL
        )
        open(path, 'w').write(fixed)
        print(f'Fixed Liquid conflict: {path}')
```

### Step 3 — Append to wiki/log.md

```bash
N=$(find wiki/papers -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
M=$(find raw -name "*.pdf" 2>/dev/null | wc -l | tr -d ' ')
echo "- [$(date "+%Y-%m-%d %H:%M")] **host** -	pushed $BRANCH — $N wiki pages, $M PDFs" >> wiki/log.md
```

### Step 4 — Commit and push

```bash
git add -A
git diff --staged --quiet || git commit -m "chore: sync knowledge base [$(date "+%Y-%m-%d %H:%M")]"
git push -u origin $BRANCH
```

### Step 5 — Enable GitHub Pages (first run only)

```bash
REPO=$(gh repo view --json owner,name --jq '"\(.owner.login)/\(.name)"')
gh api repos/$REPO/pages --method POST --field "source[branch]=$BRANCH" --field "source[path]=/" 2>/dev/null \
  || gh api repos/$REPO/pages --method PUT  --field "source[branch]=$BRANCH" --field "source[path]=/" 2>/dev/null \
  || true
```

### Step 6 — Poll the build

```bash
for i in $(seq 1 30); do
  RESULT=$(gh api repos/$REPO/pages/builds/latest --jq '{status:.status,error:.error.message}')
  STATUS=$(echo "$RESULT" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['status'])")
  ERR=$(echo "$RESULT" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['error'] or '')")
  echo "[$i] $STATUS ${ERR:+— $ERR}"
  if [ "$STATUS" = "built" ] || [ "$STATUS" = "errored" ]; then break; fi
  sleep 10
done
```

If errored: check `$ERR`. Common causes:
- **Liquid syntax error** → a wiki file has unescaped `{{` — re-run Step 2, commit, push
- **YAML exception** → malformed front matter in a wiki file — run `/lint`

### Step 7 — Report and update README

```bash
URL=$(gh api repos/$REPO/pages --jq '.html_url' 2>/dev/null)
echo "Live at: $URL"
```

Write the live URL into this branch's `README.md`:
```bash
if [ -n "$URL" ] && [ -f README.md ]; then
  sed -i '' "s|> Live:.*|> Live: $URL|" README.md
  git add README.md
  git diff --staged --quiet || git commit -m "docs: update live URL in README"
  git push origin $BRANCH
fi
```

Tell the user: built ✅ or errored ❌, the URL, and the branch name.

## Notes

- `wiki/index.md` is served at `/wiki/` (not `/`). The root `index.md` redirects `/` → `/wiki/`. Both are needed.
- `[[wikilinks]]` render as plain text on the web — Jekyll 3 limitation.
- Running `/host` again is safe: only changed files get committed.
- GitHub Pages can only serve one branch at a time. Switching branches with `/lit-switch` then running `/host` will re-point Pages to the new branch.

---
name: host
description: "Commit any uncommitted wiki/raw changes, push main to GitHub Pages, and verify the build. Everything lives on main — no branch switching."
user-invocable: true
---

# Host Knowledge Base to GitHub Pages

Everything — wiki, PDFs, Jekyll config — lives on `main`. Running `/host` is just a commit + push + build check.

## Workflow

### Step 1 — Verify on main

```bash
git branch --show-current
```

Must be on `main`. If not, stop and warn the user.

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
N=$(find wiki/papers -name "*.md" | wc -l | tr -d ' ')
M=$(find raw -name "*.pdf" | wc -l | tr -d ' ')
echo "- [$(date "+%Y-%m-%d %H:%M")] **host** | pushed main — $N wiki pages, $M PDFs" >> wiki/log.md
```

### Step 4 — Commit and push

```bash
git add -A
git diff --staged --quiet || git commit -m "chore: sync knowledge base [$(date "+%Y-%m-%d %H:%M")]"
git push origin main
```

### Step 5 — Enable GitHub Pages (first run only)

```bash
REPO=$(gh repo view --json owner,name --jq '"\(.owner.login)/\(.name)"')
gh api repos/$REPO/pages --method POST --field "source[branch]=main" --field "source[path]=/" 2>/dev/null \
  || gh api repos/$REPO/pages --method PUT  --field "source[branch]=main" --field "source[path]=/" 2>/dev/null \
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

### Step 7 — Report

Print the live URL:
```bash
gh api repos/$REPO/pages --jq '.html_url'
```

Tell the user: built ✅ or errored ❌, and the URL.

## Notes

- `wiki/index.md` is served at `/wiki/` (not `/`). The root `index.md` redirects `/` → `/wiki/`. Both are needed.
- `[[wikilinks]]` render as plain text on the web — Jekyll 3 limitation.
- Running `/host` again is safe: only changed files get committed.

# Clawiki — Workspace Rules

## Branch Policy

| Branch | Purpose | Contains |
|--------|---------|----------|
| `main` | Skills only | `.claude/skills/`, `README.md`, `CLAUDE.md` |
| `web` | GitHub Pages hosting | Everything in `main` + `wiki/`, `raw/`, `_config.yml`, root `index.md` |

**Never commit `wiki/`, `raw/`, or `bibtex/` to `main`.** The `.gitignore` on `main` enforces this. The `web` branch has a different `.gitignore` that tracks these files.

## What goes where

```
main:
  .claude/skills/**   ← all skill definitions live here
  README.md
  CLAUDE.md
  .gitignore          ← ignores wiki/, raw/, bibtex/

web:
  (everything from main, merged in)
  wiki/**             ← knowledge base markdown
  raw/**              ← downloaded PDFs
  _config.yml         ← Jekyll config for GitHub Pages
  index.md            ← homepage (mirrors wiki/index.md, root-relative links)
  .gitignore          ← does NOT ignore wiki/ or raw/
```

## Hosting

Run `/host` to publish the knowledge base to GitHub Pages. It is fully autonomous — it pushes the `web` branch and enables GitHub Pages automatically using the `gh` CLI. No manual setup required.

## Hard rules

1. **Never `git add wiki/` or `git add raw/` on `main`.** They are gitignored on main for a reason.
2. **`/host` always works on the `web` branch.** It handles checkout/merge/push automatically.
3. **Never force-push `main`.**
4. **The `web` branch `.gitignore` must not be overwritten with main's `.gitignore`.** The skill handles this.
5. **Commit skills to `main` first, then run `/host`** — the skill merges main into web automatically.

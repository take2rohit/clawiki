# Clawiki — Workspace Rules

## Everything lives on `main`

`wiki/`, `raw/`, `_config.yml`, and `index.md` are all tracked on `main`. There is no separate publishing branch.

## Hosting

Run `/host` to commit any uncommitted changes, push to `main`, and verify the GitHub Pages build. The site at `http://rohitlal.com/clawiki/` rebuilds automatically on every push.

## Hard rules

1. **Never force-push `main`.**
2. **Never modify files in `raw/`.** PDFs are immutable sources of truth.
3. **`bibtex/` is gitignored** — export only, never committed.
4. **BibTeX blocks in wiki pages must use `{% raw %}...{% endraw %}`** — Jekyll 3 interprets `{{` as Liquid otherwise.
5. **The repo must be public** for free GitHub Pages.

# Clawiki — Workspace Rules

## Branch model

`main` is a clean template — it contains only skills, config, and docs. Never commit wiki content or `raw/` to `main`.

Each `/lit-init` creates a **topic branch** (e.g. `world_models`, `jepa_research`). All research work — wiki pages, PDFs, logs — lives on that branch. Use `/lit-switch` to move between reviews.

**The wiki directory is named after the branch.** On branch `world_models`, the wiki lives in `world_models/`. On branch `jepa_research`, it lives in `jepa_research/`. This means the GitHub Pages URL matches the branch name (e.g., `/clawiki/world_models/`).

## Hosting

Run `/host` from a topic branch to commit changes, push, and verify the GitHub Pages build. The site rebuilds automatically. `/host` refuses to run on `main`.

Use `/lit-switch` to pick which review is live — switching branches and running `/host` re-points GitHub Pages to that branch.

## Hard rules

1. **Never force-push `main`.**
2. **Never modify files in `raw/`.** PDFs are immutable sources of truth.
3. **`bibtex/` is gitignored** — export only, never committed.
4. **BibTeX blocks in wiki pages must use `{% raw %}...{% endraw %}`** — Jekyll 3 interprets `{{` as Liquid otherwise.
5. **The repo must be public** for free GitHub Pages.

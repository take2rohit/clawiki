# Clawiki

A literature review system built as [Claude Code](https://claude.ai/code) skills. Give it a topic — it finds papers, downloads PDFs, and builds a cross-referenced knowledge base you can query, compare, and synthesize.

Inspired by [karpathy/llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## Quick Start

```bash
# 1. Clone and open in Claude Code
git clone git@github.com:take2rohit/clawiki.git
cd claude-research
claude

# 2. Discover papers on your topic (no downloads — populates wiki/index.md)
/lit-init "world models for robotics"

# 3. Download PDFs and build wiki pages for all discovered papers
/ingest all

# 4. Ask a question against the knowledge base
/ask "what are the key differences between RSSM and transformer-based world models?"

# 5. Find related papers and pull them in
/related hafner-2023-dreamerv3

# 6. Compare methods side-by-side
/compare hafner-2023-dreamerv3 micheli-2022-iris

# 7. Identify gaps and missing coverage
/gaps

# 8. Health-check the wiki (fixes broken links, removes singleton tags)
/lint
```

After `/ingest all` you have a fully cross-referenced wiki. Every subsequent command enriches it.

## The Idea

Most LLM + papers workflows are stateless. Upload a PDF, ask questions, close the chat, gone. This is different.

The LLM **incrementally builds and maintains a persistent wiki** — structured markdown files between you and the raw PDFs. Ingesting a paper doesn't just summarize it: the LLM reads the full PDF, creates a wiki page, updates topic/method pages, adds bidirectional cross-references, and updates the index. Each paper makes the whole knowledge base richer.

## Commands

| Command | What it does | Files Modified |
|---------|-------------|----------------|
| `/lit-init <topic>` | Discover papers via web search, create workspace structure, populate index | `wiki/index.md` (created), `wiki/log.md` (created), `wiki/overview.md` (created), `bibtex/references.bib` (created) |
| `/discover [query]` | Web-scan for new papers → adds `discovered` rows to index (no PDF download) | `wiki/index.md`, `wiki/log.md` |
| `/ingest <name\|all\|discovered>` | Download PDF(s), build wiki pages, cross-reference | `raw/{name}.pdf` (downloaded), `wiki/papers/{name}.md` (created), `wiki/index.md`, `wiki/log.md`, relevant `wiki/topics/`, `wiki/methods/`, `wiki/benchmarks/` pages |
| `/ask <question>` | Query the knowledge base with cited answers | `wiki/queries/{slug}.md` (created), `wiki/log.md` |
| `/related <name>` | Find related papers → adds `discovered` rows to index (no PDF download) | `wiki/index.md`, `wiki/log.md` |
| `/compare <n1> <n2> ...` | Side-by-side method comparison table | `wiki/queries/compare-{slug}.md` (created), `wiki/log.md` |
| `/gaps` | Find missing coverage, broken cross-references | `wiki/log.md` (report only, no structural changes) |
| `/lint` | Health-check, fix broken links, remove singleton tags | `wiki/index.md`, `wiki/papers/*.md` (tag fixes), `wiki/log.md` |
| `/bibtex [id\|all]` | Export citations to `bibtex/references.bib` | `bibtex/references.bib`, `wiki/log.md` |
| `/host` | Publish the knowledge base to GitHub Pages via the `web` branch | `web` branch: `wiki/`, `raw/`, `_config.yml`, root `index.md` |

## Discovery Flow

```
/discover "JEPA world models 2025"   # scans NeurIPS/ICML/ICLR/arXiv — index only, no downloads
/ingest discovered                   # downloads PDFs + builds wiki pages for all discovered rows
```

Or use `--cite-expand` to find papers that directly cite your existing ingested papers:

```
/discover --cite-expand
```

## Paper Status

```
discovered  ──>  (ingest downloads PDF + creates wiki page)  ──>  ingested
downloaded  ──>  (ingest reads PDF + creates wiki page)      ──>  ingested
```

`wiki/index.md` is the single source of truth for all papers and their status.

## Structure

```
├── raw/                    # Downloaded PDFs (immutable)
├── wiki/
│   ├── index.md            # Master index
│   ├── overview.md         # Narrative synthesis
│   ├── log.md              # Activity log
│   ├── papers/             # One page per ingested paper
│   ├── topics/             # Concept pages
│   ├── methods/            # Method descriptions
│   ├── benchmarks/         # Leaderboard tables
│   └── queries/            # Saved comparisons and reviews
└── bibtex/
    └── references.bib
```

## Prerequisites

- [Claude Code](https://claude.ai/code) — CLI, desktop app, or IDE extension
- Nothing else. No API keys, no dependencies.

Clone, open in Claude Code, run `/lit-init`.

## Hosting on GitHub Pages

```bash
/host   # pushes wiki/ + raw/ to the web branch, rebuilds GitHub Pages
```

First-time setup: after the first `/host`, go to repo **Settings → Pages → Branch: `web` → Save**. Every subsequent `/host` auto-updates the site.

**Branch policy:** `main` contains only skills. `wiki/` and `raw/` only live on the `web` branch. See `CLAUDE.md` for the full rules.

## Tips

- **`wiki/index.md` is your home page.** Every paper links to `[PDF]` and `[Notes]`.
- **`/ingest all` after any operation that adds papers** (`/lit-init`, `/related`, `/discover`).
- **Papers are identified flexibly:** filename (`hafner-2023-jmlr`), P-ID (`P001`), or arXiv ID (`2301.04104`).
- **`raw/` is immutable.** Notes live in `wiki/`. PDFs are never modified.
- **Git-friendly.** Everything is plaintext markdown. Commit after each session.
- **`wiki/log.md`** is your audit trail — every operation is logged.

---
title: "Full Paper Title"
type: paper
paper_id: P001
authors:
  - "Last, First"
year: 2024
venue: NeurIPS
arxiv_id: "2401.12345"
url: "https://arxiv.org/abs/2401.12345"
pdf: "../../raw/lastname-2024-neurips.pdf"
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
cites: []
cited_by: []
---

# Full Paper Title

> **One sentence.** What does this paper do and what is the single most important result?

**Authors:** First Last, First2 Last2 | **Venue:** NeurIPS 2024 | **arXiv:** [2401.12345](https://arxiv.org/abs/2401.12345)

---

## Problem & Motivation

*What problem does this paper solve? Why was it unsolved or unsatisfying before? What breaks if you use prior approaches?*

Write 3–5 sentences that explain:
- The concrete task or challenge
- The specific failure mode of existing methods on this task
- Why this failure mode matters in practice

This section should make a reader understand *why this paper exists*.

---

## Core Idea

*The key insight in plain language — no jargon, no equations. If you had to explain this paper to a smart non-expert in one paragraph, what would you say?*

Write 2–4 sentences capturing the central contribution as an idea, not a method. What did the authors realize that others hadn't? What's the conceptual shift?

---

## How It Works

### Overview

High-level architecture or approach diagram in prose. Name the main components and how data flows through them.

### [Component 1 Name]

Detailed description. Include:
- Exact input/output
- Architecture choices and why (when stated by authors)
- Key equations if they clarify something non-obvious
- Parameter counts if relevant

### [Component 2 Name]

Same depth.

### Training

- Loss function(s) — write out the terms and what each one does
- Dataset: size, collection method, domain coverage
- Compute: GPUs/TPUs, training time, batch size
- Any tricks that matter (schedulers, warmup, gradient clipping, etc.)

### Inference

How is the model actually used at test time? Any planning, beam search, decoding tricks?

---

## Results

*Don't just copy tables — explain what the numbers mean.*

### [Benchmark 1]

| Method | Metric | Score | Notes |
|--------|--------|-------|-------|
| **This paper** | ... | **X** | |
| Prior SOTA | ... | Y | -Z% worse |
| Baseline | ... | W | |

Interpretation: What does this result mean? Is the gap large or small? What does it prove?

### [Benchmark 2]

Same structure.

### Ablations

Which components actually matter? What happens when you remove each one? Summarize the ablation table as prose: "Removing X drops performance by Y%, which shows that Z is critical."

---

## Comparison to Prior Work

*This is the section that makes the page self-contained. For each closely related method, explain what it does, what this paper does differently, and what the practical consequence is.*

| | This Paper | [Prior Method A] | [Prior Method B] |
|---|---|---|---|
| Core approach | | | |
| Key difference | | | |
| Benchmark (metric) | | | |
| When to prefer | | | |

**vs [[slug-of-related-paper-1]] ([Name](../papers/slug.md)):** 2–3 sentences. What does that paper do? What does this paper do instead? Which wins where and why?

**vs [[slug-of-related-paper-2]] ([Name](../papers/slug.md)):** Same format.

*(Add one entry per directly competing method that appears in the paper's experiments or related work.)*

---

## Strengths

- **Specific strength**: Why it matters / evidence from paper.
- **Specific strength**: ...

---

## Weaknesses & Limitations

- **Specific limitation**: What breaks, under what conditions, evidence.
- **Open question left unresolved**: ...

---

## Key Takeaways

3–5 bullet points. If someone reads only this section, what must they remember?

- ...
- ...
- ...

---

## BibTeX

```bibtex
@article{key2024,
  title={},
  author={},
  journal={},
  year={2024}
}
```

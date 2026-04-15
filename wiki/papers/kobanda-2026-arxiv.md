---
title: "Intrinsic-Energy Joint Embedding Predictive Architectures Induce Quasimetric Spaces"
type: paper
paper_id: P059
authors:
  - "Kobanda, Anthony"
  - "Radji, Waris"
year: 2026
venue: arXiv
arxiv_id: "2602.12245"
url: "https://arxiv.org/abs/2602.12245"
pdf: "../../raw/kobanda-2026-arxiv.pdf"
tags: [jepa, quasimetric, energy-function, least-action-principle, goal-conditioned-control, theory]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - assran-2023-cvpr
cited_by: []
---

# Intrinsic-Energy Joint Embedding Predictive Architectures Induce Quasimetric Spaces

> **One sentence.** If a JEPA's induced energy is defined as an intrinsic (least-action) energy over admissible trajectories, then it necessarily satisfies the triangle inequality and induces a quasimetric space, placing Intrinsic-Energy JEPAs within the same function class targeted by Quasimetric Reinforcement Learning (QRL).

**Authors:** Anthony Kobanda, Waris Radji | **Venue:** arXiv 2026 | **arXiv:** [2602.12245](https://arxiv.org/abs/2602.12245)

---

## Problem & Motivation

Joint-Embedding Predictive Architectures (JEPAs) learn representations by predicting target embeddings from context embeddings, inducing a scalar compatibility energy between inputs. Quasimetric Reinforcement Learning (QRL) learns goal-conditioned control through directed distances (cost-to-go) that satisfy a triangle inequality. Despite apparent connections -- both define scalar scores over state pairs -- the structural relationship between JEPA energies and QRL quasimetric values had not been formally established. Understanding this connection matters because it determines whether JEPA representations are inherently suitable for goal-conditioned planning, or whether additional structure must be imposed.

---

## Core Idea

The paper shows that if a JEPA's energy function is defined as an "intrinsic energy" -- the infimum of accumulated local effort over admissible trajectories connecting two states (following the least-action principle from physics) -- then this energy automatically satisfies reflexivity, identity of indiscernibles, non-negativity, and the triangle inequality, making it a quasimetric. This places Intrinsic-Energy JEPAs squarely within the hypothesis class of value functions that QRL is designed to learn. The connection is conditional: it holds specifically for JEPAs whose energy admits a least-action interpretation, not for all JEPAs unconditionally. The paper also proves that symmetric finite energies cannot represent directed reachability, motivating the natural asymmetry of intrinsic energies.

---

## How It Works

### Formal Framework

**Definition (Intrinsic Energy):** For a path-connected state space X, let L(x,v) >= c*||v|| be a local effort density. The intrinsic energy between states x and y is:

E(x, y) = inf_{gamma in Gamma(x->y)} integral_0^T L(gamma(t), gamma'(t)) dt

where Gamma(x->y) is the set of admissible C^1 trajectories from x to y.

### Theorem 1 (Main Result)

**Intrinsic Energy is a Quasimetric.** E is a quasimetric on X:
- **Non-negativity:** L >= 0 implies E >= 0
- **Reflexivity:** Constant trajectory gamma(t) = x gives E(x,x) = 0
- **Identity of indiscernibles:** L(x,v) >= c*||v|| ensures E(x,y) = 0 implies x = y
- **Triangle inequality:** By concatenation of trajectories: E(x,z) <= Act(gamma_xz) = Act(gamma_xy) + Act(gamma_yz) <= E(x,y) + E(y,z) + 2*epsilon for any epsilon > 0

### Proposition 1 (Asymmetry Is Generic)

If admissibility is directed (not all paths are reversible) OR local effort is anisotropic (L(x,v) != L(x,-v)), then E(x,y) != E(y,x) in general. This naturally models physical systems where going from state s to g may have different cost than g to s.

### Connection to Goal-Conditioned Control

In reaching-cost problems, the optimal cost-to-go V*(x,g) = inf_{gamma in Gamma(x->g)} Act(gamma) is exactly an intrinsic energy (Definition 2). Therefore:

**Corollary 1:** IE-JEPA energies are quasimetrics (Theorem 1 applied to JEPA-induced E)

**Corollary 2:** In goal-reaching problems where optimal cost-to-go is intrinsic, any IE-JEPA energy approximating this cost falls within QRL's quasimetric value function class

### Proposition 2 (Symmetry Fails for Directed Reachability)

If E is symmetric and E(x,y) < +infinity iff (x,y) is reachable, then the reachability relation must be symmetric. Hence no symmetric finite energy can represent one-way reachability -- justifying the need for asymmetric (quasimetric) energies.

---

## Results

This is a theoretical short article (5 pages) with no experiments. The contributions are:

1. **Formal proof** that intrinsic-energy JEPAs induce quasimetric spaces (Theorem 1)
2. **Equivalence** placing IE-JEPAs within the QRL hypothesis class (Corollary 2)
3. **Impossibility result** showing symmetric energies cannot encode directed reachability (Proposition 2)
4. A clear conceptual bridge between the JEPA representation learning framework ([[lecun-2022-openreview]] ([LeCun 2022](../papers/lecun-2022-openreview.md))) and the QRL control framework (Wang et al., 2023)

---

## Comparison to Prior Work

| | This Paper | Destrade et al. (2025) | QRL (Wang et al., 2023) |
|---|---|---|---|
| Focus | Structural theory (when is JEPA energy a quasimetric?) | Empirical (shape JEPA space for planning) | Learning framework (learn quasimetric value functions) |
| Contribution | Formal proof + impossibility result | Algorithms + experiments | Algorithms + experiments |
| Experiments | None | Yes | Yes |
| JEPA connection | Direct (defines IE-JEPA) | Uses JEPA representations | Indirect (targets same function class) |

**vs [[lecun-2022-openreview]] ([H-JEPA / LeCun 2022](../papers/lecun-2022-openreview.md)):** LeCun's position paper proposes JEPA as the architecture for learning world models, with energy-based inference for planning. This paper formalizes a precise condition (intrinsic energy) under which the JEPA-induced energy has the right geometric structure (quasimetric) for goal-conditioned planning. It validates LeCun's intuition that JEPAs can serve as world models for control, but with an important caveat: the energy must admit a least-action form.

**vs [[balestriero-2025-iclr]] ([LeJEPA](../papers/balestriero-2025-iclr.md)):** LeJEPA focuses on preventing representation collapse in JEPAs via SIGReg. This paper operates at a different level of abstraction -- not how to train JEPAs, but what geometric structure their learned energies possess. The two are complementary: LeJEPA ensures non-degenerate representations, while this work characterizes when those representations support quasimetric planning.

**vs [[assran-2023-cvpr]] ([I-JEPA](../papers/assran-2023-cvpr.md)):** I-JEPA is a concrete instantiation of the JEPA framework. This paper analyzes the theoretical properties of the energy landscape that I-JEPA-style models induce, showing that if the energy can be interpreted as a least-action functional, it automatically has quasimetric structure useful for control.

---

## Strengths

- **Clean theoretical contribution:** The proof is short, self-contained, and connects two previously separate literatures (JEPA and QRL) through the well-understood least-action principle
- **Precise conditionality:** Does not overclaim -- explicitly states the equivalence holds for intrinsic-energy JEPAs, not all JEPAs
- **Useful impossibility result:** Proposition 2 provides a principled argument for why asymmetric energies are necessary for goal-conditioned control with one-way reachability
- **Bridges physics and ML:** Uses classical variational calculus (least-action principle) to characterize learned energy landscapes, grounding ML abstractions in physical principles

---

## Weaknesses & Limitations

- **Purely theoretical:** No experiments validate whether practical JEPA models (I-JEPA, V-JEPA) actually learn energies that approximate intrinsic energies
- **Condition may not hold in practice:** The least-action form is an assumption. Standard JEPA training with L2 comparators and learned encoders does not guarantee the resulting energy is intrinsic in the required sense
- **Narrow scope (5-page note):** Does not address how to train JEPAs to satisfy the intrinsic-energy condition, or how to verify it post-hoc
- **No connection to hierarchical/temporal JEPAs:** The analysis is for single-step JEPA energies; extension to multi-scale or video JEPAs (V-JEPA) is not discussed

---

## Key Takeaways

- Intrinsic-energy JEPAs -- those whose compatibility score equals the infimum of accumulated local effort over admissible trajectories -- automatically induce quasimetric spaces satisfying the triangle inequality
- This places IE-JEPAs within the same value function class that Quasimetric RL targets, formally connecting LeCun's JEPA vision ([[lecun-2022-openreview]]) with goal-conditioned control theory
- Symmetric finite energies provably cannot represent directed reachability, motivating the natural asymmetry of intrinsic-energy JEPAs for planning in irreversible environments
- The result is conditional: it holds for JEPAs whose energy admits a least-action interpretation, not for all JEPAs -- an important caveat for practitioners choosing energy functions

---

## BibTeX

{% raw %}
```bibtex
@article{kobanda2026intrinsic,
  title     = {Intrinsic-Energy Joint Embedding Predictive Architectures Induce Quasimetric Spaces},
  author    = {Kobanda, Anthony and Radji, Waris},
  journal   = {arXiv preprint arXiv:2602.12245},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.12245}
}
```
{% endraw %}

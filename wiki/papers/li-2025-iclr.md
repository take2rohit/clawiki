---
title: "Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers"
type: paper
paper_id: P051
authors:
  - "Li, Xianhang"
  - "Huang, Chen"
  - "Li, Chun-Liang"
  - "Malach, Eran"
  - "Susskind, Josh"
  - "Thilak, Vimal"
  - "Littwin, Etai"
year: 2025
venue: ICLR 2026
arxiv_id: "2509.24317"
url: "https://arxiv.org/abs/2509.24317"
pdf: "../../raw/li-2025-iclr.pdf"
tags: [JEPA, video-SSL, frozen-teacher, compute-efficiency, SALT]
created: 2026-04-15
updated: 2026-04-15
cites:
  - lecun-2022-openreview
  - balestriero-2025-iclr
cited_by: []
---

# Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers

> **SALT (Static-teacher Asymmetric Latent Training)** is a two-stage video SSL method that decouples the JEPA pipeline into (1) training a teacher encoder with pixel reconstruction and (2) freezing it to supervise a student via latent prediction. SALT eliminates the EMA teacher, stop-gradient, and collapse-prevention machinery of V-JEPA/V-JEPA 2, while student models outperform V-JEPA 2 at matched pretraining FLOPs across diverse benchmarks, with SALT ViT-G reaching 76.1% on SSv2 and 87.2% on K400 under frozen evaluation.

**Authors:** Xianhang Li (Apple, intern), Chen Huang, Chun-Liang Li, Eran Malach, Josh Susskind, Vimal Thilak, Etai Littwin (Apple ML Research) | **Venue:** ICLR 2026 | **arXiv:** [2509.24317](https://arxiv.org/abs/2509.24317)

---

## Problem & Motivation

Video Joint Embedding Predictive Architectures (V-JEPA) learn generalizable off-the-shelf representations by predicting masked regions in latent space using an exponential moving average (EMA)-updated teacher. While EMA prevents representation collapse, it introduces several compounding problems:

1. **Coupled teacher-student architecture**: The teacher is a lagging copy of the student, so they must share the same architecture. This prevents using small, cheap teachers with large students and complicates scalable model selection.
2. **Hyperparameter brittleness**: The EMA schedule, stop-gradient placement, and virtual early stopping all require meticulous tuning. Small perturbations can destabilize training or cause representation collapse.
3. **Uninformative training loss**: Because teacher and student co-evolve, the V-JEPA loss is not a proper loss function --- it can decrease while representation quality degrades. This forces practitioners to rely on surrogate metrics (RankMe, downstream probing) for model selection.
4. **Compute inefficiency**: The EMA teacher consumes forward-pass FLOPs at every step without receiving gradient updates, and its quality is inseparable from the student's optimization trajectory.

These issues stem from the self-distillation design pioneered by BYOL (Grill et al., 2020): the stop-gradient on the target encoder and EMA weight updates are implicit regularizers against collapse, but they conflate the optimization of two separate objectives (producing good targets vs. learning good features). The authors challenge the assumption that such elaborate online dynamics are necessary at all.

---

## Core Idea

SALT proposes that a **frozen teacher suffices** for JEPA-based video representation learning. Rather than jointly evolving teacher and student through self-distillation, SALT breaks training into two independent stages:

- **Stage 1 (V-Pixel)**: Train a teacher encoder with a pixel-space reconstruction objective (like VideoMAE) using V-JEPA's multi-block masking scheme. This produces a frozen encoder whose weights never change again.
- **Stage 2 (Latent prediction)**: Freeze the Stage 1 teacher and train a fresh student encoder + predictor using the standard JEPA latent-prediction objective (L1 loss between student predictions and teacher embeddings of masked regions).

This decoupling yields three immediate benefits: (a) the two stages use proper, interpretable loss functions; (b) the teacher and student architectures are completely independent --- a small ViT-L teacher can supervise a ViT-G student; and (c) the student training loss correlates strongly with downstream accuracy (R^2 = 0.951), enabling model selection without supervised probing.

The central empirical surprise is that **sub-optimal, small teachers produce high-quality students**: a ViT-L teacher (303M params) trained for only 40k--80k steps yields students that outperform V-JEPA 2's much more expensive EMA-based pipeline. The compute budget should overwhelmingly favor the student, not the teacher.

---

## How It Works

### Stage 1: V-Pixel (Teacher Training)

The teacher is trained to reconstruct masked pixel patches --- identical in objective to VideoMAE but using V-JEPA's multi-block masking strategy rather than random tube masking. The multi-block scheme applies short-range and long-range spatial masks with a temporal mask, matching V-JEPA's hyperparameters (short-range scale 0.15, long-range scale 0.7, temporal scale 1). This masking choice matters: ablations (Figure 5) show multi-block masking achieves 72.5% average accuracy vs. 70.7% for random tube masking, establishing a new empirical finding since VideoMAE models typically use random-tube masking.

Training details: ViT backbones (ViT-B through ViT-G) with 16x16 spatial patches, tubelet size 2, 16 frames at stride 4. Input resolution 224x224. Trained with AdamW (batch size 3072) and cosine LR schedule. The teacher does not require any collapse prevention --- pixel reconstruction is a proper supervised loss.

### Stage 2: Latent Prediction (Student Training)

The frozen teacher provides target embeddings. The student encoder processes visible (unmasked) patches, and a lightweight predictor (12-layer ViT with width 384, ~22M params) maps the student's context-conditioned representations to predict the teacher's embeddings at masked positions. The objective is:

```
min_{theta, phi} E_{x,y} || g_phi(f_theta(x), delta_y) - stop_grad(f_tilde(y)) ||_1
```

where f_theta is the student encoder, g_phi is the predictor, f_tilde is the frozen teacher, x and y are disjoint regions of the input, and delta_y encodes the spatio-temporal positions of masked regions. The stop-gradient is trivially satisfied since the teacher is frozen (no gradients flow to it regardless).

Critically, the student architecture is fully independent from the teacher. A ViT-L teacher (303M) can train a ViT-G student (1.84B). The predictor's last layer projects to the teacher's embedding dimension (not the student's), bridging the architectural gap.

### Compute Budget Allocation

Total training steps are fixed and split between teacher (Stage 1) and student (Stage 2). The optimal allocation (Figure 7) overwhelmingly favors the student: the best configuration at 240k total steps is 40k teacher + 200k student, achieving 78.2% average accuracy. Even a 20k-step teacher + 220k-step student (77.6%) outperforms V-JEPA 2 at 240k steps (75.1%). The teacher needs to be "good enough," not optimal.

### Training Data

V-3.6M dataset: Kinetics-710 (657k clips), Something-Something-v2 (169k clips), and Panda70M subset (2.8M clips) = 3.63M total clips. This differs from V-JEPA/V-JEPA 2 which additionally use Howto100M and YT-Temporal-1B.

---

## Results

### Systematic Comparison with State-of-the-Art (Table 1)

Frozen backbone evaluation on SSv2 (16x2x3) and K400 (16x2x3):

| Method | Params | Total Compute | SSv2 | K400 |
|---|---|---|---|---|
| VideoMAEv2 | 1B | 2.2 | 56.1 | 82.5 |
| PE (Bolya et al., 2025) | 1.9B | -- | 55.4 | 88.5 |
| InterVideo2-1B | 1B | -- | 67.3 | 87.9 |
| VideoPrism | 1B | 2.0B | 68.5 | 87.6 |
| DINOv2 | 1.1B | 1.9B | 50.7 | 83.6 |
| SigLIP2 | 1.2B | 4B | 49.9 | 87.3 |
| V-JEPA 2 ViT-L | 300M | 1.4 | 68.2 | 83.8 |
| V-JEPA 2 ViT-H | 600M | 2.6 | 73.4 | 84.6 |
| V-JEPA 2 ViT-g | 1B | 5.3 | 75.3 | 86.6 |
| **SALT ViT-L** | 300M | **1.2** | 74.9 | 85.4 |
| **SALT ViT-H** | 600M | **1.5** | 75.4 | 86.0 |
| **SALT ViT-g** | 1B | **1.9** | 70.2 | 86.8 |
| **SALT ViT-G** | 2B | **2.6** | **76.1** | **87.2** |

All SALT models use a frozen ViT-L (300M) teacher. SALT ViT-L outperforms V-JEPA 2 ViT-L by +6.7% on SSv2 at lower total compute (1.2 vs. 1.4 x10^21 FLOPs). SALT ViT-G achieves the highest SSv2 score (76.1%) and K400 score (87.2%) in the table.

### Head-to-Head: SALT vs. V-JEPA 2 at Matched Steps (Figure 2)

With both methods trained on V-3.6M for the same total steps (240k), SALT ViT-L outperforms V-JEPA 2 ViT-L by **+2.3% average accuracy** across six benchmarks (K400, SSv2, Diving48, COIN, ImageNet-1K, Jester). Gains are largest on temporal understanding tasks: SSv2 (+4.8 pp), Diving48 (+5.9 pp).

### Scaling Behavior (Figure 2b)

SALT's compute-accuracy curve dominates V-JEPA 2. At every parameter count from ViT-B to ViT-L, SALT achieves higher average accuracy. With a fixed ViT-L teacher, scaling the student to ViT-L yields ~82% average accuracy vs. ~78% for V-JEPA 2 at similar FLOPs.

### Interpretable Model Selection (Figure 3)

Student training loss vs. downstream SSv2 accuracy shows near-linear correlation:
- Teacher trained 80k steps: R^2 = 0.951
- Teacher trained 40k steps: R^2 = 0.972
- Teacher trained 20k steps: R^2 = 0.984

This means practitioners can select the best student checkpoint by monitoring training loss alone --- no expensive downstream probing needed. In contrast, neither the teacher's loss nor RankMe (embedding rank) are predictive of student quality (Figure 9 in appendix).

### Intuitive Physics Understanding

Following Garrido et al. (2025), SALT models were evaluated on IntPhys, GRASP, and InfLevel datasets. Intuitive physics understanding emerges in SALT-trained models as it does in V-JEPA 2, indicating that the frozen teacher does not compromise the learning of higher-level physical reasoning.

---

## Ablations

### Training Data for Teacher (Figure 4)

Six datasets tested for training the teacher, with student always trained on V-3.6M. Key findings:
- **All teacher datasets produce students that exceed V-JEPA 2** (73.7% baseline), except ImageNet-1K (59.8% teacher, 62.8% student) --- images alone are insufficient for video teachers.
- K710 alone (72.3% teacher -> 77.2% student) nearly matches the full V-3.6M mixture (72.5% -> 77.4%).
- Teachers trained on video data are robust: even SSv2-only teachers (72.8% -> 77.3%) produce strong students.

### Teacher Masking Strategy (Figure 5)

Multi-block masking (V-JEPA style) consistently outperforms random tube masking (VideoMAE style) for both teacher quality and student quality. The (S+L) block masking achieves 72.5% teacher accuracy with 77.4% student accuracy. Random tube (2x) achieves only 70.7% teacher and 76.9% student.

### Teacher Model Size (Figure 6)

Counterintuitively, **larger teachers do not always produce better students**:
- ViT-L student: Best with ViT-L teacher (77.4%), worse with ViT-H (77.3%) and ViT-G (75.2%) teachers.
- ViT-G student: Best with ViT-L teacher (79.0%), worse with ViT-H (74.2%) teacher.

All students improve over their corresponding teachers, and all improve over the V-JEPA 2 baseline. The practical implication: use a teacher the same size or smaller than the student.

### Compute Allocation (Figure 7)

At fixed total compute (120k, 160k, or 240k total steps), the Pareto-optimal allocation always favors the student:
- 120k total: best at 40k teacher + 80k student (77.2%)
- 160k total: best at 80k teacher + 80k student or 40k + 120k (77.5%)
- 240k total: best at 40k teacher + 200k student (78.2%)

SALT curves dominate V-JEPA 2 at matched FLOPs across all allocation points.

---

## Comparison to Prior Work

| | **SALT** | **V-JEPA / V-JEPA 2** | **LeJEPA** | **VideoMAE** |
|---|---|---|---|---|
| Domain | Video | Video | Images | Video |
| Teacher | Frozen (Stage 1 V-Pixel) | EMA of student (online) | None (no teacher) | None (decoder reconstructs pixels) |
| Anti-collapse | Not needed (frozen teacher) | Stop-gradient + EMA | SIGReg (provable) | Not needed (pixel reconstruction) |
| Teacher-student coupling | Decoupled (any sizes) | Coupled (same architecture) | N/A | N/A |
| Training loss interpretable | Yes (R^2 > 0.95) | No (surrogate metrics needed) | Yes (Spearman 94.5%) | Yes (pixel MSE) |
| SSv2 (frozen, ViT-L) | 74.9 | 68.2 (V-JEPA 2) | -- | -- |
| K400 (frozen, ViT-L) | 85.4 | 83.8 (V-JEPA 2) | -- | -- |

**vs. [LeCun 2022 (JEPA position paper)](lecun-2022-openreview.md):** LeCun proposed JEPA as a general architecture for autonomous intelligence, with prediction in latent space as the key principle. SALT is a concrete video instantiation that validates a surprising corollary: the teacher providing the latent targets need not be dynamic or co-evolved --- a frozen, independently trained encoder suffices, and may even be preferable. This challenges the assumption implicit in LeCun's framework that the teacher and student must jointly improve.

**vs. V-JEPA (Bardes et al., 2024) and V-JEPA 2 (Assran et al., 2025):** V-JEPA and V-JEPA 2 are the primary baselines. Both use EMA-updated teachers coupled to the student architecture. SALT replaces EMA with a frozen teacher and achieves +6.7% SSv2 accuracy at ViT-L scale with lower total compute. The architectural decoupling is the key enabler: SALT can train a 2B-parameter student from a 303M teacher, while V-JEPA 2 requires the teacher to match the student size.

**vs. I-JEPA (Assran et al., 2023):** I-JEPA applies the same EMA teacher-student paradigm to images. SALT's insight that frozen teachers suffice could in principle apply to image JEPA as well, though this paper evaluates only video.

**vs. [LeJEPA](balestriero-2025-iclr.md) (Balestriero & LeCun, 2025):** LeJEPA eliminates anti-collapse heuristics via a provable regularizer (SIGReg) and removes the teacher entirely. SALT takes a complementary approach: it keeps the teacher but freezes it, eliminating the EMA dynamics while retaining the latent-prediction objective. Both papers converge on the conclusion that elaborate online student-teacher dynamics are unnecessary. LeJEPA provides theoretical foundations (isotropic Gaussian optimality); SALT provides empirical evidence from video at scale. LeJEPA does not evaluate on video.

**vs. MVD (Masked Video Distillation, Wang et al., 2023c):** MVD also uses a two-stage pipeline with frozen teachers, but it (a) requires separate image and video teachers from strong pretrained models, (b) relies on fine-tuning for downstream evaluation, and (c) uses a teacher the same size or larger than the student. SALT uses a single video teacher, evaluates under frozen backbone protocol, uses a smaller teacher, and provides ablations showing teacher quality is surprisingly unimportant.

---

## Strengths

- **Simplicity and reproducibility**: Two-stage pipeline with standard objectives (pixel reconstruction + latent L1 prediction). No EMA schedule, no stop-gradient tuning, no virtual early stopping. Reduces implementation complexity substantially.
- **Compute efficiency**: SALT ViT-G (2B params) achieves 76.1% SSv2 at 2.6 x10^21 FLOPs; V-JEPA 2 ViT-g (1B params) achieves 75.3% at 5.3 x10^21 FLOPs. Even accounting for teacher pretraining cost, SALT dominates the Pareto frontier.
- **Architectural decoupling**: A 303M teacher can supervise a 1.84B student. This is impossible in V-JEPA 2 where teacher and student must share architecture.
- **Interpretable model selection**: R^2 > 0.95 correlation between student loss and downstream accuracy eliminates the need for expensive downstream probing during training.
- **Robustness to teacher quality**: Counterintuitive "weak-teacher, strong-student" finding. Sub-optimal teachers (trained on limited data, for few steps, or at small scale) still yield state-of-the-art students. This dramatically reduces the cost of the teacher stage.
- **Comprehensive ablations**: Systematic study of training data, masking strategy, teacher size, and compute allocation --- each independently informative.

---

## Weaknesses & Limitations

- **Two-stage pipeline overhead**: Although total compute is lower, the two-stage design requires managing separate training runs and checkpoint selection for the teacher, adding workflow complexity.
- **Teacher quality characterization is incomplete**: The paper identifies the "weak-teacher, strong-student" effect but does not fully explain the mechanism. Why does a ViT-L teacher produce better ViT-G students than a ViT-G teacher? This is acknowledged as future work.
- **Different training data from baselines**: SALT uses V-3.6M while V-JEPA 2's published results use different data mixtures including Howto100M and YT-Temporal-1B. The authors re-train V-JEPA 2 on V-3.6M for fair comparison, but this means the comparison is against their reproduction, not the official V-JEPA 2 numbers.
- **Performance plateaus at scale**: The authors note that performance plateaus as model size grows, likely reflecting data limits of the V-3.6M dataset. Larger pretraining sets may extend the scaling trend.
- **No fine-tuning evaluation**: All results use frozen backbone protocol. Fine-tuning results, which may show different trade-offs, are not reported.
- **Modest gains on appearance benchmarks**: While SALT shows large improvements on temporal understanding tasks (SSv2: +6.7%), gains on appearance-based K400 are smaller (+1.6% at ViT-L), suggesting the frozen teacher particularly helps with motion/temporal features.

---

## Key Takeaways

- **A frozen teacher suffices for JEPA video SSL**: The elaborate EMA-based self-distillation in V-JEPA/V-JEPA 2 is unnecessary. A simple pixel-reconstruction teacher, once frozen, provides targets that produce stronger student representations at lower total compute.
- **Compute should overwhelmingly favor the student**: The optimal teacher-student compute split is roughly 1:5 (40k teacher + 200k student at 240k total). Short, cheap teacher training is preferable to long, expensive teacher training.
- **Sub-optimal teachers yield strong students**: A ViT-L teacher produces better students than ViT-H or ViT-G teachers. The student's learning process amplifies the teacher's signal beyond the teacher's own quality --- challenging the assumption that better teachers always produce better students.
- **Training loss becomes a reliable model-selection signal**: SALT's decoupled design restores the loss-quality correlation (R^2 > 0.95) that is absent in EMA-based JEPAs, enabling practical hyperparameter search and checkpoint selection.
- **Convergent evidence with LeJEPA**: Both SALT and [LeJEPA](balestriero-2025-iclr.md) independently conclude that elaborate online student-teacher dynamics and EMA-based collapse prevention are unnecessary for high-quality JEPA representations. SALT shows this empirically for video; LeJEPA proves it theoretically for images. Together, they suggest the field can move toward simpler, more principled JEPA training recipes.

---

## BibTeX

{% raw %}
```bibtex
@inproceedings{li2025rethinking,
  title={Rethinking {JEPA}: Compute-Efficient Video {SSL} with Frozen Teachers},
  author={Li, Xianhang and Huang, Chen and Li, Chun-Liang and Malach, Eran and Susskind, Josh and Thilak, Vimal and Littwin, Etai},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  note={arXiv:2509.24317}
}
```
{% endraw %}

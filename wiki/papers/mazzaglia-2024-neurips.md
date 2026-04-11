---
title: "GenRL: Multimodal-foundation World Models for Generalization in Embodied Agents"
type: paper
paper_id: P020
authors:
  - "Mazzaglia, Pietro"
  - "Verbelen, Tim"
  - "Dhoedt, Bart"
  - "Courville, Aaron"
  - "Rajeswar, Sai"
year: 2024
venue: NeurIPS 2024
arxiv_id: "2406.18043"
url: "https://arxiv.org/abs/2406.18043"
pdf: "../../raw/mazzaglia-2024-neurips.pdf"
tags: [world-model, foundation-model, multi-task, embodied-AI, model-based-rl]
created: 2026-04-10
updated: 2026-04-10
cites:
  - hafner-2019-icml
  - hafner-2021-iclr
  - hafner-2023-arxiv
cited_by: []
---

# GenRL: Multimodal-foundation World Models for Generalization in Embodied Agents

> **GenRL** connects the latent space of a generative world model for RL with a pretrained vision-language model (InternVideo2) using vision-only data, enabling task specification via language or visual prompts without language annotations, and achieves a normalized score of 0.80 across 35 tasks while enabling data-free policy learning for unseen tasks.

**Authors:** Pietro Mazzaglia, Tim Verbelen, Bart Dhoedt, Aaron Courville, Sai Rajeswar | **Venue:** NeurIPS 2024 | **arXiv:** [2406.18043](https://arxiv.org/abs/2406.18043)

---

## Problem & Motivation

Scaling RL to multiple tasks requires complex reward design for each task — a costly, expert-intensive process. Language-conditioned foundation VLMs offer a natural interface for task specification, but their adoption in embodied RL faces two obstacles: (1) embodied domains typically lack the multimodal (language-annotated video) data needed to fine-tune VLMs or to use CLIP-style rewards reliably, and (2) prior approaches using CLIP similarity as a reward signal often fail without domain-specific fine-tuning and require training the agent from scratch on each new task. The fundamental gap is between the VLM's representation (trained on internet images/videos of humans) and the embodied agent's observation space (simulated locomotion or manipulation). GenRL bridges this gap without any language annotations, by connecting and aligning the VLM's joint video-language embedding with a generative world model's latent space using only visual data.

---

## Core Idea

GenRL learns a Multimodal-Foundation World Model (MFWM) by taking a pretrained vision-language model (InternVideo2) and learning two small networks — a connector and an aligner — that map VLM visual embeddings into the world model's latent space. The connector maps VLM embeddings to world model latent state predictions (bridging the two spaces), while the aligner maps language embeddings near the corresponding visual embeddings to close the modality gap without language-image pairs. At inference, a task prompt (image, video, or text) is processed through the VLM and aligner, translated into latent targets, and an actor-critic policy is trained in imagination to match those targets. Crucially, this architecture also enables fully data-free policy learning: after pretraining the MFWM, the policy can be learned by sampling latent states internally from the world model — no real observations, actions, or rewards from the environment are needed.

---

## How It Works

### Overview

GenRL has three phases: (1) train a task-agnostic generative world model on unlabeled interaction data; (2) learn connector and aligner networks to connect the VLM's representation with the world model's latent space; (3) given a task prompt, embed it with the VLM + aligner, compute target latent states with the connector, and train an actor-critic in imagination to match those targets.

### World Model (MFWM Base)

A DreamerV3-style generative world model:
- **Encoder:** q_φ(s_t | x_t) — maps observations to discrete latent states s ∈ S (independent categorical distributions)
- **Decoder:** p_φ(x_t | s_t) — reconstructs observations
- **Sequence model:** h_t = f_φ(s_{t-1}, a_{t-1}, h_{t-1}) — linear GRU cell for temporal dynamics
- **Dynamics predictor:** p_φ(s_t | h_t) — predicts next latent from hidden state

The encoder and decoder are NOT conditioned on the sequence model's hidden state h_t, unlike standard RSSMs. This decouples single-frame encoding from temporal dynamics, keeping the latent state s_t a representation only of the current observation. This design makes the encoder a "probabilistic visual tokenizer" grounded in the embodied environment.

**World model loss:**
L_φ = Σ_t D_KL[q_φ(s_t|x_t) ‖ p_φ(s_t|s_{t-1}, a_{t-1})] − E_{q_φ(s_t|x_t)}[log p_φ(x_t|s_t)]

(KL dynamics loss + reconstruction loss, trained with straight-through estimation for discrete latents.)

**Hyperparameters:** Batch size 48, sequence length 48, GRU recurrent units 1024, CNN multiplier 48, dense hidden units 1024, 4 MLP layers. Trained on V100 GPUs with 16GB VRAM; MFWM pretraining takes ~5 days for 500k gradient steps.

### Latent Connector

The connector p_ψ(s_{t:t+k} | e) learns to predict sequences of world model latent states from a VLM visual embedding e^(v) = f^(v)_PT(x_{t:t+k}):

L_conn = Σ_t D_KL[p_ψ(s_t | s_{t-1}, e) ‖ sg(q_φ(s_t | x_t))]

The connector has the same architecture as the world model's sequential dynamics model (GRU-based). It is trained with stop-gradient on the world model encoder, so it learns to predict the latent dynamics that the encoder would produce, given only the VLM visual embedding as input.

This is the inverse of the "reversed connector" (WM-CLIP baseline): instead of predicting VLM embeddings from world model states, GenRL predicts world model latent states from VLM embeddings.

### Representation Aligner

The aligner f_ψ(e^(l)) learns to map language embeddings e^(l) = f^(l)_PT(y) close to their visual embedding counterparts e^(v), without requiring language-image pairs:

L_align = ‖e^(v) − f_ψ(e^(l))‖²₂

**Key insight for language-free training:** Multimodal VLMs trained with contrastive learning have a "modality gap" where visual and language embeddings lie on different regions of the hypersphere. Prior work injects noise into visual embeddings during connector training so language embeddings (which the noise pulls toward the visual space) can generalize. GenRL instead learns an explicit aligner that maps from language embeddings toward their visual counterparts, trained using only visual embeddings with added noise. The aligner is a small U-Net with a bottleneck half the size of the 768-dimensional VLM embedding.

This two-step approach (connector trained on visual embeddings, aligner mapping language→visual) allows the connector to be trained more accurately (no noise injection) and the aligner to be retrained cheaply if noise levels change.

### Task Learning in Imagination (RL Objective)

Given task prompt τ (image, video, or text), the MFWM generates target latent states:
e_task = f_PT(τ) → aligner → connector → s^task_{t+1} ~ p_ψ(s_{t+1} | e_task)

The GenRL reward for policy learning is the cosine similarity between linear projections of dynamically generated and target latent states:

r_GenRL = cos(g_φ(s^dyn_{t+1}), g_φ(s^task_{t+1}))

where g_φ is the first linear layer of the world model decoder. The actor-critic policy π_θ(a_t | s_t) is trained to maximize this reward over imagined trajectories using the DreamerV3 actor-critic formulation (two-hot return targets, λ-returns).

**Temporal alignment (best-matching trajectory):** VLM embeddings cover k=8 frames, while the agent's policy imagines much longer trajectories. The initial state when the agent starts may not match the target sequence's initial state (e.g., agent is lying down, target shows running). GenRL uses best-path decoding (inspired by CTC in speech recognition): slide the target sequence along the imagined trajectory's time axis and align at the timestep t_a that maximizes cosine similarity with b=8 initial target states. Before t_a, use target initial states; after t_a, use the k-step offset target states.

### Data-Free Policy Learning

In data-free mode, initial latent states are sampled internally without accessing any real observations:
1. Sample random discrete latent states uniformly
2. Mix with states generated by the connector from randomly sampled VLM embeddings (connector-generated states have more coherent structure)
3. Warm up with 5 steps of the GRU using a mix of trained policy and random actions
4. Roll out policy in imagination; compute GenRL rewards from target embeddings

Data-free policy learning converges within ~30 minutes of training (10k gradient steps), eliminates CPU-GPU data transfers, and halves total training time compared to offline RL mode.

---

## Results

### Language-to-Action In-Distribution (21 Tasks, Offline RL)

Tasks: Walker (6), Quadruped (3), Stickman (2), Cheetah (2), Kitchen (4), tested across 10 seeds. Scores rescaled: 0 = random policy, 1 = expert agent.

| Method | Overall Score |
|--------|--------------|
| **GenRL** | **0.80 ± 0.02** |
| WM-CLIP-V | 0.70 ± 0.04 |
| TD3-V | 0.41 ± 0.05 |
| TD3+BC-V | 0.24 ± 0.02 |
| IQL-V | 0.29 ± 0.02 |
| WM-CLIP-I | 0.30 ± 0.02 |
| TD3-I | 0.23 ± 0.04 |

GenRL outperforms all model-free and model-based baselines across locomotion (Walker, Quadruped, Cheetah, Stickman) and manipulation (Kitchen) domains. Particularly strong gains in dynamic locomotion tasks (walking, running, quadruped movements). Kitchen domain shows some static tasks where other methods occasionally match or exceed GenRL (due to GenRL's video embeddings implying motion even for static targets).

### Language-to-Action Generalization (Unseen Tasks)

Evaluated on tasks not present in training data, GenRL achieves the best overall performance across all domains, substantially outperforming all model-free baselines. GenRL's performance approaches that of specialized agents on Walker and Quadruped (dynamic tasks). The advantage is most pronounced in quadruped and cheetah domains.

### Video-to-Action

GenRL accepts short video clips as task prompts and generalizes across different visual styles: drawings, realistic footage, AI-generated images, different camera viewpoints, and different morphologies (e.g., cheetah motions specified by a spider video). Performance with video prompts is comparable to language prompts on the same tasks.

### Data-Free Policy Learning (35 Tasks)

| Method | Overall Score (35 tasks) |
|--------|------------------------|
| **GenRL (offline)** | **~0.80** |
| **GenRL (data-free)** | **~0.75** |
| WM-CLIP-V (offline) | ~0.67 |
| TD3-V (offline) | ~0.40 |

Data-free GenRL achieves performance close to offline GenRL (~5% lower overall), and outperforms TD3-V offline by ~35%. The data-free variant even outperforms offline GenRL in the Kitchen domain (Appendix K), likely because offline RL overfits to limited kitchen data.

### Ablations

**Aligner ablation:** Removing the aligner (feeding language embeddings directly to connector) drops overall score from 0.76 to 0.17 — the aligner is essential for language generalization. Adding the aligner to the WM-CLIP baseline provides only marginal gains (0.67→0.68), confirming the aligner's benefit is specific to the connector-based architecture.

**Temporal alignment ablation:** Best-matching trajectory (0.80 overall) > best-matching initial state > no alignment. The full sliding-window approach is most beneficial for dynamic tasks where the agent's starting state differs from the target sequence's starting state.

**Training data distribution:** Using all training data (exploration + task-specific) achieves best performance (0.65 normalized on Walker). Exploration data alone achieves 0.38; task-specific data without exploration achieves lower generalization. Diverse data distribution is critical for the MFWM to capture a wide range of behaviors.

---

## Comparison to Prior Work

| Method | Task spec. | Language annot. needed | Model-based | Data-free | Multi-task |
|--------|-----------|----------------------|-------------|-----------|-----------|
| CLIP reward (VLM-as-reward) | Language | Yes (fine-tuning) | No | No | No |
| LIV | Language/video | Yes (language-image pairs) | No | No | No |
| WM-CLIP (baseline) | Language | No (vision only) | Yes | No | Partial |
| **GenRL** | Language/video/image | No (vision only) | Yes | Yes | Yes |

**WM-CLIP (reversed connector, based on [[hafner-2023-arxiv]] ([DreamerV3](../papers/hafner-2023-arxiv.md))):** The ablation baseline most similar to GenRL. It learns a "reversed connector" predicting VLM embeddings from world model states, then uses cosine similarity in VLM space as a reward. GenRL uses the opposite direction (VLM embeddings → world model states), which is more accurate for training (predicting the well-structured world model latents is easier than predicting the VLM's high-dimensional embedding). Overall: GenRL 0.76 vs. WM-CLIP 0.67 on the aligner ablation setting.

**LIV** (Ma et al., 2023): Learns language-image-value functions using contrastive learning for reward specification. Requires language-image paired data for fine-tuning. Without domain-specific fine-tuning, LIV scores near 0 on Kitchen tasks; GenRL achieves 0.76 on the same tasks without any language annotations.

**Model-free VLM-reward approaches (TD3-V, IQL-V, TD3+BC-V):** Use VLM cosine similarity directly as a reward for offline RL. Must train the full agent from scratch per task. GenRL outperforms all by large margins, especially on generalization tasks, because model-based world models enable richer imagination-based learning from the same data.

**DynaLang** (Lin et al., 2023): Shares language-vision representations in the world model, but trains the world model with language predictions as part of the objective. GenRL's world model remains task-agnostic (no language involvement in world model training), with language/vision grounding done via the connector-aligner modules.

---

## Strengths
- First framework to connect and align a pretrained VLM with a generative world model for RL using vision-only data — no language annotations required in embodied domains.
- Enables data-free policy learning: after MFWM pretraining, new task policies can be learned in ~30 minutes with no access to any real data, opening a path toward foundation models for embodied RL.
- Multi-modal task specification: accepts text, images, or videos as prompts and generalizes across visual styles, viewpoints, and morphologies.
- The connector-aligner design cleanly separates two distinct problems (visual grounding and modality alignment), making each tractable without cross-modal annotated data.
- Strong multi-task generalization: GenRL approaches specialized expert performance on held-out locomotion tasks without any task-specific data.
- Decoder visualization of latent targets provides an interpretable view of what behavior the model infers from a prompt, enabling rapid prompt iteration.

## Weaknesses & Limitations
- Performance on static tasks (e.g., Kitchen) is weaker than on dynamic tasks, because InternVideo2 (a video model) encodes motion even for static behaviors, producing misleading latent targets.
- Requires a large, diverse pre-training dataset — complex behaviors not covered during MFWM training cannot be reproduced (demonstrated by Kitchen generalization failures for behaviors outside training distribution).
- Reconstruction quality degrades in complex visual environments (e.g., Minecraft), limiting the accuracy of latent targets from prompts in open-ended settings.
- Inherits VLM biases: prompts must sometimes specify the embodiment explicitly (e.g., "spider running fast") to avoid VLM defaulting to human-motion representations.
- Dependent on the modality gap in VLMs — the aligner addresses this but does not eliminate it entirely, and would need to be retrained for different VLMs.
- Precise fine-grained manipulation tasks (beyond the 4 kitchen tasks) are currently out of reach due to the difficulty of exploring meaningful states in manipulation environments.

## Key Takeaways
- Connecting a pretrained video-language model to a world model's latent space via vision-only data (connector + aligner) enables language and visual task specification in embodied domains without language annotations — a key bottleneck for foundation models in RL.
- The connector-aligner direction (VLM → world model) substantially outperforms the reversed direction (world model → VLM): GenRL 0.80 vs. WM-CLIP 0.70 overall, with the aligner being critical (removing it drops to 0.17).
- Data-free policy learning is viable with a pretrained MFWM: GenRL data-free achieves 0.75 vs. 0.80 offline, converges in ~30 minutes, and eliminates all data dependencies for new task learning.
- Video prompts generalize across diverse visual domains (drawings, AI-generated, different morphologies) without any retraining, demonstrating the VLM's cross-domain embedding structure transfers through the connector.
- Diverse exploratory pre-training data is the most critical data type: training on all data (exploration + task-specific) substantially outperforms any subset, indicating that broad coverage of the behavior space is key to MFWM generalization.

---

## BibTeX
{% raw %}
```bibtex
@inproceedings{mazzaglia2024genrl,
  title={{GenRL}: Multimodal-foundation World Models for Generalization in Embodied Agents},
  author={Mazzaglia, Pietro and Verbelen, Tim and Dhoedt, Bart and Courville, Aaron and Rajeswar, Sai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
{% endraw %}

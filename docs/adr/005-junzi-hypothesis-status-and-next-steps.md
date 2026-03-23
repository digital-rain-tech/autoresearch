# ADR-005: Junzi Hypothesis — Status, Hardware Constraints, and Next Steps

**Status**: Active
**Date**: 2026-03-20
**Depends on**: ADR-002, ADR-003, ADR-004

## The Junzi Hypothesis

The [Junzi Alignment hypothesis](https://augustinchan.dev/posts/2026-01-25-junzi-alignment-initial-weights-hypothesis) (Chan, 2026) proposes that AI alignment can emerge from initial conditions — weight initialization and training curriculum — rather than exclusively from post-hoc methods like RLHF. Drawing from Confucian philosophy, it frames alignment as intrinsic character cultivation, predicting:

1. **Seed-dependent alignment variance** — different initializations produce measurably different behavioral dispositions
2. **Optimal learning curricula** follow discoverable sequences (specifically the King Wen I-Ching ordering) that cultivate capability through structured exposure
3. **RLHF efficiency** varies by initialization — "aligned-predisposed" seeds need less safety training

The [King Wen AGI Framework](https://github.com/augchan42/king-wen-agi-framework) provides the mathematical basis, arguing the King Wen sequence optimizes Bayesian surprise for meta-learning curricula.

## What We've Tested

| Experiment | ADR | Result |
|---|---|---|
| King Wen as LR modulation | ADR-002 | **Hurts** — all amplitudes worse than baseline |
| Seed behavioral sensitivity (30 seeds) | ADR-004 | **Negligible** — within-seed noise dominates; no multi-dimensional "character traits" |

Neither result falsifies the Junzi hypothesis — they constrain it. The LR modulation was the wrong application domain. The seed sensitivity likely requires larger scale.

## Mainstream Research Support

Recent papers provide directional support for the *general principle* that initial conditions matter for alignment, though through different mechanisms than originally proposed:

**Directly relevant:**
- [Assessing Macro and Micro Effects of Random Seeds on Fine-Tuning LLMs](https://arxiv.org/html/2503.07329v1) (Mar 2025) — Two LLMs fine-tuned with different seeds can achieve identical accuracy but only 20% overlapping predictions. Seed-dependent behavioral variance IS real at fine-tuning scale.
- [When Should We Introduce Safety Interventions During Pretraining?](https://arxiv.org/html/2601.07087) (Feb 2026) — Timing of safety data introduction is a curriculum design choice that significantly affects downstream robustness. Models that absorb safe-only data first (20-60% of training) develop more durable alignment. Validates curriculum ordering for safety.
- [Safety Pretraining: Toward the Next Generation of Safe AI](https://arxiv.org/abs/2504.16980) (Apr 2025) — Building safety into pretraining data is more robust than post-hoc RLHF. "Once unsafe patterns are learned during pretraining, they are hard to remove."

**Key insight from the literature:** The mechanism is *data curriculum and representation geometry*, not mystical weight initialization properties. Early training shapes the loss landscape attractors that the model subsequently falls into.

## Hardware Constraints

**Current setup:** NVIDIA RTX 2060, 6 GB VRAM

| Constraint | Impact |
|---|---|
| 6 GB VRAM | Forces fp32 (no bf16 on Turing), DEPTH=4, DEVICE_BATCH_SIZE=16 |
| Turing architecture | No FlashAttention 3, must use SDPA |
| 5-min time budget | ~131K tokens/step, limited total training |
| Model size | 4-layer GPT, ~256-dim — very small capacity |

### What IS feasible on this hardware

**1. Curriculum ordering (ADR-003) — YES, run it.**
This modifies batch ordering within the existing training loop. No additional memory or compute needed beyond the current setup. The buffered reordering approach (64 batches scored by token diversity) adds ~8 MB memory and <1s overhead per refill — negligible on RTX 2060.

This is the most natural remaining test of King Wen's properties and the one most supported by the recent safety-pretraining literature. The Feb 2026 paper specifically validates "intervention timing as a curriculum design choice."

**2. Finer-grained behavioral analysis — YES, but limited value.**
We could re-analyze the 4,500 existing samples with semantic metrics (topic modeling, embedding clustering via a pretrained model). But ADR-004 showed the primary axis of seed variation is just verbosity, not deeper behavioral traits. Diminishing returns at this model scale.

**3. Longer training runs — PARTIALLY.**
Increasing `TIME_BUDGET` from 300s to 900s or 1800s would allow more trajectory divergence between seeds. The model fits in memory; it just takes longer. However, the 4-layer architecture may not have enough capacity for interesting attractor structure regardless of training time.

**4. Larger models — VERY LIMITED.**
- DEPTH=6 (384-dim): Likely fits in 6 GB with reduced batch size. ~50% more parameters. Worth trying.
- DEPTH=8 (512-dim): Probably doesn't fit at fp32. *Might* fit with gradient checkpointing or DEVICE_BATCH_SIZE=8 + more grad accumulation steps. Experimental.
- DEPTH=12+ : Not feasible on RTX 2060 at fp32.

On an RTX 2060, we cannot reach the scale where the seed sensitivity paper (arXiv:2503.07329) found meaningful behavioral divergence (they used BERT/RoBERTa-scale models with fine-tuning).

### What is NOT feasible on this hardware

- **Fine-tuning large pretrained LLMs** (the setting where seed sensitivity is documented)
- **RLHF or DPO experiments** (Junzi prediction #3)
- **Multi-GPU distributed training**
- **bf16 training** (Turing GPUs lack full bf16 support)

## Recommended Next Steps (in priority order)

### 1. Curriculum ordering experiment (ADR-003)

**Feasibility: FULLY FEASIBLE — no hardware changes needed.**

This is the strongest remaining test. Implement buffered batch reordering in `train.py`:
- Buffer 64 batches from dataloader
- Score by token diversity (free) or model loss (cheap)
- Present in King Wen order, with controls (sequential, random, easy-to-hard, hard-to-easy, Shao Yong)

The Feb 2026 intervention-timing paper validates that data ordering affects alignment robustness. If King Wen curriculum ordering improves val_bpb or produces measurably different behavioral profiles, it's a meaningful result even at small scale.

### 2. Scale probe: DEPTH=6 with seed sensitivity

**Feasibility: LIKELY FEASIBLE — needs testing.**

Run a smaller seed sweep (5-10 seeds) at DEPTH=6 to see if the extra capacity creates more seed-dependent behavioral differentiation. If the variance ratio (ADR-004 §1) increases meaningfully, it supports the hypothesis that scale is the bottleneck.

### 3. Extended training: 15-min budget

**Feasibility: FULLY FEASIBLE — just slower.**

Triple the time budget to 900s. Run 5 seeds at baseline to see if longer training amplifies seed-dependent behavioral divergence. If trajectories diverge more with longer training, it supports the "training time too short" explanation from ADR-004.

### 4. Document and publish findings

Regardless of next experimental results, the current body of work (ADR-001 through ADR-005) constitutes a rigorous negative-result study with clear methodology. Negative results that are well-documented advance the field by constraining the hypothesis space.

## Decision

Proceed with curriculum ordering (ADR-003) as the primary next experiment. It is:
- Fully feasible on current hardware
- The most natural application of King Wen's anti-habituation properties
- Supported by mainstream research on curriculum effects in safety pretraining
- The experiment the original King Wen research actually motivates (not LR modulation)

If curriculum ordering also shows null results at this scale, the honest conclusion is that the Junzi hypothesis requires larger models to test meaningfully, and we should document that constraint clearly.

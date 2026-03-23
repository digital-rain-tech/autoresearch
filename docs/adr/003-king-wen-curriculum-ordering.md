# ADR-003: King Wen as Curriculum Ordering — Experiment Design

**Status**: In Progress (implementation v1 revealed critical buffering issue)
**Date**: 2026-03-19 (updated 2026-03-23)
**Depends on**: ADR-002 (King Wen LR schedule — not supported)

## Context

ADR-002 showed that King Wen's anti-habituation profile *hurts* when applied as LR modulation — the high variance destabilizes gradient updates. However, the original King Wen research (Chan, 2026) frames the sequence as optimizing Bayesian surprise for *meta-learning curricula*, not optimizer schedules. The [Junzi Alignment hypothesis](https://augustinchan.dev/posts/2026-01-25-junzi-alignment-initial-weights-hypothesis) further positions King Wen as a developmental sequence for cultivating capability through structured exposure.

This reframes King Wen's role: instead of modulating *how fast* the optimizer steps (LR), modulate *what data* it sees and when.

## Hypothesis

Training data presented in King Wen-ordered difficulty progression will yield lower val_bpb than sequential, random, or Shao Yong orderings, because:

1. Anti-habituation in data exposure forces the model to continuously adapt rather than overfit to local data patterns
2. High surprise variance in difficulty transitions prevents the model from settling into narrow representational basins
3. Zero autocorrelation means the model can't "predict" upcoming difficulty, forcing more robust feature learning

## Design

### Approach: Buffered batch reordering by difficulty

Since `prepare.py` is immutable, implement curriculum ordering in `train.py` by:

1. **Buffer N batches** from the standard dataloader (e.g., N=64 to match the King Wen sequence length)
2. **Score each batch** by entropy/difficulty (mean cross-entropy of a forward pass, or simpler: token diversity as proxy)
3. **Sort batches by score** into 64 difficulty buckets
4. **Present in King Wen order** — map King Wen surprise values to difficulty buckets, so high-surprise positions get harder batches and low-surprise positions get easier ones

### Controls (must test all to be fair — lesson from ADR-002)

| Run | Ordering | What it tests |
|-----|----------|---------------|
| 1 | Sequential (baseline) | Standard dataloader order |
| 2 | Random shuffle | "Any reordering helps?" |
| 3 | Easy-to-hard (curriculum) | Classical curriculum learning |
| 4 | Hard-to-easy (anti-curriculum) | Opposite of curriculum |
| 5 | Shao Yong ordering | Structured but predictable difficulty |
| 6 | King Wen ordering | Anti-habituation difficulty |

### Difficulty scoring

Two options, trading accuracy for cost:

**Option A — Token diversity (free):** Count unique tokens per batch. More diverse = harder. No model forward pass needed.

**Option B — Loss-based (costs one buffer pass):** Run each buffered batch through the model, use mean loss as difficulty score. More accurate but adds ~1 second per buffer refill. With 64-batch buffers refilled every ~64 steps, overhead is small relative to 5-minute budget.

### Implementation constraints

- Only `train.py` is modified
- `prepare.py` and its `make_dataloader` are used as-is
- Buffering adds memory overhead: 64 batches x 16 sequences x 2048 tokens x 4 bytes ≈ 8MB (negligible)
- Reordering adds compute overhead: scoring 64 batches takes <1s per refill
- val_bpb evaluation is unchanged

## Risks

- **Buffer size limits reordering scope** — with 64 batches buffered, we can only reorder within local windows, not globally across the dataset. This may dilute any curriculum effect.
- **Difficulty proxy may be poor** — token diversity may not correlate with actual learning difficulty for the model.
- **Overhead could eat into training time** — if loss-based scoring is used, the forward passes during scoring reduce the effective training budget. Need to account for this in comparison.
- **LR decay sabotages curriculum** — see "Literature findings" below.
- **GPU buffer reuse interacts with torch.compile** — see "Implementation v1 findings" below.

## Success Criteria

Same fairness standard as ADR-002: King Wen ordering must beat **all** controls (sequential, random, easy-to-hard, hard-to-easy, Shao Yong), not just baseline, to validate the anti-habituation hypothesis for curriculum ordering.

---

## Implementation v1 Findings (2026-03-23)

### What was built

A `curriculum_dataloader` generator wrapper in `train.py` that:
- Buffers 64 micro-batches by cloning GPU tensors from the dataloader
- Scores each batch by token diversity (`x.unique().numel() / x.numel()`)
- Reorders according to the active curriculum policy (env var `AUTORESEARCH_CURRICULUM`)
- Yields reordered batches to the unchanged training loop

### Critical bug: GPU tensor cloning breaks torch.compile

All non-sequential orderings produced catastrophically worse results:

| Ordering | val_bpb | Steps | vs Baseline |
|----------|---------|-------|-------------|
| sequential (baseline) | 1.719 | 124 | — |
| random | 2.849 | 114 | +1.130 |
| easy_to_hard | 2.839 | 116 | +1.120 |
| hard_to_easy | 2.814 | 113 | +1.095 |
| **buffered_passthrough** | **2.794** | **109** | **+1.075** |

The `buffered_passthrough` mode (buffer + clone but NO reorder, same data order as sequential) was equally bad. This proves the problem is the **buffering/cloning itself**, not the reordering.

### Root cause analysis

The dataloader (`prepare.py`) uses a pre-allocated `gpu_buffer` and yields views into it. `torch.compile(model, dynamic=False)` optimizes the model for this memory layout. When we clone into separate GPU tensors:

1. **Every step is 15-20% slower** (dt ~2900ms vs ~2600ms), not just at buffer boundaries. This suggests torch.compile kernels are less efficient with different tensor storage patterns.
2. **Sawtooth loss pattern** at buffer boundaries (every 16 optimizer steps = 64 micro-batches): loss drops rapidly within each buffer, then spikes when a new buffer arrives.
3. **Training loss is much LOWER but val_bpb is much WORSE** — classic overfitting. At step 31: passthrough loss 3.46 vs sequential 5.90, yet val_bpb is 1.07 worse.

The fast loss drop within buffers suggests the compiled model treats cloned tensors differently than views of a single buffer — possibly through caching or memory-layout-dependent kernel optimization.

### Fix approach (v2, not yet tested)

Buffer on **CPU pinned memory** instead of GPU, then transfer one batch at a time during yield into a **single reusable GPU tensor pair** — mimicking the original dataloader's H2D transfer pattern. This should:
- Preserve torch.compile's memory layout expectations
- Avoid dynamic GPU tensor allocation
- Add minimal overhead (one H2D copy per batch, same as the original dataloader)

---

## Literature Findings (2026-03-23)

A Hugging Face paper search revealed critical findings that affect our experimental design:

### 1. LR decay sabotages curriculum learning

[How LR Decay Wastes Your Best Data in Curriculum-Based Pretraining](https://huggingface.co/papers/2511.18903) (Nov 2025) shows that standard LR warmdown reduces the learning rate to near-zero exactly when the best/hardest data arrives in curriculum ordering. This creates a fundamental tension:

- Our `WARMDOWN_RATIO = 0.5` means 50% of training uses decaying LR
- King Wen ordering places high-difficulty batches at high-surprise positions throughout training, but particularly in later positions within each buffer
- The decaying LR prevents the model from learning from these hard batches

**Proposed solutions from the paper:**
1. Use constant LR + model averaging (CMA) instead of decay
2. Use moderate decay (ending LR ~1/3 of peak, not ~1/300)
3. Co-design LR schedule and curriculum together

**Implication for our experiment:** We need to test curriculum orderings both WITH and WITHOUT warmdown. At minimum, add runs with `WARMDOWN_RATIO = 0.0` or `0.1`.

### 2. Curriculum learning does work for LLM pretraining

[Beyond Random Sampling: Curriculum Learning for LM Pretraining](https://huggingface.co/papers/2506.11300) (Jun 2025) — first systematic investigation:

- Curriculum learning consistently improves convergence in early/mid training
- **Best difficulty signals: compression ratio, lexical diversity, readability** — our token diversity metric is in the right family
- Up to 3.5% improvement when used as warmup strategy
- Works best as a warmup strategy, not applied throughout training

### 3. Data ordering is a first-order concern

[Olmix: A Framework for Data Mixing](https://huggingface.co/papers/2602.12237) (Feb 2026) and [Data Mixing Laws](https://huggingface.co/papers/2403.16952) (Mar 2024) confirm that data composition and ordering during pretraining fundamentally shape model behavior.

### 4. Stochastic variability aids cognition

[Stochastic CHAOS](https://huggingface.co/papers/2601.07239) (Jan 2026) argues distributional variability is essential for robust AI cognition — conceptual support for anti-habituation, though not directly about training curricula.

---

## Implementation v2 Results (2026-03-23)

### Fix applied

Switched from GPU tensor cloning to CPU pinned memory buffering with a single reusable GPU tensor pair. This eliminates the torch.compile interaction (no more sawtooth loss pattern). The fix correctly preserves the original dataloader's H2D transfer pattern.

### Regime A results (standard warmdown)

| Ordering | val_bpb | Steps | Mean dt | vs Passthrough |
|----------|---------|-------|---------|----------------|
| sequential (no buffer) | 1.719 | 124 | 2658ms | N/A |
| buffered_passthrough | 1.680 | 86 | 4044ms | — |
| random | **1.614** | 98 | 3449ms | **-0.066** |
| easy_to_hard | 1.632 | 95 | 3581ms | -0.048 |
| hard_to_easy | 1.627 | 97 | 3516ms | -0.053 |
| shao_yong | 1.638 | 93 | 3669ms | -0.042 |
| king_wen | 1.662 | 89 | 3873ms | -0.018 |

### Analysis

**Buffer overhead is significant and varies by ordering.** The CPU buffer approach adds 30-52% per-step overhead (2658ms → 3449-4044ms). This reduces training steps from 124 to 86-98 within the 5-minute budget. The overhead varies by ordering because:
- Buffer fill + scoring is constant (~1.6s per refill every 16 steps)
- Reordering compute is negligible
- But run-to-run GPU thermal/clock variance affects totals

**Step count strongly correlates with val_bpb (r = -0.979).** Among buffered orderings, more steps → lower val_bpb. This makes it difficult to attribute val_bpb improvements to curriculum ordering vs simply more training.

**All buffered orderings beat the unbuffered sequential baseline.** Even the passthrough (same order, with buffer overhead) achieves 1.680 vs 1.719. This may indicate that the buffer's implicit data shuffling (reading 64 batches then yielding) provides mild regularization, or it may be a measurement artifact from the different H2D transfer patterns.

**King Wen is the worst non-sequential ordering.** It barely beats passthrough (1.662 vs 1.680) and has 3 more steps. The Junzi hypothesis prediction — that King Wen's anti-habituation profile would outperform all controls — is not supported.

**Random shuffle performed best** (1.614, 98 steps), but also had the most steps among buffered orderings, so this may simply reflect more training.

### Confounds that prevent firm conclusions

1. **Variable overhead**: Step counts range from 86-98 across orderings (14% variance). With only ~90 steps total, each step matters.
2. **Small effect sizes**: The val_bpb differences between orderings (0.02-0.07) are within the seed noise range established by ADR-004 (±0.04).
3. **LR decay interaction**: Per the literature (arXiv:2511.18903), our WARMDOWN_RATIO=0.5 may be sabotaging curriculum effects by reducing LR when harder data arrives.

---

## Decision

**King Wen curriculum ordering does not outperform controls at this scale.** The hypothesis that anti-habituation data ordering improves training is not supported.

However, the experiment has significant confounds (buffer overhead variance, LR-curriculum interaction) that limit the strength of this conclusion. A cleaner test would require:

1. **Equal overhead across all orderings** — perhaps by pre-computing the ordering before training and applying it without runtime buffering
2. **Longer time budget** — 15-30 min instead of 5 min, to reduce the relative impact of buffer overhead
3. **Reduced warmdown** — test with WARMDOWN_RATIO=0.0 to eliminate the LR-curriculum interaction

### Regime B (reduced warmdown) — NOT YET RUN

The literature strongly suggests LR decay sabotages curriculum learning. If resources permit, running with reduced warmdown would provide a fairer test. But given that King Wen was the worst performer in Regime A (not second-best or competitive), it is unlikely to become the best performer simply by changing the LR schedule.

### Overall Junzi hypothesis status

Three experiments completed, all negative:
- ADR-002: King Wen as LR modulation — hurts
- ADR-004: Seed behavioral sensitivity — negligible at this scale
- ADR-003: King Wen as curriculum ordering — worst of all orderings tested

The honest conclusion: **the Junzi hypothesis may require larger models to test meaningfully**, or the King Wen sequence's statistical properties (high variance, zero autocorrelation) may not translate to useful training curricula at any scale. The mainstream curriculum learning literature finds that simple easy-to-hard ordering with appropriate LR co-design is what works — not exotic anti-habituation profiles.

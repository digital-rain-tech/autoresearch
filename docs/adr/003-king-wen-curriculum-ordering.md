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

## Revised Experiment Plan

Based on implementation findings and literature review, the experiment needs two fixes before proceeding:

### Fix 1: Buffer implementation (required)

Switch from GPU tensor cloning to CPU pinned memory buffering with single-tensor GPU yield. This eliminates the torch.compile interaction. Must verify the fix produces identical val_bpb to sequential baseline before running any curriculum experiments.

### Fix 2: LR schedule co-design (important)

Run curriculum experiments in TWO LR regimes:

**Regime A — Standard warmdown (current):**
- `WARMDOWN_RATIO = 0.5`, `FINAL_LR_FRAC = 0.0`
- Tests curriculum ordering under the existing setup
- May underestimate curriculum benefits due to LR-curriculum interaction

**Regime B — Reduced warmdown:**
- `WARMDOWN_RATIO = 0.1`, `FINAL_LR_FRAC = 0.1`
- Preserves higher LR throughout training
- Gives curriculum ordering a fair chance to show effects
- Aligned with literature recommendations

### Full run matrix

| Run | Ordering | LR Regime | Env vars |
|-----|----------|-----------|----------|
| 1 | sequential | A (standard) | `CURRICULUM=sequential` |
| 2 | random | A | `CURRICULUM=random` |
| 3 | easy_to_hard | A | `CURRICULUM=easy_to_hard` |
| 4 | hard_to_easy | A | `CURRICULUM=hard_to_easy` |
| 5 | shao_yong | A | `CURRICULUM=shao_yong` |
| 6 | king_wen | A | `CURRICULUM=king_wen` |
| 7 | sequential | B (reduced) | `CURRICULUM=sequential` + warmdown override |
| 8 | random | B | `CURRICULUM=random` + warmdown override |
| 9 | easy_to_hard | B | `CURRICULUM=easy_to_hard` + warmdown override |
| 10 | hard_to_easy | B | `CURRICULUM=hard_to_easy` + warmdown override |
| 11 | shao_yong | B | `CURRICULUM=shao_yong` + warmdown override |
| 12 | king_wen | B | `CURRICULUM=king_wen` + warmdown override |

12 runs x 5 min = 1 hour total. All use seed 42.

### Implementation sequence

1. Fix buffer implementation (CPU pinned memory approach)
2. Verify sequential ordering matches unmodified baseline
3. Add `AUTORESEARCH_WARMDOWN_RATIO` env var override for Regime B
4. Run all 12 experiments
5. Record results, update ADR with findings

# ADR-003: King Wen as Curriculum Ordering — Experiment Design

**Status**: Proposed
**Date**: 2026-03-19
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

### Approach: Buffered batch reordering by entropy

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

## Success Criteria

Same fairness standard as ADR-002: King Wen ordering must beat **all** controls (sequential, random, easy-to-hard, hard-to-easy, Shao Yong), not just baseline, to validate the anti-habituation hypothesis for curriculum ordering.

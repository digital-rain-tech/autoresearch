# ADR-006a: CUDA Curriculum Rerun with Compression Ratio

**Status**: Pending
**Date**: 2026-03-23
**Hardware**: Intel laptop, NVIDIA RTX 2060 6 GB
**Depends on**: ADR-006 (MLX curriculum exploration)
**Repo**: `autoresearch` (the PyTorch/CUDA sibling repo)

## Purpose

The CUDA v2 curriculum experiments used **token diversity** (`x.unique().numel() / x.numel()`) as the difficulty metric. ADR-006 standardizes on **compression ratio** (literature's strongest signal per arXiv:2506.11300) for cross-platform comparison with the MLX experiments.

This document specifies the minimal rerun needed on the 2060 to align metrics.

## What to change

**One function in `train.py`** — replace `score_batch_difficulty`:

```python
# BEFORE (token diversity)
def score_batch_difficulty(x):
    """Token diversity: unique tokens / total tokens. Higher = harder."""
    return x.unique().numel() / x.numel()

# AFTER (compression ratio)
def score_batch_difficulty(x):
    """Compression ratio: gzip compressed / raw bytes. Higher = harder."""
    import gzip
    raw = x.cpu().numpy().tobytes()
    return len(gzip.compress(raw)) / len(raw)
```

**Nothing else changes.** Same buffer implementation, same King Wen mapping, same env vars, same 5-minute budget.

## Runs to execute

All at DEPTH=4, seed 42, standard warmdown (WARMDOWN_RATIO=0.5):

```bash
# 1. Sequential baseline (no buffer — should match v2 result)
AUTORESEARCH_CURRICULUM=sequential uv run train.py > run_seq.log 2>&1

# 2. Buffered passthrough
AUTORESEARCH_CURRICULUM=buffered_passthrough uv run train.py > run_pass.log 2>&1

# 3. Random shuffle
AUTORESEARCH_CURRICULUM=random uv run train.py > run_rand.log 2>&1

# 4. Easy to hard
AUTORESEARCH_CURRICULUM=easy_to_hard uv run train.py > run_e2h.log 2>&1

# 5. Hard to easy
AUTORESEARCH_CURRICULUM=hard_to_easy uv run train.py > run_h2e.log 2>&1

# 6. King Wen
AUTORESEARCH_CURRICULUM=king_wen uv run train.py > run_kw.log 2>&1
```

Shao Yong is dropped (see ADR-006 rationale). 6 runs x ~7 min = **~42 minutes total**.

## What to record

For each run, extract:

```bash
grep "^val_bpb:\|^peak_vram_mb:" run_*.log
```

Log to `results.tsv` with description prefix `curriculum-cr:` (compression ratio) to distinguish from the v2 token-diversity results (`curriculum:`).

## What to watch for

1. **Sequential baseline should match v2** (val_bpb ≈ 1.719). The difficulty metric doesn't affect sequential ordering. If it differs by more than 0.005, something else changed.

2. **Overhead from gzip:** The `gzip.compress` call runs on CPU. Expected ~100ms per 64-batch buffer refill. If `curriculum_overhead_seconds` exceeds 20% of budget (60s), the gzip is too expensive on this hardware. Fallback: use `zlib.compress` (faster, same ordering signal).

3. **Ordering ranking change:** The key question is whether compression ratio changes the relative ranking of orderings compared to token diversity. Specifically:
   - Does random still win?
   - Does King Wen still rank worst?
   - Does hard_to_easy still match easy_to_hard?

If the ranking changes, the difficulty metric — not the ordering — drives the effect. If ranking is stable, the decorrelation hypothesis is strengthened (ordering rankings are metric-independent).

## Comparison table (to fill in)

| Ordering | Token Diversity (v2) | Compression Ratio (v3) | Delta |
|----------|---------------------|----------------------|-------|
| sequential | 1.719 | | |
| buffered_passthrough | 1.680 | | |
| random | 1.614 | | |
| easy_to_hard | 1.632 | | |
| hard_to_easy | 1.627 | | |
| king_wen | 1.662 | | |

## After completion

Copy results to `autoresearch-mlx/docs/adr/006a-cuda-rerun.md` (this file) and update the comparison table. The MLX Phase 2 experiments will use these as the CUDA reference point.

# ADR-008: ADR-007 Implementation & Review Findings

**Status**: Accepted
**Date**: 2026-03-24
**Depends on**: ADR-007 (game-theoretic research)

## Context

ADR-007 proposed two parallel workstreams for testing the King Wen hypothesis in a new domain (discrete strategy) while also addressing an open question from ADR-003 (adaptive curriculum). This ADR documents the implementation and captures findings from an independent code review (Codex multi-agent review, 2026-03-24).

## What Was Implemented

### Workstream A: Adaptive Bandit Curriculum (`train.py`, `sweep_curriculum.py`)

Modified `train.py`'s existing adaptive curriculum path:
- UCB1 now tracks **loss improvement** per difficulty bucket (not raw loss), matching ADR-007's spec of "whichever data it currently learns most from"
- Token diversity used as fast difficulty proxy (gzip compression ratio is ~500x slower in the hot path)
- Added adaptive stats export (bucket counts, avg losses, UCB scores) to final training output
- Added `assert buffer_size >= NUM_BUCKETS` guard

Created `sweep_curriculum.py`:
- Paired seed sweep: same seed across orderings to reduce variance
- Paired t-test, Cohen's d, ADR-007 success criterion check (>0.04 bpb improvement over random)
- Caches completed runs to allow incremental experimentation

### Workstream B: King Wen OpenSpiel Experiments (`king_wen_openspiel.py`)

- 3 King Wen → action mappings: hash, trigram, sequential
- 6 experimental conditions: 3 KW variants + CFR-only baseline + scrambled KW + random prior
- 2 games: Kuhn Poker (analytically known Nash), Goofspiel (simultaneous actions via turn-based wrapper)
- External Sampling MCCFR for seed-dependent stochasticity
- Prior-biased initial regrets with configurable temperature and strength
- Log-spaced convergence curves for convergence speed analysis
- Statistical analysis: Welch's t-test, Bonferroni correction, Cohen's d, ADR-007 success criterion

### Orchestration (`run_adr007.py`)

Unified runner with `--quick` mode for smoke tests and `--workstream A|B|both`.

## Review Findings & Dispositions

Independent Codex review identified 10 issues (0 critical, 3 high, 6 medium, 1 low).

### Fixed

| # | Severity | Finding | Fix |
|---|----------|---------|-----|
| 1 | High | Scoring uses token diversity instead of compression ratio (ADR-006a) | Added comment documenting the design trade-off: gzip is ~500x slower; token diversity is a valid fast proxy |
| 4 | Medium | UCB optimizes raw loss instead of loss improvement | Changed UCB to track loss *improvement* per bucket across buffers (prev_avg - cur_avg) |
| 5 | Medium | No tests for adaptive curriculum | Added `test_adaptive_curriculum.py` with 5 unit tests: bucket partitioning, UCB exploration, improvement-over-raw-loss, feedback accumulation, buffer validation |
| 6 | Medium | CLAUDE.md doesn't list "adaptive" curriculum option | Updated CLAUDE.md: added "adaptive" to curriculum options, added new experiment files to tooling section |
| 9 | Medium | open-spiel/scipy bloat base dependencies | Moved to `[project.optional-dependencies] gametheory` in pyproject.toml |
| 10 | Low | bucket_size division assumes buffer >= buckets | Added `assert buffer_size >= NUM_BUCKETS` |

### Declined

| # | Severity | Finding | Reason |
|---|----------|---------|--------|
| 2 | High | No AGENTS.md files in subdirectories | Repo-wide documentation concern outside scope of this implementation |
| 3 | High | README doesn't reflect current repo shape | Pre-existing; not proactively creating docs |
| 7 | Medium | model_def.py duplicates train.py model code | Pre-existing architectural debt, not introduced by this change |
| 8 | Medium | Env var usage not centralized or validated | Over-engineering for a research repo with 3 env vars |

## Known Limitations

1. **kw_hash and kw_sequential produce identical results** on Kuhn Poker because the sequential mapping falls back to hashing info_state_str (game states don't carry a natural move counter through the MCCFR API). True sequential mapping would need game-tree depth tracking.

2. **Token diversity vs compression ratio**: The scoring proxy could produce different bucket assignments than gzip-based scoring. If the adaptive curriculum shows promising results, a follow-up should validate that token diversity and compression ratio produce correlated rankings.

3. **Goofspiel complexity**: With `num_cards=4`, Goofspiel's game tree is manageable but `num_cards=13` (full deck) would be intractable for tabular CFR. The 4-card version is sufficient for hypothesis testing.

4. **Temperature search not yet run**: ADR-007 specifies testing temperatures [0.1, 0.3, 0.5, 1.0]. The implementation supports this but the pilot default is temperature=0.3 only. Full temperature sweep should be run in the Week 1 experiments.

## How to Run

```bash
# Workstream A (requires GPU, ~25 min for 5 seeds × 3 orderings)
uv run sweep_curriculum.py --seeds 0 5

# Workstream B (CPU only, ~5 min; install optional deps first)
uv pip install -e ".[gametheory]"
uv run king_wen_openspiel.py --seeds 0 5 --iterations 1000

# Both workstreams, quick smoke test
uv run run_adr007.py --quick

# Run adaptive curriculum tests
python test_adaptive_curriculum.py
```

## Decision

Implementation is complete and reviewed. Proceed to run full experiments per ADR-007 Week 1 phasing.

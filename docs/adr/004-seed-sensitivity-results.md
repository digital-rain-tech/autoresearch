# ADR-004: Seed Sensitivity Experiment — No Behavioral Differentiation at Small Scale

**Status**: Completed
**Date**: 2026-03-20
**Branch**: `autoresearch/seed-sensitivity`
**Depends on**: ADR-002 (King Wen LR schedule — not supported)

## Context

ADR-002 found that King Wen LR modulation hurt training, but left open whether the Phase 1 results (random perturbation and Shao Yong appearing to help) were genuine or seed noise. Separately, the [Junzi Alignment hypothesis](https://augustinchan.dev/posts/2026-01-25-junzi-alignment-initial-weights-hypothesis) predicts that different random initializations produce measurably different behavioral dispositions — "seed-dependent alignment variance."

This experiment addresses both questions by training 30 models with different seeds on the identical baseline configuration, then sampling and scoring their text outputs.

## Experiment Design

- **Seeds**: 0–29 (30 runs)
- **Configuration**: Identical baseline — 4-layer GPT, standard LR schedule, 5-min time budget, RTX 2060
- **Sampling**: 15 prompts × 2 temperatures (0.8, 1.0) × 5 samples = 150 samples per seed, 4,500 total
- **Metrics**: 12 behavioral metrics per sample (word count, lexical diversity, distinct-n, word entropy, sentence length, repetition rate, punctuation rate, degeneration score, vocab coverage)
- **Tooling**: `sweep_seeds.py` orchestrates train → checkpoint → sample → score via env vars (`AUTORESEARCH_SEED`, `AUTORESEARCH_CHECKPOINT_PATH`), no file mutation

## Results

### val_bpb across seeds

| Statistic | Value |
|-----------|-------|
| Range | 1.7317 – 1.7732 |
| Mean | 1.7560 |
| Std | 0.0089 |
| CV | 0.51% |

Best seed: 28 (1.7317). Worst seed: 6 (1.7732). Spread of ~0.041.

### Phase 1 results were seed noise

Comparing ADR-002 results against the seed sweep distribution:

| Run | val_bpb | Within seed noise? |
|-----|---------|-------------------|
| Baseline (seed 42) | 1.753 | Yes — near mean |
| Random perturbation (amp=0.3) | 1.731 | Yes — lucky end of range |
| Shao Yong (amp=0.3) | 1.732 | Yes — lucky end of range |
| King Wen (amp=0.3) | 1.785 | **No** — genuinely worse |
| King Wen (amp=0.15) | 1.777 | Borderline outside |
| King Wen (amp=0.5) | 1.790 | **No** — genuinely worse |

The "improvements" from random perturbation and Shao Yong (both ~0.02 better than baseline) fall within the natural seed variance range. Only King Wen's *degradation* is statistically meaningful.

### Behavioral metrics: seeds don't differentiate

**Between-seed vs within-seed variance ratio** (higher = seed matters more):

| Metric | Between-seed σ | Within-seed σ | Ratio |
|--------|---------------|---------------|-------|
| word_count | 6.73 | 31.99 | 0.21 |
| unique_words | 2.51 | 12.11 | 0.21 |
| lexical_diversity | 0.019 | 0.100 | 0.19 |
| distinct_2 | 0.010 | 0.061 | 0.17 |
| word_entropy | 0.050 | 0.329 | 0.15 |
| degeneration_score | 0.020 | 0.127 | 0.16 |

No metric has a ratio above 0.21. Within-seed sampling noise dominates.

### PCA: one dimension explains most variance

PC1 captures 66.4% of between-seed variance. Its loadings reveal a single "verbosity axis":

| Metric | PC1 loading |
|--------|-------------|
| word_count | -0.361 |
| distinct_2 | +0.347 |
| lexical_diversity | +0.346 |
| degeneration_score | -0.335 |

Seeds vary primarily in "long repetitive output" vs "short diverse output" — not in multi-dimensional behavioral traits.

### Outlier seeds

Seed 4 is a 3σ outlier on 7 metrics (shorter, more lexically diverse, lower degeneration). Seed 15 shows a similar but milder pattern (cosine similarity 0.95 with seed 4). These appear to be weaker models producing shorter outputs, not models with distinctive "personalities."

### val_bpb does not predict behavior

Only one moderate correlation: avg_sentence_length (r=0.34). All others |r| < 0.25. Better-trained seeds do not generate qualitatively different text.

### Temperature amplifies seed differences

Between-seed σ is ~1.5x larger at temperature 0.8 vs 1.0 across all metrics. Higher temperature adds noise that washes out seed-specific patterns.

### Prompt sensitivity

Some prompts are more seed-sensitive than others (CV range 0.042–0.077), but the differences are small. Narrative prompts ("The city had changed...") show more seed sensitivity than abstract prompts ("What is remembered...").

## Decision

**Seed-dependent behavioral variance is negligible at this scale.** The Junzi hypothesis's prediction of "measurably different behavioral dispositions" from different initializations is not supported for 4-layer GPTs trained for 5 minutes.

## Analysis

### Why seeds don't differentiate

1. **Model capacity is too small.** A 4-layer GPT has a relatively simple loss landscape. Most seeds converge to similar solutions because there aren't many distinct attractor basins to fall into.

2. **Training time is too short.** 5 minutes of training may not be enough for seed-dependent trajectory divergence to accumulate. All seeds are still in early learning and converge toward the same general capability level.

3. **Sampling noise dominates.** With top-k=50 sampling and only 200 new tokens per sample, the stochastic decoding process contributes far more variance than any seed-dependent model differences.

4. **Behavioral metrics are coarse.** Word-level statistics may not capture subtle differences in semantic content, reasoning patterns, or stylistic preferences that could exist at finer granularity.

### When seed sensitivity might emerge

- **Larger models** with richer loss landscapes and more attractor basins
- **Longer training** where trajectory divergence has time to compound
- **Finer-grained evaluation** — semantic similarity, topic modeling, or LLM-as-judge rather than lexical statistics
- **Post-RLHF** where alignment training might amplify latent dispositional differences

## Consequences

- The Junzi hypothesis is not falsified — it may simply require larger scale than this framework supports
- King Wen LR modulation is confirmed harmful (outside seed noise); random/Shao Yong "improvements" were seed luck
- The seed sweep tooling (`sweep_seeds.py`, `sample.py`, `score.py`) is validated and reusable for future experiments
- Curriculum ordering (ADR-003) remains the most promising untested application of King Wen in this framework

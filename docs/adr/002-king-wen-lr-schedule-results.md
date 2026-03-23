# ADR-002: King Wen LR Schedule Experiment — Hypothesis Not Supported

**Status**: Completed
**Date**: 2026-03-19
**Branch**: `autoresearch/kingwen-mar19`

## Context

The King Wen sequence (c. 1000 BC) orders the 64 I-Ching hexagrams in a pattern with statistically unusual properties: random-like mean surprise, significantly higher variance than all baselines, and zero autocorrelation. We hypothesized that this "anti-habituation" surprise profile would prevent optimizer habituation when applied as learning rate modulation during neural network training.

Reference: [King Wen AGI Framework](https://github.com/augchan42/king-wen-agi-framework)

## Experiment Design

All runs used identical setup: 4-layer GPT, 5-minute fixed time budget, RTX 2060, ClimbMix-400B dataset.

Three LR schedule variants were tested against a standard baseline, all using the same warmup/warmdown envelope with modulation on top:

- **Random perturbation** — pseudo-random values, controls for "any noise helps"
- **Shao Yong** — highly autocorrelated sawtooth, controls for "any structured perturbation helps"
- **King Wen** — high variance, zero autocorrelation (the hypothesis)

King Wen was additionally tested at three amplitudes (0.15, 0.3, 0.5).

## Results

| Run | Schedule | val_bpb | vs baseline |
|-----|----------|---------|-------------|
| 1 | Baseline (standard) | 1.753376 | — |
| 2 | Random perturbation (amp=0.3) | 1.730856 | **-0.023** |
| 3 | Shao Yong (amp=0.3) | 1.732407 | **-0.021** |
| 4 | King Wen (amp=0.3) | 1.785356 | +0.032 |
| 5 | King Wen (amp=0.15) | 1.777449 | +0.024 |
| 6 | King Wen (amp=0.5) | 1.789786 | +0.036 |

## Decision

**The hypothesis is not supported.** King Wen LR modulation consistently hurt training across all amplitudes tested.

## Analysis

### What worked
Both random perturbation and Shao Yong improved over baseline by ~0.02 bpb. This suggests mild LR noise injection acts as stochastic regularization — helping the optimizer escape sharp minima, regardless of the noise structure.

### Why King Wen hurt
King Wen was selected *because* it has higher variance than random and zero autocorrelation. These properties are exactly what made it harmful:

1. **Excessive variance** — King Wen's defining feature (higher variance than random) means the effective perturbation magnitude exceeds the sweet spot. At amp=0.3, King Wen swings harder than random at amp=0.3.
2. **Zero autocorrelation disrupts optimization** — Optimizers benefit from temporal coherence. Shao Yong (highly autocorrelated) lets the optimizer make progress at a consistent LR for several steps. Random noise has some statistical clustering. King Wen is designed to *never* repeat, so the optimizer is constantly whiplashed.
3. **Anti-habituation is premature at this scale** — A 4-layer model trained for 5 minutes is far from converging. It's still in rapid learning mode. Disrupting a learner that hasn't plateaued just slows it down.

### Amplitude sweep was unfair
Only King Wen was tested at multiple amplitudes (0.15, 0.3, 0.5). A fair comparison would sweep amplitudes for all three methods. However, since King Wen performed worse than baseline at *every* amplitude — and worse than both controls at the one shared amplitude — an amplitude sweep for controls would only widen the gap.

### When King Wen might work
- **Late-stage training** on large models that have genuinely plateaued, where violent disruption could help escape sharp minima
- **Much lower effective amplitude** (e.g., amp=0.05) to compensate for King Wen's inherently higher variance
- **Non-LR applications** — modulating weight decay or momentum, where high variance is less directly disruptive
- **Curriculum ordering** rather than LR scheduling (see ADR-003)

## Consequences

- King Wen LR scheduling should not be pursued further at this scale
- The finding that *any* mild LR perturbation helps is independently interesting and worth exploring
- King Wen's properties may be better suited to data curriculum ordering, where "anti-habituation" means varying training data exposure rather than destabilizing gradient updates

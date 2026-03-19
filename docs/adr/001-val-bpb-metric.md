# ADR-001: Use val_bpb (Validation Bits Per Byte) as the Single Experiment Metric

**Status**: Accepted
**Date**: 2026-03-19
**Context**: Autoresearch King Wen LR schedule experiment

## Context

Autoresearch needs a single metric to compare training experiments across runs where the agent may change architecture, vocabulary size, tokenizer, batch size, or any other hyperparameter. Standard cross-entropy loss (per-token) is not suitable because it depends on vocabulary size — a smaller vocab produces more tokens that are individually easier to predict, yielding a lower loss that isn't actually a better model.

## Decision

Use **val_bpb** (validation bits per byte) as the sole metric for experiment comparison.

### How it works

```
val_bpb = total_nats / (ln(2) * total_bytes)
```

1. Run the model on validation data, collecting per-token cross-entropy loss (in nats)
2. Look up how many UTF-8 bytes each target token encodes (special tokens = 0 bytes, excluded)
3. Sum total nats and total bytes across all validation batches
4. Convert nats to bits and normalize by byte count

Implementation: `prepare.py:evaluate_bpb()`, evaluated over `EVAL_TOKENS = 40 * 524288` tokens.

### Theoretical bounds

| Estimate | Value (bpb) | Source |
|----------|-------------|--------|
| Shannon's lower bound (English) | ~0.6 | Shannon 1950, human prediction experiments |
| Shannon's upper bound (English) | ~1.3 | Shannon 1950 |
| Neural LM extrapolation (English) | ~1.12 | Cross Entropy at Infinity (PMC 2020) |
| SOTA compressors (Wikipedia) | ~0.85 | Transformer-based lossless compression |
| Autoresearch baseline (ClimbMix, 4-layer, 5 min) | ~1.73 | Current experiment |

The ClimbMix-400B dataset is mixed web text (multilingual, code, noisy), which has higher entropy than clean English prose. The 1.73 baseline reflects both the data difficulty and the small model/short training budget.

## Consequences

**Benefits:**
- Vocabulary-size-independent: changing tokenizer doesn't invalidate prior results
- Architecture-independent: any model that predicts tokens can be compared
- Intuitive: measures compression efficiency in bits per byte of source text
- Aligns with information-theoretic fundamentals

**Tradeoffs:**
- Slightly more expensive to compute than raw loss (requires byte-length lookup per token)
- Not directly comparable to published perplexity numbers without conversion
- Depends on the specific validation dataset — results are comparable within autoresearch but not directly to external benchmarks using different data

## References

- Shannon, C.E. (1950). "Prediction and Entropy of Printed English"
- [Cross Entropy of Neural Language Models at Infinity](https://pmc.ncbi.nlm.nih.gov/articles/PMC7512401/)
- [Matt Mahoney - Refining the Estimated Entropy of English](https://mattmahoney.net/dc/entropy1.html)
- Implementation: `prepare.py:evaluate_bpb()` (lines 343-365)

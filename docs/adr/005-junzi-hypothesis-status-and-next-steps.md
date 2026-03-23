# ADR-005: Junzi Hypothesis — Final Status and Conclusions

**Status**: Concluded
**Date**: 2026-03-20 (updated 2026-03-23)
**Depends on**: ADR-002, ADR-003, ADR-004

## The Junzi Hypothesis

The [Junzi Alignment hypothesis](https://augustinchan.dev/posts/2026-01-25-junzi-alignment-initial-weights-hypothesis) (Chan, 2026) proposes that AI alignment can emerge from initial conditions — weight initialization and training curriculum — rather than exclusively from post-hoc methods like RLHF. Drawing from Confucian philosophy, it frames alignment as intrinsic character cultivation, predicting:

1. **Seed-dependent alignment variance** — different initializations produce measurably different behavioral dispositions
2. **Optimal learning curricula** follow discoverable sequences (specifically the King Wen I-Ching ordering) that cultivate capability through structured exposure
3. **RLHF efficiency** varies by initialization — "aligned-predisposed" seeds need less safety training

The [King Wen AGI Framework](https://github.com/augchan42/king-wen-agi-framework) provides the mathematical basis, arguing the King Wen sequence optimizes Bayesian surprise for meta-learning curricula.

## All Experiments Completed

| # | Experiment | ADR | Result |
|---|---|---|---|
| 1 | King Wen as LR modulation | ADR-002 | **Hurts** — worse than baseline at all amplitudes (0.15, 0.3, 0.5). Outside seed noise. |
| 2 | Seed behavioral sensitivity (30 seeds) | ADR-004 | **Negligible** — within-seed noise dominates. No multi-dimensional "character traits." |
| 3 | King Wen as curriculum ordering | ADR-003 | **Worst of all orderings** — random, easy-to-hard, hard-to-easy, and Shao Yong all outperform it. |

### Summary of findings

**Prediction 1 (Seed-dependent alignment variance):** Not supported at this scale. 30-seed sweep with 4,500 text samples shows between-seed variance ratio < 0.21 on all behavioral metrics. PCA reveals a single "verbosity axis" (PC1 = 66.4% variance), not multi-dimensional character traits. val_bpb does not correlate with behavioral metrics (max |r| = 0.34).

**Prediction 2 (King Wen as optimal curriculum):** Not supported. Tested as both LR modulation (ADR-002) and data curriculum ordering (ADR-003). In both cases, King Wen's defining properties — high variance and zero autocorrelation — actively harm training. As LR modulation, the high variance destabilizes gradient updates. As curriculum ordering, the unpredictable difficulty transitions prevent the model from building on recent learning.

**Prediction 3 (RLHF efficiency):** Not testable on this hardware (no RLHF pipeline).

## Mainstream Research Context

Recent papers provide directional support for the *general principle* that initial conditions matter for alignment, but through different mechanisms than the Junzi hypothesis proposes:

**Supporting the general principle:**
- [Assessing Macro and Micro Effects of Random Seeds on Fine-Tuning LLMs](https://arxiv.org/html/2503.07329v1) (Mar 2025) — Seed-dependent behavioral variance IS real at fine-tuning scale (BERT/RoBERTa), just not at our 4-layer pretraining scale.
- [When Should We Introduce Safety Interventions During Pretraining?](https://arxiv.org/html/2601.07087) (Feb 2026) — Data ordering during training IS a curriculum design choice that affects alignment robustness.
- [Safety Pretraining](https://arxiv.org/abs/2504.16980) (Apr 2025) — Building safety into pretraining data IS more robust than post-hoc RLHF.

**Contradicting King Wen specifically:**
- [Beyond Random Sampling: Curriculum Learning for LM Pretraining](https://huggingface.co/papers/2506.11300) (Jun 2025) — Simple difficulty metrics (compression ratio, lexical diversity, readability) with easy-to-hard ordering work. No evidence for exotic anti-habituation profiles.
- [How LR Decay Wastes Your Best Data in Curriculum-Based Pretraining](https://huggingface.co/papers/2511.18903) (Nov 2025) — Curriculum learning requires co-designed LR schedules. The LR-curriculum interaction is a first-order concern, not the specific ordering pattern.

**Key insight:** The mechanism that makes initial conditions matter is *data curriculum and representation geometry* — not the specific mathematical properties of the King Wen sequence. Simple, well-understood approaches (easy-to-hard ordering, moderate LR decay, data quality sorting) outperform exotic sequences.

## Hardware Constraints

**Current setup:** NVIDIA RTX 2060, 6 GB VRAM

| Constraint | Impact |
|---|---|
| 6 GB VRAM | Forces fp32 (no bf16 on Turing), DEPTH=4, DEVICE_BATCH_SIZE=16 |
| Turing architecture | No FlashAttention 3, must use SDPA |
| 5-min time budget | ~131K tokens/step, ~90-124 steps per run |
| Model size | 4-layer GPT, ~256-dim, ~11.5M params |

The seed sensitivity literature (arXiv:2503.07329) found meaningful behavioral divergence at BERT/RoBERTa scale with fine-tuning — a regime we cannot reach on this hardware. The curriculum learning literature (arXiv:2506.11300) tested at 1B+ parameter scale. Our 11.5M parameter model may simply lack the capacity for these effects to manifest.

## What This Work Contributes

Despite negative results, this body of work is valuable:

1. **Rigorous experimental methodology** — Each experiment has proper controls, the seed sweep (ADR-004) established baseline variance, and the curriculum experiment (ADR-003) tested 6 orderings with a fair comparison structure.

2. **Reusable tooling** — `sweep_seeds.py`, `sample.py`, `score.py`, and `analyze_seeds.py` form a behavioral evaluation pipeline that can test future hypotheses.

3. **Implementation lessons** — The torch.compile + GPU tensor cloning interaction (ADR-003 v1) is a non-obvious pitfall. The CPU pinned memory buffer approach is a reusable pattern for dataloader wrappers.

4. **Literature synthesis** — ADR-003 and ADR-005 connect the King Wen hypothesis to mainstream curriculum learning, safety pretraining, and seed sensitivity research, providing context for future work.

5. **Constraints on the hypothesis space** — Three negative results at small scale don't falsify the Junzi hypothesis, but they establish that King Wen's statistical properties (high variance, zero autocorrelation) do not help at 4-layer / 5-minute scale. Any future work must either use larger models or propose a different mechanism.

## Possible Future Directions

If pursuing the Junzi hypothesis further:

| Direction | Feasibility (RTX 2060) | Expected value |
|---|---|---|
| DEPTH=6 seed sensitivity probe | Likely feasible | Low — probably too small still |
| 15-min training budget | Feasible, just slower | Low — more time unlikely to change the picture |
| Curriculum with co-designed LR (no warmdown) | Feasible | Medium — literature says this matters, but KW was worst ordering regardless |
| Fine-tune a pretrained model with KW curriculum | Not feasible (VRAM) | High — this is where seed effects are documented |
| Test at 1B+ scale on cloud GPU | Feasible with budget | High — matches the scale of positive curriculum results |

## Decision

**The Junzi hypothesis experiments are concluded at this scale.** Three rigorous experiments all returned negative results. The King Wen sequence's statistical properties do not translate to useful training signals for a 4-layer GPT.

The broader principle — that initial conditions and data ordering matter for alignment — has mainstream support but operates through simpler mechanisms (data quality sorting, easy-to-hard curriculum, moderate LR co-design) than the King Wen sequence provides.

Further work on this hypothesis would require either:
- **Larger models** (1B+ params) where the loss landscape has richer attractor structure
- **A different application domain** for King Wen — perhaps weight initialization patterns rather than training dynamics
- **A refined theory** that specifies when and why anti-habituation helps, grounded in the curriculum learning literature rather than I-Ching numerology

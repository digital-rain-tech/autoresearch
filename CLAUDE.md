# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Autoresearch is an autonomous AI research framework by Andrej Karpathy. AI agents iteratively improve a small LLM by modifying `train.py`, running 5-minute training experiments, and keeping or discarding changes based on validation performance (val_bpb — bits per byte, lower is better).

## Commands

```bash
uv sync                          # Install dependencies
uv run prepare.py                # One-time data download + tokenizer training (~2 min)
uv run train.py                  # Run a single training experiment (~5 min)
uv run train.py > run.log 2>&1   # Run and capture output
grep "^val_bpb:\|^peak_vram_mb:" run.log  # Extract results
```

No test suite, linter, or CI exists. The "test" is whether val_bpb improves.

## Architecture

### Core (3 files — the autoresearch framework)
- **`prepare.py`** — Fixed constants, data prep, tokenizer, dataloader, evaluation utilities. **Never modify this file.**
- **`train.py`** — GPT model, Muon+AdamW optimizer, training loop. **This is the only file agents edit.**
- **`program.md`** — Experiment instructions written by humans for agents. Agents read this, humans edit it.

### Experiment Tooling
- **`model_def.py`** — GPT model classes extracted from `train.py` (must stay in sync). Used by `sample.py` to load checkpoints without triggering training.
- **`sample.py`** — Text generation from trained checkpoints. 15 fixed prefixes, configurable temperature/top-k.
- **`score.py`** — Behavioral metrics on generated text (lexical diversity, distinct-n, entropy, degeneration, etc.).
- **`sweep_seeds.py`** — Orchestrates seed sweep: train → checkpoint → sample → score → aggregate.
- **`analyze_seeds.py`** — Deep analysis of seed behavioral metrics (variance decomposition, outlier detection, PCA clustering, prompt sensitivity, val_bpb correlation).
- **`king_wen_schedules.py`** — LR schedule variants and King Wen surprise values. Used by ADR-002 (LR) and ADR-003 (curriculum).

### train.py internals
- GPT with RoPE, optional sliding window attention, value embeddings (ResFormer-style), per-layer scaling
- Hybrid optimizer: Muon for 2D matrix params, AdamW for embeddings/scalars/lm_head
- Key hyperparams: `DEPTH`, `ASPECT_RATIO`, `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`, `MATRIX_LR`, `WARMDOWN_RATIO`
- Training runs for exactly `TIME_BUDGET` seconds (300s default, set in `prepare.py`)
- Seed and checkpoint path configurable via env vars: `AUTORESEARCH_SEED`, `AUTORESEARCH_CHECKPOINT_PATH`
- Curriculum ordering configurable via env var: `AUTORESEARCH_CURRICULUM` (sequential, random, easy_to_hard, hard_to_easy, shao_yong, king_wen)
- Curriculum buffering uses CPU pinned memory with single-tensor GPU yield (avoids torch.compile interaction)

### prepare.py constants
- `MAX_SEQ_LEN = 2048`, `TIME_BUDGET = 300`, vocab size 8192
- Data: ClimbMix-400B dataset cached at `~/.cache/autoresearch/`

## The Experiment Loop

1. Read `program.md` for what to run next
2. Modify `train.py` (typically just the LR schedule or hyperparams)
3. `git commit` with descriptive message
4. `uv run train.py > run.log 2>&1`
5. Extract val_bpb and peak_vram_mb from run.log
6. Append to `results.tsv` (format: `commit | val_bpb | memory_gb | status | description`)
7. If val_bpb improved → keep commit. If not → `git reset --hard` to discard.

## Current Branch Context

Branch `autoresearch/seed-sensitivity` contains the completed Junzi hypothesis experiments:
- **ADR-002**: King Wen LR schedule — not supported (all amplitudes worse than baseline)
- **ADR-003**: King Wen curriculum ordering — worst of all orderings tested
- **ADR-004**: Seed behavioral sensitivity — negligible at 4-layer scale (30-seed sweep)
- **ADR-005**: Overall status — three negative results; hypothesis requires larger scale to test meaningfully

## Key Constraints

- Only `train.py` is agent-modifiable; `prepare.py` and `program.md` are not touched by agents
- val_bpb is the single metric — vocab-size-independent, comparable across architecture changes
- Fixed 5-min time budget makes experiments comparable regardless of what changes
- Single NVIDIA GPU only (no distributed training)
- Package manager is `uv`, not pip

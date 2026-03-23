# ADR-007: Game-Theoretic Research — King Wen in Discrete Strategy

**Status**: Proposed
**Date**: 2026-03-24
**Depends on**: ADR-005 (Junzi hypothesis concluded), warringstates-day ADR-002 (game engine architecture)

## Context

ADR-005 concluded the Junzi hypothesis experiments with three negative results at 4-layer / 5-minute scale. The King Wen sequence's statistical properties (high variance, zero autocorrelation) do not help continuous gradient-based LLM training. However, ADR-005 explicitly noted:

> Further work on this hypothesis would require either **larger models** (1B+ params) or **a different application domain** for King Wen.

The Warring States game engine (warringstates-day ADR-002) IS that different domain. It shifts the testbed from continuous optimization (gradient descent on language modeling loss) to **discrete multi-agent strategic decision-making** — a domain where unpredictability and diverse state transitions are strategic *advantages*, not liabilities.

This ADR proposes research directions that bridge autoresearch's experimental methodology with game-theoretic evaluation, using OpenSpiel (DeepMind) as the algorithmic foundation.

## Why the Domain Shift Matters

| Property | LLM Training (ADR-002/003) | Multi-Agent Game |
|----------|---------------------------|------------------|
| Action space | Continuous (gradient steps) | Discrete (ally/attack/fortify/reform/betray) |
| High variance | Destabilizes optimizer | Creates unpredictable opponents |
| Zero autocorrelation | Prevents building on recent learning | Prevents opponents from modeling your strategy |
| Anti-habituation | Disrupts convergence | Disrupts opponent exploitation |
| Sequence structure | No clear mapping to loss landscape | Maps to state transitions and decision cycles |

King Wen's defining properties — the ones that actively *hurt* LLM training — are exactly what game theory values in mixed strategies. A strategy that is predictable (high autocorrelation) is exploitable.

## Pareto Analysis: Effort vs Research Value

Building the full Warring States engine (warringstates-day ADR-002 Phases 1-5) is months of work before any research results. This ADR takes a **Pareto-optimal approach**: test the core hypothesis on existing OpenSpiel games first, then invest in the custom engine only if results warrant it.

| Approach | Effort | Novelty | P(signal) | Time to first result |
|----------|--------|---------|-----------|---------------------|
| Full Warring States engine, then test | Very High | Very High | Medium | Months |
| **King Wen priors on existing OpenSpiel games** | **Low-Med** | **High** | **Medium** | **~1 week** |
| Adaptive bandit curriculum in train.py | Very Low | Medium | Medium | ~1 day |

The middle row dominates: OpenSpiel ships ~100 games with imperfect information, simultaneous actions, and built-in exploitability computation. We can test King Wen's value in discrete strategy *today* without building anything from scratch.

## Strategy: Two Parallel Workstreams

### Workstream A — Adaptive Bandit Curriculum in train.py (autoresearch repo)

**Gap addressed**: ADR-003/006a tested six *static* curriculum orderings and found effects smaller than seed noise. But *adaptive* curriculum — where the ordering responds to live model performance — was never tested.

**What**: Modify `train.py`'s batch selection to use UCB1 (Upper Confidence Bound) based on per-batch loss during training. Instead of a fixed ordering, the model trains on whichever data it currently learns most from.

**Implementation** (~30 lines in train.py):
1. Partition the 64-batch curriculum buffer into K=8 difficulty buckets (using existing compression-ratio scoring from ADR-006a)
2. Track running average loss-reduction per bucket
3. At each buffer refill, select buckets using UCB1: `score = mean_improvement + c * sqrt(ln(total) / n_bucket)`
4. Compare against: sequential baseline, random, and best static ordering (random, from ADR-006a)

**Success criterion**: Adaptive ordering produces val_bpb improvement outside seed noise (> 0.04 bpb better than random ordering, measured over 5+ seeds).

**Why it's Pareto-optimal**: Reuses ALL existing infrastructure (train.py, curriculum buffer, compression-ratio scoring, sweep_seeds.py). One day of work. Answers a clean open question.

### Workstream B — King Wen Priors on Existing OpenSpiel Games (new repo)

**Gap addressed**: All prior King Wen experiments tested it on continuous optimization. The hypothesis that King Wen helps in *discrete strategic decisions* has never been tested.

**What**: Test King Wen sequence-informed policy priors on 2-3 existing OpenSpiel games that share structural features with the planned Warring States game. No custom game engine needed.

#### Game Selection

Candidate OpenSpiel games, chosen for structural similarity to Warring States:

| Game | Players | Info | Simultaneous? | Why relevant |
|------|---------|------|---------------|-------------|
| **Goofspiel** | 2-3 | Perfect | Yes (simultaneous bids) | Simultaneous action selection mirrors Warring States diplomacy phase |
| **Leduc Poker** | 2 | Imperfect | Sequential with hidden info | Hidden intent + limited action space + bluffing |
| **Kuhn Poker** | 2 | Imperfect | Sequential with hidden info | Simplest imperfect-info game; good for validating methodology |
| **Sheriff** | 2 | Imperfect | Yes (negotiation + hidden) | Deception, trust, negotiation — closest to diplomacy dynamics |
| **Phantom Tic-Tac-Toe** | 2 | Imperfect | Sequential with hidden state | Imperfect information over a simple substrate |

Start with **Kuhn Poker** (simplest, exploitability is analytically known) and **Goofspiel** (simultaneous actions, closest to Warring States).

#### King Wen Policy Prior Design

Map game state to a hexagram index, then use the King Wen sequence position to bias action probabilities:

```
game_state → feature_vector → hexagram_index (mod 64) → KW_position → action_weights
```

Three candidate mappings (must be pre-registered before experiments):

1. **Hash mapping**: Hash game state features to hexagram index. Simple, arbitrary, tests whether *any* structured sequence helps.
2. **Trigram mapping**: Map state features to upper/lower trigrams based on semantic correspondence (e.g., resource level → Earth/Mountain/Water trigram). Tests whether the trigram structure encodes useful state abstractions.
3. **Sequential mapping**: Use move number mod 64 as index into King Wen sequence. Tests whether the sequence's temporal structure (pair relationships, surprise profile) helps across a game's trajectory.

Each mapping produces action weights: hexagram's associated line meanings (moving lines) bias toward defensive (yin) or aggressive (yang) actions. The bias is soft — a temperature parameter controls how strongly King Wen overrides the base policy.

#### Experimental Design

For each game:

| Condition | Agent | What it tests |
|-----------|-------|---------------|
| Treatment | CFR/MCTS + King Wen prior (trigram mapping) | Does King Wen structure help? |
| Control 1 | CFR/MCTS alone (no prior) | Pure ML baseline |
| Control 2 | CFR/MCTS + scrambled King Wen | Does *any* fixed sequence help? |
| Control 3 | CFR/MCTS + random prior | Does *any* structured bias help? |
| Control 4 | Random agent | Lower bound |

**Metrics** (all available in OpenSpiel out of the box):
- **Exploitability / NashConv**: Distance from Nash equilibrium. The core metric — does King Wen produce less exploitable strategies?
- **Win rate**: Against each control, over 10K+ games per matchup
- **Convergence speed**: How many iterations to reach a given exploitability threshold?

**Statistical rigor**: p < 0.05, confidence intervals, effect size (Cohen's d), Bonferroni correction for multiple comparisons across games and mappings.

**Success criterion**: King Wen-biased agent achieves statistically significant lower exploitability than pure ML agent on at least one game, with effect size d > 0.3.

#### Why Existing Games First

The full Warring States engine (warringstates-day ADR-002) has:
- 7 asymmetric players with unique resource models
- 5 action types with complex resolution
- 4 resources per state (territory, army, treasury, stability)
- Custom lore instrumentation

All of this is interesting but irrelevant to the core question: **does King Wen help in discrete strategy?** Testing on Kuhn Poker and Goofspiel isolates this question from game-balance confounds, asymmetry tuning, and implementation bugs. If King Wen doesn't help on simple games, it won't help on complex ones either.

## Research Directions (retained from initial analysis)

The seven directions from the initial proposal remain valid but are now sequenced behind the Pareto-optimal entry points:

### Tier 1: Do First (Workstreams A + B above)

- **Adaptive bandit curriculum** in train.py (Workstream A)
- **King Wen priors on existing OpenSpiel games** (Workstream B)

### Tier 2: Do If Workstream B Shows Signal

These require the custom Warring States engine and are justified only if King Wen shows measurable benefit on simpler games:

- **Direction 1: Warring States with King Wen Han agent** — The full experiment from warringstates-day ADR-002 Phase 4. Treatment vs 3 controls, Han as maximum-difficulty test case.
- **Direction 5: PSRO population evaluation** — Does King Wen agent survive in a game-theoretically rigorous population? Requires tournament infrastructure.
- **Direction 6: Exploitability as diagnostic** — NashConv measurement on 7-player game. Approximate via best-response computation.

### Tier 3: Do If Tier 2 Produces Results

- **Direction 3: Regret-matched meta-loop** — CFR over experiment configurations. Useful once we have enough variants to select among.
- **Direction 4: Minimax policy scheduling** — Game-theoretic explanation for why simple strategies win. Requires the N×M payoff matrix.
- **Direction 7: Replicator dynamics for architecture search** — Evolutionary population of agent architectures. Interesting but tangential to King Wen hypothesis.

### Tier 2a: Independent of Game Engine (can run anytime)

- **Direction 2: Adaptive curriculum for agent training** — Bandit-based scenario selection during RL training. Can apply to any OpenSpiel game, not just Warring States.

## Phasing

```
Week 1 (parallel):
├── Workstream A: Adaptive bandit curriculum in train.py
│   ├── Implement UCB1 batch selection (~30 lines)
│   ├── Run 5-seed sweep: adaptive vs random vs sequential
│   └── Result: does adaptive curriculum help at 4-layer scale?
│
└── Workstream B: King Wen priors on existing OpenSpiel games
    ├── Set up OpenSpiel environment
    ├── Implement King Wen policy prior (3 mapping variants)
    ├── Pre-register hexagram-to-action mappings
    └── Run Kuhn Poker + Goofspiel experiments

Week 2:
├── Analyze Workstream A results → append to ADR-003 or new ADR-008
├── Analyze Workstream B results → new ADR-009
└── Decision gate: do King Wen priors help in discrete strategy?
    ├── YES → proceed to Tier 2 (build Warring States engine)
    └── NO → King Wen hypothesis is concluded across both domains
         └── Pivot: adaptive curriculum (Direction 2) + regret matching
             (Direction 3) as standalone research on existing games

Week 3+ (conditional on Week 2 results):
├── If YES: Warring States engine Phase 1-2 (warringstates-day ADR-002)
└── If NO: Write up negative results, explore Directions 2-4 as
    game-theoretic training optimization research (no King Wen dependency)
```

## Relation to Prior Work

| Prior ADR | Finding | How This ADR Builds On It |
|-----------|---------|--------------------------|
| ADR-002 | King Wen LR hurts training | Domain shift: test in discrete decisions, not continuous gradients |
| ADR-003 | Static curriculum effects < noise | Workstream A tests *adaptive* curriculum |
| ADR-004 | Seed sensitivity negligible | Direction 6 measures *exploitability* instead of variance |
| ADR-005 | Junzi hypothesis needs different domain | Game theory IS the different domain |
| ADR-006a | Ordering effects metric-dependent | Game tournaments provide unambiguous win/loss metrics |

## Hardware Considerations

Both workstreams run comfortably on existing hardware:

| Resource | Workstream A (train.py) | Workstream B (OpenSpiel) |
|----------|------------------------|-------------------------|
| GPU | Same as before (RTX 2060) | Minimal (no neural nets in Tier 1) |
| CPU | Same as before | Bottleneck (game simulation, CFR iterations) |
| Time per experiment | 5 min (1 training run) | Seconds-minutes (Kuhn/Goofspiel are tiny) |
| Metric noise | ±0.04 bpb (seed variance) | ±~0.1% (10K games, analytical exploitability for Kuhn) |

Kuhn Poker's Nash equilibrium is analytically known (alpha = 1/3 for optimal bluffing frequency). This means exploitability can be computed exactly, not approximated — a much cleaner signal than val_bpb.

## Open Questions

1. **King Wen → action mapping**: The trigram mapping (mapping 2) has the strongest theoretical basis but is the most complex. Start with hash mapping (mapping 1) as a sanity check, then test trigram mapping if hash shows any signal.
2. **Prior temperature**: How strongly should King Wen bias the base policy? Too strong and it overrides learning; too weak and it's noise. Test temperatures [0.1, 0.3, 0.5, 1.0] and report the Pareto frontier of exploitability vs convergence speed.
3. **Game selection**: Goofspiel may be too simple (perfect information) to test imperfect-info hypotheses. Sheriff is a better Warring States proxy but more complex. Start simple, expand if needed.
4. **Negative result plan**: If neither workstream produces signal, the King Wen hypothesis is concluded across both continuous and discrete domains. The game-theoretic infrastructure (OpenSpiel setup, tournament scripts, exploitability measurement) remains valuable for non-King-Wen research on adaptive curriculum and population-based training.

## Decision

Adopt a **Pareto-optimal, gate-based approach**:

1. **Run Workstreams A and B in parallel** (~1 week). Both are low-effort and high-information-value.
2. **Workstream A** (adaptive bandit curriculum in train.py) addresses the open question from ADR-003 with minimal implementation.
3. **Workstream B** (King Wen on existing OpenSpiel games) tests the discrete-strategy hypothesis without building a custom game engine.
4. **Decision gate at Week 2**: Only invest in the full Warring States engine (warringstates-day ADR-002) if Workstream B shows King Wen provides measurable benefit on simpler games.

This avoids the trap of building months of infrastructure before knowing whether the hypothesis has legs in this domain. If King Wen helps on Kuhn Poker, it justifies the Warring States engine. If it doesn't, we've saved months and have a clean negative result across both continuous and discrete domains.

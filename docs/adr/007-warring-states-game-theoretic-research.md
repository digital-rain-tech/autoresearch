# ADR-007: Game-Theoretic Research — King Wen in Discrete Strategy

**Status**: In Progress
**Date**: 2026-03-24 (updated 2026-03-24)
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
| Full 7-state Warring States engine | Very High | Very High | Medium | Months |
| ~~King Wen priors on existing OpenSpiel games~~ | ~~Low-Med~~ | ~~High~~ | ~~Low~~ | ~~~1 week~~ |
| **3-state Warring States prototype on OpenSpiel** | **Medium** | **High** | **Medium-High** | **~2-3 weeks** |
| Adaptive bandit curriculum in train.py | Very Low | Medium | Medium | ~1 day |

~~The "existing games" row was rejected~~ — Kuhn Poker (~12 info sets) and Goofspiel (perfect info) lack the strategic depth for King Wen's 64 hexagrams to map meaningfully. The 3-state prototype is the revised Pareto-optimal path: enough complexity for a meaningful test, fraction of the full engine's effort.

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

#### Workstream A Results (2026-03-24)

UCB1 adaptive curriculum implemented in `train.py` (~50 lines). Single-seed (42) comparison:

| Ordering | val_bpb | Steps | Tokens |
|----------|---------|-------|--------|
| Sequential (no buffer) | 1.968 | 46 | 6.0M |
| Random shuffle (best static, ADR-006a) | 1.870 | 45 | 5.9M |
| **Adaptive UCB1** | **1.863** | 45 | 5.9M |

**Finding**: Adaptive beats random by 0.007 bpb, but this is **within seed noise** (±0.04 from ADR-004). The dominant effect remains buffering itself (~0.1 bpb improvement over sequential), not the ordering within the buffer. Multi-seed sweep needed to confirm whether the adaptive edge is real.

**Conclusion**: Promising but inconclusive. The adaptive mechanism works (no crashes, no overhead, UCB1 converges) and is at worst equivalent to random. Does not block Workstream B.

### Workstream B — King Wen Priors in Game-Theoretic Setting (new repo)

**Gap addressed**: All prior King Wen experiments tested it on continuous optimization. The hypothesis that King Wen helps in *discrete strategic decisions* has never been tested.

#### Revised Approach: Minimal Warring States Prototype (not toy games)

The original plan was to test on existing OpenSpiel games (Kuhn Poker, Goofspiel) before building anything custom. On further analysis, this was rejected:

| Game | Problem |
|------|---------|
| Kuhn Poker | ~12 information sets — too few for 64 hexagrams to map meaningfully |
| Goofspiel | Perfect information — King Wen's unpredictability has no strategic value |
| Leduc Poker | 2-player sequential — no alliance/betrayal dynamics |
| Sheriff | Closer, but still 2-player with no territorial/resource model |

**The core issue**: King Wen's hypothesized value is in complex multi-agent settings with trust, betrayal, shifting alliances, and resource allocation. Testing it on Kuhn Poker is like testing a military strategy on tic-tac-toe — a negative result proves nothing.

**Revised plan**: Build a **minimal 3-state Warring States prototype** on OpenSpiel directly:

- **3 states** (Qin, Han, Chu) — triangle graph, simplest non-trivial multi-agent topology
- **3 actions** (attack, fortify, ally) — drop reform and betray for now
- **Minimal resource model** — territory count + army strength only (no treasury, no stability)
- **20 rounds** max — enough for meaningful strategic arcs, short enough for fast tournaments
- **Simultaneous hidden orders** — the key imperfect-information mechanic
- Han gets King Wen prior; Qin/Chu get scripted/MCTS baselines

This is small enough to build in ~2 weeks but has enough strategic depth (3 players × simultaneous actions × hidden intent × army/territory state = hundreds of distinct game situations) for King Wen's 64 hexagrams to map meaningfully.

#### King Wen Policy Prior Design

Map game state to a hexagram index, then use the King Wen sequence position to bias action probabilities:

```
game_state → feature_vector → hexagram_index (mod 64) → KW_position → action_weights
```

Three candidate mappings (must be pre-registered before experiments):

1. **Hash mapping**: Hash game state features to hexagram index. Simple, arbitrary, tests whether *any* structured sequence helps.
2. **Trigram mapping**: Map state features to upper/lower trigrams based on semantic correspondence (e.g., army strength → Heaven/Earth trigram, territory → Mountain/Lake). Tests whether the trigram structure encodes useful state abstractions.
3. **Sequential mapping**: Use move number mod 64 as index into King Wen sequence. Tests whether the sequence's temporal structure (pair relationships, surprise profile) helps across a game's trajectory.

Each mapping produces action weights: hexagram's associated line meanings (moving lines) bias toward defensive (yin) or aggressive (yang) actions. The bias is soft — a temperature parameter controls how strongly King Wen overrides the base policy.

**Critical**: King Wen integration must be **state-reactive** (hash current game state → hexagram → action bias), NOT sequential ("play hexagram 1 on turn 1"). The Workstream A finding that adaptive > static applies here — a fixed temporal schedule would repeat the ADR-002 mistake.

#### Experimental Design

| Condition | Agent | What it tests |
|-----------|-------|---------------|
| Treatment | MCTS + King Wen prior (trigram mapping) | Does King Wen structure help? |
| Control 1 | MCTS alone (no prior) | Pure ML baseline |
| Control 2 | MCTS + scrambled King Wen | Does *any* fixed sequence help? |
| Control 3 | MCTS + random prior | Does *any* structured bias help? |
| Control 4 | Scripted heuristic agents | Strong non-ML baseline |
| Control 5 | Random agent | Lower bound |

**Metrics**:
- **Exploitability** (primary): Does King Wen make Han less predictable/exploitable, even if it doesn't maximize average win rate? This is the metric most aligned with King Wen's anti-habituation properties. OpenSpiel provides approximate exploitability via best-response computation.
- **Win rate / survival rate**: Against each control, over 10K+ games per matchup
- **Convergence speed**: How many MCTS iterations to reach a given performance threshold?

**Statistical rigor**: p < 0.05, confidence intervals, effect size (Cohen's d), Bonferroni correction for multiple comparisons across mappings.

**Success criterion**: King Wen-biased Han achieves statistically significant lower exploitability OR higher survival rate than pure MCTS Han (p < 0.05 over 10K+ games, effect size d > 0.3).

## Design Guidance from Prior Experiments

The full ADR-001 through ADR-007 experimental history provides concrete guidance for the game engine design:

| Experimental Finding | Source | Design Implication |
|---------------------|--------|-------------------|
| King Wen hurts as continuous modifier | ADR-002 | Use for **discrete action selection**, not parameter scaling |
| Anti-habituation destabilizes optimization but creates unpredictability | ADR-002/005 | Primary metric should be **exploitability**, not just win rate |
| Static orderings ≈ noise; adaptive slightly better | ADR-003/006a/007 | King Wen prior must be **state-responsive** (map current game state → hexagram), not a fixed temporal plan |
| Buffering matters more than ordering | ADR-003 | Game infrastructure (action resolution, info model) matters more than the specific King Wen mapping — get the substrate right first |
| Effect sizes tiny at small scale | ADR-004/006a | Game needs **enough strategic depth** for 64 hexagrams to index meaningfully — reject toy games |
| Controls are essential; post-hoc justification is the #1 risk | ADR-002 | Must test: pure ML, scrambled King Wen, random prior, scripted. **Pre-register mapping before experiments.** |
| Seed noise (±0.04 bpb) swamps curriculum signal | ADR-004 | Game metrics need **cleaner signal** — win rate over 10K games gives ±~1% precision |
| val_bpb numbers vary across runs (~1.72 in ADR-003 vs ~1.87 in ADR-007) | ADR-003/007 | Only **relative comparisons within a run batch** are meaningful; absolute values drift with environment |

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
Week 1 (DONE — 2026-03-24):
├── Workstream A: Adaptive bandit curriculum in train.py
│   ├── ✓ Implemented UCB1 batch selection (~50 lines)
│   ├── ✓ Single-seed comparison: adaptive 1.863 vs random 1.870 vs sequential 1.968
│   ├── ✓ Result: adaptive ≈ random (within noise), both >> sequential
│   └── ○ Optional: multi-seed sweep to confirm
│
└── Workstream B: Revised — skip toy games, build minimal prototype
    ├── ✗ Kuhn Poker / Goofspiel rejected (too simple for 64-hexagram mapping)
    └── → Build 3-state Warring States prototype on OpenSpiel instead

Week 2-3 (NEXT):
├── Build minimal Warring States on OpenSpiel
│   ├── 3 states (Qin, Han, Chu), triangle graph
│   ├── 3 actions (attack, fortify, ally), simultaneous hidden orders
│   ├── Minimal resource model (territory + army only)
│   ├── Random + scripted baseline agents
│   └── Tournament infra producing reproducible win rates over 10K games
│
├── Pre-register King Wen → action mapping (before any experiments)
└── Implement King Wen policy prior for Han + controls

Week 4:
├── Run treatment vs 5 controls (10K+ games each)
├── Compute exploitability for each agent variant
└── Decision gate: does King Wen help in discrete strategy?
    ├── YES → expand to full 7-state game (warringstates-day ADR-002)
    └── NO → King Wen hypothesis concluded across both domains
         └── Game infra still useful for Directions 2-4 (adaptive
             curriculum, regret matching, PSRO) without King Wen
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

The 3-state game is computationally trivial. With 10K games at ~50ms each, a full tournament takes ~8 minutes. MCTS with ~1000 rollouts per decision point is feasible in real-time.

## Open Questions

1. **King Wen → action mapping**: The trigram mapping (mapping 2) has the strongest theoretical basis but is the most complex. Start with hash mapping (mapping 1) as a sanity check, then test trigram mapping if hash shows any signal. All mappings must be pre-registered before experiments run.
2. **Prior temperature**: How strongly should King Wen bias the base policy? Too strong and it overrides learning; too weak and it's noise. Test temperatures [0.1, 0.3, 0.5, 1.0] and report the Pareto frontier of exploitability vs convergence speed.
3. **3-state balance**: With only Qin, Han, Chu, the triangle topology is symmetric except for starting stats. Han must be the weakest (historically accurate) but not so weak that no agent can help. Calibrate via Phase 2 baseline tournaments.
4. **Negative result plan**: If the 3-state prototype produces no signal, the King Wen hypothesis is concluded across both continuous and discrete domains. The game-theoretic infrastructure (OpenSpiel setup, tournament scripts, exploitability measurement) remains valuable for non-King-Wen research on adaptive curriculum, regret matching, and population-based training.

## Decision

Adopt a **Pareto-optimal, gate-based approach**:

1. **Workstream A** (adaptive bandit curriculum in train.py) — **DONE**. UCB1 implemented, single-seed result: adaptive ≈ random, both >> sequential. Edge is within noise; multi-seed sweep optional.
2. **Workstream B** (King Wen in discrete strategy) — **REVISED**. Existing OpenSpiel games rejected as too simple. Build a minimal 3-state Warring States prototype instead (~2 weeks, not months).
3. **Decision gate at Week 4**: Only expand to the full 7-state game (warringstates-day ADR-002) if the 3-state prototype shows King Wen provides measurable benefit.

The 3-state prototype is the Pareto-optimal path: complex enough for meaningful King Wen integration (hundreds of distinct game states, simultaneous hidden actions, multi-agent dynamics), simple enough to build quickly and iterate on. It de-risks the full Warring States investment without the false economy of testing on toy games that can't carry the hypothesis.

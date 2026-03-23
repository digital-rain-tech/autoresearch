"""
Workstream B: King Wen policy priors on OpenSpiel games.

Tests whether the King Wen sequence provides useful structure for discrete
strategic decision-making, using CFR (Counterfactual Regret Minimization)
with biased initialization on Kuhn Poker and Goofspiel.

ADR-007: The King Wen hypothesis shifts from continuous optimization (where it
failed, ADR-002/003) to discrete multi-agent strategy (where unpredictability
and diverse state transitions are strategic advantages).

Three King Wen → action mappings:
  1. Hash mapping: hash game state → hexagram index → action weights
  2. Trigram mapping: state features → upper/lower trigrams → action weights
  3. Sequential mapping: move number mod 64 → KW position → action weights

Five experimental conditions per game:
  - Treatment: CFR + King Wen prior (each mapping variant)
  - Control 1: CFR alone (no prior)
  - Control 2: CFR + scrambled King Wen
  - Control 3: CFR + random prior
  - Control 4: Random agent (lower bound)

Metrics:
  - Exploitability (NashConv): distance from Nash equilibrium
  - Convergence speed: iterations to reach exploitability threshold

Usage:
    uv run king_wen_openspiel.py                        # Run all experiments
    uv run king_wen_openspiel.py --game kuhn_poker      # Single game
    uv run king_wen_openspiel.py --iterations 10000     # More iterations
    uv run king_wen_openspiel.py --skip-experiments     # Analyze only
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability, external_sampling_mccfr

from king_wen_schedules import KING_WEN_SURPRISE

# ---------------------------------------------------------------------------
# King Wen sequence data
# ---------------------------------------------------------------------------

# Full King Wen ordering (hexagram numbers 1-64 in King Wen sequence)
KING_WEN_ORDER = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
]

# Line types: 0=yin (defensive), 1=yang (aggressive)
# Each hexagram has 6 lines (bottom to top). We use this for trigram mapping.
HEXAGRAM_LINES = {
    i: [(i >> bit) & 1 for bit in range(6)]
    for i in range(64)
}

# Trigram associations (lower 3 bits, upper 3 bits)
TRIGRAM_NAMES = {
    0b000: "earth",    # Kun: receptive, defensive
    0b001: "thunder",  # Zhen: initiative, movement
    0b010: "water",    # Kan: danger, adaptability
    0b011: "lake",     # Dui: joy, exchange
    0b100: "mountain", # Gen: stillness, restraint
    0b101: "fire",     # Li: clarity, action
    0b110: "wind",     # Xun: gentle penetration
    0b111: "heaven",   # Qian: creative, aggressive
}

# Aggression score per trigram (0=defensive, 1=aggressive)
TRIGRAM_AGGRESSION = {
    0b000: 0.1,   # earth: very defensive
    0b001: 0.6,   # thunder: moderate initiative
    0b010: 0.3,   # water: cautious/adaptive
    0b011: 0.5,   # lake: neutral/diplomatic
    0b100: 0.2,   # mountain: restrained
    0b101: 0.7,   # fire: active
    0b110: 0.4,   # wind: gentle
    0b111: 0.9,   # heaven: very aggressive
}


# ---------------------------------------------------------------------------
# King Wen → action mappings
# ---------------------------------------------------------------------------

def kw_hash_mapping(info_state_str, num_actions, temperature=0.3):
    """Mapping 1: Hash game state to hexagram index, derive action weights.

    Tests whether *any* structured, deterministic sequence helps.
    """
    h = int(hashlib.md5(info_state_str.encode()).hexdigest(), 16)
    hex_idx = h % 64
    kw_pos = KING_WEN_ORDER.index(hex_idx + 1)  # position in King Wen sequence
    surprise = KING_WEN_SURPRISE[kw_pos % len(KING_WEN_SURPRISE)]

    # Higher surprise → more aggressive (favor higher-indexed actions)
    weights = []
    for a in range(num_actions):
        action_frac = a / max(num_actions - 1, 1)
        # Bias toward aggressive actions when surprise is high
        bias = temperature * (2 * surprise - 1) * (2 * action_frac - 1)
        weights.append(math.exp(bias))

    total = sum(weights)
    return [w / total for w in weights]


def kw_trigram_mapping(info_state_str, num_actions, temperature=0.3):
    """Mapping 2: Map state features to trigrams, derive aggression bias.

    Tests whether trigram structure encodes useful state abstractions.
    """
    h = int(hashlib.md5(info_state_str.encode()).hexdigest(), 16)
    lower_trigram = h & 0b111
    upper_trigram = (h >> 3) & 0b111

    # Combine trigram aggression scores
    lower_agg = TRIGRAM_AGGRESSION[lower_trigram]
    upper_agg = TRIGRAM_AGGRESSION[upper_trigram]
    aggression = (lower_agg + upper_agg) / 2

    # Map aggression to action distribution
    weights = []
    for a in range(num_actions):
        action_frac = a / max(num_actions - 1, 1)
        bias = temperature * (2 * aggression - 1) * (2 * action_frac - 1)
        weights.append(math.exp(bias))

    total = sum(weights)
    return [w / total for w in weights]


def kw_sequential_mapping(move_number, num_actions, temperature=0.3):
    """Mapping 3: Use move number mod 64 to index King Wen sequence.

    Tests whether the sequence's temporal structure (pair relationships,
    surprise profile) helps across a game's trajectory.
    """
    kw_pos = move_number % 64
    surprise = KING_WEN_SURPRISE[kw_pos % len(KING_WEN_SURPRISE)]

    weights = []
    for a in range(num_actions):
        action_frac = a / max(num_actions - 1, 1)
        bias = temperature * (2 * surprise - 1) * (2 * action_frac - 1)
        weights.append(math.exp(bias))

    total = sum(weights)
    return [w / total for w in weights]


def scrambled_kw_mapping(info_state_str, num_actions, temperature=0.3, seed=12345):
    """Control 2: Scrambled King Wen — same values, random permutation."""
    rng = random.Random(seed)
    scrambled = list(KING_WEN_SURPRISE)
    rng.shuffle(scrambled)

    h = int(hashlib.md5(info_state_str.encode()).hexdigest(), 16)
    idx = h % len(scrambled)
    surprise = scrambled[idx]

    weights = []
    for a in range(num_actions):
        action_frac = a / max(num_actions - 1, 1)
        bias = temperature * (2 * surprise - 1) * (2 * action_frac - 1)
        weights.append(math.exp(bias))

    total = sum(weights)
    return [w / total for w in weights]


def random_prior_mapping(info_state_str, num_actions, temperature=0.3):
    """Control 3: Random but deterministic prior (no King Wen structure)."""
    h = int(hashlib.md5(info_state_str.encode()).hexdigest(), 16)
    rng = random.Random(h)
    surprise = rng.random()

    weights = []
    for a in range(num_actions):
        action_frac = a / max(num_actions - 1, 1)
        bias = temperature * (2 * surprise - 1) * (2 * action_frac - 1)
        weights.append(math.exp(bias))

    total = sum(weights)
    return [w / total for w in weights]


# ---------------------------------------------------------------------------
# CFR with prior-biased regret initialization
# ---------------------------------------------------------------------------

class PriorBiasedCFR:
    """CFR solver with King Wen (or other) prior for initial regret values.

    Uses External Sampling MCCFR which has natural stochasticity (samples
    opponent actions randomly each iteration), giving different results per seed.
    The prior biases initial cumulative regrets toward the mapping's recommendation.
    """

    def __init__(self, game, prior_fn=None, prior_strength=1.0,
                 mapping_type="hash", temperature=0.3, use_mccfr=True):
        self.game = game
        self.prior_fn = prior_fn
        self.prior_strength = prior_strength
        self.mapping_type = mapping_type
        self.temperature = temperature
        self.use_mccfr = use_mccfr
        if use_mccfr:
            self._solver = external_sampling_mccfr.ExternalSamplingSolver(game)
        else:
            self._solver = cfr.CFRSolver(game)
        self._initialized_states = set()
        self._priors_applied = False

    def _apply_priors(self):
        """Apply prior bias to all discovered info states' initial regrets."""
        if self.prior_fn is None:
            return

        if self.use_mccfr:
            infostates = self._solver._infostates
        else:
            infostates = self._solver._info_state_nodes

        for info_state_key in infostates:
            if info_state_key in self._initialized_states:
                continue

            info_state_str = str(info_state_key)

            if self.use_mccfr:
                # MCCFR: _infostates[key] = [regret_array, avg_strategy_array]
                regret_arr = infostates[info_state_key][0]
                num_actions = len(regret_arr)
            else:
                node = infostates[info_state_key]
                num_actions = len(node.cumulative_regret)

            if self.mapping_type == "sequential":
                h = int(hashlib.md5(info_state_str.encode()).hexdigest(), 16)
                weights = self.prior_fn(h % 64, num_actions, self.temperature)
            else:
                weights = self.prior_fn(info_state_str, num_actions, self.temperature)

            # Bias initial cumulative regrets
            uniform = 1.0 / num_actions
            if self.use_mccfr:
                for a in range(num_actions):
                    regret_arr[a] += (weights[a] - uniform) * self.prior_strength
            else:
                for a in range(num_actions):
                    node.cumulative_regret[a] += (weights[a] - uniform) * self.prior_strength

            self._initialized_states.add(info_state_key)

    def evaluate_and_update(self):
        """Run one CFR iteration."""
        if self.use_mccfr:
            self._solver.iteration()
        else:
            self._solver.evaluate_and_update_policy()
        # Apply priors to newly discovered info states after first iteration
        if self.prior_fn is not None:
            self._apply_priors()

    def average_policy(self):
        return self._solver.average_policy()

    @property
    def iterations(self):
        return self._solver._iteration


def compute_exploitability(game, policy):
    """Compute exploitability (NashConv) of a policy."""
    return exploitability.exploitability(game, policy)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

GAMES = {
    "kuhn_poker": {},
    "goofspiel": {"num_cards": 4, "imp_info": True, "points_order": "descending"},
}

CONDITIONS = {
    "kw_hash": ("King Wen hash mapping", kw_hash_mapping, "hash"),
    "kw_trigram": ("King Wen trigram mapping", kw_trigram_mapping, "trigram"),
    "kw_sequential": ("King Wen sequential mapping", kw_sequential_mapping, "sequential"),
    "cfr_only": ("CFR baseline (no prior)", None, None),
    "scrambled_kw": ("Scrambled King Wen", scrambled_kw_mapping, "hash"),
    "random_prior": ("Random prior", random_prior_mapping, "hash"),
}

TEMPERATURES = [0.1, 0.3, 0.5, 1.0]


def run_experiment(game_name, condition_name, num_iterations=1000,
                   temperature=0.3, prior_strength=1.0, seed=42):
    """Run a single experiment: CFR with or without prior, return convergence curve."""
    random.seed(seed)
    np.random.seed(seed)

    game_params = GAMES[game_name]
    if game_name == "goofspiel":
        base_game = pyspiel.load_game("goofspiel", game_params)
        # Goofspiel is simultaneous; CFR needs sequential form
        game = pyspiel.load_game_as_turn_based("goofspiel", game_params)
    else:
        game = pyspiel.load_game(game_name)

    condition = CONDITIONS[condition_name]
    prior_fn = condition[1]
    mapping_type = condition[2]

    solver = PriorBiasedCFR(
        game,
        prior_fn=prior_fn,
        prior_strength=prior_strength,
        mapping_type=mapping_type,
        temperature=temperature,
    )

    # Convergence curve: measure exploitability at log-spaced intervals
    checkpoints = sorted(set(
        [int(10 ** (i * math.log10(num_iterations) / 50)) for i in range(51)]
        + [num_iterations]
    ))
    checkpoints = [c for c in checkpoints if 1 <= c <= num_iterations]

    curve = []
    checkpoint_idx = 0

    for iteration in range(1, num_iterations + 1):
        solver.evaluate_and_update()

        if checkpoint_idx < len(checkpoints) and iteration == checkpoints[checkpoint_idx]:
            policy = solver.average_policy()
            exploit = compute_exploitability(game, policy)
            curve.append({
                "iteration": iteration,
                "exploitability": exploit,
            })
            checkpoint_idx += 1

    final_policy = solver.average_policy()
    final_exploit = compute_exploitability(game, final_policy)

    return {
        "game": game_name,
        "condition": condition_name,
        "condition_desc": condition[0],
        "temperature": temperature,
        "prior_strength": prior_strength,
        "num_iterations": num_iterations,
        "final_exploitability": final_exploit,
        "convergence_curve": curve,
        "seed": seed,
    }


def find_convergence_iteration(curve, threshold):
    """Find first iteration where exploitability drops below threshold."""
    for point in curve:
        if point["exploitability"] <= threshold:
            return point["iteration"]
    return None


def run_all_experiments(games=None, num_iterations=1000, temperatures=None,
                        seeds=None, prior_strength=1.0):
    """Run all conditions across games, temperatures, and seeds."""
    if games is None:
        games = list(GAMES.keys())
    if temperatures is None:
        temperatures = [0.3]  # default: single temperature pilot
    if seeds is None:
        seeds = [42]

    results = []
    total = len(games) * len(CONDITIONS) * len(temperatures) * len(seeds)
    done = 0

    for game_name in games:
        for condition_name in CONDITIONS:
            for temp in temperatures:
                for seed in seeds:
                    t0 = time.time()
                    # No-prior conditions don't use temperature
                    if condition_name in ("cfr_only",) and temp != temperatures[0]:
                        continue

                    result = run_experiment(
                        game_name, condition_name, num_iterations,
                        temperature=temp, prior_strength=prior_strength, seed=seed,
                    )
                    elapsed = time.time() - t0
                    done += 1
                    print(f"  [{done}/{total}] {game_name}/{condition_name} "
                          f"temp={temp} seed={seed}: "
                          f"exploit={result['final_exploitability']:.6f} "
                          f"({elapsed:.1f}s)")
                    results.append(result)

    return results


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def analyze_experiments(results):
    """Produce statistical analysis of experiment results."""
    from scipy import stats

    lines = []
    lines.append("=" * 70)
    lines.append("KING WEN OPENSPIEL ANALYSIS (ADR-007 Workstream B)")
    lines.append("=" * 70)

    # Group by game
    by_game = defaultdict(list)
    for r in results:
        by_game[r["game"]].append(r)

    for game_name, game_results in by_game.items():
        lines.append(f"\n{'='*50}")
        lines.append(f"Game: {game_name}")
        lines.append(f"{'='*50}")

        # Group by condition (across seeds/temperatures)
        by_condition = defaultdict(list)
        for r in game_results:
            key = (r["condition"], r["temperature"])
            by_condition[key].append(r)

        # Summary table
        lines.append(f"\n{'Condition':>20} {'Temp':>5} {'n':>3} {'Exploit (mean)':>14} {'Exploit (std)':>13}")
        lines.append("-" * 60)

        condition_exploits = {}
        for (cond, temp), runs in sorted(by_condition.items()):
            exploits = [r["final_exploitability"] for r in runs]
            n = len(exploits)
            mean = sum(exploits) / n
            std = (sum((x - mean) ** 2 for x in exploits) / max(n - 1, 1)) ** 0.5
            lines.append(f"{cond:>20} {temp:>5.1f} {n:>3} {mean:>14.6f} {std:>13.6f}")
            condition_exploits[(cond, temp)] = exploits

        # Pairwise comparisons: each KW variant vs cfr_only baseline
        baseline_key = None
        for key in condition_exploits:
            if key[0] == "cfr_only":
                baseline_key = key
                break

        if baseline_key is None:
            lines.append("\n  No CFR baseline found — skipping comparisons")
            continue

        baseline_exploits = condition_exploits[baseline_key]

        lines.append(f"\nPairwise comparisons vs CFR baseline:")
        num_comparisons = sum(1 for k in condition_exploits if k[0] != "cfr_only")
        bonferroni_alpha = 0.05 / max(num_comparisons, 1)
        lines.append(f"  Bonferroni-corrected alpha = {bonferroni_alpha:.4f} ({num_comparisons} comparisons)")

        for (cond, temp), exploits in sorted(condition_exploits.items()):
            if cond == "cfr_only":
                continue

            n_treat = len(exploits)
            n_base = len(baseline_exploits)

            if n_treat < 2 or n_base < 2:
                lines.append(f"\n  {cond} (temp={temp}): insufficient data")
                continue

            # Welch's t-test (unequal variances, unpaired)
            t_stat, p_value = stats.ttest_ind(exploits, baseline_exploits, equal_var=False)

            # Cohen's d
            mean_diff = sum(exploits) / n_treat - sum(baseline_exploits) / n_base
            var_treat = sum((x - sum(exploits) / n_treat) ** 2 for x in exploits) / (n_treat - 1)
            var_base = sum((x - sum(baseline_exploits) / n_base) ** 2 for x in baseline_exploits) / (n_base - 1)
            pooled_std = ((var_treat + var_base) / 2) ** 0.5
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            sig = "SIGNIFICANT" if p_value < bonferroni_alpha else "not significant"
            better = "treatment" if mean_diff < 0 else "baseline"

            lines.append(f"\n  {cond} (temp={temp}) vs cfr_only:")
            lines.append(f"    mean exploit: {sum(exploits)/n_treat:.6f} vs {sum(baseline_exploits)/n_base:.6f}")
            lines.append(f"    diff = {mean_diff:+.6f}")
            lines.append(f"    t = {t_stat:.3f}, p = {p_value:.4f}")
            lines.append(f"    Cohen's d = {cohens_d:.3f}")
            lines.append(f"    → {sig} (Bonferroni α={bonferroni_alpha:.4f}). Lower exploit = {better}")

        # Convergence analysis
        lines.append(f"\nConvergence speed (iterations to reach exploit < threshold):")
        # Use 2x the final CFR exploitability as threshold
        if baseline_exploits:
            threshold = 2 * sum(baseline_exploits) / len(baseline_exploits)
            lines.append(f"  Threshold: {threshold:.6f} (2× CFR final mean)")

            for (cond, temp), runs in sorted(by_condition.items()):
                conv_iters = []
                for r in runs:
                    ci = find_convergence_iteration(r["convergence_curve"], threshold)
                    if ci is not None:
                        conv_iters.append(ci)

                if conv_iters:
                    mean_ci = sum(conv_iters) / len(conv_iters)
                    lines.append(f"    {cond:>20} (temp={temp}): mean={mean_ci:.0f} iters ({len(conv_iters)}/{len(runs)} converged)")
                else:
                    lines.append(f"    {cond:>20} (temp={temp}): did not converge")

    # ADR-007 success criterion
    lines.append(f"\n{'='*50}")
    lines.append("ADR-007 SUCCESS CRITERION")
    lines.append(f"{'='*50}")
    lines.append("Criterion: King Wen-biased agent achieves statistically significant")
    lines.append("lower exploitability than pure CFR on at least one game,")
    lines.append("with effect size d > 0.3.")

    criterion_met = False
    for game_name, game_results in by_game.items():
        by_condition = defaultdict(list)
        for r in game_results:
            by_condition[(r["condition"], r["temperature"])].append(r)

        baseline_key = None
        for key in by_condition:
            if key[0] == "cfr_only":
                baseline_key = key
                break
        if baseline_key is None:
            continue

        baseline_vals = [r["final_exploitability"] for r in by_condition[baseline_key]]

        for (cond, temp), runs in by_condition.items():
            if cond in ("cfr_only", "scrambled_kw", "random_prior"):
                continue
            treat_vals = [r["final_exploitability"] for r in runs]
            if len(treat_vals) < 2 or len(baseline_vals) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(treat_vals, baseline_vals, equal_var=False)
            mean_diff = sum(treat_vals) / len(treat_vals) - sum(baseline_vals) / len(baseline_vals)
            var_t = sum((x - sum(treat_vals)/len(treat_vals))**2 for x in treat_vals) / (len(treat_vals)-1)
            var_b = sum((x - sum(baseline_vals)/len(baseline_vals))**2 for x in baseline_vals) / (len(baseline_vals)-1)
            pooled = ((var_t + var_b) / 2) ** 0.5
            d = abs(mean_diff) / pooled if pooled > 0 else 0

            if p_value < 0.05 and mean_diff < 0 and d > 0.3:
                lines.append(f"  CRITERION MET: {cond} on {game_name} (p={p_value:.4f}, d={d:.3f})")
                criterion_met = True

    if not criterion_met:
        lines.append("  CRITERION NOT MET across all games and conditions.")
        lines.append("  King Wen priors do not significantly improve exploitability.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser(description="King Wen OpenSpiel experiments (ADR-007 Workstream B)")
    parser.add_argument("--game", choices=list(GAMES.keys()), default=None,
                        help="Run only this game (default: all)")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="CFR iterations per experiment (default: 1000)")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.3],
                        help="Prior temperatures to test (default: 0.3)")
    parser.add_argument("--seeds", type=int, nargs=2, default=[0, 5],
                        metavar=("START", "END"),
                        help="Seed range [start, end) — default 0 5")
    parser.add_argument("--prior-strength", type=float, default=1.0,
                        help="How strongly prior biases initial regrets (default: 1.0)")
    parser.add_argument("--skip-experiments", action="store_true",
                        help="Skip experiments, only analyze existing results")
    args = parser.parse_args()

    games = [args.game] if args.game else list(GAMES.keys())
    seeds = list(range(args.seeds[0], args.seeds[1]))

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, "king_wen_openspiel.json")

    if not args.skip_experiments:
        print(f"Running experiments: {len(games)} games × {len(CONDITIONS)} conditions "
              f"× {len(args.temperatures)} temps × {len(seeds)} seeds")

        results = run_all_experiments(
            games=games,
            num_iterations=args.iterations,
            temperatures=args.temperatures,
            seeds=seeds,
            prior_strength=args.prior_strength,
        )

        # Save results (without full convergence curves for readability)
        save_results = []
        for r in results:
            r_save = dict(r)
            # Keep only first/last/key points of convergence curve
            curve = r_save.pop("convergence_curve", [])
            r_save["convergence_first"] = curve[0] if curve else None
            r_save["convergence_last"] = curve[-1] if curve else None
            r_save["convergence_midpoints"] = [
                curve[len(curve) // 4],
                curve[len(curve) // 2],
                curve[3 * len(curve) // 4],
            ] if len(curve) >= 4 else curve
            save_results.append(r_save)

        with open(results_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        # Also save full convergence curves as CSV
        curves_path = os.path.join(RESULTS_DIR, "king_wen_convergence.csv")
        with open(curves_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["game", "condition", "temperature", "seed", "iteration", "exploitability"])
            for r in results:
                for point in r["convergence_curve"]:
                    writer.writerow([
                        r["game"], r["condition"], r["temperature"],
                        r["seed"], point["iteration"], point["exploitability"],
                    ])
        print(f"Convergence curves saved to {curves_path}")
    else:
        # Load existing results
        with open(results_path) as f:
            save_results = json.load(f)
        # Reconstruct minimal results for analysis
        results = []
        for r in save_results:
            r["convergence_curve"] = []
            if r.get("convergence_first"):
                r["convergence_curve"].append(r["convergence_first"])
            for mp in r.get("convergence_midpoints", []):
                r["convergence_curve"].append(mp)
            if r.get("convergence_last"):
                r["convergence_curve"].append(r["convergence_last"])
            results.append(r)

    # Analysis
    analysis = analyze_experiments(results)
    print(analysis)

    analysis_path = os.path.join(RESULTS_DIR, "king_wen_openspiel_analysis.txt")
    with open(analysis_path, "w") as f:
        f.write(analysis)
    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    main()

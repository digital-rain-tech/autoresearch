"""
Deep analysis of seed sensitivity in behavioral metrics.

Questions:
1. Are behavioral metrics truly stable across seeds, or do means hide structure?
2. Do some seeds produce qualitatively different text (outlier detection)?
3. Are specific prompts more seed-sensitive than others?
4. Do seeds cluster into behavioral groups?
5. Is there a relationship between val_bpb and behavioral metrics?

Uses only the existing scored data — no GPU needed.
"""

import csv
import os
import json
import math
import numpy as np
from collections import defaultdict
from pathlib import Path

SCORES_DIR = "scores"
RESULTS_DIR = "results"
SAMPLES_DIR = "samples"

METRICS = [
    "word_count", "unique_words", "lexical_diversity", "distinct_1",
    "distinct_2", "distinct_3", "word_entropy", "avg_sentence_length",
    "repetition_rate", "punctuation_rate", "degeneration_score", "vocab_coverage"
]

# Skip repetition_rate (near-zero everywhere) for most analyses
INTERESTING_METRICS = [m for m in METRICS if m != "repetition_rate"]


def load_all_scores():
    """Load all per-sample scores into a structured dict."""
    all_data = {}  # seed -> list of row dicts
    for seed in range(30):
        path = os.path.join(SCORES_DIR, f"seed_{seed}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Convert numeric fields
            for r in rows:
                for m in METRICS:
                    r[m] = float(r[m])
                r["sample_idx"] = int(r["sample_idx"])
            all_data[seed] = rows
    return all_data


def load_val_bpbs():
    """Load val_bpb per seed from aggregated results."""
    bpbs = {}
    path = os.path.join(RESULTS_DIR, "seed_sweep.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            bpbs[int(r["seed"])] = float(r["val_bpb"])
    return bpbs


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_cross_seed_variance(all_data):
    """Compare between-seed vs within-seed variance for each metric."""
    section("1. BETWEEN-SEED vs WITHIN-SEED VARIANCE")
    print("\nIf between-seed variance >> within-seed variance, seeds matter.")
    print("Ratio > 1 means seed choice matters more than sampling noise.\n")

    print(f"{'Metric':<25} {'Between-seed σ':>15} {'Within-seed σ':>15} {'Ratio':>10} {'Signal?':>10}")
    print("-" * 80)

    for metric in INTERESTING_METRICS:
        # Per-seed means
        seed_means = []
        within_vars = []
        for seed, rows in sorted(all_data.items()):
            vals = [r[metric] for r in rows]
            seed_mean = np.mean(vals)
            seed_means.append(seed_mean)
            within_vars.append(np.var(vals))

        between_var = np.var(seed_means)
        mean_within_var = np.mean(within_vars)

        between_std = math.sqrt(between_var)
        within_std = math.sqrt(mean_within_var)

        ratio = between_std / within_std if within_std > 0 else float('inf')
        signal = "YES" if ratio > 0.3 else "no"

        print(f"{metric:<25} {between_std:>15.4f} {within_std:>15.4f} {ratio:>10.3f} {signal:>10}")


def analyze_outlier_seeds(all_data):
    """Find seeds that are statistical outliers on any metric."""
    section("2. OUTLIER SEED DETECTION")
    print("\nSeeds with mean metric > 2σ from the grand mean on any metric.\n")

    seed_profiles = {}  # seed -> {metric: mean}
    for seed, rows in sorted(all_data.items()):
        profile = {}
        for m in INTERESTING_METRICS:
            profile[m] = np.mean([r[m] for r in rows])
        seed_profiles[seed] = profile

    # Compute z-scores
    outliers = defaultdict(list)  # seed -> [(metric, z-score, direction)]
    for m in INTERESTING_METRICS:
        vals = [seed_profiles[s][m] for s in sorted(seed_profiles)]
        mu = np.mean(vals)
        sigma = np.std(vals)
        if sigma == 0:
            continue
        for seed in sorted(seed_profiles):
            z = (seed_profiles[seed][m] - mu) / sigma
            if abs(z) > 2.0:
                direction = "HIGH" if z > 0 else "LOW"
                outliers[seed].append((m, z, direction))

    if not outliers:
        print("No outlier seeds found (none > 2σ on any metric).")
    else:
        for seed in sorted(outliers):
            print(f"Seed {seed:>2}:")
            for m, z, d in outliers[seed]:
                print(f"  {m:<25} z={z:>+6.2f} ({d})")
        print(f"\n{len(outliers)} outlier seeds out of {len(seed_profiles)}")

    # Also print seeds ranked by number of outlier metrics
    if outliers:
        print(f"\nMost unusual seeds (by # outlier metrics):")
        for seed, metrics in sorted(outliers.items(), key=lambda x: -len(x[1])):
            print(f"  Seed {seed:>2}: {len(metrics)} outlier metrics")


def analyze_prompt_sensitivity(all_data):
    """Which prompts are most sensitive to seed choice?"""
    section("3. PROMPT SENSITIVITY ANALYSIS")
    print("\nFor each prompt, how much does seed choice affect behavioral metrics?")
    print("Higher CV (coefficient of variation) = more seed-sensitive.\n")

    # Group by prefix
    prefix_data = defaultdict(lambda: defaultdict(list))  # prefix -> metric -> [per-seed means]
    for seed, rows in sorted(all_data.items()):
        by_prefix = defaultdict(list)
        for r in rows:
            by_prefix[r["prefix"]].append(r)
        for prefix, prefix_rows in by_prefix.items():
            for m in INTERESTING_METRICS:
                prefix_data[prefix][m].append(np.mean([r[m] for r in prefix_rows]))

    # Compute CV per prefix across seeds
    prefix_sensitivity = {}
    for prefix in prefix_data:
        cvs = []
        for m in ["lexical_diversity", "word_entropy", "degeneration_score", "distinct_2"]:
            vals = prefix_data[prefix][m]
            mu = np.mean(vals)
            if mu > 0:
                cvs.append(np.std(vals) / mu)
        prefix_sensitivity[prefix] = np.mean(cvs) if cvs else 0

    print(f"{'Prompt (truncated)':<55} {'Mean CV':>10}")
    print("-" * 67)
    for prefix, cv in sorted(prefix_sensitivity.items(), key=lambda x: -x[1]):
        trunc = prefix[:52] + "..." if len(prefix) > 52 else prefix
        print(f"{trunc:<55} {cv:>10.4f}")

    # Detailed breakdown for most and least sensitive
    most = max(prefix_sensitivity, key=prefix_sensitivity.get)
    least = min(prefix_sensitivity, key=prefix_sensitivity.get)

    for label, prefix in [("MOST SENSITIVE", most), ("LEAST SENSITIVE", least)]:
        print(f"\n{label}: \"{prefix[:60]}\"")
        print(f"  {'Metric':<25} {'Seed mean':>10} {'Seed σ':>10} {'CV':>10}")
        for m in INTERESTING_METRICS:
            vals = prefix_data[prefix][m]
            mu = np.mean(vals)
            sigma = np.std(vals)
            cv = sigma / mu if mu > 0 else 0
            print(f"  {m:<25} {mu:>10.4f} {sigma:>10.4f} {cv:>10.4f}")


def analyze_seed_clustering(all_data):
    """Do seeds form behavioral clusters?"""
    section("4. SEED BEHAVIORAL CLUSTERING")
    print("\nUsing pairwise distance between seed behavioral profiles.")
    print("If seeds cluster, some initializations produce similar 'personalities'.\n")

    # Build feature matrix: seeds x metrics
    seeds = sorted(all_data.keys())
    profiles = np.zeros((len(seeds), len(INTERESTING_METRICS)))
    for i, seed in enumerate(seeds):
        for j, m in enumerate(INTERESTING_METRICS):
            profiles[i, j] = np.mean([r[m] for r in all_data[seed]])

    # Z-score normalize
    mu = profiles.mean(axis=0)
    sigma = profiles.std(axis=0)
    sigma[sigma == 0] = 1
    normed = (profiles - mu) / sigma

    # Pairwise cosine similarity
    norms = np.linalg.norm(normed, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit = normed / norms
    sim_matrix = unit @ unit.T

    # Find most similar and most different pairs
    n = len(seeds)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((seeds[i], seeds[j], sim_matrix[i, j]))

    pairs.sort(key=lambda x: -x[2])

    print("Most similar seed pairs (cosine similarity on z-scored metrics):")
    for s1, s2, sim in pairs[:5]:
        print(f"  Seed {s1:>2} — Seed {s2:>2}: {sim:.4f}")

    print("\nMost different seed pairs:")
    for s1, s2, sim in pairs[-5:]:
        print(f"  Seed {s1:>2} — Seed {s2:>2}: {sim:.4f}")

    # Check if similarity distribution is bimodal (would indicate clustering)
    sims = [p[2] for p in pairs]
    print(f"\nSimilarity distribution:")
    print(f"  Mean:   {np.mean(sims):.4f}")
    print(f"  Std:    {np.std(sims):.4f}")
    print(f"  Min:    {np.min(sims):.4f}")
    print(f"  Max:    {np.max(sims):.4f}")
    print(f"  Skew:   {float(np.mean(((sims - np.mean(sims)) / np.std(sims))**3)):.4f}")

    # Simple k=2 check: split seeds into two groups by first principal component
    # and see if groups are meaningfully different
    U, S, Vt = np.linalg.svd(normed, full_matrices=False)
    pc1 = U[:, 0]
    explained = S[0]**2 / (S**2).sum()

    print(f"\nPCA: PC1 explains {explained:.1%} of variance")

    group_a = [seeds[i] for i in range(n) if pc1[i] < 0]
    group_b = [seeds[i] for i in range(n) if pc1[i] >= 0]
    print(f"  Split by PC1 sign: Group A ({len(group_a)} seeds) vs Group B ({len(group_b)} seeds)")

    # Show which metrics drive PC1
    loadings = Vt[0]
    print(f"\n  PC1 loadings (what drives differentiation):")
    ranked = sorted(zip(INTERESTING_METRICS, loadings), key=lambda x: -abs(x[1]))
    for m, l in ranked[:5]:
        print(f"    {m:<25} {l:>+.4f}")


def analyze_bpb_behavior_correlation(all_data, bpbs):
    """Is val_bpb correlated with behavioral metrics?"""
    section("5. val_bpb vs BEHAVIORAL METRICS CORRELATION")
    print("\nDo seeds that train better (lower val_bpb) also generate differently?\n")

    seeds = sorted(all_data.keys())
    bpb_vals = [bpbs[s] for s in seeds]

    print(f"{'Metric':<25} {'Pearson r':>12} {'Direction':>15}")
    print("-" * 55)

    for m in INTERESTING_METRICS:
        metric_vals = [np.mean([r[m] for r in all_data[s]]) for s in seeds]

        # Pearson correlation
        x = np.array(bpb_vals)
        y = np.array(metric_vals)
        r = np.corrcoef(x, y)[0, 1]

        direction = ""
        if abs(r) > 0.3:
            direction = "lower bpb → " + ("higher" if r < 0 else "lower") + f" {m}"

        marker = " ***" if abs(r) > 0.5 else " *" if abs(r) > 0.3 else ""
        print(f"{m:<25} {r:>+12.4f} {direction:>15}{marker}")

    print("\n*** = strong (|r|>0.5), * = moderate (|r|>0.3)")


def analyze_per_temperature(all_data):
    """Does temperature interact with seed sensitivity?"""
    section("6. TEMPERATURE x SEED INTERACTION")
    print("\nAre seeds more differentiated at one temperature vs another?\n")

    for temp in ["0.8", "1.0"]:
        seed_means = {}
        for seed, rows in sorted(all_data.items()):
            temp_rows = [r for r in rows if r["temperature"] == temp]
            if temp_rows:
                seed_means[seed] = {m: np.mean([r[m] for r in temp_rows]) for m in INTERESTING_METRICS}

        print(f"Temperature = {temp}:")
        print(f"  {'Metric':<25} {'Between-seed σ':>15} {'CV of seed means':>18}")
        for m in ["lexical_diversity", "word_entropy", "degeneration_score", "distinct_2", "word_count"]:
            vals = [seed_means[s][m] for s in sorted(seed_means)]
            mu = np.mean(vals)
            sigma = np.std(vals)
            cv = sigma / mu if mu > 0 else 0
            print(f"  {m:<25} {sigma:>15.4f} {cv:>18.4f}")
        print()


def summarize_findings(all_data, bpbs):
    """Executive summary."""
    section("SUMMARY: SEED SENSITIVITY FINDINGS")

    seeds = sorted(all_data.keys())
    bpb_vals = [bpbs[s] for s in seeds]

    print(f"""
Seeds analyzed: {len(seeds)}
Samples per seed: {len(all_data[seeds[0]])}
Total samples: {sum(len(v) for v in all_data.values())}

val_bpb:
  Range: {min(bpb_vals):.6f} — {max(bpb_vals):.6f}
  Mean:  {np.mean(bpb_vals):.6f}
  Std:   {np.std(bpb_vals):.6f}
  CV:    {np.std(bpb_vals)/np.mean(bpb_vals):.4f}
""")


def main():
    print("Loading data...")
    all_data = load_all_scores()
    bpbs = load_val_bpbs()
    print(f"Loaded {len(all_data)} seeds, {sum(len(v) for v in all_data.values())} total samples")

    summarize_findings(all_data, bpbs)
    analyze_cross_seed_variance(all_data)
    analyze_outlier_seeds(all_data)
    analyze_prompt_sensitivity(all_data)
    analyze_seed_clustering(all_data)
    analyze_bpb_behavior_correlation(all_data, bpbs)
    analyze_per_temperature(all_data)


if __name__ == "__main__":
    main()

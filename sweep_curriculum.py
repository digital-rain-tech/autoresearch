"""
Paired curriculum sweep: compare adaptive vs baseline orderings across seeds.

Uses paired comparison (same seed, different ordering) to reduce variance.
This is Workstream A from ADR-007.

Usage:
    uv run sweep_curriculum.py                          # Seeds 0-4, all orderings
    uv run sweep_curriculum.py --seeds 0 10             # Seeds 0-9
    uv run sweep_curriculum.py --orderings adaptive random  # Only these two
    uv run sweep_curriculum.py --skip-training           # Analyze existing logs only

Output:
    results/curriculum_sweep.csv   — per-seed, per-ordering val_bpb
    results/curriculum_analysis.txt — paired statistical analysis
"""

import argparse
import csv
import math
import os
import re
import subprocess
import time


RESULTS_DIR = "results"
LOGS_DIR = "results/curriculum_logs"

ORDERINGS = ["sequential", "random", "adaptive"]


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def run_single(seed, ordering):
    """Run training for one seed+ordering pair. Returns dict with results."""
    log_path = os.path.join(LOGS_DIR, f"seed{seed}_{ordering}.log")

    # Check for existing result
    if os.path.exists(log_path):
        result = parse_log(log_path)
        if result and result.get("val_bpb") is not None:
            print(f"  seed={seed} ordering={ordering}: cached val_bpb={result['val_bpb']:.6f}")
            return result

    print(f"\n{'='*60}")
    print(f"  seed={seed}  ordering={ordering}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["AUTORESEARCH_SEED"] = str(seed)
    env["AUTORESEARCH_CURRICULUM"] = ordering

    t0 = time.time()
    process = subprocess.Popen(
        ["uv", "run", "train.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    stdout_lines = []
    try:
        assert process.stdout is not None
        assert process.stderr is not None
        for line in process.stdout:
            stdout_lines.append(line)
            # Print only non-progress lines to reduce noise
            if not line.startswith("\r"):
                print(f"  [{ordering}] {line}", end="", flush=True)
        process.wait(timeout=600)
        stderr_text = process.stderr.read()
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print(f"  TIMEOUT: seed={seed} ordering={ordering}")
        return None

    elapsed = time.time() - t0
    stdout_full = "".join(stdout_lines)

    # Save log
    with open(log_path, "w") as f:
        f.write(stdout_full)
        if stderr_text.strip():
            f.write(f"\n--- STDERR ---\n{stderr_text}")

    if process.returncode != 0:
        print(f"  FAILED: seed={seed} ordering={ordering}")
        return None

    result = parse_log(log_path)
    if result:
        result["elapsed"] = elapsed
        print(f"  val_bpb={result['val_bpb']:.6f} ({elapsed:.0f}s)")
    return result


def parse_log(log_path):
    """Extract key metrics from a training log."""
    result = {}
    try:
        with open(log_path) as f:
            text = f.read()
    except FileNotFoundError:
        return None

    for line in text.split("\n"):
        if line.startswith("val_bpb:"):
            result["val_bpb"] = float(line.split(":")[1].strip())
        elif line.startswith("peak_vram_mb:"):
            result["peak_vram_mb"] = float(line.split(":")[1].strip())
        elif line.startswith("num_steps:"):
            result["num_steps"] = int(line.split(":")[1].strip())
        elif line.startswith("total_tokens_M:"):
            result["total_tokens_M"] = float(line.split(":")[1].strip())
        elif line.startswith("adaptive_bucket_avg_loss:"):
            result["adaptive_bucket_avg_loss"] = line.split(":")[1].strip()

    return result if "val_bpb" in result else None


def analyze_results(results, orderings):
    """Paired statistical analysis of curriculum orderings."""
    lines = []
    lines.append("=" * 70)
    lines.append("CURRICULUM SWEEP ANALYSIS (ADR-007 Workstream A)")
    lines.append("=" * 70)

    # Group by seed
    by_seed = {}
    for r in results:
        by_seed.setdefault(r["seed"], {})[r["ordering"]] = r

    # Summary table
    lines.append(f"\n{'Seed':>6} | " + " | ".join(f"{o:>12}" for o in orderings))
    lines.append("-" * (8 + 15 * len(orderings)))
    for seed in sorted(by_seed.keys()):
        vals = []
        for o in orderings:
            if o in by_seed[seed] and by_seed[seed][o].get("val_bpb") is not None:
                vals.append(f"{by_seed[seed][o]['val_bpb']:.6f}")
            else:
                vals.append("     —")
        lines.append(f"{seed:>6} | " + " | ".join(f"{v:>12}" for v in vals))

    # Per-ordering statistics
    lines.append("\nPer-ordering summary:")
    ordering_bpbs = {}
    for o in orderings:
        bpbs = [by_seed[s][o]["val_bpb"] for s in by_seed
                if o in by_seed[s] and by_seed[s][o].get("val_bpb") is not None]
        ordering_bpbs[o] = bpbs
        if bpbs:
            mean = sum(bpbs) / len(bpbs)
            std = (sum((x - mean) ** 2 for x in bpbs) / len(bpbs)) ** 0.5
            lines.append(f"  {o:>12}: mean={mean:.6f} std={std:.6f} n={len(bpbs)}")

    # Paired comparisons (all pairs)
    lines.append("\nPaired comparisons:")
    for i, o1 in enumerate(orderings):
        for o2 in orderings[i + 1:]:
            paired_seeds = [s for s in by_seed
                           if o1 in by_seed[s] and o2 in by_seed[s]
                           and by_seed[s][o1].get("val_bpb") is not None
                           and by_seed[s][o2].get("val_bpb") is not None]
            if len(paired_seeds) < 2:
                lines.append(f"  {o1} vs {o2}: insufficient paired data ({len(paired_seeds)} seeds)")
                continue

            diffs = [by_seed[s][o1]["val_bpb"] - by_seed[s][o2]["val_bpb"] for s in paired_seeds]
            n = len(diffs)
            mean_diff = sum(diffs) / n
            std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)) ** 0.5
            se = std_diff / n ** 0.5
            t_stat = mean_diff / se if se > 0 else 0

            # Two-tailed p-value approximation (t-distribution with n-1 df)
            # Using normal approximation for simplicity (accurate for n >= 5)
            p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

            # Cohen's d
            pooled_std = std_diff  # for paired, this is std of differences
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            lines.append(f"  {o1} vs {o2} (n={n} paired seeds):")
            lines.append(f"    mean diff = {mean_diff:+.6f} bpb ({o1} - {o2})")
            lines.append(f"    std diff  = {std_diff:.6f}")
            lines.append(f"    t-stat    = {t_stat:.3f}")
            lines.append(f"    p-value   = {p_value:.4f}")
            lines.append(f"    Cohen's d = {cohens_d:.3f}")

            # Interpret
            sig = "SIGNIFICANT" if p_value < 0.05 else "not significant"
            better = o1 if mean_diff < 0 else o2
            lines.append(f"    → {sig} (p<0.05). Lower bpb = {better}")

    # ADR-007 success criterion
    lines.append("\nADR-007 Success Criterion (adaptive > 0.04 bpb better than random):")
    if "adaptive" in ordering_bpbs and "random" in ordering_bpbs:
        paired = [s for s in by_seed
                  if "adaptive" in by_seed[s] and "random" in by_seed[s]
                  and by_seed[s]["adaptive"].get("val_bpb") is not None
                  and by_seed[s]["random"].get("val_bpb") is not None]
        if paired:
            diffs = [by_seed[s]["random"]["val_bpb"] - by_seed[s]["adaptive"]["val_bpb"] for s in paired]
            mean_improvement = sum(diffs) / len(diffs)
            lines.append(f"  Mean improvement (random - adaptive) = {mean_improvement:+.6f} bpb")
            if mean_improvement > 0.04:
                lines.append("  → CRITERION MET: adaptive is >0.04 bpb better than random")
            else:
                lines.append("  → CRITERION NOT MET: improvement < 0.04 bpb")

    return "\n".join(lines)


def _normal_cdf(x):
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def main():
    parser = argparse.ArgumentParser(description="Paired curriculum sweep (ADR-007 Workstream A)")
    parser.add_argument("--seeds", type=int, nargs=2, default=[0, 5],
                        metavar=("START", "END"),
                        help="Seed range [start, end) — default 0 5")
    parser.add_argument("--orderings", nargs="+", default=ORDERINGS,
                        help=f"Orderings to test — default {ORDERINGS}")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only analyze existing logs")
    args = parser.parse_args()

    seeds = list(range(args.seeds[0], args.seeds[1]))
    orderings = args.orderings

    print(f"Curriculum sweep: {len(seeds)} seeds × {len(orderings)} orderings = {len(seeds) * len(orderings)} runs")
    print(f"Seeds: {seeds}")
    print(f"Orderings: {orderings}")

    ensure_dirs()

    results = []
    for seed in seeds:
        for ordering in orderings:
            if args.skip_training:
                log_path = os.path.join(LOGS_DIR, f"seed{seed}_{ordering}.log")
                result = parse_log(log_path)
            else:
                result = run_single(seed, ordering)

            if result:
                result["seed"] = seed
                result["ordering"] = ordering
                results.append(result)

    # Save raw results
    csv_path = os.path.join(RESULTS_DIR, "curriculum_sweep.csv")
    if results:
        fieldnames = ["seed", "ordering", "val_bpb", "peak_vram_mb", "num_steps", "total_tokens_M"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

    # Analysis
    analysis = analyze_results(results, orderings)
    print(analysis)

    analysis_path = os.path.join(RESULTS_DIR, "curriculum_analysis.txt")
    with open(analysis_path, "w") as f:
        f.write(analysis)
    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    main()

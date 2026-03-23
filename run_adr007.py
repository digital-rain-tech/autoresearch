"""
ADR-007 unified experiment runner: both workstreams in parallel.

Workstream A: Adaptive bandit curriculum in train.py (GPU-bound, ~25 min for 5 seeds × 3 orderings)
Workstream B: King Wen priors on OpenSpiel games (CPU-bound, ~5 min for 2 games × 6 conditions × 5 seeds)

Usage:
    uv run run_adr007.py                     # Run both workstreams
    uv run run_adr007.py --workstream A      # Only curriculum sweep
    uv run run_adr007.py --workstream B      # Only OpenSpiel experiments
    uv run run_adr007.py --quick             # Quick smoke test (2 seeds, 100 iters)
"""

import argparse
import os
import subprocess
import sys
import time


def run_workstream_a(seeds_start, seeds_end, orderings):
    """Run curriculum sweep (Workstream A)."""
    print("=" * 70)
    print("WORKSTREAM A: Adaptive Bandit Curriculum")
    print("=" * 70)

    cmd = [
        "uv", "run", "sweep_curriculum.py",
        "--seeds", str(seeds_start), str(seeds_end),
        "--orderings", *orderings,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, timeout=7200)
    return result.returncode == 0


def run_workstream_b(seeds_start, seeds_end, iterations, temperatures):
    """Run OpenSpiel experiments (Workstream B)."""
    print("=" * 70)
    print("WORKSTREAM B: King Wen Priors on OpenSpiel Games")
    print("=" * 70)

    cmd = [
        "uv", "run", "king_wen_openspiel.py",
        "--seeds", str(seeds_start), str(seeds_end),
        "--iterations", str(iterations),
        "--temperatures", *[str(t) for t in temperatures],
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, timeout=3600)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="ADR-007 unified experiment runner")
    parser.add_argument("--workstream", choices=["A", "B", "both"], default="both",
                        help="Which workstream to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (2 seeds, fewer iterations)")
    parser.add_argument("--seeds", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Override seed range")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.quick:
        seeds = args.seeds or [0, 2]
        iterations = 100
        temperatures = [0.3]
        orderings = ["sequential", "random", "adaptive"]
    else:
        seeds = args.seeds or [0, 5]
        iterations = 1000
        temperatures = [0.1, 0.3, 0.5, 1.0]
        orderings = ["sequential", "random", "adaptive"]

    t0 = time.time()
    results = {}

    if args.workstream in ("A", "both"):
        results["A"] = run_workstream_a(seeds[0], seeds[1], orderings)

    if args.workstream in ("B", "both"):
        results["B"] = run_workstream_b(seeds[0], seeds[1], iterations, temperatures)

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("ADR-007 EXPERIMENT SUMMARY")
    print("=" * 70)
    for ws, ok in results.items():
        status = "COMPLETED" if ok else "FAILED"
        print(f"  Workstream {ws}: {status}")
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"\nResults in: results/")
    print(f"  Workstream A: results/curriculum_sweep.csv, results/curriculum_analysis.txt")
    print(f"  Workstream B: results/king_wen_openspiel.json, results/king_wen_openspiel_analysis.txt")


if __name__ == "__main__":
    main()

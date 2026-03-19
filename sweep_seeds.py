"""
Seed sensitivity sweep: train models with different seeds, sample, and score.

Uses environment variables AUTORESEARCH_SEED and AUTORESEARCH_CHECKPOINT_PATH
to configure train.py without modifying it.

Usage:
    uv run sweep_seeds.py                        # Seeds 0-29 (default)
    uv run sweep_seeds.py --seeds 0 5            # Seeds 0-4
    uv run sweep_seeds.py --seeds 0 3 --skip-training  # Score existing checkpoints only

Output structure:
    checkpoints/seed_{N}.pt     — trained model checkpoints
    samples/seed_{N}.json       — generated text samples
    scores/seed_{N}.csv         — per-sample behavioral scores
    results/seed_sweep.csv      — aggregated per-seed summary
"""

import argparse
import csv
import os
import subprocess


CHECKPOINT_DIR = "checkpoints"
SAMPLES_DIR = "samples"
SCORES_DIR = "scores"
RESULTS_DIR = "results"


def ensure_dirs():
    for d in [CHECKPOINT_DIR, SAMPLES_DIR, SCORES_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


def run_training(seed):
    """Run training for a single seed. Returns val_bpb or None on failure."""
    print(f"\n{'='*60}")
    print(f"  SEED {seed}: Training...")
    print(f"{'='*60}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"seed_{seed}.pt")

    # Configure seed and checkpoint via environment variables — no file mutation
    env = os.environ.copy()
    env["AUTORESEARCH_SEED"] = str(seed)
    env["AUTORESEARCH_CHECKPOINT_PATH"] = ckpt_path

    # Stream stdout so user sees training progress
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
            print(line, end="", flush=True)
        process.wait(timeout=600)
        stderr_text = process.stderr.read()
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print(f"  SEED {seed}: Training TIMED OUT")
        return None

    stdout_full = "".join(stdout_lines)

    # Extract val_bpb from output
    val_bpb = None
    for line in stdout_full.split('\n'):
        if line.startswith('val_bpb:'):
            val_bpb = float(line.split(':')[1].strip())
            break

    if process.returncode != 0:
        print(f"  SEED {seed}: Training FAILED")
        print(f"  stderr: {stderr_text[-500:]}")
        return None

    print(f"  SEED {seed}: val_bpb = {val_bpb}")
    return val_bpb


def run_sampling(seed):
    """Generate samples from a trained checkpoint."""
    ckpt = os.path.join(CHECKPOINT_DIR, f"seed_{seed}.pt")
    output = os.path.join(SAMPLES_DIR, f"seed_{seed}.json")

    if not os.path.exists(ckpt):
        print(f"  SEED {seed}: No checkpoint found, skipping sampling")
        return False

    print(f"  SEED {seed}: Sampling...")
    result = subprocess.run(
        ["uv", "run", "sample.py",
         "--checkpoint", ckpt,
         "--output", output,
         "--seed", str(seed)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"  SEED {seed}: Sampling FAILED")
        print(f"  stderr: {result.stderr[-500:]}")
        return False

    print(f"  SEED {seed}: Sampling complete")
    return True


def run_scoring(seed):
    """Score generated samples."""
    input_path = os.path.join(SAMPLES_DIR, f"seed_{seed}.json")
    output_path = os.path.join(SCORES_DIR, f"seed_{seed}.csv")

    if not os.path.exists(input_path):
        print(f"  SEED {seed}: No samples found, skipping scoring")
        return False

    print(f"  SEED {seed}: Scoring...")
    result = subprocess.run(
        ["uv", "run", "score.py",
         "--input", input_path,
         "--output", output_path],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        print(f"  SEED {seed}: Scoring FAILED")
        print(f"  stderr: {result.stderr[-500:]}")
        return False

    print(result.stdout)
    return True


def aggregate_results(seeds):
    """Aggregate per-seed scores into a summary CSV."""
    from score import METRIC_NAMES

    summary_rows = []
    for seed in seeds:
        score_path = os.path.join(SCORES_DIR, f"seed_{seed}.csv")
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"seed_{seed}.pt")

        if not os.path.exists(score_path):
            continue

        # Get val_bpb from checkpoint
        val_bpb = None
        if os.path.exists(ckpt_path):
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            val_bpb = ckpt.get("val_bpb")

        # Read scores and compute means
        with open(score_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        means = {}
        for metric in METRIC_NAMES:
            vals = [float(r[metric]) for r in rows]
            means[f"mean_{metric}"] = round(sum(vals) / len(vals), 4)
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            means[f"std_{metric}"] = round(variance ** 0.5, 4)

        summary_rows.append({
            "seed": seed,
            "val_bpb": val_bpb,
            "n_samples": len(rows),
            **means,
        })

    if not summary_rows:
        print("No results to aggregate")
        return

    output_path = os.path.join(RESULTS_DIR, "seed_sweep.csv")
    fieldnames = ["seed", "val_bpb", "n_samples"]
    for metric in METRIC_NAMES:
        fieldnames.extend([f"mean_{metric}", f"std_{metric}"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nAggregated results → {output_path}")
    print(f"Seeds completed: {len(summary_rows)}/{len(seeds)}")

    if summary_rows:
        bpbs = [r["val_bpb"] for r in summary_rows if r["val_bpb"] is not None]
        if bpbs:
            print(f"val_bpb range: {min(bpbs):.6f} — {max(bpbs):.6f}")
            print(f"val_bpb mean:  {sum(bpbs)/len(bpbs):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Seed sensitivity sweep")
    parser.add_argument("--seeds", type=int, nargs=2, default=[0, 30],
                        metavar=("START", "END"),
                        help="Seed range [start, end) — default 0 30")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only sample and score existing checkpoints")
    parser.add_argument("--skip-sampling", action="store_true",
                        help="Skip sampling, only aggregate existing scores")
    args = parser.parse_args()

    seeds = list(range(args.seeds[0], args.seeds[1]))
    print(f"Seed sweep: {len(seeds)} seeds ({seeds[0]}–{seeds[-1]})")

    ensure_dirs()

    failed_seeds = []

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Processing seed {seed}")

        # Train
        if not args.skip_training:
            val_bpb = run_training(seed)
            if val_bpb is None:
                failed_seeds.append(seed)
                continue

        # Sample
        if not args.skip_sampling:
            ok = run_sampling(seed)
            if not ok:
                failed_seeds.append(seed)
                continue

            # Score
            run_scoring(seed)

    # Aggregate
    aggregate_results(seeds)

    if failed_seeds:
        print(f"\nFailed seeds: {failed_seeds}")

    print("\nDone.")


if __name__ == "__main__":
    main()

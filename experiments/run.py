"""Run a paper experiment with fixed parameters and independently generated data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from experiments.common import ROOT

# name: (module, function, iterations, first evaluation seed, seed count, checkpoint)
EXPERIMENTS = {
    "sparse_recovery": ("recovery", "sparse_recovery", 900, 10, 10, 25),
    "group_sparse": ("recovery", "group_recovery", 950, 10, 10, 25),
    "sparse_phase": ("recovery", "sparse_phase_recovery", 1200, 10, 10, 25),
    "sparse_coverage": ("recovery", "sparse_coverage", 900, 10, 10, 25),
    "extra_baselines": ("recovery", "extra_baselines", 900, 10, 10, 25),
    "smooth_benchmarks": ("recovery", "smooth_benchmarks", 1200, 10, 10, 25),
    "smooth_stationarity": ("diagnostics", "smooth_stationarity", 2000, 0, 5, 10),
    "recovery_trajectories": ("recovery", "recovery_trajectories", 900, 10, 10, 25),
    "imaging_phantom": ("imaging", "imaging", 1000, 30, 10, 20),
    "imaging_cameraman": ("imaging", "imaging", 1000, 40, 10, 20),
    "modern_fourier": ("imaging", "modern_comparison", 1000, 50, 10, 25),
    "modern_stochastic": ("imaging", "modern_comparison", 900, 50, 10, 25),
    "quartic": ("geometry", "quartic_recovery", 600, 30, 10, 10),
    "weighted_quadratic": ("geometry", "weighted_quadratic", 700, 0, 10, 10),
    "noise_sweep": ("stochastic", "noise_sweep", 900, 10, 20, 25),
    "stochastic_trajectories": ("stochastic", "residual_trajectories", 900, 10, 20, 25),
    "defect_identification": ("diagnostics", "defect_identification", 3000, 0, 10, 10),
    "lifted_residuals": ("diagnostics", "lifted_residuals", 700, 10, 10, 10),
    "sensitivity": ("diagnostics", "sensitivity", 2000, 0, 5, 10),
    "hitting_times": ("diagnostics", "hitting_times", 1200, 0, 5, 10),
}


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=EXPERIMENTS)
    parser.add_argument("--list", action="store_true", help="list paper experiments and their default budgets")
    parser.add_argument("--seeds", help="comma-separated evaluation seeds; defaults match the paper")
    parser.add_argument("--iterations", type=positive_int, help="override the paper iteration budget")
    parser.add_argument("--checkpoint", type=positive_int, help="override checkpoint spacing where applicable")
    parser.add_argument("--output", type=Path, help="output directory (default: timestamped directory in results/)")
    parser.add_argument("--smoke", action="store_true", help="use the first seed and five iterations to check execution")
    args = parser.parse_args()
    if args.list or args.experiment is None:
        for name, (_, _, iterations, first, count, _) in EXPERIMENTS.items():
            print(f"{name:26} {iterations:4} iterations; seeds {first}..{first + count - 1}")
        return
    module, function, iterations, first, count, checkpoint = EXPERIMENTS[args.experiment]
    try:
        args.seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else list(range(first, first + count))
    except ValueError:
        parser.error("--seeds must contain comma-separated integers")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("--seeds must contain distinct nonnegative integers")
    args.iterations = args.iterations if args.iterations is not None else (5 if args.smoke else iterations)
    args.checkpoint = args.checkpoint if args.checkpoint is not None else checkpoint
    if args.smoke:
        args.seeds = args.seeds[:1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    args.output = (args.output or ROOT / "results" / f"{args.experiment}_{stamp}").resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty; use a new directory for each run")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for directory, pattern in (("sdbpg", "*.py"), ("experiments", "*.py"), ("configs", "*.json"))
              for p in sorted((ROOT / directory).rglob(pattern))}
    metadata = {"experiment": args.experiment, "evaluation_seeds": args.seeds,
                "iterations": args.iterations, "checkpoint": args.checkpoint, "smoke": args.smoke,
                "python": platform.python_version(), "source_sha256": hashes, "status": "running"}
    metadata["dependencies"] = {name: version(name) for name in
                                ("numpy", "scipy", "networkx", "pandas", "PyWavelets", "scikit-image")}
    path = args.output / "run.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        getattr(importlib.import_module(f"experiments.{module}"), function)(args)
    except Exception:
        metadata["status"] = "failed"
        path.write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    metadata["status"] = "complete"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()

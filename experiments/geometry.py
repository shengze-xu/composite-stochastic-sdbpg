"""Quartic and weighted quadratic kernel experiments."""

from experiments.common import config, summarize, write
from experiments._impl import quartic, weighted
import numpy as np


def quartic_recovery(args):
    rows, histories = [], []
    selected = config("quartic")
    for heterogeneity in (2.5, 3.0, 3.25):
        for seed in args.seeds:
            for method, parameters in selected.items():
                row, history = quartic.run_one(method, parameters, seed, heterogeneity,
                                               args.iterations, args.checkpoint)
                rows.append(row)
                histories.extend(history)
    frame = write(args.output, "seed_level", rows)
    history = write(args.output, "trajectories", histories)
    write(args.output, "summary", quartic.aggregate(frame))
    write(args.output, "paired", [quartic.paired_bootstrap(frame, method, h)
        for h in (2.5, 3.0, 3.25) for method in ("Quadratic-SDBPG", "ProxGT")])
    write(args.output, "cost_to_target", quartic.cost_to_target(history))


def weighted_quadratic(args):
    rows, histories = [], []
    for condition in weighted.CONDITION_NUMBERS:
        for parameters in weighted.CONFIGS:
            for seed in args.seeds:
                result = weighted._run_seed(parameters, seed, condition, args.iterations)
                identity = {"condition_number": condition, "algorithm": parameters.name, "seed": seed}
                rows.append({**identity, **{k: v for k, v in result.items() if np.isscalar(v)}})
                histories.extend({**identity, "iteration": t, "running_min_gradient_sq": g,
                                  "consensus": c} for t, (g, c) in enumerate(
                    zip(result["running_grad_norm_sq"], result["consensus"]), 1))
    summarize(args.output, rows, ["condition_number", "algorithm"])
    write(args.output, "trajectories", histories)

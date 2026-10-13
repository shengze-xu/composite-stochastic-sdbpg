"""Sparse, shared-support, sparse phase, and smooth recovery experiments."""

from __future__ import annotations

import numpy as np

from experiments.common import config, summarize, write
from experiments._impl import sparse, sparse_baselines, group_sparse
from experiments._impl import sparse_phase, sparse_phase_baselines, coverage, extra
from experiments._impl import phase, matrix, trajectories
from sdbpg.core import generate_network_topology


def sparse_recovery(args):
    _recovery(args, "sparse_recovery", sparse, sparse_baselines, "L1-SDBPG")


def group_recovery(args):
    _recovery(args, "group_sparse", group_sparse, group_sparse, "L21-SDBPG")


def sparse_phase_recovery(args):
    _recovery(args, "sparse_phase", sparse_phase, sparse_phase_baselines, "L1-SDBPG")


def _recovery(args, name, model, runner, reference):
    task = next(iter(model.TASKS.values()))
    rows = []
    for case in config(name):
        scenario = case["scenario"]
        for method, parameters in case["methods"].items():
            for seed in args.seeds:
                if "SDBPG" in method:
                    metrics = runner._run_one_seed_sdbpg(
                        task, model.SCENARIOS[scenario], seed, parameters["step_mult"],
                        parameters["lambda_reg"], args.iterations, model.SUPPORT_TOL,
                    )
                else:
                    metrics = runner._run_one_seed_baseline(
                        task, model.SCENARIOS[scenario], seed, method, parameters,
                        args.iterations, model.SUPPORT_TOL,
                    )
                rows.append({"scenario": scenario, "algorithm": method, "seed": seed, **metrics})
        print(f"Completed {scenario}", flush=True)
    summarize(args.output, rows, ["scenario", "algorithm"], reference)


def sparse_coverage(args):
    task = sparse.TASKS["sparse_recovery"]
    rows = []
    for case in config("sparse_coverage"):
        scenario = {**sparse.SCENARIOS[case["scenario"]], **{
            k: case[k] for k in ("sparsity", "samples", "noise_std", "hetero_strength")
        }}
        for method, parameters in case["methods"].items():
            for seed in args.seeds:
                if "SDBPG" in method:
                    metrics = coverage._run_one_seed_sdbpg(
                        task, scenario, seed, parameters["step_mult"], parameters["lambda_reg"],
                        args.iterations, sparse.SUPPORT_TOL, case["topology"],
                    )
                else:
                    metrics = coverage._run_one_seed_baseline(
                        task, scenario, seed, method, parameters, args.iterations,
                        sparse.SUPPORT_TOL, case["topology"],
                    )
                rows.append({"setting_id": case["setting_id"], "algorithm": method, "seed": seed, **metrics})
        print(f"Completed {case['setting_id']}", flush=True)
    summarize(args.output, rows, ["setting_id", "algorithm"], "L1-SDBPG")


def extra_baselines(args):
    rows = []
    for case in config("extra"):
        parameters = extra.ProxExtraConfig(float(case["stepsize"]), float(case["lambda_reg"]))
        for seed in args.seeds:
            rows.append(extra._run_config_seed(case["scenario"], parameters, seed, args.iterations))
    # Compute the reference afresh so this run needs no historical result files.
    for case in config("sparse_recovery"):
        p = case["methods"]["L1-SDBPG"]
        for seed in args.seeds:
            metrics = sparse_baselines._run_one_seed_sdbpg(
                sparse.TASKS["sparse_recovery"], sparse.SCENARIOS[case["scenario"]], seed,
                p["step_mult"], p["lambda_reg"], args.iterations, sparse.SUPPORT_TOL,
            )
            rows.append({"scenario": case["scenario"], "algorithm": "L1-SDBPG", "seed": seed, **metrics})
    summarize(args.output, rows, ["scenario", "algorithm"], "L1-SDBPG")


def smooth_benchmarks(args):
    rows, curves = [], []
    selected = config("smooth_baselines")
    for task, module in (("phase_retrieval", phase), ("matrix_factorization", matrix)):
        for scenario, case in module.SCENARIOS.items():
            for method in ("SDBPG", "DGD", "DGT"):
                parameters = next((r["parameters"] for r in selected if
                    r["task"] == task and r["scenario"] == scenario and r["algorithm"] == method), {})
                for seed in args.seeds:
                    common = dict(n_agents=6, samples_per_agent=case["samples"],
                                  noise_std=case["noise_std"], hetero_strength=case["hetero_strength"], seed=seed)
                    if module is phase:
                        problem = module.PhaseRetrievalProblem(dim=12, **common)
                    else:
                        problem = module.MatrixFactorizationProblem(mat_dim=8, rank=2, **common)
                    graph = generate_network_topology(6, "random", connectivity=0.5, seed=seed)
                    if method == "SDBPG":
                        algorithm = module.L2SDBPG(problem, graph, lambda_reg=0.0, **module.BASE_PARAMS)
                    elif method == "DGD":
                        algorithm = module.DecentralizedGradientDescent(problem, graph, **parameters)
                    else:
                        algorithm = module.DecentralizedGradientTracking(problem, graph, **parameters)
                    rng = np.random.default_rng(2026 + seed)
                    initial = module._make_init_vector(problem, rng, "spectral")
                    module._initialize_algorithm(algorithm, problem, initial, rng)
                    result = module._run_one(algorithm, problem, args.iterations)
                    identity = {"task": task, "scenario": scenario, "algorithm": method, "seed": seed}
                    rows.append({**identity, **{k: v for k, v in result.items() if np.isscalar(v)}})
                    for t, (g2, recovery) in enumerate(zip(result["run_min"], result["rec_err"]), 1):
                        curves.append({**identity, "iteration": t, "running_min_gradient_sq": g2, "recovery_error": recovery})
    summarize(args.output, rows, ["task", "scenario", "algorithm"], "SDBPG")
    write(args.output, "trajectories", curves)


def recovery_trajectories(args):
    rows = []
    for scenario in trajectories.SPARSE_RECOVERY_CONFIGS:
        for seed in args.seeds:
            rows.extend(trajectories._run_recovery_trajectory(scenario, seed, args.iterations, args.checkpoint))
    write(args.output, "trajectories", rows)

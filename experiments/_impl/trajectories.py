"""Trajectories numerical routines."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple


import numpy as np


from sdbpg.algorithm import DecentralizedGradientDescent
from sdbpg.algorithm import DecentralizedGradientTracking
from sdbpg.algorithm import ProximalDecentralizedGradientDescent
from sdbpg.algorithm import ProximalDecentralizedGradientTracking


from sdbpg.core import generate_network_topology


from experiments._impl import sparse as sr


SPARSE_RECOVERY_CONFIGS = {
    "sparse_noisy_undersampled": {
        "display": "Noisy undersampled",
        "step_mult": 1.50,
        "lambda_reg": 0.02,
        "baseline": {
            "DGD": {"stepsize": 0.008, "consensus_weight": 0.2, "lambda_reg": 0.0},
            "DGT": {"stepsize": 0.01, "lambda_reg": 0.0},
            "ProxDGD": {"stepsize": 0.008, "consensus_weight": 0.4, "lambda_reg": 0.02},
            "ProxDGT": {"stepsize": 0.01, "lambda_reg": 0.02},
        },
    },
    "sparse_noisy_moderate": {
        "display": "Noisy moderate",
        "step_mult": 1.50,
        "lambda_reg": 0.02,
        "baseline": {
            "DGD": {"stepsize": 0.008, "consensus_weight": 0.2, "lambda_reg": 0.0},
            "DGT": {"stepsize": 0.01, "lambda_reg": 0.0},
            "ProxDGD": {"stepsize": 0.008, "consensus_weight": 0.4, "lambda_reg": 0.02},
            "ProxDGT": {"stepsize": 0.01, "lambda_reg": 0.02},
        },
    },
}


def _init_baseline(algo: Any, problem: sr.SparseLinearRecoveryProblem, seed: int) -> None:
    rng = np.random.default_rng(20260410 + int(seed))
    x0 = rng.standard_normal(problem.dim)
    x0 /= max(np.linalg.norm(x0), 1e-12)
    xs = []
    for _ in range(problem.n_agents):
        xi = x0 + sr.INIT_PERTURB_STD * rng.standard_normal(problem.dim)
        xs.append(xi.copy())
    algo.x = xs
    if hasattr(algo, "y"):
        algo.y = [problem.local_gradient(i, algo.x[i]) for i in range(problem.n_agents)]


def _make_baseline(algorithm: str, params: Dict[str, float], problem: sr.SparseLinearRecoveryProblem, graph: Any) -> Any:
    if algorithm == "DGD":
        return DecentralizedGradientDescent(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
            consensus_weight=float(params["consensus_weight"]),
        )
    if algorithm == "DGT":
        return DecentralizedGradientTracking(problem=problem, graph=graph, stepsize=float(params["stepsize"]))
    if algorithm == "ProxDGD":
        return ProximalDecentralizedGradientDescent(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
            consensus_weight=float(params["consensus_weight"]),
            lambda_reg=float(params["lambda_reg"]),
        )
    if algorithm == "ProxDGT":
        return ProximalDecentralizedGradientTracking(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
            lambda_reg=float(params["lambda_reg"]),
        )
    raise ValueError(f"Unknown baseline: {algorithm}")


def _xbar(algo: Any) -> np.ndarray:
    if hasattr(algo, "agents"):
        xs = [ag.x for ag in algo.agents.values()]
    else:
        xs = algo.x
    return np.mean(xs, axis=0)


def _record_metrics(
    problem: sr.SparseLinearRecoveryProblem,
    algo: Any,
    iteration: int,
    label: str,
    scenario: str,
) -> Dict[str, Any]:
    x_bar = _xbar(algo)
    grad = problem.global_gradient(x_bar)
    eta = sr.algorithm_prox_eta(algo)
    lam = sr.algorithm_lambda_reg(algo)
    mapping = sr.l1_prox_gradient_mapping(x_bar, grad, lambda_reg=lam, eta=eta)
    f1, precision, recall = problem.support_f1(x_bar, tol=sr.SUPPORT_TOL)
    return {
        "scenario": scenario,
        "algorithm": label,
        "iteration": int(iteration),
        "recovery_error": float(problem.recovery_error(x_bar)),
        "support_f1": float(f1),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_size": int(np.count_nonzero(np.abs(x_bar) >= sr.SUPPORT_TOL)),
        "composite_g2": float(np.dot(mapping, mapping)),
    }


def _run_recovery_trajectory(
    scenario: str,
    seed: int,
    max_iter: int = 900,
    every: int = 25,
) -> List[Dict[str, Any]]:
    scenario_cfg = dict(sr.SCENARIOS[scenario])
    cfg = SPARSE_RECOVERY_CONFIGS[scenario]
    problem = sr.TASKS["sparse_recovery"].make_problem(scenario_cfg, seed)
    graph = generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)

    algorithms: List[Tuple[str, Any]] = []

    l1_algo = sr._make_factory(step_mult=cfg["step_mult"], lambda_reg=cfg["lambda_reg"])(problem, graph)
    sr._initialize_algorithm(l1_algo, problem.dim, seed)
    algorithms.append(("L1-SDBPG", l1_algo))

    sdbpg_algo = sr._make_factory(step_mult=cfg["step_mult"], lambda_reg=0.0)(problem, graph)
    sr._initialize_algorithm(sdbpg_algo, problem.dim, seed)
    algorithms.append(("SDBPG", sdbpg_algo))

    for name in ("DGD", "DGT", "ProxDGD", "ProxDGT"):
        algo = _make_baseline(name, cfg["baseline"][name], problem, graph)
        _init_baseline(algo, problem, seed)
        algorithms.append((name, algo))

    records: List[Dict[str, Any]] = []
    for label, algo in algorithms:
        records.append(_record_metrics(problem, algo, 0, label, scenario))
    for it in range(1, max_iter + 1):
        for _, algo in algorithms:
            algo.step()
        if it % every == 0 or it == max_iter:
            for label, algo in algorithms:
                records.append(_record_metrics(problem, algo, it, label, scenario))
    for rec in records:
        rec["seed"] = seed
    return records

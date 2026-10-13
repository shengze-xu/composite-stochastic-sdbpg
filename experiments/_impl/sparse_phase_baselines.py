"""Sparse phase baselines numerical routines."""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple


import numpy as np


from sdbpg.algorithm import DecentralizedGradientDescent
from sdbpg.algorithm import DecentralizedGradientTracking
from sdbpg.algorithm import ProximalDecentralizedGradientDescent
from sdbpg.algorithm import ProximalDecentralizedGradientTracking


from experiments._impl import sparse_phase as spr


def _initialize_baseline(algo: Any, problem: spr.SparsePhaseRetrievalProblem, seed: int) -> None:
    rng = np.random.default_rng(20260410 + int(seed))
    x0 = problem.spectral_init()

    xs = []
    for _ in range(problem.n_agents):
        xi = x0 + spr.INIT_PERTURB_STD * rng.standard_normal(problem.dim)
        xs.append(xi.copy())

    algo.x = xs
    if hasattr(algo, "y"):
        algo.y = [problem.local_gradient(i, algo.x[i]) for i in range(problem.n_agents)]


def _make_sdbpg_factory(step_mult: float, lambda_reg: float) -> Callable:
    return spr._make_factory(step_mult=step_mult, lambda_reg=lambda_reg)


def _make_baseline_factory(algorithm: str, params: Dict[str, float]) -> Callable:
    if algorithm == "DGD":
        return lambda problem, graph: DecentralizedGradientDescent(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
            consensus_weight=float(params["consensus_weight"]),
        )
    if algorithm == "DGT":
        return lambda problem, graph: DecentralizedGradientTracking(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
        )
    if algorithm == "ProxDGD":
        return lambda problem, graph: ProximalDecentralizedGradientDescent(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
            consensus_weight=float(params["consensus_weight"]),
            lambda_reg=float(params["lambda_reg"]),
        )
    if algorithm == "ProxDGT":
        return lambda problem, graph: ProximalDecentralizedGradientTracking(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
            lambda_reg=float(params["lambda_reg"]),
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _run_one_generic(
    algo: Any,
    problem: spr.SparsePhaseRetrievalProblem,
    max_iter: int,
    success_threshold: float,
    support_tol: float,
) -> Dict[str, Any]:
    g2_curve: List[float] = []
    composite_g2_curve: List[float] = []
    best_g2 = float("inf")
    best_composite_g2 = float("inf")
    diverged = False
    x_bar = np.zeros(problem.dim, dtype=float)
    lambda_reg = spr.algorithm_lambda_reg(algo)
    prox_eta = spr.algorithm_prox_eta(algo)

    for _ in range(max_iter):
        algo.step()

        if hasattr(algo, "agents"):
            xs = [ag.x for ag in algo.agents.values()]
        else:
            xs = algo.x
        x_bar = np.mean(xs, axis=0)

        grad = problem.global_gradient(x_bar)
        g2 = float(np.dot(grad, grad))
        mapping = spr.l1_prox_gradient_mapping(x_bar, grad, lambda_reg=lambda_reg, eta=prox_eta)
        composite_g2 = float(np.dot(mapping, mapping))

        if (
            not np.isfinite(g2)
            or not np.isfinite(composite_g2)
            or not np.isfinite(np.linalg.norm(x_bar))
            or g2 > spr.DIVERGENCE_THRESHOLD
            or composite_g2 > spr.DIVERGENCE_THRESHOLD
            or np.linalg.norm(x_bar) > 1e4
        ):
            diverged = True
            break

        g2_curve.append(g2)
        composite_g2_curve.append(composite_g2)
        best_g2 = min(best_g2, g2)
        best_composite_g2 = min(best_composite_g2, composite_g2)

    if diverged or len(g2_curve) == 0:
        return {
            "final_g2": float(spr.DIVERGENCE_THRESHOLD),
            "composite_g2": float(spr.DIVERGENCE_THRESHOLD),
            "final_rec": float("inf"),
            "support_f1": 0.0,
            "support_precision": 0.0,
            "support_recall": 0.0,
            "final_obj": float("inf"),
            "final_l1": float("inf"),
            "final_composite_obj": float("inf"),
            "support_size": int(problem.dim),
            "success": 0,
            "diverged": 1,
            "hit_t": int(max_iter + 1),
            "composite_hit_t": int(max_iter + 1),
        }

    final_rec = problem.recovery_error(x_bar)
    f1, precision, recall = problem.support_f1(x_bar, tol=support_tol)
    final_obj = problem.global_objective(x_bar)
    final_l1 = float(np.linalg.norm(x_bar, 1))
    final_composite_obj = float(final_obj + lambda_reg * final_l1)
    support_size = int(np.count_nonzero(np.abs(x_bar) >= support_tol))
    hit_t = spr._first_sustained_below(np.asarray(g2_curve, dtype=float), spr.GRAD_HIT_EPS, spr.GRAD_HIT_HOLD)
    composite_hit_t = spr._first_sustained_below(
        np.asarray(composite_g2_curve, dtype=float),
        spr.GRAD_HIT_EPS,
        spr.GRAD_HIT_HOLD,
    )

    return {
        "final_g2": float(best_g2),
        "composite_g2": float(best_composite_g2),
        "final_rec": float(final_rec),
        "support_f1": float(f1),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "final_obj": float(final_obj),
        "final_l1": float(final_l1),
        "final_composite_obj": float(final_composite_obj),
        "support_size": int(support_size),
        "success": int(final_rec < float(success_threshold)),
        "diverged": 0,
        "hit_t": int(hit_t) if int(hit_t) > 0 else int(max_iter + 1),
        "composite_hit_t": int(composite_hit_t) if int(composite_hit_t) > 0 else int(max_iter + 1),
    }


def _run_one_seed_sdbpg(
    task: spr.TaskSpec,
    scenario_cfg: Dict[str, Any],
    seed: int,
    step_mult: float,
    lambda_reg: float,
    max_iter: int,
    support_tol: float,
) -> Dict[str, Any]:
    problem = task.make_problem(scenario_cfg, seed)
    graph = spr.generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)
    algo = _make_sdbpg_factory(step_mult=step_mult, lambda_reg=lambda_reg)(problem, graph)
    spr._initialize_algorithm(algo, problem, seed)
    return _run_one_generic(algo, problem, max_iter=max_iter, success_threshold=task.success_threshold, support_tol=support_tol)


def _run_one_seed_baseline(
    task: spr.TaskSpec,
    scenario_cfg: Dict[str, Any],
    seed: int,
    algorithm: str,
    params: Dict[str, float],
    max_iter: int,
    support_tol: float,
) -> Dict[str, Any]:
    problem = task.make_problem(scenario_cfg, seed)
    graph = spr.generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)
    algo = _make_baseline_factory(algorithm, params)(problem, graph)
    _initialize_baseline(algo, problem, seed)
    return _run_one_generic(algo, problem, max_iter=max_iter, success_threshold=task.success_threshold, support_tol=support_tol)

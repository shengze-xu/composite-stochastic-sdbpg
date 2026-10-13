"""Complexity numerical routines."""

from __future__ import annotations

from dataclasses import dataclass


from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple


import numpy as np


from scipy.optimize import minimize


from sdbpg.core import QuadraticRegressionProblem
from sdbpg.core import generate_network_topology


N_AGENTS = 6


DIM = 10


N_BOOTSTRAP = 2000


INIT_PERTURB_STD = 0.1


BASE_PARAMS = dict(rho=2.0, alpha=0.02, lambda_param=0.025, beta=0.80, C=1.5)


@dataclass(frozen=True)
class Scenario:
    name: str
    topology: str
    samples_per_agent: int
    connectivity: float = 0.5


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("random_baseline", "random", 20, 0.5),
    Scenario("ring_baseline", "ring", 20, 0.0),
    Scenario("complete_baseline", "complete", 20, 0.0),
    Scenario("random_low_sample", "random", 10, 0.5),
)


def _global_gradient(problem: QuadraticRegressionProblem, x: np.ndarray) -> np.ndarray:
    return np.mean([problem.local_gradient(i, x) for i in range(problem.n_agents)], axis=0)


def _global_grad_norm_sq(x_list: Sequence[np.ndarray], problem: QuadraticRegressionProblem) -> float:
    x_bar = np.mean(x_list, axis=0)
    grad = _global_gradient(problem, x_bar)
    return float(np.dot(grad, grad))


def _consensus_error(x_list: Sequence[np.ndarray]) -> float:
    x_bar = np.mean(x_list, axis=0)
    return float(np.mean([np.dot(x - x_bar, x - x_bar) for x in x_list]))


def _bootstrap_ci_mean(values: Sequence[float], rng: np.random.Generator) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        val = float(arr[0])
        return val, val
    idx = rng.integers(0, arr.size, size=(N_BOOTSTRAP, arr.size))
    boots = arr[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _fit_loglog_slope(x_vals: Sequence[float], y_vals: Sequence[float]) -> Tuple[float, float, float]:
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    if np.count_nonzero(mask) < 3:
        return float("nan"), float("nan"), float("nan")
    lx = np.log(x_arr[mask])
    ly = np.log(y_arr[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    denom = float(np.sum((ly - ly.mean()) ** 2))
    if denom <= 1e-15:
        r2 = 1.0
    else:
        r2 = 1.0 - float(np.sum((ly - pred) ** 2)) / denom
    return float(slope), float(intercept), float(r2)


def _first_checkpoint_hit(
    iterations: Sequence[int],
    values: Sequence[float],
    eps: float,
    hold: int,
) -> Optional[int]:
    vals = np.asarray(values, dtype=float)
    iters = np.asarray(iterations, dtype=int)
    if vals.size == 0:
        return None
    hold = max(1, min(hold, vals.size))
    below = vals <= eps
    for idx in range(0, vals.size - hold + 1):
        if np.all(below[idx : idx + hold]):
            return int(iters[idx])
    return None


def _solve_bregman_proximal_point(
    problem: QuadraticRegressionProblem,
    degrees: np.ndarray,
    z_list: Sequence[np.ndarray],
    x_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    z_arr = np.asarray(z_list, dtype=float)
    mean_z = z_arr.mean(axis=0)
    degree_weighted_mean_z = (degrees[:, None] * z_arr).sum(axis=0) / float(problem.n_agents)

    penalty_coeff = BASE_PARAMS["C"] + BASE_PARAMS["rho"] * float(degrees.mean())
    bias = BASE_PARAMS["C"] * mean_z + BASE_PARAMS["rho"] * degree_weighted_mean_z

    def objective(x: np.ndarray) -> float:
        return float(
            problem.global_objective(x)
            + 0.5 * penalty_coeff * np.dot(x, x)
            - np.dot(bias, x)
        )

    def gradient(x: np.ndarray) -> np.ndarray:
        return _global_gradient(problem, x) + penalty_coeff * x - bias

    x0 = mean_z.copy() if x_init is None else np.asarray(x_init, dtype=float).copy()
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 120},
    )

    x_hat = np.asarray(result.x if np.all(np.isfinite(result.x)) else x0, dtype=float)
    stats = {
        "solver_success": float(bool(result.success)),
        "solver_nit": float(getattr(result, "nit", np.nan)),
        "solver_fun": float(getattr(result, "fun", np.nan)),
    }
    return x_hat, stats


def _run_seed(
    scenario: Scenario,
    seed: int,
    max_iter: int,
    checkpoint_every: int,
) -> List[Dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    problem = QuadraticRegressionProblem(N_AGENTS, DIM, scenario.samples_per_agent, seed=seed)
    graph = generate_network_topology(
        N_AGENTS,
        scenario.topology,
        connectivity=scenario.connectivity,
        seed=seed,
    )

    neighbors = {i: list(graph.neighbors(i)) for i in range(N_AGENTS)}
    degrees = np.asarray([graph.degree(i) for i in range(N_AGENTS)], dtype=float)

    x = [rng.standard_normal(DIM) * INIT_PERTURB_STD for _ in range(N_AGENTS)]
    z = [xi.copy() for xi in x]
    y = {i: {j: np.zeros(DIM) for j in neighbors[i]} for i in range(N_AGENTS)}

    records: List[Dict[str, float | int | str]] = []
    x_hat_prev: Optional[np.ndarray] = None

    for iteration in range(1, max_iter + 1):
        for i in range(N_AGENTS):
            for j in neighbors[i]:
                y[i][j] = y[i][j] + BASE_PARAMS["alpha"] * (x[i] - x[j])

        x_new = []
        for i in range(N_AGENTS):
            grad_f = problem.local_gradient(i, x[i])
            grad_breg = BASE_PARAMS["C"] * (x[i] - z[i])
            for j in neighbors[i]:
                grad_breg += BASE_PARAMS["rho"] * (x[i] - z[j])
            grad_dual = sum(y[i][j] for j in neighbors[i]) if neighbors[i] else np.zeros(DIM)
            grad_k = grad_f + grad_breg + grad_dual
            x_new.append(x[i] - BASE_PARAMS["lambda_param"] * grad_k)
        x = x_new

        z = [(xi + BASE_PARAMS["beta"] * zi) / (1.0 + BASE_PARAMS["beta"]) for xi, zi in zip(x, z)]

        should_record = iteration == 1 or iteration == max_iter or iteration % checkpoint_every == 0
        if not should_record:
            continue

        grad_norm_sq = _global_grad_norm_sq(x, problem)
        grad_norm = float(np.sqrt(max(grad_norm_sq, 0.0)))
        consensus_err = _consensus_error(x)

        x_hat, solver_stats = _solve_bregman_proximal_point(problem, degrees, z, x_hat_prev)
        x_hat_prev = x_hat
        residuals = np.linalg.norm(np.asarray(z) - x_hat[None, :], axis=1)

        records.append(
            {
                "scenario": scenario.name,
                "topology": scenario.topology,
                "samples_per_agent": scenario.samples_per_agent,
                "seed": seed,
                "iteration": iteration,
                "theory_proxy_mean": float(np.mean(residuals)),
                "theory_proxy_max": float(np.max(residuals)),
                "grad_norm": grad_norm,
                "grad_norm_sq": grad_norm_sq,
                "consensus_error": consensus_err,
                "solver_success": int(solver_stats["solver_success"]),
                "solver_nit": int(solver_stats["solver_nit"]),
                "solver_fun": float(solver_stats["solver_fun"]),
            }
        )

    return records


def _group_seed_records(
    records: Sequence[Dict[str, float | int | str]]
) -> Dict[Tuple[str, int], List[Dict[str, float | int | str]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, float | int | str]]] = {}
    for row in records:
        key = (str(row["scenario"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: int(item["iteration"]))
    return grouped


def build_complexity_summary(
    records: Sequence[Dict[str, float | int | str]],
    eps_grid: Sequence[float],
    hold_checkpoints: int,
) -> Tuple[List[Dict[str, float | int | str]], List[Dict[str, float | int | str]]]:
    grouped = _group_seed_records(records)
    rng = np.random.default_rng(20260409)

    summary_rows: List[Dict[str, float | int | str]] = []
    slope_rows: List[Dict[str, float | int | str]] = []

    scenario_names = sorted({str(row["scenario"]) for row in records})
    for scenario in scenario_names:
        per_eps_mean_hit: List[float] = []
        per_eps_values: List[float] = []
        valid_eps_for_fit: List[float] = []

        for eps in eps_grid:
            hits: List[float] = []
            for seed in sorted({int(row["seed"]) for row in records if str(row["scenario"]) == scenario}):
                seed_rows = grouped[(scenario, seed)]
                iterations = [int(row["iteration"]) for row in seed_rows]
                proxy_vals = [float(row["theory_proxy_max"]) for row in seed_rows]
                hit_iter = _first_checkpoint_hit(iterations, proxy_vals, eps, hold_checkpoints)
                hits.append(float(hit_iter) if hit_iter is not None else float("nan"))

            finite_hits = [hit for hit in hits if np.isfinite(hit)]
            success_count = len(finite_hits)
            success_rate = success_count / max(len(hits), 1)
            mean_hit = float(np.mean(finite_hits)) if finite_hits else float("nan")
            median_hit = float(np.median(finite_hits)) if finite_hits else float("nan")
            ci_lo, ci_hi = _bootstrap_ci_mean(finite_hits, rng)

            summary_rows.append(
                {
                    "scenario": scenario,
                    "epsilon": eps,
                    "mean_hit_iter": mean_hit,
                    "median_hit_iter": median_hit,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "success_count": success_count,
                    "num_seeds": len(hits),
                    "success_rate": success_rate,
                }
            )

            if np.isfinite(mean_hit) and success_rate >= 0.60:
                valid_eps_for_fit.append(eps)
                per_eps_mean_hit.append(mean_hit)
                per_eps_values.append(mean_hit)

        slope, intercept, r2 = _fit_loglog_slope(valid_eps_for_fit, per_eps_values)
        slope_rows.append(
            {
                "scenario": scenario,
                "complexity_slope": slope,
                "complexity_intercept": intercept,
                "complexity_r2": r2,
                "num_fit_points": len(valid_eps_for_fit),
            }
        )

    return summary_rows, slope_rows

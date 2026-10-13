"""Diagnostics numerical routines."""

from __future__ import annotations


from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple


import numpy as np


from sdbpg.core import generate_network_topology


from experiments._impl import sparse as sr


from experiments._impl import stochastic as st


MAX_ITER = 700


RECORD_EVERY = 10


SCENARIO = "sparse_noisy_undersampled"


STOCH_SCENARIO = "sparse_noisy_moderate"


def _agent_arrays(algo: Any) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    xs: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    dual_sums: List[np.ndarray] = []
    for i in sorted(algo.agents):
        ag = algo.agents[i]
        xs.append(np.asarray(ag.x, dtype=float).copy())
        zs.append(np.asarray(ag.z, dtype=float).copy())
        if ag.neighbors:
            dual_sums.append(np.sum([ag.y[j] for j in ag.neighbors], axis=0))
        else:
            dual_sums.append(np.zeros_like(ag.x))
    return xs, zs, dual_sums


def _local_grad(problem: Any, agent_id: int, x: np.ndarray) -> np.ndarray:
    if hasattr(problem, "full_local_gradient"):
        return np.asarray(problem.full_local_gradient(agent_id, x), dtype=float)
    return np.asarray(problem.local_gradient(agent_id, x), dtype=float)


def _consensus_sq(xs: Sequence[np.ndarray], graph: Any) -> float:
    total = 0.0
    for i, j in graph.edges():
        diff = np.asarray(xs[i]) - np.asarray(xs[j])
        total += float(np.dot(diff, diff))
    return total / max(graph.number_of_edges(), 1)


def _l1_stationarity_dist_sq(v: np.ndarray, x: np.ndarray, lambda_reg: float) -> float:
    lam = float(lambda_reg)
    if lam <= 0.0:
        return float(np.dot(v, v))
    x_arr = np.asarray(x, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    nonzero = np.abs(x_arr) > 1e-10
    residual = np.zeros_like(v_arr)
    residual[nonzero] = v_arr[nonzero] + lam * np.sign(x_arr[nonzero])
    zero_idx = ~nonzero
    residual[zero_idx] = np.sign(v_arr[zero_idx]) * np.maximum(np.abs(v_arr[zero_idx]) - lam, 0.0)
    return float(np.dot(residual, residual))


def _lifted_metrics(problem: Any, algo: Any, graph: Any) -> Dict[str, float]:
    xs, zs, dual_sums = _agent_arrays(algo)
    lambda_reg = float(getattr(algo, "lambda_reg", 0.0))
    stationarity = 0.0
    for i, x_i in enumerate(xs):
        v_i = _local_grad(problem, i, x_i) + dual_sums[i]
        stationarity += _l1_stationarity_dist_sq(v_i, x_i, lambda_reg=lambda_reg)
    stationarity /= max(len(xs), 1)
    consensus = _consensus_sq(xs, graph)
    e_val = float(sum(np.dot(x - z, x - z) for x, z in zip(xs, zs)))
    x_bar = np.mean(np.asarray(xs), axis=0)
    support = np.abs(x_bar) >= sr.SUPPORT_TOL
    return {
        "lifted_residual_sq": float(stationarity + consensus),
        "stationarity_sq": float(stationarity),
        "consensus_sq": float(consensus),
        "e": float(e_val),
        "support_size": float(np.count_nonzero(support)),
    }


def _support_switch(prev_support: np.ndarray | None, algo: Any) -> Tuple[np.ndarray, float]:
    xs, _, _ = _agent_arrays(algo)
    x_bar = np.mean(np.asarray(xs), axis=0)
    support = np.abs(x_bar) >= sr.SUPPORT_TOL
    if prev_support is None:
        return support, 0.0
    return support, float(np.count_nonzero(support != prev_support))


def _run_deterministic_case(case_id: str, label: str, lambda_reg: float, seed: int, max_iter: int = MAX_ITER) -> List[Dict[str, Any]]:
    cfg = dict(sr.SCENARIOS[SCENARIO])
    problem = sr.TASKS["sparse_recovery"].make_problem(cfg, seed)
    graph = generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)
    algo = sr._make_factory(step_mult=1.5, lambda_reg=lambda_reg)(problem, graph)
    sr._initialize_algorithm(algo, problem.dim, seed)

    rows: List[Dict[str, Any]] = []
    support_prev: np.ndarray | None = None
    support_switch_window = 0.0
    for iteration in range(0, max_iter + 1):
        if iteration > 0:
            algo.step()
        support_prev, support_switch = _support_switch(support_prev, algo)
        support_switch_window += support_switch
        if iteration % RECORD_EVERY != 0 and iteration != max_iter:
            continue
        metrics = _lifted_metrics(problem, algo, graph)
        rows.append(
            {
                "case_id": case_id,
                "case_label": label,
                "seed": seed,
                "iteration": iteration,
                "oracle_noise_sq": float("nan"),
                "support_switch": support_switch_window,
                **metrics,
            }
        )
        support_switch_window = 0.0
    return rows


def _run_stochastic_case(seed: int, max_iter: int = MAX_ITER) -> List[Dict[str, Any]]:
    setting = st.build_setting_manifest([STOCH_SCENARIO], "mini_batch", [4], [])[0]
    step_mult, lambda_reg = _selected_stochastic_config(setting["setting_id"])
    problem = st._make_problem(setting, seed)
    graph = generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)
    algo = st._make_sdbpg_factory(step_mult=step_mult, lambda_reg=lambda_reg)(problem, graph)
    st._initialize_sdbpg(algo, problem.dim, seed)

    rows: List[Dict[str, Any]] = []
    support_prev: np.ndarray | None = None
    support_switch_window = 0.0
    for iteration in range(0, max_iter + 1):
        oracle_noise = float("nan")
        if iteration > 0:
            problem.begin_step()
            algo.step()
            oracle_noise = float(problem.consume_step_oracle_summary()["avg_oracle_noise_sq"])
        support_prev, support_switch = _support_switch(support_prev, algo)
        support_switch_window += support_switch
        if iteration % RECORD_EVERY != 0 and iteration != max_iter:
            continue
        metrics = _lifted_metrics(problem, algo, graph)
        rows.append(
            {
                "case_id": "stochastic_l1",
                "case_label": "Stochastic L1-SDBPG",
                "seed": seed,
                "iteration": iteration,
                "oracle_noise_sq": oracle_noise,
                "support_switch": support_switch_window,
                **metrics,
            }
        )
        support_switch_window = 0.0
    return rows


def _selected_stochastic_config(setting_id: str) -> Tuple[float, float]:
    """Fixed mini-batch-four parameters used for the residual diagnostics."""
    from experiments.common import config
    parameters = config("diagnostics")
    return parameters["step_mult"], parameters["lambda_reg"]

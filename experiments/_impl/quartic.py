"""Quartic numerical routines."""

from __future__ import annotations

import json


from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Tuple


import networkx as nx


import numpy as np


import pandas as pd


from experiments._impl.phase import PhaseRetrievalProblem


from sdbpg.baselines import ProxGT, identity_prox


from sdbpg.geometry import GeometrySDBPG


N_AGENTS = 6


DIM = 16


SAMPLES_PER_AGENT = 60


NOISE_STD = 0.02


def benchmark_graph() -> nx.Graph:
    graph = nx.cycle_graph(N_AGENTS)
    graph.add_edge(0, 3)
    graph.add_edge(1, 4)
    return graph


def make_problem(seed: int, heterogeneity: float) -> PhaseRetrievalProblem:
    return PhaseRetrievalProblem(
        n_agents=N_AGENTS,
        dim=DIM,
        samples_per_agent=SAMPLES_PER_AGENT,
        noise_std=NOISE_STD,
        hetero_strength=float(heterogeneity),
        seed=int(seed),
    )


def common_initialization(problem: PhaseRetrievalProblem, seed: int) -> np.ndarray:
    center = problem.spectral_init()
    rng = np.random.default_rng(20260803 + int(seed))
    return np.stack([center + 0.01 * rng.standard_normal(problem.dim) for _ in range(problem.n_agents)])


def make_algorithm(
    method: str,
    config: Mapping[str, float],
    problem: PhaseRetrievalProblem,
    graph: nx.Graph,
    initial: np.ndarray,
) -> Any:
    if method in {"Quadratic-SDBPG", "Quartic-SDBPG"}:
        geometry = "quadratic" if method == "Quadratic-SDBPG" else "quartic"
        return GeometrySDBPG(
            problem=problem,
            graph=graph,
            x0=initial,
            geometry=geometry,
            rho=float(config.get("rho", 2.0)),
            dual_stepsize=float(config.get("dual_stepsize", 0.01)),
            primal_stepsize=float(config["primal_stepsize"]),
            beta=float(config.get("beta", 0.8)),
            anchor_weight=float(config.get("anchor_weight", 1.5)),
            lambda_reg=0.0,
            samples_per_call=SAMPLES_PER_AGENT,
        )
    if method == "ProxGT":
        return ProxGT(
            graph=graph,
            x0=initial,
            gradient_oracle=problem.local_gradient,
            prox_operator=identity_prox,
            stepsize=float(config["stepsize"]),
            consensus_rounds=int(config["consensus_rounds"]),
            samples_per_call=SAMPLES_PER_AGENT,
        )
    raise ValueError(f"unknown method: {method}")


def _consensus_mse(states: np.ndarray) -> float:
    average = np.mean(states, axis=0)
    return float(np.mean(np.sum((states - average) ** 2, axis=1)))


def run_one(
    method: str,
    config: Mapping[str, float],
    seed: int,
    heterogeneity: float,
    iterations: int,
    checkpoint: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    problem = make_problem(seed, heterogeneity)
    graph = benchmark_graph()
    initial = common_initialization(problem, seed)
    algorithm = make_algorithm(method, config, problem, graph, initial)
    history: List[Dict[str, Any]] = []
    diverged = False
    for iteration in range(1, int(iterations) + 1):
        algorithm.step()
        states = np.asarray(algorithm.states, dtype=float)
        norms = np.linalg.norm(states, axis=1)
        if not np.all(np.isfinite(norms)) or float(np.max(norms)) > 1e4:
            diverged = True
            break
        if iteration == 1 or iteration % int(checkpoint) == 0 or iteration == iterations:
            average = np.mean(states, axis=0)
            cost = algorithm.cost.snapshot()
            history.append(
                {
                    "seed": int(seed),
                    "heterogeneity": float(heterogeneity),
                    "algorithm": method,
                    **cost,
                    "recovery_error": problem.recovery_error(average),
                    "objective": problem.global_objective(average),
                    "consensus_mse": _consensus_mse(states),
                }
            )

    states = np.asarray(algorithm.states, dtype=float)
    average = np.mean(states, axis=0)
    if diverged or not np.all(np.isfinite(average)):
        recovery_error = 1e6
        objective = 1e12
        stationarity = 1e12
        consensus = 1e12
    else:
        diagnostic_gradient = problem.global_gradient(average)
        recovery_error = problem.recovery_error(average)
        objective = problem.global_objective(average)
        stationarity = float(np.dot(diagnostic_gradient, diagnostic_gradient))
        consensus = _consensus_mse(states)
    final = {
        "seed": int(seed),
        "heterogeneity": float(heterogeneity),
        "algorithm": method,
        "config": json.dumps(dict(config), sort_keys=True),
        **algorithm.cost.snapshot(),
        "recovery_error": float(recovery_error),
        "objective": float(objective),
        "stationarity_g2": float(stationarity),
        "consensus_mse": float(consensus),
        "diverged": int(diverged),
        "diagnostic_full_gradients_charged": 0,
    }
    return final, history


def paired_bootstrap(
    frame: pd.DataFrame,
    baseline: str,
    heterogeneity: float,
    rounds: int = 10000,
) -> Dict[str, Any]:
    subset = frame[np.isclose(frame["heterogeneity"], float(heterogeneity))]
    pivot = subset.pivot(index="seed", columns="algorithm", values="recovery_error").dropna()
    differences = (
        pivot["Quartic-SDBPG"].to_numpy(dtype=float)
        - pivot[baseline].to_numpy(dtype=float)
    )
    rng = np.random.default_rng(20260803 + int(round(float(heterogeneity) * 10)))
    indices = rng.integers(0, differences.size, size=(int(rounds), differences.size))
    boot = differences[indices].mean(axis=1)
    return {
        "heterogeneity": float(heterogeneity),
        "ours": "Quartic-SDBPG",
        "baseline": baseline,
        "n_pairs": int(differences.size),
        "mean_recovery_difference": float(np.mean(differences)),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "favorable_seed_rate": float(np.mean(differences < 0.0)),
    }


def cost_to_target(history: pd.DataFrame, target: float = 0.03) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (heterogeneity, seed, method), group in history.groupby(
        ["heterogeneity", "seed", "algorithm"]
    ):
        ordered = group.sort_values("iterations")
        reached = ordered[ordered["recovery_error"] <= float(target)]
        if reached.empty:
            rows.append(
                {
                    "heterogeneity": heterogeneity,
                    "seed": seed,
                    "algorithm": method,
                    "target": target,
                    "reached": 0,
                    "communication_rounds_to_target": np.nan,
                    "communication_mib_to_target": np.nan,
                    "oracle_samples_to_target": np.nan,
                    "wall_time_sec_to_target": np.nan,
                }
            )
        else:
            first = reached.iloc[0]
            rows.append(
                {
                    "heterogeneity": heterogeneity,
                    "seed": seed,
                    "algorithm": method,
                    "target": target,
                    "reached": 1,
                    "communication_rounds_to_target": first["communication_rounds"],
                    "communication_mib_to_target": first["communication_mib"],
                    "oracle_samples_to_target": first["oracle_samples"],
                    "wall_time_sec_to_target": first["wall_time_sec"],
                }
            )
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["heterogeneity", "algorithm"], as_index=False)
        .agg(
            recovery_error_mean=("recovery_error", "mean"),
            recovery_error_median=("recovery_error", "median"),
            objective_mean=("objective", "mean"),
            stationarity_g2_mean=("stationarity_g2", "mean"),
            consensus_mse_mean=("consensus_mse", "mean"),
            divergence_rate=("diverged", "mean"),
            communication_rounds=("communication_rounds", "mean"),
            communication_mib=("communication_mib", "mean"),
            oracle_samples=("oracle_samples", "mean"),
            wall_time_sec=("wall_time_sec", "mean"),
        )
    )

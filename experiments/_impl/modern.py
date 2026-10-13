"""Modern numerical routines."""

from __future__ import annotations

from experiments._impl import sparse as sparse


from experiments._impl import fourier as fourier


from experiments._impl import stochastic as stochastic


import json


import time


from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Tuple


import networkx as nx


import numpy as np


import pandas as pd


from sdbpg.baselines import CostLedger, ProxDASAGT, ProxGT, l1_prox


def _states(algorithm: Any) -> np.ndarray:
    if hasattr(algorithm, "states"):
        return np.asarray(algorithm.states, dtype=float)
    if hasattr(algorithm, "agents"):
        return np.stack([agent.x for agent in algorithm.agents.values()])
    return np.asarray(algorithm.x, dtype=float)


def _charge_sdbpg_step(
    ledger: CostLedger,
    graph: nx.Graph,
    dim: int,
    n_agents: int,
    samples_per_call: int,
) -> None:
    ledger.iterations += 1
    ledger.charge_oracle(n_agents, samples_per_call)
    # SDBPG broadcasts x and z together in one neighbor exchange.
    ledger.charge_communication(graph, dim, sequential_rounds=1, vectors_per_round=2)


def _charge_proxdgt_step(
    ledger: CostLedger,
    graph: nx.Graph,
    dim: int,
    n_agents: int,
    samples_per_call: int,
) -> None:
    ledger.iterations += 1
    ledger.charge_oracle(n_agents, samples_per_call)
    ledger.charge_communication(graph, dim, sequential_rounds=1, vectors_per_round=2)


def _common_initialization(n_agents: int, dim: int, seed: int, zero: bool) -> np.ndarray:
    if zero:
        return np.zeros((int(n_agents), int(dim)), dtype=float)
    rng = np.random.default_rng(20260803 + int(seed))
    center = rng.standard_normal(int(dim))
    center /= max(float(np.linalg.norm(center)), 1e-12)
    return np.stack(
        [center + sparse.INIT_PERTURB_STD * rng.standard_normal(int(dim)) for _ in range(int(n_agents))]
    )


def _initialize_legacy(algorithm: Any, initial: np.ndarray) -> None:
    if hasattr(algorithm, "agents"):
        for i, agent in algorithm.agents.items():
            agent.x = initial[int(i)].copy()
            agent.z = initial[int(i)].copy()
            for neighbor in agent.neighbors:
                agent.y[neighbor] = np.zeros(initial.shape[1], dtype=float)
            agent.neighbors_x = {}
            agent.neighbors_z = {}
    else:
        algorithm.x = [row.copy() for row in initial]
        if hasattr(algorithm, "grad"):
            algorithm.grad = [
                algorithm.problem.local_gradient(i, algorithm.x[i])
                for i in range(algorithm.problem.n_agents)
            ]
        if hasattr(algorithm, "y"):
            if hasattr(algorithm, "grad"):
                algorithm.y = [gradient.copy() for gradient in algorithm.grad]
            else:
                algorithm.y = [
                    algorithm.problem.local_gradient(i, algorithm.x[i])
                    for i in range(algorithm.problem.n_agents)
                ]


def _consensus_mse(states: np.ndarray) -> float:
    average = np.mean(states, axis=0)
    return float(np.mean(np.sum((states - average) ** 2, axis=1)))


def _metric_history_row(
    task: str,
    seed: int,
    algorithm: str,
    ledger: CostLedger,
    metrics: Mapping[str, float],
) -> Dict[str, Any]:
    return {
        "task": task,
        "seed": int(seed),
        "algorithm": algorithm,
        **ledger.snapshot(),
        **{key: float(value) for key, value in metrics.items()},
    }


def _run_fourier_method(
    seed: int,
    name: str,
    config: Mapping[str, float],
    iterations: int,
    checkpoint: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    problem = fourier.DistributedFourierWaveletProblem(
        fourier.FourierImagingConfig(image_name="phantom", image_size=64),
        seed=int(seed),
    )
    graph = fourier.imaging_graph(problem.n_agents)
    initial = _common_initialization(problem.n_agents, problem.dim, seed, zero=True)
    lambda_reg = float(config.get("lambda_reg", 0.02))
    ledger: CostLedger
    if name == "ProxGT":
        algorithm = ProxGT(
            graph=graph,
            x0=initial,
            gradient_oracle=problem.local_gradient,
            prox_operator=l1_prox(lambda_reg),
            stepsize=float(config["stepsize"]),
            consensus_rounds=int(config["consensus_rounds"]),
            samples_per_call=1,
        )
        ledger = algorithm.cost
    elif name == "L1-SDBPG":
        algorithm = fourier.make_algorithm(name, config, problem, graph)
        _initialize_legacy(algorithm, initial)
        ledger = CostLedger()
    elif name == "ProxDGT":
        # Construct directly so the paper-required initial gradient is evaluated
        # exactly once and both its oracle count and elapsed time are charged.
        initialization_started = time.perf_counter()
        algorithm = fourier.CachedProxDGT(
            problem,
            graph,
            float(config["stepsize"]),
            lambda_reg,
        )
        ledger = CostLedger()
        ledger.charge_oracle(problem.n_agents, 1)
        ledger.wall_time_sec += float(time.perf_counter() - initialization_started)
    else:
        raise ValueError(f"unknown Fourier method: {name}")

    history: List[Dict[str, Any]] = []
    for iteration in range(1, int(iterations) + 1):
        started = time.perf_counter()
        algorithm.step()
        if name == "L1-SDBPG":
            _charge_sdbpg_step(ledger, graph, problem.dim, problem.n_agents, 1)
            ledger.wall_time_sec += float(time.perf_counter() - started)
        elif name == "ProxDGT":
            _charge_proxdgt_step(ledger, graph, problem.dim, problem.n_agents, 1)
            ledger.wall_time_sec += float(time.perf_counter() - started)

        if iteration == 1 or iteration % int(checkpoint) == 0 or iteration == iterations:
            states = _states(algorithm)
            average = np.mean(states, axis=0)
            quality = fourier.image_metrics(problem, average)
            history.append(
                _metric_history_row(
                    "fourier_imaging",
                    seed,
                    name,
                    ledger,
                    {**quality, "consensus_mse": _consensus_mse(states)},
                )
            )

    states = _states(algorithm)
    average = np.mean(states, axis=0)
    quality = fourier.image_metrics(problem, average)
    diagnostic_gradient = problem.global_gradient(average)
    mapping = sparse.l1_prox_gradient_mapping(average, diagnostic_gradient, lambda_reg, eta=0.25)
    final = {
        "task": "fourier_imaging",
        "seed": int(seed),
        "algorithm": name,
        "config": json.dumps(dict(config), sort_keys=True),
        **ledger.snapshot(),
        **quality,
        "consensus_mse": _consensus_mse(states),
        "composite_stationarity_g2": float(np.dot(mapping, mapping)),
        "composite_objective": float(problem.global_objective(average) + lambda_reg * np.linalg.norm(average, 1)),
        "diagnostic_full_gradients_charged": 0,
    }
    return final, history


def _stochastic_problem(seed: int, batch_size: int) -> stochastic.StochasticSparseLinearRecoveryProblem:
    scenario = sparse.SCENARIOS["sparse_noisy_moderate"]
    return stochastic.StochasticSparseLinearRecoveryProblem(
        n_agents=int(scenario["n_agents"]),
        dim=int(scenario["dim"]),
        sparsity=float(scenario["sparsity"]),
        samples_per_agent=int(scenario["samples"]),
        noise_std=float(scenario["noise_std"]),
        hetero_strength=float(scenario["hetero_strength"]),
        seed=int(seed),
        oracle_mode="mini_batch",
        batch_size=int(batch_size),
        noise_sigma=0.0,
    )


def _run_stochastic_method(
    seed: int,
    name: str,
    config: Mapping[str, float],
    iterations: int,
    checkpoint: int,
    batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    problem = _stochastic_problem(seed, batch_size)
    graph = nx.cycle_graph(problem.n_agents)
    if problem.n_agents >= 6:
        graph.add_edge(0, problem.n_agents // 2)
        graph.add_edge(1, problem.n_agents // 2 + 1)
    initial = _common_initialization(problem.n_agents, problem.dim, seed, zero=False)
    lambda_reg = float(config.get("lambda_reg", 0.02))

    if name == "L1-SDBPG":
        params = dict(sparse.BASE_PARAMS)
        multiplier = float(config.get("step_multiplier", 1.0))
        params["alpha"] *= multiplier
        params["lambda_param"] *= multiplier
        algorithm = sparse.SparseL1SDBPG(
            problem=problem,
            graph=graph,
            lambda_reg=lambda_reg,
            **params,
        )
        _initialize_legacy(algorithm, initial)
        ledger = CostLedger()
    elif name == "ProxGT-SA":
        algorithm = ProxGT(
            graph=graph,
            x0=initial,
            gradient_oracle=problem.local_gradient,
            prox_operator=l1_prox(lambda_reg),
            stepsize=float(config["stepsize"]),
            consensus_rounds=int(config["consensus_rounds"]),
            samples_per_call=int(batch_size),
        )
        ledger = algorithm.cost
    elif name == "Prox-DASA-GT":
        algorithm = ProxDASAGT(
            graph=graph,
            x0=initial,
            gradient_oracle=problem.local_gradient,
            prox_operator=l1_prox(lambda_reg),
            gamma=float(config["gamma"]),
            alpha=float(config["alpha"]),
            consensus_rounds=int(config["consensus_rounds"]),
            samples_per_call=int(batch_size),
        )
        ledger = algorithm.cost
    else:
        raise ValueError(f"unknown stochastic method: {name}")

    history: List[Dict[str, Any]] = []
    for iteration in range(1, int(iterations) + 1):
        started = time.perf_counter()
        algorithm.step()
        if name == "L1-SDBPG":
            _charge_sdbpg_step(
                ledger,
                graph,
                problem.dim,
                problem.n_agents,
                int(batch_size),
            )
            ledger.wall_time_sec += float(time.perf_counter() - started)

        if iteration == 1 or iteration % int(checkpoint) == 0 or iteration == iterations:
            states = _states(algorithm)
            average = np.mean(states, axis=0)
            f1, precision, recall = problem.support_f1(average)
            history.append(
                _metric_history_row(
                    "stochastic_sparse_recovery",
                    seed,
                    name,
                    ledger,
                    {
                        "recovery_error": problem.recovery_error(average),
                        "support_f1": f1,
                        "support_precision": precision,
                        "support_recall": recall,
                        "consensus_mse": _consensus_mse(states),
                    },
                )
            )

    states = _states(algorithm)
    average = np.mean(states, axis=0)
    f1, precision, recall = problem.support_f1(average)
    diagnostic_gradient = problem.full_global_gradient(average)
    mapping = sparse.l1_prox_gradient_mapping(average, diagnostic_gradient, lambda_reg, eta=0.25)
    final = {
        "task": "stochastic_sparse_recovery",
        "seed": int(seed),
        "algorithm": name,
        "config": json.dumps(dict(config), sort_keys=True),
        **ledger.snapshot(),
        "recovery_error": problem.recovery_error(average),
        "support_f1": f1,
        "support_precision": precision,
        "support_recall": recall,
        "consensus_mse": _consensus_mse(states),
        "composite_stationarity_g2": float(np.dot(mapping, mapping)),
        "composite_objective": float(problem.global_objective(average) + lambda_reg * np.linalg.norm(average, 1)),
        "diagnostic_full_gradients_charged": 0,
    }
    return final, history


def _paired_bootstrap(
    frame: pd.DataFrame,
    task: str,
    ours: str,
    baseline: str,
    metric: str,
    lower_is_better: bool,
    rounds: int = 10000,
) -> Dict[str, Any]:
    subset = frame[frame["task"] == task]
    pivot = subset.pivot(index="seed", columns="algorithm", values=metric).dropna()
    differences = pivot[ours].to_numpy(dtype=float) - pivot[baseline].to_numpy(dtype=float)
    rng = np.random.default_rng(20260803)
    indices = rng.integers(0, differences.size, size=(int(rounds), differences.size))
    bootstrap = differences[indices].mean(axis=1)
    favorable = differences < 0.0 if lower_is_better else differences > 0.0
    return {
        "task": task,
        "metric": metric,
        "ours": ours,
        "baseline": baseline,
        "n_pairs": int(differences.size),
        "mean_difference_ours_minus_baseline": float(np.mean(differences)),
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
        "favorable_seed_rate": float(np.mean(favorable)),
    }


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in (
            "nrmse",
            "psnr",
            "ssim",
            "recovery_error",
            "support_f1",
            "consensus_mse",
            "composite_stationarity_g2",
            "communication_rounds",
            "communication_mib",
            "oracle_calls",
            "oracle_samples",
            "wall_time_sec",
        )
        if column in frame.columns
    ]
    return frame.groupby(["task", "algorithm"], as_index=False)[metric_columns].mean(numeric_only=True)

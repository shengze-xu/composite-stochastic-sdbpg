"""Coverage numerical routines."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple


from experiments._impl import sparse_baselines as sc


from experiments._impl import sparse as sr


def _make_graph(n_agents: int, topology: str, seed: int):
    if topology == "random":
        return sr.generate_network_topology(n_agents, topology, connectivity=0.5, seed=seed)
    return sr.generate_network_topology(n_agents, topology, seed=seed)


def _run_one_seed_sdbpg(
    task: sr.TaskSpec,
    scenario_cfg: Dict[str, Any],
    seed: int,
    step_mult: float,
    lambda_reg: float,
    max_iter: int,
    support_tol: float,
    topology: str,
) -> Dict[str, Any]:
    problem = task.make_problem(scenario_cfg, seed)
    graph = _make_graph(problem.n_agents, topology, seed)
    algo = sr._make_factory(step_mult=step_mult, lambda_reg=lambda_reg)(problem, graph)
    sr._initialize_algorithm(algo, problem.dim, seed)
    return sc._run_one_generic(algo, problem, max_iter=max_iter, success_threshold=task.success_threshold, support_tol=support_tol)


def _run_one_seed_baseline(
    task: sr.TaskSpec,
    scenario_cfg: Dict[str, Any],
    seed: int,
    algorithm: str,
    params: Dict[str, float],
    max_iter: int,
    support_tol: float,
    topology: str,
) -> Dict[str, Any]:
    problem = task.make_problem(scenario_cfg, seed)
    graph = _make_graph(problem.n_agents, topology, seed)
    algo = sc._make_baseline_factory(algorithm, params)(problem, graph)
    sc._initialize_baseline(algo, problem, seed)
    return sc._run_one_generic(algo, problem, max_iter=max_iter, success_threshold=task.success_threshold, support_tol=support_tol)

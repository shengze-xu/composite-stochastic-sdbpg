"""Extra numerical routines."""

from __future__ import annotations

from dataclasses import dataclass


from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple


import networkx as nx


import numpy as np


from experiments._impl import sparse as sr


from sdbpg.core import generate_network_topology
from sdbpg.core import soft_threshold


@dataclass(frozen=True)
class ProxExtraConfig:
    stepsize: float
    lambda_reg: float


class ProxExtraSparse:
    """PG-EXTRA/Prox-EXTRA style composite decentralized baseline."""

    def __init__(self, problem: sr.SparseLinearRecoveryProblem, graph: nx.Graph, stepsize: float, lambda_reg: float):
        self.problem = problem
        self.graph = graph
        self.stepsize = float(stepsize)
        self.lambda_reg = float(lambda_reg)
        self.W = _metropolis_mixing_matrix(graph, problem.n_agents)
        self.W_tilde = 0.5 * (np.eye(problem.n_agents) + self.W)
        self.x = [np.zeros(problem.dim, dtype=float) for _ in range(problem.n_agents)]
        self.x_prev: List[np.ndarray] | None = None
        self.grad_prev: List[np.ndarray] | None = None
        self.initialized = False

    def step(self) -> None:
        x_old = [x.copy() for x in self.x]
        grad_old = [self.problem.local_gradient(i, x_old[i]) for i in range(self.problem.n_agents)]

        if not self.initialized:
            x_new = []
            for i in range(self.problem.n_agents):
                mixed = sum(self.W[i, j] * x_old[j] for j in range(self.problem.n_agents))
                v = mixed - self.stepsize * grad_old[i]
                x_new.append(soft_threshold(v, self.stepsize * self.lambda_reg))
            self.x_prev = x_old
            self.grad_prev = grad_old
            self.x = x_new
            self.initialized = True
            return

        assert self.x_prev is not None
        assert self.grad_prev is not None
        x_new = []
        for i in range(self.problem.n_agents):
            mixed_now = x_old[i] + sum(self.W[i, j] * x_old[j] for j in range(self.problem.n_agents))
            mixed_prev = sum(self.W_tilde[i, j] * self.x_prev[j] for j in range(self.problem.n_agents))
            correction = self.stepsize * (grad_old[i] - self.grad_prev[i])
            v = mixed_now - mixed_prev - correction
            x_new.append(soft_threshold(v, self.stepsize * self.lambda_reg))

        self.x_prev = x_old
        self.grad_prev = grad_old
        self.x = x_new


def _metropolis_mixing_matrix(graph: nx.Graph, n_agents: int) -> np.ndarray:
    W = np.zeros((n_agents, n_agents), dtype=float)
    degrees = dict(graph.degree())
    for i, j in graph.edges():
        weight = 1.0 / (1.0 + max(float(degrees[i]), float(degrees[j])))
        W[i, j] = weight
        W[j, i] = weight
    for i in range(n_agents):
        W[i, i] = 1.0 - float(np.sum(W[i]))
    return W


def _initialize(algo: ProxExtraSparse, problem: sr.SparseLinearRecoveryProblem, seed: int) -> None:
    rng = np.random.default_rng(20260410 + int(seed))
    x0 = rng.standard_normal(problem.dim)
    x0 /= max(np.linalg.norm(x0), 1e-12)
    algo.x = [x0 + sr.INIT_PERTURB_STD * rng.standard_normal(problem.dim) for _ in range(problem.n_agents)]
    algo.x_prev = None
    algo.grad_prev = None
    algo.initialized = False


def _algorithm_label(config: ProxExtraConfig) -> str:
    return "PG-EXTRA" if float(config.lambda_reg) == 0.0 else "Prox-EXTRA"


def _run_config_seed(scenario: str, config: ProxExtraConfig, seed: int, max_iter: int) -> Dict[str, Any]:
    scenario_cfg = dict(sr.SCENARIOS[scenario])
    task = sr.TASKS["sparse_recovery"]
    problem = task.make_problem(scenario_cfg, seed)
    graph = generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)
    algo = ProxExtraSparse(problem=problem, graph=graph, stepsize=config.stepsize, lambda_reg=config.lambda_reg)
    _initialize(algo, problem, seed)
    metrics = _run_one(algo, problem, max_iter=max_iter, success_threshold=task.success_threshold)
    return {
        "scenario": scenario,
        "seed": int(seed),
        "algorithm": _algorithm_label(config),
        "stepsize": float(config.stepsize),
        "lambda_reg": float(config.lambda_reg),
        **metrics,
    }


def _run_one(
    algo: ProxExtraSparse,
    problem: sr.SparseLinearRecoveryProblem,
    max_iter: int,
    success_threshold: float,
) -> Dict[str, Any]:
    g2_curve: List[float] = []
    composite_g2_curve: List[float] = []
    best_g2 = float("inf")
    best_composite_g2 = float("inf")
    diverged = False
    x_bar = np.mean(algo.x, axis=0)

    for _ in range(max_iter):
        algo.step()
        x_bar = np.mean(algo.x, axis=0)
        grad = problem.global_gradient(x_bar)
        g2 = float(np.dot(grad, grad))
        mapping = sr.l1_prox_gradient_mapping(x_bar, grad, lambda_reg=algo.lambda_reg, eta=algo.stepsize)
        composite_g2 = float(np.dot(mapping, mapping))
        if (
            not np.isfinite(g2)
            or not np.isfinite(composite_g2)
            or not np.isfinite(np.linalg.norm(x_bar))
            or g2 > sr.DIVERGENCE_THRESHOLD
            or composite_g2 > sr.DIVERGENCE_THRESHOLD
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
            "final_g2": float(sr.DIVERGENCE_THRESHOLD),
            "composite_g2": float(sr.DIVERGENCE_THRESHOLD),
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
    f1, precision, recall = problem.support_f1(x_bar, tol=sr.SUPPORT_TOL)
    final_obj = problem.global_objective(x_bar)
    final_l1 = float(np.linalg.norm(x_bar, 1))
    final_composite_obj = float(final_obj + algo.lambda_reg * final_l1)
    support_size = int(np.count_nonzero(np.abs(x_bar) >= sr.SUPPORT_TOL))
    hit_t = sr._first_sustained_below(np.asarray(g2_curve, dtype=float), sr.GRAD_HIT_EPS, sr.GRAD_HIT_HOLD)
    composite_hit_t = sr._first_sustained_below(
        np.asarray(composite_g2_curve, dtype=float),
        sr.GRAD_HIT_EPS,
        sr.GRAD_HIT_HOLD,
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

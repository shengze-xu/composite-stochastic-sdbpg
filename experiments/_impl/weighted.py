"""Weighted numerical routines."""

from __future__ import annotations

from dataclasses import dataclass


from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple


import networkx as nx


import numpy as np


N_AGENTS = 6


DIM = 12


MAX_ITER = 700


SEEDS = list(range(10))


CONDITION_NUMBERS = (10, 30, 100, 300)


THRESHOLD = 1e-4


DIVERGENCE_CAP = 1e12


ALPHA = 0.02


RHO = 1.0


BETA = 0.8


C_PARAM = 1.0


@dataclass(frozen=True)
class KernelConfig:
    name: str
    label: str
    lambda_param: float
    weighted: bool


CONFIGS = (
    KernelConfig(
        name="isotropic_conservative",
        label=r"Isotropic $\lambda=0.006$",
        lambda_param=0.006,
        weighted=False,
    ),
    KernelConfig(
        name="isotropic_same_step",
        label=r"Isotropic $\lambda=0.025$",
        lambda_param=0.025,
        weighted=False,
    ),
    KernelConfig(
        name="weighted_same_step",
        label=r"Weighted diagonal $\lambda=0.025$",
        lambda_param=0.025,
        weighted=True,
    ),
)


class DiagonalQuadraticConsensusProblem:
    def __init__(self, seed: int, condition_number: float):
        rng = np.random.default_rng(seed)
        self.n_agents = N_AGENTS
        self.dim = DIM
        self.condition_number = float(condition_number)
        self.base_curvature = np.logspace(0.0, np.log10(self.condition_number), DIM)
        self.weight_diag = self.base_curvature / float(np.mean(self.base_curvature))
        x_star = rng.normal(size=DIM)
        self.x_star = x_star / np.linalg.norm(x_star)
        self.local_curvatures = []
        self.local_centers = []
        for _ in range(N_AGENTS):
            scale = rng.uniform(0.85, 1.15, size=DIM)
            self.local_curvatures.append(self.base_curvature * scale)
            center_noise = 0.03 * rng.normal(size=DIM) / np.sqrt(self.base_curvature)
            self.local_centers.append(self.x_star + center_noise)

    def local_gradient(self, agent_id: int, x: np.ndarray) -> np.ndarray:
        return self.local_curvatures[agent_id] * (x - self.local_centers[agent_id])

    def global_gradient(self, x: np.ndarray) -> np.ndarray:
        grads = [self.local_gradient(i, x) for i in range(self.n_agents)]
        return np.mean(grads, axis=0)


def _make_graph(seed: int) -> nx.Graph:
    graph = nx.cycle_graph(N_AGENTS)
    rng = np.random.default_rng(1000 + seed)
    for i in range(N_AGENTS):
        for j in range(i + 2, N_AGENTS):
            if (i == 0 and j == N_AGENTS - 1):
                continue
            if rng.random() < 0.25:
                graph.add_edge(i, j)
    return graph


def _initial_states(problem: DiagonalQuadraticConsensusProblem, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(2000 + seed)
    return [problem.x_star + 0.25 * rng.normal(size=problem.dim) for _ in range(problem.n_agents)]


def _consensus_error(xs: Sequence[np.ndarray]) -> float:
    x_bar = np.mean(xs, axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.mean([np.dot(x - x_bar, x - x_bar) for x in xs]))
    if not np.isfinite(value):
        return 1e12
    return min(value, 1e12)


def _run_seed(config: KernelConfig, seed: int, condition_number: float, max_iter: int = MAX_ITER) -> Dict[str, object]:
    problem = DiagonalQuadraticConsensusProblem(seed, condition_number=condition_number)
    graph = _make_graph(seed)
    x_list = [x.copy() for x in _initial_states(problem, seed)]
    z_list = [x.copy() for x in x_list]
    y = {(i, j): np.zeros(DIM) for i in range(N_AGENTS) for j in graph.neighbors(i)}
    weight = problem.weight_diag if config.weighted else np.ones(DIM)
    inv_weight = 1.0 / weight

    grad_norm_sq = []
    consensus = []
    best_grad = DIVERGENCE_CAP
    hit_iteration = max_iter + 1
    diverged = False

    for iteration in range(1, max_iter + 1):
        x_old = [x.copy() for x in x_list]
        z_old = [z.copy() for z in z_list]

        for i, j in list(y.keys()):
            y[(i, j)] = y[(i, j)] + ALPHA * (x_old[i] - x_old[j])

        for i in range(N_AGENTS):
            with np.errstate(over="ignore", invalid="ignore"):
                grad = problem.local_gradient(i, x_old[i])
                grad = grad + C_PARAM * weight * (x_old[i] - z_old[i])
                for j in graph.neighbors(i):
                    grad = grad + RHO * weight * (x_old[i] - z_old[j])
                    grad = grad + y[(i, j)]
                x_list[i] = x_old[i] - config.lambda_param * inv_weight * grad

        for i in range(N_AGENTS):
            with np.errstate(over="ignore", invalid="ignore"):
                z_list[i] = (x_list[i] + BETA * weight * z_old[i]) / (1.0 + BETA * weight)

        x_bar = np.mean(x_list, axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            g = problem.global_gradient(x_bar)
            g2 = float(np.dot(g, g))
        if not np.isfinite(g2) or g2 > DIVERGENCE_CAP:
            diverged = True
            grad_norm_sq.extend([DIVERGENCE_CAP] * (max_iter - len(grad_norm_sq)))
            consensus.extend([DIVERGENCE_CAP] * (max_iter - len(consensus)))
            break

        best_grad = min(best_grad, g2)
        if hit_iteration == max_iter + 1 and best_grad <= THRESHOLD:
            hit_iteration = iteration
        grad_norm_sq.append(g2)
        consensus.append(min(_consensus_error(x_list), DIVERGENCE_CAP))

    return {
        "running_grad_norm_sq": np.minimum.accumulate(np.asarray(grad_norm_sq, dtype=float)),
        "consensus": np.asarray(consensus, dtype=float),
        "final_running_grad_norm_sq": float(np.minimum.accumulate(np.asarray(grad_norm_sq, dtype=float))[-1]),
        "final_consensus": float(np.asarray(consensus, dtype=float)[-1]),
        "hit_iteration": int(hit_iteration),
        "diverged": bool(diverged),
    }

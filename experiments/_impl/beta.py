"""Beta numerical routines."""

from __future__ import annotations

from typing import Dict
from typing import List


import numpy as np


from sdbpg.core import QuadraticRegressionProblem
from sdbpg.core import exact_quadratic_primal_step
from sdbpg.core import exact_quadratic_smoothing_step
from sdbpg.core import generate_network_topology


from sdbpg.algorithm import SDBPG


BETA_CONFIGS = [
    (0.10, "β=0.10"),
    (0.30, "β=0.30"),
    (0.50, "β=0.50"),
    (0.80, "β=0.80"),
    (1.20, "β=1.20"),
    (2.00, "β=2.00"),
]


N_AGENTS   = 6


DIM        = 10


SAMPLES    = 20


MAX_ITER   = 2000


SEEDS      = list(range(5))


LAMBDA_REG = 0.0


BASE_PARAMS = dict(rho=2.0, alpha=0.02, lambda_param=0.025, C=1.5)


DIVERGENCE_THRESHOLD = 1e6   # treat as diverged if ‖∇f‖² exceeds this


class ExactSDBPG(SDBPG):
    def __init__(self, *args, lambda_reg: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = float(lambda_reg)

    def step(self):
        states = {i: ag.get_state() for i, ag in self.agents.items()}
        for i, ag in self.agents.items():
            ag.receive_from_neighbors({j: states[j] for j in ag.neighbors if j in states})
        for ag in self.agents.values():
            ag.dual_update()
        for ag in self.agents.values():
            grad_K = ag.compute_local_surrogate_gradient() + self.lambda_reg * ag.x
            ag.x = exact_quadratic_primal_step(ag.x, grad_K, ag.lambda_param, mu=1.0)
        for ag in self.agents.values():
            ag.z = exact_quadratic_smoothing_step(ag.x, ag.z, ag.beta, mu=1.0)


def _grad_norm_sq(xs: List[np.ndarray], problem) -> float:
    x_bar = np.mean(xs, axis=0)
    grad  = np.mean([problem.local_gradient(i, x_bar) for i in range(len(xs))], axis=0)
    return float(np.dot(grad, grad))


def _run_one(beta: float, seed: int, max_iter: int = MAX_ITER) -> np.ndarray:
    prob  = QuadraticRegressionProblem(N_AGENTS, DIM, SAMPLES, seed=seed)
    graph = generate_network_topology(N_AGENTS, "random", connectivity=0.5, seed=seed)
    algo  = ExactSDBPG(problem=prob, graph=graph, lambda_reg=LAMBDA_REG,
                       beta=beta, **BASE_PARAMS)

    grad_sq = []
    for _ in range(max_iter):
        algo.step()
        xs = [ag.x for ag in algo.agents.values()]
        g  = _grad_norm_sq(xs, prob)
        if np.isnan(g) or g > DIVERGENCE_THRESHOLD:
            # Fill remaining with large sentinel value
            grad_sq += [DIVERGENCE_THRESHOLD] * (max_iter - len(grad_sq))
            break
        grad_sq.append(g)

    return np.minimum.accumulate(np.array(grad_sq[:max_iter]))

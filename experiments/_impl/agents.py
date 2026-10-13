"""Agents numerical routines."""

from __future__ import annotations

from typing import Dict
from typing import List


import numpy as np


from sdbpg.core import QuadraticRegressionProblem
from sdbpg.core import exact_quadratic_primal_step
from sdbpg.core import exact_quadratic_smoothing_step
from sdbpg.core import generate_network_topology


from sdbpg.algorithm import SDBPG


N_LIST   = [4, 8, 12, 20]       # agent counts to sweep


DIM      = 10


SAMPLES  = 20


MAX_ITER = 2000


SEEDS    = list(range(5))


BASE_PARAMS  = dict(rho=2.0, alpha=0.02, lambda_param=0.025, beta=0.80, C=1.5)


class ExactSDBPG(SDBPG):
    def __init__(self, *args, lambda_reg: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = float(lambda_reg)

    def step(self):
        states = {i: agent.get_state() for i, agent in self.agents.items()}
        for i, agent in self.agents.items():
            nd = {j: states[j] for j in agent.neighbors if j in states}
            agent.receive_from_neighbors(nd)
        for agent in self.agents.values():
            agent.dual_update()
        for agent in self.agents.values():
            grad_K = (agent.compute_local_surrogate_gradient()
                      + self.lambda_reg * agent.x)
            agent.x = exact_quadratic_primal_step(agent.x, grad_K, agent.lambda_param, mu=1.0)
        for agent in self.agents.values():
            agent.z = exact_quadratic_smoothing_step(agent.x, agent.z, agent.beta, mu=1.0)


def _grad_norm_sq(xs: List[np.ndarray], problem: QuadraticRegressionProblem) -> float:
    x_bar = np.mean(xs, axis=0)
    grad  = np.mean([problem.local_gradient(i, x_bar) for i in range(len(xs))], axis=0)
    return float(np.dot(grad, grad))


def _run_one(n: int, lambda_reg: float, seed: int, max_iter: int = MAX_ITER) -> np.ndarray:
    prob  = QuadraticRegressionProblem(n, DIM, SAMPLES, seed=seed)
    graph = generate_network_topology(n, "random", connectivity=0.5, seed=seed)

    algo = ExactSDBPG(problem=prob, graph=graph,
                      lambda_reg=lambda_reg, **BASE_PARAMS)

    grad_sq = []
    for _ in range(max_iter):
        algo.step()
        xs = [ag.x for ag in algo.agents.values()]
        grad_sq.append(_grad_norm_sq(xs, prob))

    return np.minimum.accumulate(np.array(grad_sq))

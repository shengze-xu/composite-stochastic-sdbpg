"""Exact quadratic updates and smooth stationarity diagnostics."""

from typing import Dict, List
import numpy as np
from sdbpg.algorithm import SDBPG
from sdbpg.core import (QuadraticRegressionProblem, exact_quadratic_primal_step,
                        exact_quadratic_smoothing_step)

class L2SDBPG(SDBPG):
    """
    SDBPG with L2 regularisation:   min (1/N) Σ f_i(x) + (λ/2)‖x‖²

    The L2 gradient  λ·x  is folded into the surrogate gradient of the
    primal step without altering the Bregman proximal structure.
    """

    def __init__(self, *args, lambda_reg: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = float(lambda_reg)

    def step(self):
        # Broadcast current states
        states = {i: agent.get_state() for i, agent in self.agents.items()}
        for i, agent in self.agents.items():
            nd = {j: states[j] for j in agent.neighbors if j in states}
            agent.receive_from_neighbors(nd)
        # Dual update
        for agent in self.agents.values():
            agent.dual_update()
        # Primal update with regularised gradient
        for agent in self.agents.values():
            grad_K = (agent.compute_local_surrogate_gradient()
                      + self.lambda_reg * agent.x)
            agent.x = exact_quadratic_primal_step(agent.x, grad_K, agent.lambda_param, mu=1.0)
        # Smoothing update
        for agent in self.agents.values():
            agent.z = exact_quadratic_smoothing_step(agent.x, agent.z, agent.beta, mu=1.0)


def _global_grad_norm_sq(agents_x: List[np.ndarray],
                         problem: QuadraticRegressionProblem,
                         lambda_reg: float = 0.0) -> float:
    """
    ‖∇F(x̄)‖²  where  ∇F = (1/N)Σ∇f_i + λ·x̄
    """
    x_bar = np.mean(agents_x, axis=0)
    grad_f = np.mean(
        [problem.local_gradient(i, x_bar) for i in range(len(agents_x))], axis=0
    )
    grad_F = grad_f + lambda_reg * x_bar
    return float(np.dot(grad_F, grad_F))


def _consensus_error(agents_x: List[np.ndarray]) -> float:
    """(1/N) Σ_i ‖x_i - x̄‖²"""
    x_bar = np.mean(agents_x, axis=0)
    return float(np.mean([np.dot(x - x_bar, x - x_bar) for x in agents_x]))


def _run_and_record(algo,
                    problem: QuadraticRegressionProblem,
                    lambda_reg: float,
                    max_iter: int) -> Dict:
    """
    Run algorithm for max_iter steps, recording per-iteration metrics.
    Works for any algo that exposes .step() and .agents / .x (DGD).
    Returns dict with keys: grad_norm_sq, running_min, consensus_err, x_bar_final.
    """
    grad_norm_sq   = []
    consensus_err  = []

    is_sdbpg = hasattr(algo, "agents")

    for t in range(max_iter):
        algo.step()

        if is_sdbpg:
            xs = [ag.x for ag in algo.agents.values()]
        else:
            xs = algo.x

        g2  = _global_grad_norm_sq(xs, problem, lambda_reg)
        ce  = _consensus_error(xs)
        grad_norm_sq.append(g2)
        consensus_err.append(ce)

    grad_norm_sq  = np.array(grad_norm_sq)
    running_min   = np.minimum.accumulate(grad_norm_sq)

    if is_sdbpg:
        xs_final = [ag.x for ag in algo.agents.values()]
    else:
        xs_final = algo.x
    x_bar_final = np.mean(xs_final, axis=0)

    return {
        "grad_norm_sq":  grad_norm_sq,
        "running_min":   running_min,
        "consensus_err": np.array(consensus_err),
        "x_bar_final":   x_bar_final,
    }

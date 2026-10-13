"""Dimension numerical routines."""

from __future__ import annotations

from typing import List
from typing import Dict


import networkx as nx


import numpy as np


from sdbpg.core import QuadraticRegressionProblem
from sdbpg.core import generate_network_topology


N_AGENTS = 6


SAMPLES = 20


SEEDS = list(range(5))


DIM_LIST  = [5, 10, 20, 50]


MAX_ITER  = 2000


RHO      = 2.0


ALPHA    = 0.02


LAM_STEP = 0.025


BETA     = 0.80


C_PARAM  = 1.5


def _global_grad_norm_sq(
    agents_x: List[np.ndarray],
    problem: QuadraticRegressionProblem,
    lambda_reg: float = 0.0,
) -> float:
    x_bar = np.mean(agents_x, axis=0)
    gradient = np.mean(
        [problem.local_gradient(i, x_bar) for i in range(len(agents_x))],
        axis=0,
    )
    gradient = gradient + lambda_reg * x_bar
    return float(np.dot(gradient, gradient))


def _fast_sdbpg_run(problem: QuadraticRegressionProblem,
                    graph: nx.Graph,
                    lambda_reg: float,
                    max_iter: int,
                    seed: int) -> np.ndarray:
    """
    Run SDBPG with *closed-form* updates for quadratic Bregman.
    A matrices in QuadraticRegressionProblem are now normalised by dim in
    core.py, so gradient magnitude is O(1) regardless of d.  No per-dimension
    step-size scaling is needed here.

    Returns running_min array of ‖∇f(x̄_t)‖² (shape: max_iter).
    """
    np.random.seed(seed)
    N   = problem.n_agents
    d   = problem.dim
    nbrs = {i: list(graph.neighbors(i)) for i in range(N)}

    # Fixed base step sizes (same for all dimensions)
    lam_step = LAM_STEP
    alpha    = ALPHA
    rho      = RHO

    # State
    x = [np.random.randn(d) * 0.1 for _ in range(N)]
    z = [xi.copy() for xi in x]
    y = {i: {j: np.zeros(d) for j in nbrs[i]} for i in range(N)}

    grad_sq = []

    for _ in range(max_iter):
        # ── Step 1: dual update ──────────────────────────────────
        for i in range(N):
            for j in nbrs[i]:
                y[i][j] = y[i][j] + alpha * (x[i] - x[j])

        # ── Step 2: primal update (closed form) ──────────────────
        x_new = []
        for i in range(N):
            # surrogate gradient ∇K_i  (quadratic Bregman: ∇φ(x)=x)
            grad_f  = problem.local_gradient(i, x[i])
            g_breg  = C_PARAM * (x[i] - z[i])
            for j in nbrs[i]:
                g_breg += rho * (x[i] - z[j])
            g_dual  = sum(y[i][j] for j in nbrs[i]) if nbrs[i] else np.zeros(d)
            grad_K  = grad_f + g_breg + g_dual + lambda_reg * x[i]
            # closed-form prox: x ← x - lam_step·∇K
            x_new.append(x[i] - lam_step * grad_K)
        x = x_new

        # ── Step 3: smoothing update (closed form) ───────────────
        z = [(xi + BETA * zi) / (1.0 + BETA) for xi, zi in zip(x, z)]

        # ── Record ‖∇f(x̄)‖² (not ‖∇F_λ‖²) ─────────────────────
        g2 = _global_grad_norm_sq(x, problem, lambda_reg=0.0)
        grad_sq.append(g2)

    return np.minimum.accumulate(np.array(grad_sq))

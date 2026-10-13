"""Geometry numerical routines."""

from __future__ import annotations

import time


from typing import Dict
from typing import Literal


import networkx as nx


import numpy as np


from sdbpg.baselines import CostLedger, soft_threshold


Array = np.ndarray


Geometry = Literal["quadratic", "quartic"]


def quartic_grad_phi(x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    return (1.0 + float(np.dot(arr, arr))) * arr


def _positive_root_cubic(a: float, b: float, target: float) -> float:
    """Solve ``a*r^3 + b*r = target`` for its nonnegative root."""
    aa = float(a)
    bb = float(b)
    tt = max(float(target), 0.0)
    if not np.isfinite(tt):
        return float("nan")
    if tt == 0.0:
        return 0.0
    if aa == 0.0:
        return tt / bb
    # Monotone safeguarded Newton iteration.  The bracket makes this reliable
    # for both tiny and very large norms.
    lower = 0.0
    upper = max(tt / bb, (tt / aa) ** (1.0 / 3.0))
    while aa * upper**3 + bb * upper < tt:
        upper *= 2.0
    value = min(tt / bb, upper)
    for _ in range(60):
        residual = aa * value**3 + bb * value - tt
        if abs(residual) <= 1e-14 * max(1.0, tt):
            break
        if residual > 0.0:
            upper = value
        else:
            lower = value
        derivative = 3.0 * aa * value**2 + bb
        proposal = value - residual / derivative
        if not (lower < proposal < upper):
            proposal = 0.5 * (lower + upper)
        value = proposal
    return float(value)


def inverse_quartic_gradient(dual: Array) -> Array:
    arr = np.asarray(dual, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm):
        return np.full_like(arr, np.nan)
    if norm == 0.0:
        return np.zeros_like(arr)
    radius = _positive_root_cubic(1.0, 1.0, norm)
    return (radius / norm) * arr


def quartic_l1_prox_step(
    x: Array,
    gradient: Array,
    stepsize: float,
    lambda_reg: float = 0.0,
) -> Array:
    """Exact Bregman proximal-gradient step for the quartic kernel and L1."""
    step = float(stepsize)
    if step <= 0.0:
        raise ValueError("stepsize must be positive")
    dual = quartic_grad_phi(x) - step * np.asarray(gradient, dtype=float)
    thresholded = soft_threshold(dual, step * max(float(lambda_reg), 0.0))
    return inverse_quartic_gradient(thresholded)


def quartic_smoothing_step(x_new: Array, z_old: Array, beta: float) -> Array:
    """Exact minimizer of ``.5||z-x||^2 + beta D_phi(z,z_old)``."""
    smoothing = float(beta)
    if smoothing <= 0.0:
        raise ValueError("beta must be positive")
    target = np.asarray(x_new, dtype=float) + smoothing * quartic_grad_phi(z_old)
    norm = float(np.linalg.norm(target))
    if not np.isfinite(norm):
        return np.full_like(target, np.nan)
    if norm == 0.0:
        return np.zeros_like(target)
    radius = _positive_root_cubic(smoothing, 1.0 + smoothing, norm)
    return (radius / norm) * target


class GeometrySDBPG:
    """Standalone exact SDBPG for quadratic-versus-quartic geometry tests."""

    def __init__(
        self,
        problem: object,
        graph: nx.Graph,
        x0: Array,
        geometry: Geometry,
        rho: float,
        dual_stepsize: float,
        primal_stepsize: float,
        beta: float,
        anchor_weight: float,
        lambda_reg: float = 0.0,
        samples_per_call: int = 1,
    ):
        initial = np.asarray(x0, dtype=float)
        if initial.ndim != 2:
            raise ValueError("x0 must have shape (n_agents, dim)")
        self.problem = problem
        self.graph = graph
        self.x = initial.copy()
        self.z = initial.copy()
        self.n_agents, self.dim = self.x.shape
        self.geometry = str(geometry)
        if self.geometry not in {"quadratic", "quartic"}:
            raise ValueError(f"unknown geometry: {geometry}")
        self.rho = float(rho)
        self.dual_stepsize = float(dual_stepsize)
        self.primal_stepsize = float(primal_stepsize)
        self.beta = float(beta)
        self.anchor_weight = float(anchor_weight)
        self.lambda_reg = max(float(lambda_reg), 0.0)
        self.samples_per_call = int(samples_per_call)
        self.dual = np.zeros((self.n_agents, self.n_agents, self.dim), dtype=float)
        self.cost = CostLedger()

    def _grad_phi(self, x: Array) -> Array:
        if self.geometry == "quadratic":
            return np.asarray(x, dtype=float)
        return quartic_grad_phi(x)

    def step(self) -> None:
        started = time.perf_counter()
        old_x = self.x.copy()
        old_z = self.z.copy()
        for i, j in self.graph.edges():
            difference = old_x[int(i)] - old_x[int(j)]
            self.dual[int(i), int(j)] += self.dual_stepsize * difference
            self.dual[int(j), int(i)] -= self.dual_stepsize * difference

        new_x = np.zeros_like(old_x)
        for i in range(self.n_agents):
            gradient = self.problem.local_gradient(i, old_x[i])
            model_gradient = gradient + self.anchor_weight * (
                self._grad_phi(old_x[i]) - self._grad_phi(old_z[i])
            )
            for j in self.graph.neighbors(i):
                model_gradient += self.rho * (
                    self._grad_phi(old_x[i]) - self._grad_phi(old_z[int(j)])
                )
                model_gradient += self.dual[i, int(j)]
            if self.geometry == "quadratic":
                candidate = old_x[i] - self.primal_stepsize * model_gradient
                new_x[i] = soft_threshold(
                    candidate,
                    self.primal_stepsize * self.lambda_reg,
                )
            else:
                new_x[i] = quartic_l1_prox_step(
                    old_x[i],
                    model_gradient,
                    self.primal_stepsize,
                    self.lambda_reg,
                )

        new_z = np.zeros_like(old_z)
        for i in range(self.n_agents):
            if self.geometry == "quadratic":
                new_z[i] = (new_x[i] + self.beta * old_z[i]) / (1.0 + self.beta)
            else:
                new_z[i] = quartic_smoothing_step(new_x[i], old_z[i], self.beta)
        self.x = new_x
        self.z = new_z
        self.cost.charge_oracle(self.n_agents, self.samples_per_call)
        self.cost.charge_communication(
            self.graph,
            self.dim,
            sequential_rounds=1,
            vectors_per_round=2,
        )
        self.cost.iterations += 1
        self.cost.wall_time_sec += float(time.perf_counter() - started)

    @property
    def states(self) -> Array:
        return self.x.copy()

    def geometry_diagnostics(self) -> Dict[str, float]:
        gradients = np.stack([self._grad_phi(row) for row in self.x])
        return {
            "max_state_norm": float(np.max(np.linalg.norm(self.x, axis=1))),
            "max_mirror_norm": float(np.max(np.linalg.norm(gradients, axis=1))),
        }

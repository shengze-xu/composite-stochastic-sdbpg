"""Defects numerical routines."""

from __future__ import annotations

from dataclasses import dataclass


import numpy as np


SEEDS = tuple(range(10))


T = 3000


@dataclass
class RunResult:
    penalty: str
    seed: int
    eta: np.ndarray
    zeta: np.ndarray
    cumulative: np.ndarray
    support_size: np.ndarray
    identification_time: int
    identified_true_support: bool
    strict_slack: float
    min_active_magnitude: float
    active_kkt_residual: float
    recovery_error: float
    consensus_error: float
    total_defect: float
    post_identification_tail: float


def _incidence_operators(num_agents: int):
    edges = [(i, (i + 1) % num_agents) for i in range(num_agents)] + [(0, num_agents // 2)]
    heads = np.asarray([edge[0] for edge in edges], dtype=int)
    tails = np.asarray([edge[1] for edge in edges], dtype=int)
    degree = np.zeros(num_agents)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    def bt(values: np.ndarray) -> np.ndarray:
        return values[heads] - values[tails]

    def b(edge_values: np.ndarray) -> np.ndarray:
        out = np.zeros((num_agents,) + edge_values.shape[1:])
        np.add.at(out, heads, edge_values)
        np.add.at(out, tails, -edge_values)
        return out

    return heads, tails, degree, bt, b


def _prox(values: np.ndarray, threshold: float, penalty: str) -> np.ndarray:
    if penalty == "l1":
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)
    row_norms = np.linalg.norm(values, axis=-1, keepdims=True)
    factors = np.maximum(1.0 - threshold / np.maximum(row_norms, 1e-30), 0.0)
    return factors * values


def _aggregation_distance_sq(selected_sum: np.ndarray, xbar: np.ndarray, radius: float, penalty: str) -> float:
    if penalty == "l1":
        active = np.abs(xbar[..., 0]) > 1e-7
        selected = selected_sum[..., 0]
        target = radius * np.sign(xbar[..., 0])
        active_distance = np.where(active, selected - target, 0.0)
        inactive_distance = np.where(active, 0.0, selected - np.clip(selected, -radius, radius))
        return float(np.sum(active_distance**2 + inactive_distance**2))

    row_norms = np.linalg.norm(xbar, axis=1)
    active = row_norms > 1e-7
    distance_sq = 0.0
    if np.any(active):
        targets = radius * xbar[active] / row_norms[active, None]
        distance_sq += float(np.sum((selected_sum[active] - targets) ** 2))
    if np.any(~active):
        inactive_values = selected_sum[~active]
        inactive_norms = np.linalg.norm(inactive_values, axis=1)
        excess = np.maximum(inactive_norms - radius, 0.0)
        distance_sq += float(np.sum(excess**2))
    return distance_sq


def _support_pattern(xbar: np.ndarray, penalty: str) -> np.ndarray:
    if penalty == "l1":
        values = xbar[..., 0]
        return np.sign(np.where(np.abs(values) > 1e-7, values, 0.0))
    return (np.linalg.norm(xbar, axis=1) > 1e-7).astype(int)


def _run(seed: int, penalty: str, iterations: int = T) -> RunResult:
    T = iterations
    rng = np.random.default_rng(seed)
    num_agents, dimension, sparsity = 6, 40, 5
    num_tasks = 1 if penalty == "l1" else 4
    samples_per_agent = 25
    regularization = 0.05
    noise = 1e-3

    true_support = rng.choice(dimension, sparsity, replace=False)
    xstar = np.zeros((dimension, num_tasks))
    if penalty == "l1":
        active_values = rng.standard_normal((sparsity, 1))
        active_values += np.sign(active_values)
    else:
        directions = rng.standard_normal((sparsity, num_tasks))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        active_values = directions * (1.4 + rng.random((sparsity, 1)))
    xstar[true_support] = active_values

    matrices = [
        rng.standard_normal((samples_per_agent, dimension)) / np.sqrt(samples_per_agent)
        for _ in range(num_agents)
    ]
    observations = [
        matrix @ xstar + noise * rng.standard_normal((samples_per_agent, num_tasks))
        for matrix in matrices
    ]
    gram = [matrix.T @ matrix for matrix in matrices]
    cross = [matrix.T @ observation for matrix, observation in zip(matrices, observations)]

    def gradf(agent: int, value: np.ndarray) -> np.ndarray:
        return gram[agent] @ value - cross[agent]

    heads, tails, degree, bt, b = _incidence_operators(num_agents)

    primal_step = 0.04
    dual_step = 0.02
    smoothing_weight = 0.5
    anchor_weight = 0.5
    neighbor_weight = 0.1
    motion_absorption = 3.0
    consensus_absorption = 3.0

    x = np.zeros((num_agents, dimension, num_tasks))
    z = x.copy()
    y = np.zeros((len(heads), dimension, num_tasks))
    selected_previous = None
    dx_previous = 0.0
    dz_previous = 0.0

    raw_motion = np.full(T, np.nan)
    eta = np.zeros(T)
    zeta = np.zeros(T)
    support_size = np.zeros(T)
    patterns = []

    for iteration in range(T):
        y_next = y + dual_step * bt(x)
        node_force = b(y_next)
        neighbor_reference_sum = np.zeros_like(x)
        np.add.at(neighbor_reference_sum, heads, z[tails])
        np.add.at(neighbor_reference_sum, tails, z[heads])
        local_gradient = np.stack([gradf(agent, x[agent]) for agent in range(num_agents)])
        model_gradient = (
            local_gradient
            + anchor_weight * (x - z)
            + neighbor_weight * (degree[:, None, None] * x - neighbor_reference_sum)
            + node_force
        )
        x_next = _prox(x - primal_step * model_gradient, primal_step * regularization, penalty)
        z_next = (x_next + smoothing_weight * z) / (1.0 + smoothing_weight)
        selected = -model_gradient - (x_next - x) / primal_step

        dx = 0.5 * float(np.sum((x_next - x) ** 2))
        dz = 0.5 * float(np.sum((z_next - z) ** 2))
        if selected_previous is not None:
            raw_motion[iteration] = float(np.sum((selected - selected_previous) ** 2))
            budget = motion_absorption * (dx + dx_previous + dz_previous)
            eta[iteration] = max(raw_motion[iteration] - budget, 0.0)

        xbar_next = x_next.mean(axis=0)
        consensus_next = float(np.sum(bt(x_next) ** 2))
        aggregation_raw = _aggregation_distance_sq(
            selected.sum(axis=0), xbar_next, num_agents * regularization, penalty
        )
        zeta[iteration] = max(aggregation_raw - consensus_absorption * consensus_next, 0.0)

        pattern = _support_pattern(xbar_next, penalty)
        patterns.append(pattern)
        support_size[iteration] = np.count_nonzero(pattern)

        selected_previous = selected.copy()
        dx_previous, dz_previous = dx, dz
        x, z, y = x_next, z_next, y_next

    patterns_array = np.asarray(patterns)
    final_pattern = patterns_array[-1]
    identification_time = 0
    for iteration, pattern in enumerate(patterns_array):
        if not np.array_equal(pattern, final_pattern):
            identification_time = iteration + 1

    expected_pattern = np.zeros(dimension, dtype=int)
    if penalty == "l1":
        expected_pattern[true_support] = np.sign(xstar[true_support, 0]).astype(int)
    else:
        expected_pattern[true_support] = 1
    identified_true_support = bool(np.array_equal(final_pattern, expected_pattern))

    xbar = x.mean(axis=0)
    global_gradient = sum(gradf(agent, xbar) for agent in range(num_agents))
    radius = num_agents * regularization
    if penalty == "l1":
        active = np.abs(xbar[:, 0]) > 1e-7
        inactive_norm = np.abs(global_gradient[~active, 0])
        strict_slack = radius - float(np.max(inactive_norm)) if inactive_norm.size else float("nan")
        min_active_magnitude = float(np.min(np.abs(xbar[active, 0]))) if np.any(active) else float("nan")
        active_kkt_residual = float(
            np.max(np.abs(global_gradient[active, 0] + radius * np.sign(xbar[active, 0])), initial=0.0)
        )
    else:
        row_norms = np.linalg.norm(xbar, axis=1)
        active = row_norms > 1e-7
        inactive_norm = np.linalg.norm(global_gradient[~active], axis=1)
        strict_slack = radius - float(np.max(inactive_norm)) if inactive_norm.size else float("nan")
        min_active_magnitude = float(np.min(row_norms[active])) if np.any(active) else float("nan")
        active_subgradient = radius * xbar[active] / row_norms[active, None]
        active_kkt_residual = float(
            np.max(np.linalg.norm(global_gradient[active] + active_subgradient, axis=1), initial=0.0)
        )

    cumulative = np.cumsum(eta + zeta)
    tail_start = min(identification_time + 50, T - 1)
    post_identification_tail = float(np.max(eta[tail_start:] + zeta[tail_start:]))
    recovery_error = float(np.linalg.norm(xbar - xstar) / np.linalg.norm(xstar))
    consensus_error = float(np.sum(bt(x) ** 2))

    return RunResult(
        penalty=penalty,
        seed=seed,
        eta=eta,
        zeta=zeta,
        cumulative=cumulative,
        support_size=support_size,
        identification_time=identification_time,
        identified_true_support=identified_true_support,
        strict_slack=strict_slack,
        min_active_magnitude=min_active_magnitude,
        active_kkt_residual=active_kkt_residual,
        recovery_error=recovery_error,
        consensus_error=consensus_error,
        total_defect=float(cumulative[-1]),
        post_identification_tail=post_identification_tail,
    )

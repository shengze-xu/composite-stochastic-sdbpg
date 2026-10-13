"""Modern baselines numerical routines."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass


import time


from typing import Callable
from typing import Dict
from typing import Optional


import networkx as nx


import numpy as np


Array = np.ndarray


GradientOracle = Callable[[int, Array], Array]


ProxOperator = Callable[[Array, float], Array]


def metropolis_matrix(graph: nx.Graph, n_agents: int) -> Array:
    """Return a symmetric doubly-stochastic Metropolis mixing matrix."""
    matrix = np.zeros((int(n_agents), int(n_agents)), dtype=float)
    degrees = dict(graph.degree())
    for i, j in graph.edges():
        weight = 1.0 / (1.0 + max(float(degrees[i]), float(degrees[j])))
        matrix[int(i), int(j)] = weight
        matrix[int(j), int(i)] = weight
    for i in range(int(n_agents)):
        matrix[i, i] = 1.0 - float(np.sum(matrix[i]))
    return matrix


def soft_threshold(values: Array, threshold: float) -> Array:
    arr = np.asarray(values, dtype=float)
    tau = max(float(threshold), 0.0)
    return np.sign(arr) * np.maximum(np.abs(arr) - tau, 0.0)


def l1_prox(lambda_reg: float) -> ProxOperator:
    weight = max(float(lambda_reg), 0.0)
    return lambda values, step: soft_threshold(values, float(step) * weight)


def identity_prox(values: Array, step: float) -> Array:
    del step
    return np.asarray(values, dtype=float).copy()


@dataclass
class CostLedger:
    """Cumulative algorithmic cost, excluding diagnostic evaluations."""

    iterations: int = 0
    communication_rounds: int = 0
    transmitted_scalars: int = 0
    oracle_calls: int = 0
    oracle_samples: int = 0
    wall_time_sec: float = 0.0

    def charge_oracle(self, n_agents: int, samples_per_call: int) -> None:
        self.oracle_calls += int(n_agents)
        self.oracle_samples += int(n_agents) * int(samples_per_call)

    def charge_communication(
        self,
        graph: nx.Graph,
        dim: int,
        sequential_rounds: int,
        vectors_per_round: int,
    ) -> None:
        rounds = int(sequential_rounds)
        vectors = int(vectors_per_round)
        self.communication_rounds += rounds
        self.transmitted_scalars += int(
            2 * graph.number_of_edges() * int(dim) * rounds * vectors
        )

    def snapshot(self) -> Dict[str, float | int]:
        result: Dict[str, float | int] = asdict(self)
        result["communication_mib"] = float(self.transmitted_scalars) * 8.0 / (1024.0**2)
        return result


class _ModernBaselineBase:
    def __init__(
        self,
        graph: nx.Graph,
        x0: Array,
        gradient_oracle: GradientOracle,
        prox_operator: ProxOperator,
        samples_per_call: int,
    ):
        initial = np.asarray(x0, dtype=float)
        if initial.ndim != 2:
            raise ValueError("x0 must have shape (n_agents, dim)")
        self.x = initial.copy()
        self.n_agents, self.dim = self.x.shape
        if self.n_agents != graph.number_of_nodes():
            raise ValueError("x0 agent count does not match graph")
        self.graph = graph
        self.W = metropolis_matrix(graph, self.n_agents)
        self.gradient_oracle = gradient_oracle
        self.prox_operator = prox_operator
        self.samples_per_call = int(samples_per_call)
        if self.samples_per_call <= 0:
            raise ValueError("samples_per_call must be positive")
        self.cost = CostLedger()

    def _gradients(self, points: Array) -> Array:
        gradients = np.stack(
            [self.gradient_oracle(i, points[i]) for i in range(self.n_agents)]
        )
        self.cost.charge_oracle(self.n_agents, self.samples_per_call)
        return gradients

    def _mix(self, values: Array, rounds: int) -> Array:
        result = np.asarray(values, dtype=float)
        for _ in range(int(rounds)):
            result = self.W @ result
        return result

    @property
    def states(self) -> Array:
        return self.x.copy()


class ProxGT(_ModernBaselineBase):
    """ProxGT with a user-supplied exact or stochastic gradient estimator.

    One iteration performs ``K`` sequential mixing rounds for the gradient
    tracker and another ``K`` rounds for the proximal point, exactly as in the
    published generic recursion.
    """

    def __init__(
        self,
        graph: nx.Graph,
        x0: Array,
        gradient_oracle: GradientOracle,
        prox_operator: ProxOperator,
        stepsize: float,
        consensus_rounds: int = 1,
        samples_per_call: int = 1,
    ):
        super().__init__(
            graph=graph,
            x0=x0,
            gradient_oracle=gradient_oracle,
            prox_operator=prox_operator,
            samples_per_call=samples_per_call,
        )
        self.stepsize = float(stepsize)
        self.consensus_rounds = int(consensus_rounds)
        if self.stepsize <= 0.0 or self.consensus_rounds <= 0:
            raise ValueError("stepsize and consensus_rounds must be positive")
        self.y = np.zeros_like(self.x)
        self.v_previous = np.zeros_like(self.x)

    def step(self) -> None:
        started = time.perf_counter()
        v_current = self._gradients(self.x)
        self.y = self._mix(
            self.y + v_current - self.v_previous,
            self.consensus_rounds,
        )
        z = np.stack(
            [
                self.prox_operator(
                    self.x[i] - self.stepsize * self.y[i],
                    self.stepsize,
                )
                for i in range(self.n_agents)
            ]
        )
        self.x = self._mix(z, self.consensus_rounds)
        self.v_previous = v_current
        self.cost.charge_communication(
            self.graph,
            self.dim,
            sequential_rounds=2 * self.consensus_rounds,
            vectors_per_round=1,
        )
        self.cost.iterations += 1
        self.cost.wall_time_sec += float(time.perf_counter() - started)


class ProxDASAGT(_ModernBaselineBase):
    """Prox-DASA-GT following the three-state Algorithm 2 recursion."""

    def __init__(
        self,
        graph: nx.Graph,
        x0: Array,
        gradient_oracle: GradientOracle,
        prox_operator: ProxOperator,
        gamma: float,
        alpha: float,
        consensus_rounds: int = 1,
        samples_per_call: int = 1,
    ):
        super().__init__(
            graph=graph,
            x0=x0,
            gradient_oracle=gradient_oracle,
            prox_operator=prox_operator,
            samples_per_call=samples_per_call,
        )
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.consensus_rounds = int(consensus_rounds)
        if self.gamma <= 0.0 or not (0.0 < self.alpha <= 1.0):
            raise ValueError("gamma must be positive and alpha must lie in (0, 1]")
        if self.consensus_rounds <= 0:
            raise ValueError("consensus_rounds must be positive")

        # Algorithm 2 uses u^0 = v^0 and z^0 = v^0.  The initialization is an
        # algorithmic oracle cost and is intentionally charged to the ledger.
        initialization_started = time.perf_counter()
        self.v_previous = self._gradients(self.x)
        self.u = self.v_previous.copy()
        self.z = self.v_previous.copy()
        self.cost.wall_time_sec += float(time.perf_counter() - initialization_started)

    def step(self) -> None:
        started = time.perf_counter()
        y = np.stack(
            [
                self.prox_operator(
                    self.x[i] - self.gamma * self.z[i],
                    self.gamma,
                )
                for i in range(self.n_agents)
            ]
        )
        x_tilde = (1.0 - self.alpha) * self.x + self.alpha * y
        v_current = self._gradients(self.x)
        u_tilde = self.u + v_current - self.v_previous
        z_tilde = (1.0 - self.alpha) * self.z + self.alpha * u_tilde

        self.x = self._mix(x_tilde, self.consensus_rounds)
        self.u = self._mix(u_tilde, self.consensus_rounds)
        self.z = self._mix(z_tilde, self.consensus_rounds)
        self.v_previous = v_current
        self.cost.charge_communication(
            self.graph,
            self.dim,
            sequential_rounds=self.consensus_rounds,
            vectors_per_round=3,
        )
        self.cost.iterations += 1
        self.cost.wall_time_sec += float(time.perf_counter() - started)

"""Sparse phase numerical routines."""

from __future__ import annotations

from dataclasses import dataclass


from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple


import numpy as np


from sdbpg.algorithm import SDBPG


from sdbpg.core import exact_l1_quadratic_prox_step
from sdbpg.core import exact_quadratic_smoothing_step
from sdbpg.core import generate_network_topology
from sdbpg.core import quadratic_bregman


SOLVER_MODE = "exact_quadratic"


INIT_PERTURB_STD = 0.01


DIVERGENCE_THRESHOLD = 1e8


GRAD_HIT_EPS = 1e-5


GRAD_HIT_HOLD = 20


SUPPORT_TOL = 5e-2


SPARSE_INIT_TOPK_FACTOR = 2.0


BASE_PARAMS = dict(rho=2.0, alpha=0.015, lambda_param=0.010, beta=0.80, C=1.5)


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "sparse_phase_noisy_undersampled": {
        "n_agents": 8,
        "dim": 40,
        "sparsity": 0.10,
        "samples": 8,
        "noise_std": 0.08,
        "hetero_strength": 1.6,
    },
    "sparse_phase_noisy_moderate": {
        "n_agents": 8,
        "dim": 50,
        "sparsity": 0.10,
        "samples": 10,
        "noise_std": 0.08,
        "hetero_strength": 1.6,
    },
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    display_name: str
    success_threshold: float
    scenarios: Dict[str, Dict[str, Any]]
    make_problem: Callable[[Dict[str, Any], int], Any]


class SparsePhaseRetrievalProblem:
    """Decentralized sparse phase retrieval with controllable heterogeneity."""

    def __init__(
        self,
        n_agents: int,
        dim: int,
        sparsity: float,
        samples_per_agent: int,
        noise_std: float,
        hetero_strength: float,
        seed: int,
    ):
        self.n_agents = int(n_agents)
        self.dim = int(dim)
        self.samples_per_agent = int(samples_per_agent)

        rng = np.random.default_rng(seed)
        k = max(1, int(round(self.dim * float(sparsity))))
        support_idx = rng.choice(self.dim, size=k, replace=False)
        self.support_mask = np.zeros(self.dim, dtype=bool)
        self.support_mask[support_idx] = True

        self.x_star = np.zeros(self.dim, dtype=float)
        self.x_star[support_idx] = rng.standard_normal(k)
        self.x_star /= max(np.linalg.norm(self.x_star), 1e-12)

        self.As: List[np.ndarray] = []
        self.Ys: List[np.ndarray] = []
        for i in range(self.n_agents):
            scale = 1.0 + float(hetero_strength) * i / max(self.n_agents - 1, 1)
            A_i = (scale / np.sqrt(self.dim)) * rng.standard_normal((self.samples_per_agent, self.dim))
            y_clean = (A_i @ self.x_star) ** 2
            noise = float(noise_std) * rng.standard_normal(self.samples_per_agent)
            y_i = y_clean + noise
            self.As.append(A_i)
            self.Ys.append(y_i)

    def local_gradient(self, agent_id: int, x: np.ndarray) -> np.ndarray:
        A_i = self.As[agent_id]
        y_i = self.Ys[agent_id]
        u = A_i @ x
        residual = (u ** 2) - y_i
        return (4.0 / max(self.samples_per_agent, 1)) * (A_i.T @ (residual * u))

    def global_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.mean([self.local_gradient(i, x) for i in range(self.n_agents)], axis=0)

    def local_objective(self, agent_id: int, x: np.ndarray) -> float:
        A_i = self.As[agent_id]
        y_i = self.Ys[agent_id]
        residual = (A_i @ x) ** 2 - y_i
        return float(np.mean(residual**2))

    def global_objective(self, x: np.ndarray) -> float:
        return float(np.mean([self.local_objective(i, x) for i in range(self.n_agents)]))

    def recovery_error(self, x: np.ndarray) -> float:
        e1 = np.linalg.norm(x - self.x_star)
        e2 = np.linalg.norm(x + self.x_star)
        return float(min(e1, e2) / max(np.linalg.norm(self.x_star), 1e-12))

    def support_f1(self, x: np.ndarray, tol: float = SUPPORT_TOL) -> Tuple[float, float, float]:
        pred = np.abs(np.asarray(x, dtype=float)) >= float(tol)
        truth = self.support_mask

        tp = int(np.sum(pred & truth))
        fp = int(np.sum(pred & (~truth)))
        fn = int(np.sum((~pred) & truth))

        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        if precision + recall <= 0.0:
            return 0.0, precision, recall
        f1 = float(2.0 * precision * recall / (precision + recall))
        return f1, precision, recall

    def spectral_init(self) -> np.ndarray:
        mat = np.zeros((self.dim, self.dim), dtype=float)
        count = 0
        for A_i, y_i in zip(self.As, self.Ys):
            for a, y in zip(A_i, y_i):
                mat += float(y) * np.outer(a, a)
                count += 1
        mat /= max(count, 1)

        vals, vecs = np.linalg.eigh(mat)
        idx = int(np.argmax(vals))
        lead = vecs[:, idx]
        scale = np.sqrt(max(float(vals[idx]), 1e-8))
        x0 = scale * lead

        # The task family is explicitly sparse, so we warm-start from a sparse
        # spectral estimate instead of a dense eigenvector.
        target_k = max(1, int(np.sum(self.support_mask)))
        keep_k = min(self.dim, max(target_k, int(round(SPARSE_INIT_TOPK_FACTOR * target_k))))
        if keep_k < self.dim:
            sparse_x0 = np.zeros_like(x0)
            top_idx = np.argpartition(np.abs(x0), -keep_k)[-keep_k:]
            sparse_x0[top_idx] = x0[top_idx]
            x0 = sparse_x0

        x0 /= max(np.linalg.norm(x0), 1e-12)
        return x0


class SparseL1SDBPG(SDBPG):
    """SDBPG with exact L1 proximal step under quadratic Bregman geometry."""

    def __init__(self, *args, lambda_reg: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = float(lambda_reg)
        self.solver_mode = SOLVER_MODE
        self.bregman = quadratic_bregman(mu=1.0)
        for ag in self.agents.values():
            ag.bregman = self.bregman

    def step(self):
        states = {i: ag.get_state() for i, ag in self.agents.items()}
        for i, ag in self.agents.items():
            ag.receive_from_neighbors({j: states[j] for j in ag.neighbors if j in states})

        for ag in self.agents.values():
            ag.dual_update()

        for ag in self.agents.values():
            grad_k = ag.compute_local_surrogate_gradient()
            ag.x = exact_l1_quadratic_prox_step(
                ag.x,
                grad_k,
                ag.lambda_param,
                self.lambda_reg,
                mu=1.0,
            )

        for ag in self.agents.values():
            ag.z = exact_quadratic_smoothing_step(ag.x, ag.z, ag.beta, mu=1.0)


TASKS: Dict[str, TaskSpec] = {
    "sparse_phase_retrieval": TaskSpec(
        name="sparse_phase_retrieval",
        display_name="Sparse Phase Retrieval",
        success_threshold=0.90,
        scenarios=SCENARIOS,
        make_problem=lambda sc, seed: SparsePhaseRetrievalProblem(
            n_agents=int(sc["n_agents"]),
            dim=int(sc["dim"]),
            sparsity=float(sc["sparsity"]),
            samples_per_agent=int(sc["samples"]),
            noise_std=float(sc["noise_std"]),
            hetero_strength=float(sc["hetero_strength"]),
            seed=seed,
        ),
    )
}


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise L1 proximal operator."""
    arr = np.asarray(x, dtype=float)
    return np.sign(arr) * np.maximum(np.abs(arr) - float(tau), 0.0)


def l1_prox_gradient_mapping(
    x: np.ndarray,
    grad_f: np.ndarray,
    lambda_reg: float,
    eta: float,
) -> np.ndarray:
    """Composite stationarity mapping for f(x) + lambda ||x||_1."""
    step = max(float(eta), 1e-12)
    lam = max(float(lambda_reg), 0.0)
    x_arr = np.asarray(x, dtype=float)
    grad_arr = np.asarray(grad_f, dtype=float)
    prox = soft_threshold(x_arr - step * grad_arr, step * lam)
    return (x_arr - prox) / step


def algorithm_lambda_reg(algo: Any) -> float:
    return float(getattr(algo, "lambda_reg", 0.0))


def algorithm_prox_eta(algo: Any) -> float:
    if hasattr(algo, "agents"):
        agents = list(getattr(algo, "agents").values())
        if agents and hasattr(agents[0], "lambda_param"):
            return float(np.mean([float(ag.lambda_param) for ag in agents]))
    if hasattr(algo, "stepsize"):
        return float(getattr(algo, "stepsize"))
    return 1.0


def _first_sustained_below(curve: np.ndarray, eps: float, hold: int) -> int:
    n = len(curve)
    if n == 0:
        return -1
    hold = max(1, min(hold, n))
    mask = (curve <= eps).astype(np.int32)
    csum = np.cumsum(np.concatenate(([0], mask)))
    for i in range(0, n - hold + 1):
        if csum[i + hold] - csum[i] == hold:
            return i + 1
    return -1


def _make_factory(step_mult: float, lambda_reg: float) -> Callable:
    params = dict(BASE_PARAMS)
    params["alpha"] = float(params["alpha"]) * float(step_mult)
    params["lambda_param"] = float(params["lambda_param"]) * float(step_mult)
    return lambda problem, graph: SparseL1SDBPG(
        problem=problem,
        graph=graph,
        lambda_reg=float(lambda_reg),
        **params,
    )


def _initialize_algorithm(algo: SparseL1SDBPG, problem: SparsePhaseRetrievalProblem, seed: int) -> None:
    rng = np.random.default_rng(20260410 + int(seed))
    x0 = problem.spectral_init()

    for ag in algo.agents.values():
        xi = x0 + INIT_PERTURB_STD * rng.standard_normal(problem.dim)
        ag.x = xi.copy()
        ag.z = xi.copy()
        for j in ag.neighbors:
            ag.y[j] = np.zeros(problem.dim)
        ag.neighbors_x = {}
        ag.neighbors_z = {}

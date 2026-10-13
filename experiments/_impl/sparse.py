"""Sparse numerical routines."""

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


N_BOOTSTRAP = 4000


INIT_PERTURB_STD = 0.01


DIVERGENCE_THRESHOLD = 1e8


GRAD_HIT_EPS = 1e-5


GRAD_HIT_HOLD = 20


SUPPORT_TOL = 5e-2


BASE_PARAMS = dict(rho=2.0, alpha=0.02, lambda_param=0.015, beta=0.85, C=1.5)


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "sparse_noisy_undersampled": {
        "n_agents": 8,
        "dim": 40,
        "sparsity": 0.10,
        "samples": 6,
        "noise_std": 0.08,
        "hetero_strength": 1.6,
    },
    "sparse_noisy_moderate": {
        "n_agents": 8,
        "dim": 50,
        "sparsity": 0.10,
        "samples": 8,
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


class SparseLinearRecoveryProblem:
    """Decentralized sparse linear recovery with controllable heterogeneity."""

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
        self.bs: List[np.ndarray] = []
        for i in range(self.n_agents):
            scale = 1.0 + float(hetero_strength) * i / max(self.n_agents - 1, 1)
            A_i = (scale / np.sqrt(self.dim)) * rng.standard_normal((self.samples_per_agent, self.dim))
            noise = float(noise_std) * rng.standard_normal(self.samples_per_agent)
            b_i = A_i @ self.x_star + noise
            self.As.append(A_i)
            self.bs.append(b_i)

    def local_gradient(self, agent_id: int, x: np.ndarray) -> np.ndarray:
        A_i = self.As[agent_id]
        b_i = self.bs[agent_id]
        residual = A_i @ x - b_i
        return (A_i.T @ residual) / max(self.samples_per_agent, 1)

    def global_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.mean([self.local_gradient(i, x) for i in range(self.n_agents)], axis=0)

    def local_objective(self, agent_id: int, x: np.ndarray) -> float:
        A_i = self.As[agent_id]
        b_i = self.bs[agent_id]
        residual = A_i @ x - b_i
        return float(0.5 * np.mean(residual**2))

    def global_objective(self, x: np.ndarray) -> float:
        return float(np.mean([self.local_objective(i, x) for i in range(self.n_agents)]))

    def recovery_error(self, x: np.ndarray) -> float:
        return float(np.linalg.norm(x - self.x_star) / max(np.linalg.norm(self.x_star), 1e-12))

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
    "sparse_recovery": TaskSpec(
        name="sparse_recovery",
        display_name="Sparse Recovery",
        success_threshold=0.90,
        scenarios=SCENARIOS,
        make_problem=lambda sc, seed: SparseLinearRecoveryProblem(
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


def _bootstrap_ci_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOTSTRAP) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = arr[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _bootstrap_ci_paired_mean(
    model_vals: np.ndarray,
    baseline_vals: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOTSTRAP,
) -> Tuple[float, float]:
    diffs = np.asarray(model_vals, dtype=float) - np.asarray(baseline_vals, dtype=float)
    return _bootstrap_ci_mean(diffs, rng, n_boot=n_boot)


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


def _initialize_algorithm(algo: SparseL1SDBPG, dim: int, seed: int) -> None:
    rng = np.random.default_rng(20260410 + int(seed))
    x0 = rng.standard_normal(dim)
    x0 /= max(np.linalg.norm(x0), 1e-12)

    for ag in algo.agents.values():
        xi = x0 + INIT_PERTURB_STD * rng.standard_normal(dim)
        ag.x = xi.copy()
        ag.z = xi.copy()
        for j in ag.neighbors:
            ag.y[j] = np.zeros(dim)
        ag.neighbors_x = {}
        ag.neighbors_z = {}


def _run_one(
    algo: SparseL1SDBPG,
    problem: SparseLinearRecoveryProblem,
    max_iter: int,
    success_threshold: float,
    support_tol: float,
) -> Dict[str, Any]:
    g2_curve: List[float] = []
    composite_g2_curve: List[float] = []
    best_g2 = float("inf")
    best_composite_g2 = float("inf")
    diverged = False
    x_bar = np.zeros(problem.dim, dtype=float)
    lambda_reg = algorithm_lambda_reg(algo)
    prox_eta = algorithm_prox_eta(algo)

    for _ in range(max_iter):
        algo.step()

        xs = [ag.x for ag in algo.agents.values()]
        x_bar = np.mean(xs, axis=0)

        grad = problem.global_gradient(x_bar)
        g2 = float(np.dot(grad, grad))
        mapping = l1_prox_gradient_mapping(x_bar, grad, lambda_reg=lambda_reg, eta=prox_eta)
        composite_g2 = float(np.dot(mapping, mapping))

        if (
            not np.isfinite(g2)
            or not np.isfinite(composite_g2)
            or not np.isfinite(np.linalg.norm(x_bar))
            or g2 > DIVERGENCE_THRESHOLD
            or composite_g2 > DIVERGENCE_THRESHOLD
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
            "final_g2": float(DIVERGENCE_THRESHOLD),
            "composite_g2": float(DIVERGENCE_THRESHOLD),
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
    f1, precision, recall = problem.support_f1(x_bar, tol=support_tol)
    final_obj = problem.global_objective(x_bar)
    final_l1 = float(np.linalg.norm(x_bar, 1))
    final_composite_obj = float(final_obj + lambda_reg * final_l1)
    support_size = int(np.count_nonzero(np.abs(x_bar) >= support_tol))
    hit_t = _first_sustained_below(np.asarray(g2_curve, dtype=float), GRAD_HIT_EPS, GRAD_HIT_HOLD)
    composite_hit_t = _first_sustained_below(
        np.asarray(composite_g2_curve, dtype=float),
        GRAD_HIT_EPS,
        GRAD_HIT_HOLD,
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


def _run_one_seed(
    task: TaskSpec,
    scenario_cfg: Dict[str, Any],
    seed: int,
    step_mult: float,
    lambda_reg: float,
    max_iter: int,
    support_tol: float,
) -> Dict[str, Any]:
    problem = task.make_problem(scenario_cfg, seed)
    graph = generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)
    algo = _make_factory(step_mult=step_mult, lambda_reg=lambda_reg)(problem, graph)
    _initialize_algorithm(algo, problem.dim, seed)
    return _run_one(algo, problem, max_iter=max_iter, success_threshold=task.success_threshold, support_tol=support_tol)

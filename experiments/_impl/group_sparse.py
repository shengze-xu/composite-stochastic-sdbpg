"""Group sparse numerical routines."""

from __future__ import annotations

from dataclasses import dataclass


from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple


import numpy as np


from sdbpg.algorithm import DecentralizedGradientDescent
from sdbpg.algorithm import DecentralizedGradientTracking
from sdbpg.algorithm import SDBPG


from sdbpg.core import exact_group_l21_quadratic_prox_step
from sdbpg.core import exact_quadratic_smoothing_step
from sdbpg.core import generate_network_topology
from sdbpg.core import quadratic_bregman


from experiments._impl import sparse as sr


SOLVER_MODE = "exact_quadratic"


INIT_PERTURB_STD = 0.01


DIVERGENCE_THRESHOLD = 1e8


SUPPORT_TOL = 0.12


GRAD_HIT_EPS = 1e-5


GRAD_HIT_HOLD = 20


BASE_PARAMS = dict(rho=2.0, alpha=0.02, lambda_param=0.015, beta=0.85, C=1.5)


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "group_sparse_noisy_undersampled": {
        "n_agents": 8,
        "dim": 36,
        "n_tasks": 4,
        "row_sparsity": 0.14,
        "samples": 8,
        "noise_std": 0.08,
        "hetero_strength": 1.8,
    },
    "group_sparse_noisy_heterogeneous": {
        "n_agents": 8,
        "dim": 40,
        "n_tasks": 4,
        "row_sparsity": 0.10,
        "samples": 9,
        "noise_std": 0.08,
        "hetero_strength": 2.2,
    },
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    display_name: str
    success_threshold: float
    scenarios: Dict[str, Dict[str, Any]]
    make_problem: Callable[[Dict[str, Any], int], Any]


def _reshape_matrix(x: np.ndarray, dim: int, n_tasks: int) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(dim, n_tasks)


class GroupSparseRecoveryProblem:
    """Shared-support multitask linear recovery with controllable heterogeneity."""

    def __init__(
        self,
        n_agents: int,
        dim: int,
        n_tasks: int,
        row_sparsity: float,
        samples_per_agent: int,
        noise_std: float,
        hetero_strength: float,
        seed: int,
    ):
        self.n_agents = int(n_agents)
        self.feature_dim = int(dim)
        self.n_tasks = int(n_tasks)
        self.dim = self.feature_dim * self.n_tasks
        self.samples_per_agent = int(samples_per_agent)

        rng = np.random.default_rng(seed)
        k = max(1, int(round(self.feature_dim * float(row_sparsity))))
        support_idx = rng.choice(self.feature_dim, size=k, replace=False)
        self.support_mask = np.zeros(self.feature_dim, dtype=bool)
        self.support_mask[support_idx] = True

        self.X_star = np.zeros((self.feature_dim, self.n_tasks), dtype=float)
        self.X_star[support_idx] = rng.standard_normal((k, self.n_tasks))
        self.X_star /= max(np.linalg.norm(self.X_star), 1e-12)
        self.x_star = self.X_star.reshape(-1)

        self.As: List[np.ndarray] = []
        self.Bs: List[np.ndarray] = []
        for i in range(self.n_agents):
            scale = 1.0 + float(hetero_strength) * i / max(self.n_agents - 1, 1)
            column_scale = 1.0 + 0.35 * float(hetero_strength) * rng.uniform(-1.0, 1.0, size=self.feature_dim)
            A_i = rng.standard_normal((self.samples_per_agent, self.feature_dim))
            A_i = (scale / np.sqrt(self.feature_dim)) * A_i * column_scale[None, :]
            noise = float(noise_std) * rng.standard_normal((self.samples_per_agent, self.n_tasks))
            B_i = A_i @ self.X_star + noise
            self.As.append(A_i)
            self.Bs.append(B_i)

    def local_gradient(self, agent_id: int, x: np.ndarray) -> np.ndarray:
        X = _reshape_matrix(x, self.feature_dim, self.n_tasks)
        A_i = self.As[agent_id]
        B_i = self.Bs[agent_id]
        residual = A_i @ X - B_i
        grad = (A_i.T @ residual) / max(self.samples_per_agent, 1)
        return grad.reshape(-1)

    def global_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.mean([self.local_gradient(i, x) for i in range(self.n_agents)], axis=0)

    def local_objective(self, agent_id: int, x: np.ndarray) -> float:
        X = _reshape_matrix(x, self.feature_dim, self.n_tasks)
        residual = self.As[agent_id] @ X - self.Bs[agent_id]
        return float(0.5 * np.mean(residual**2))

    def global_objective(self, x: np.ndarray) -> float:
        return float(np.mean([self.local_objective(i, x) for i in range(self.n_agents)]))

    def recovery_error(self, x: np.ndarray) -> float:
        X = _reshape_matrix(x, self.feature_dim, self.n_tasks)
        return float(np.linalg.norm(X - self.X_star) / max(np.linalg.norm(self.X_star), 1e-12))

    def support_f1(self, x: np.ndarray, tol: float = SUPPORT_TOL) -> Tuple[float, float, float]:
        X = _reshape_matrix(x, self.feature_dim, self.n_tasks)
        pred = np.linalg.norm(X, axis=1) >= float(tol)
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

    def support_size(self, x: np.ndarray, tol: float = SUPPORT_TOL) -> int:
        X = _reshape_matrix(x, self.feature_dim, self.n_tasks)
        return int(np.count_nonzero(np.linalg.norm(X, axis=1) >= float(tol)))

    def group_l21_norm(self, x: np.ndarray) -> float:
        X = _reshape_matrix(x, self.feature_dim, self.n_tasks)
        return float(np.sum(np.linalg.norm(X, axis=1)))


class GroupL21SDBPG(SDBPG):
    """SDBPG with exact row-wise L2,1 proximal step under quadratic geometry."""

    def __init__(self, *args, lambda_reg: float = 0.0, feature_dim: int, n_tasks: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = float(lambda_reg)
        self.feature_dim = int(feature_dim)
        self.n_tasks = int(n_tasks)
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
            ag.x = exact_group_l21_quadratic_prox_step(
                ag.x,
                grad_k,
                ag.lambda_param,
                self.lambda_reg,
                self.feature_dim,
                self.n_tasks,
                mu=1.0,
            )

        for ag in self.agents.values():
            ag.z = exact_quadratic_smoothing_step(ag.x, ag.z, ag.beta, mu=1.0)


TASKS: Dict[str, TaskSpec] = {
    "group_sparse_recovery": TaskSpec(
        name="group_sparse_recovery",
        display_name="Group Sparse Recovery",
        success_threshold=0.75,
        scenarios=SCENARIOS,
        make_problem=lambda sc, seed: GroupSparseRecoveryProblem(
            n_agents=int(sc["n_agents"]),
            dim=int(sc["dim"]),
            n_tasks=int(sc["n_tasks"]),
            row_sparsity=float(sc["row_sparsity"]),
            samples_per_agent=int(sc["samples"]),
            noise_std=float(sc["noise_std"]),
            hetero_strength=float(sc["hetero_strength"]),
            seed=seed,
        ),
    )
}


def group_l21_prox_gradient_mapping(
    x: np.ndarray,
    grad_f: np.ndarray,
    lambda_reg: float,
    eta: float,
    feature_dim: int,
    n_tasks: int,
) -> np.ndarray:
    step = max(float(eta), 1e-12)
    prox = exact_group_l21_quadratic_prox_step(
        x,
        grad_f,
        step,
        lambda_reg,
        feature_dim,
        n_tasks,
        mu=1.0,
    )
    return (np.asarray(x, dtype=float) - prox) / step


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


def _make_sdbpg_factory(step_mult: float, lambda_reg: float) -> Callable:
    params = dict(BASE_PARAMS)
    params["alpha"] = float(params["alpha"]) * float(step_mult)
    params["lambda_param"] = float(params["lambda_param"]) * float(step_mult)

    def factory(problem: GroupSparseRecoveryProblem, graph: Any) -> GroupL21SDBPG:
        return GroupL21SDBPG(
            problem=problem,
            graph=graph,
            lambda_reg=float(lambda_reg),
            feature_dim=problem.feature_dim,
            n_tasks=problem.n_tasks,
            **params,
        )

    return factory


def _initialize_sdbpg(algo: GroupL21SDBPG, problem: GroupSparseRecoveryProblem, seed: int) -> None:
    rng = np.random.default_rng(20260414 + int(seed))
    x0 = rng.standard_normal(problem.dim)
    x0 /= max(np.linalg.norm(x0), 1e-12)

    for ag in algo.agents.values():
        xi = x0 + INIT_PERTURB_STD * rng.standard_normal(problem.dim)
        ag.x = xi.copy()
        ag.z = xi.copy()
        for j in ag.neighbors:
            ag.y[j] = np.zeros(problem.dim)
        ag.neighbors_x = {}
        ag.neighbors_z = {}


def _initialize_baseline(algo: Any, problem: GroupSparseRecoveryProblem, seed: int) -> None:
    rng = np.random.default_rng(20260414 + int(seed))
    x0 = rng.standard_normal(problem.dim)
    x0 /= max(np.linalg.norm(x0), 1e-12)

    xs = []
    for _ in range(problem.n_agents):
        xi = x0 + INIT_PERTURB_STD * rng.standard_normal(problem.dim)
        xs.append(xi.copy())

    algo.x = xs
    if hasattr(algo, "y"):
        algo.y = [problem.local_gradient(i, algo.x[i]) for i in range(problem.n_agents)]


def _extract_xs(algo: Any) -> List[np.ndarray]:
    if hasattr(algo, "agents"):
        return [ag.x for ag in algo.agents.values()]
    return list(algo.x)


def _run_one_generic(
    algo: Any,
    problem: GroupSparseRecoveryProblem,
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
    lambda_reg = float(getattr(algo, "lambda_reg", 0.0) or 0.0)
    prox_eta = sr.algorithm_prox_eta(algo)

    for _ in range(max_iter):
        algo.step()

        xs = _extract_xs(algo)
        x_bar = np.mean(xs, axis=0)

        grad = problem.global_gradient(x_bar)
        g2 = float(np.dot(grad, grad))
        mapping = group_l21_prox_gradient_mapping(
            x_bar,
            grad,
            lambda_reg=lambda_reg,
            eta=prox_eta,
            feature_dim=problem.feature_dim,
            n_tasks=problem.n_tasks,
        )
        composite_g2 = float(np.dot(mapping, mapping))

        if (
            np.isnan(g2)
            or np.isnan(composite_g2)
            or np.isnan(np.linalg.norm(x_bar))
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
            "final_l21": float("inf"),
            "support_size": int(problem.feature_dim),
            "success": 0,
            "diverged": 1,
            "hit_t": int(max_iter + 1),
            "composite_hit_t": int(max_iter + 1),
        }

    final_rec = problem.recovery_error(x_bar)
    f1, precision, recall = problem.support_f1(x_bar, tol=support_tol)
    final_obj = problem.global_objective(x_bar)
    final_l21 = problem.group_l21_norm(x_bar)
    support_size = problem.support_size(x_bar, tol=support_tol)
    hit_t = _first_sustained_below(np.asarray(g2_curve, dtype=float), GRAD_HIT_EPS, GRAD_HIT_HOLD)
    composite_hit_t = _first_sustained_below(np.asarray(composite_g2_curve, dtype=float), GRAD_HIT_EPS, GRAD_HIT_HOLD)

    return {
        "final_g2": float(best_g2),
        "composite_g2": float(best_composite_g2),
        "final_rec": float(final_rec),
        "support_f1": float(f1),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "final_obj": float(final_obj),
        "final_l21": float(final_l21),
        "support_size": int(support_size),
        "success": int(final_rec < float(success_threshold)),
        "diverged": 0,
        "hit_t": int(hit_t) if int(hit_t) > 0 else int(max_iter + 1),
        "composite_hit_t": int(composite_hit_t) if int(composite_hit_t) > 0 else int(max_iter + 1),
    }


def _run_one_seed_sdbpg(
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
    algo = _make_sdbpg_factory(step_mult=step_mult, lambda_reg=lambda_reg)(problem, graph)
    _initialize_sdbpg(algo, problem, seed)
    return _run_one_generic(algo, problem, max_iter=max_iter, success_threshold=task.success_threshold, support_tol=support_tol)


def _make_baseline_factory(algorithm: str, params: Dict[str, float]) -> Callable:
    if algorithm == "DGD":
        return lambda problem, graph: DecentralizedGradientDescent(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
            consensus_weight=float(params["consensus_weight"]),
        )
    if algorithm == "DGT":
        return lambda problem, graph: DecentralizedGradientTracking(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _run_one_seed_baseline(
    task: TaskSpec,
    scenario_cfg: Dict[str, Any],
    seed: int,
    algorithm: str,
    params: Dict[str, float],
    max_iter: int,
    support_tol: float,
) -> Dict[str, Any]:
    problem = task.make_problem(scenario_cfg, seed)
    graph = generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)
    algo = _make_baseline_factory(algorithm, params)(problem, graph)
    _initialize_baseline(algo, problem, seed)
    return _run_one_generic(algo, problem, max_iter=max_iter, success_threshold=task.success_threshold, support_tol=support_tol)

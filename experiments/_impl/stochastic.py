"""Stochastic numerical routines."""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple


import numpy as np


from sdbpg.algorithm import DecentralizedGradientDescent
from sdbpg.algorithm import DecentralizedGradientTracking
from sdbpg.algorithm import ProximalDecentralizedGradientDescent
from sdbpg.algorithm import ProximalDecentralizedGradientTracking


from sdbpg.core import generate_network_topology


from experiments._impl import sparse as sr


class StochasticSparseLinearRecoveryProblem:
    """Finite-sum sparse recovery with a controllable stochastic gradient oracle."""

    def __init__(
        self,
        n_agents: int,
        dim: int,
        sparsity: float,
        samples_per_agent: int,
        noise_std: float,
        hetero_strength: float,
        seed: int,
        oracle_mode: str,
        batch_size: Optional[int],
        noise_sigma: float,
    ):
        self.n_agents = int(n_agents)
        self.dim = int(dim)
        self.samples_per_agent = int(samples_per_agent)
        self.oracle_mode = str(oracle_mode)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.noise_sigma = float(noise_sigma)

        rng = np.random.default_rng(seed)
        self.oracle_rng = np.random.default_rng(700000 + int(seed))

        k = max(1, int(round(self.dim * float(sparsity))))
        support_idx = rng.choice(self.dim, size=k, replace=False)
        self.support_mask = np.zeros(self.dim, dtype=bool)
        self.support_mask[support_idx] = True

        self.x_star = np.zeros(self.dim, dtype=float)
        self.x_star[support_idx] = rng.standard_normal(k)
        self.x_star /= max(np.linalg.norm(self.x_star), 1e-12)

        self.A_samples: List[np.ndarray] = []
        self.b_samples: List[np.ndarray] = []
        for i in range(self.n_agents):
            scale = 1.0 + float(hetero_strength) * i / max(self.n_agents - 1, 1)
            A_i = (scale / np.sqrt(self.dim)) * rng.standard_normal((self.samples_per_agent, self.dim))
            noise = float(noise_std) * rng.standard_normal(self.samples_per_agent)
            b_i = A_i @ self.x_star + noise
            self.A_samples.append(A_i)
            self.b_samples.append(b_i)

        self._step_oracle_noise_sq: List[float] = []

    def begin_step(self) -> None:
        self._step_oracle_noise_sq = []

    def consume_step_oracle_summary(self) -> Dict[str, float]:
        if not self._step_oracle_noise_sq:
            return {
                "avg_oracle_noise_sq": float("nan"),
                "max_oracle_noise_sq": float("nan"),
                "oracle_calls": 0.0,
            }
        arr = np.asarray(self._step_oracle_noise_sq, dtype=float)
        return {
            "avg_oracle_noise_sq": float(arr.mean()),
            "max_oracle_noise_sq": float(arr.max()),
            "oracle_calls": float(arr.size),
        }

    def local_objective(self, agent_id: int, x: np.ndarray) -> float:
        A_i = self.A_samples[agent_id]
        b_i = self.b_samples[agent_id]
        residual = A_i @ x - b_i
        return float(0.5 * np.mean(residual**2))

    def global_objective(self, x: np.ndarray) -> float:
        return float(np.mean([self.local_objective(i, x) for i in range(self.n_agents)]))

    def full_local_gradient(self, agent_id: int, x: np.ndarray) -> np.ndarray:
        A_i = self.A_samples[agent_id]
        b_i = self.b_samples[agent_id]
        residual = A_i @ x - b_i
        return (A_i.T @ residual) / max(self.samples_per_agent, 1)

    def local_gradient(self, agent_id: int, x: np.ndarray) -> np.ndarray:
        full_grad = self.full_local_gradient(agent_id, x)
        grad = self.sample_local_gradient(agent_id, x, rng=self.oracle_rng, full_grad=full_grad)
        diff = grad - full_grad
        self._step_oracle_noise_sq.append(float(np.dot(diff, diff)))
        return grad

    def sample_local_gradient(
        self,
        agent_id: int,
        x: np.ndarray,
        rng: np.random.Generator,
        full_grad: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if full_grad is None:
            full_grad = self.full_local_gradient(agent_id, x)

        if self.oracle_mode == "full":
            return full_grad

        if self.oracle_mode == "mini_batch":
            batch_size = self.samples_per_agent if self.batch_size is None else int(self.batch_size)
            if batch_size >= self.samples_per_agent:
                return full_grad
            idx = rng.integers(0, self.samples_per_agent, size=batch_size)
            A_b = self.A_samples[agent_id][idx]
            b_b = self.b_samples[agent_id][idx]
            residual = A_b @ x - b_b
            return (A_b.T @ residual) / max(batch_size, 1)

        if self.oracle_mode == "additive":
            sigma = float(self.noise_sigma)
            if sigma <= 0.0:
                return full_grad
            noise = (sigma / np.sqrt(self.dim)) * rng.standard_normal(self.dim)
            return full_grad + noise

        raise ValueError(f"Unknown oracle_mode: {self.oracle_mode}")

    def full_global_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.mean([self.full_local_gradient(i, x) for i in range(self.n_agents)], axis=0)

    def recovery_error(self, x: np.ndarray) -> float:
        return float(np.linalg.norm(x - self.x_star) / max(np.linalg.norm(self.x_star), 1e-12))

    def support_f1(self, x: np.ndarray, tol: float = sr.SUPPORT_TOL) -> Tuple[float, float, float]:
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


def build_setting_manifest(
    scenario_names: Sequence[str],
    oracle_mode: str,
    batch_sizes: Sequence[int],
    noise_sigmas: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario_name in scenario_names:
        if scenario_name not in sr.SCENARIOS:
            raise ValueError(f"Unknown sparse-recovery scenario: {scenario_name}")
        base_cfg = sr.SCENARIOS[scenario_name]

        if oracle_mode == "mini_batch":
            for batch_size in batch_sizes:
                rows.append(
                    {
                        "setting_id": f"{scenario_name}__mini_batch__{int(batch_size)}",
                        "task": "stochastic_sparse_recovery",
                        "scenario": scenario_name,
                        "oracle_mode": "mini_batch",
                        "batch_size": int(batch_size),
                        "noise_sigma": float("nan"),
                        "n_agents": int(base_cfg["n_agents"]),
                        "dim": int(base_cfg["dim"]),
                        "sparsity": float(base_cfg["sparsity"]),
                        "samples": int(base_cfg["samples"]),
                        "noise_std": float(base_cfg["noise_std"]),
                        "hetero_strength": float(base_cfg["hetero_strength"]),
                    }
                )
        elif oracle_mode == "additive":
            for noise_sigma in noise_sigmas:
                rows.append(
                    {
                        "setting_id": f"{scenario_name}__additive__{float(noise_sigma):.3f}",
                        "task": "stochastic_sparse_recovery",
                        "scenario": scenario_name,
                        "oracle_mode": "additive",
                        "batch_size": int(base_cfg["samples"]),
                        "noise_sigma": float(noise_sigma),
                        "n_agents": int(base_cfg["n_agents"]),
                        "dim": int(base_cfg["dim"]),
                        "sparsity": float(base_cfg["sparsity"]),
                        "samples": int(base_cfg["samples"]),
                        "noise_std": float(base_cfg["noise_std"]),
                        "hetero_strength": float(base_cfg["hetero_strength"]),
                    }
                )
        else:
            raise ValueError(f"Unknown oracle_mode: {oracle_mode}")
    return rows


def _make_problem(setting: Dict[str, Any], seed: int) -> StochasticSparseLinearRecoveryProblem:
    return StochasticSparseLinearRecoveryProblem(
        n_agents=int(setting["n_agents"]),
        dim=int(setting["dim"]),
        sparsity=float(setting["sparsity"]),
        samples_per_agent=int(setting["samples"]),
        noise_std=float(setting["noise_std"]),
        hetero_strength=float(setting["hetero_strength"]),
        seed=seed,
        oracle_mode=str(setting["oracle_mode"]),
        batch_size=int(setting["batch_size"]),
        noise_sigma=0.0 if np.isnan(float(setting["noise_sigma"])) else float(setting["noise_sigma"]),
    )


def _initialize_sdbpg(algo: sr.SparseL1SDBPG, dim: int, seed: int) -> None:
    sr._initialize_algorithm(algo, dim, seed)


def _initialize_baseline(algo: Any, problem: StochasticSparseLinearRecoveryProblem, seed: int) -> None:
    rng = np.random.default_rng(20260414 + int(seed))
    x0 = rng.standard_normal(problem.dim)
    x0 /= max(np.linalg.norm(x0), 1e-12)

    xs = []
    for _ in range(problem.n_agents):
        xi = x0 + sr.INIT_PERTURB_STD * rng.standard_normal(problem.dim)
        xs.append(xi.copy())

    algo.x = xs
    if hasattr(algo, "y"):
        algo.y = [problem.local_gradient(i, algo.x[i]) for i in range(problem.n_agents)]


def _make_sdbpg_factory(step_mult: float, lambda_reg: float) -> Callable:
    return sr._make_factory(step_mult=step_mult, lambda_reg=lambda_reg)


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
    if algorithm == "ProxDGD":
        return lambda problem, graph: ProximalDecentralizedGradientDescent(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
            consensus_weight=float(params["consensus_weight"]),
            lambda_reg=float(params["lambda_reg"]),
        )
    if algorithm == "ProxDGT":
        return lambda problem, graph: ProximalDecentralizedGradientTracking(
            problem=problem,
            graph=graph,
            stepsize=float(params["stepsize"]),
            lambda_reg=float(params["lambda_reg"]),
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _snapshot_sdbpg_state(algo: Any) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    if not hasattr(algo, "agents"):
        return None
    return {i: (ag.x.copy(), ag.z.copy()) for i, ag in algo.agents.items()}


def _compute_theory_step_metrics(
    algo: Any,
    state_before: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]],
) -> Dict[str, float]:
    if state_before is None:
        return {
            "d_x": float("nan"),
            "d_x_rev": float("nan"),
            "d_z": float("nan"),
            "e": float("nan"),
        }

    d_x = 0.0
    d_x_rev = 0.0
    d_z = 0.0
    e = 0.0
    for agent_id, ag in algo.agents.items():
        x_old, z_old = state_before[agent_id]
        d_x += float(ag.bregman.divergence(ag.x, x_old))
        d_x_rev += float(ag.bregman.divergence(x_old, ag.x))
        d_z += float(ag.bregman.divergence(ag.z, z_old))
        e += float(np.dot(ag.x - ag.z, ag.x - ag.z))

    return {
        "d_x": float(d_x),
        "d_x_rev": float(d_x_rev),
        "d_z": float(d_z),
        "e": float(e),
    }


def _run_one_generic(
    algo: Any,
    problem: StochasticSparseLinearRecoveryProblem,
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
    lambda_reg = sr.algorithm_lambda_reg(algo)
    prox_eta = sr.algorithm_prox_eta(algo)

    d_x_curve: List[float] = []
    d_x_rev_curve: List[float] = []
    d_z_curve: List[float] = []
    e_curve: List[float] = []
    oracle_noise_curve: List[float] = []
    oracle_max_curve: List[float] = []
    oracle_calls_curve: List[float] = []

    for _ in range(max_iter):
        if hasattr(problem, "begin_step"):
            problem.begin_step()
        state_before = _snapshot_sdbpg_state(algo)
        algo.step()
        theory = _compute_theory_step_metrics(algo, state_before)
        oracle = problem.consume_step_oracle_summary()

        if hasattr(algo, "agents"):
            xs = [ag.x for ag in algo.agents.values()]
        else:
            xs = algo.x
        x_bar = np.mean(xs, axis=0)

        grad = problem.full_global_gradient(x_bar)
        g2 = float(np.dot(grad, grad))
        mapping = sr.l1_prox_gradient_mapping(x_bar, grad, lambda_reg=lambda_reg, eta=prox_eta)
        composite_g2 = float(np.dot(mapping, mapping))
        if (
            not np.isfinite(g2)
            or not np.isfinite(composite_g2)
            or not np.isfinite(np.linalg.norm(x_bar))
            or g2 > sr.DIVERGENCE_THRESHOLD
            or composite_g2 > sr.DIVERGENCE_THRESHOLD
            or np.linalg.norm(x_bar) > 1e4
        ):
            diverged = True
            break

        g2_curve.append(g2)
        composite_g2_curve.append(composite_g2)
        best_g2 = min(best_g2, g2)
        best_composite_g2 = min(best_composite_g2, composite_g2)
        d_x_curve.append(float(theory["d_x"]))
        d_x_rev_curve.append(float(theory["d_x_rev"]))
        d_z_curve.append(float(theory["d_z"]))
        e_curve.append(float(theory["e"]))
        oracle_noise_curve.append(float(oracle["avg_oracle_noise_sq"]))
        oracle_max_curve.append(float(oracle["max_oracle_noise_sq"]))
        oracle_calls_curve.append(float(oracle["oracle_calls"]))

    if diverged or len(g2_curve) == 0:
        return {
            "final_g2": float(sr.DIVERGENCE_THRESHOLD),
            "composite_g2": float(sr.DIVERGENCE_THRESHOLD),
            "final_rec": float("inf"),
            "support_f1": 0.0,
            "support_precision": 0.0,
            "support_recall": 0.0,
            "final_obj": float("inf"),
            "final_l1": float("inf"),
            "support_size": int(problem.dim),
            "success": 0,
            "diverged": 1,
            "hit_t": int(max_iter + 1),
            "composite_hit_t": int(max_iter + 1),
            "avg_d_x": float("nan"),
            "avg_d_x_rev": float("nan"),
            "avg_d_z": float("nan"),
            "avg_e": float("nan"),
            "last_d_x": float("nan"),
            "last_d_x_rev": float("nan"),
            "last_d_z": float("nan"),
            "last_e": float("nan"),
            "avg_oracle_noise_sq": float("nan"),
            "max_oracle_noise_sq": float("nan"),
            "oracle_calls_per_iter": float("nan"),
        }

    final_rec = problem.recovery_error(x_bar)
    f1, precision, recall = problem.support_f1(x_bar, tol=support_tol)
    final_obj = problem.global_objective(x_bar)
    final_l1 = float(np.linalg.norm(x_bar, 1))
    support_size = int(np.count_nonzero(np.abs(x_bar) >= support_tol))
    hit_t = sr._first_sustained_below(np.asarray(g2_curve, dtype=float), sr.GRAD_HIT_EPS, sr.GRAD_HIT_HOLD)
    composite_hit_t = sr._first_sustained_below(
        np.asarray(composite_g2_curve, dtype=float),
        sr.GRAD_HIT_EPS,
        sr.GRAD_HIT_HOLD,
    )

    def _safe_mean(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return float("nan")
        if np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    def _safe_max(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmax(arr))

    return {
        "final_g2": float(best_g2),
        "composite_g2": float(best_composite_g2),
        "final_rec": float(final_rec),
        "support_f1": float(f1),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "final_obj": float(final_obj),
        "final_l1": float(final_l1),
        "support_size": int(support_size),
        "success": int(final_rec < float(success_threshold)),
        "diverged": 0,
        "hit_t": int(hit_t) if int(hit_t) > 0 else int(max_iter + 1),
        "composite_hit_t": int(composite_hit_t) if int(composite_hit_t) > 0 else int(max_iter + 1),
        "avg_d_x": _safe_mean(d_x_curve),
        "avg_d_x_rev": _safe_mean(d_x_rev_curve),
        "avg_d_z": _safe_mean(d_z_curve),
        "avg_e": _safe_mean(e_curve),
        "last_d_x": float(d_x_curve[-1]) if d_x_curve else float("nan"),
        "last_d_x_rev": float(d_x_rev_curve[-1]) if d_x_rev_curve else float("nan"),
        "last_d_z": float(d_z_curve[-1]) if d_z_curve else float("nan"),
        "last_e": float(e_curve[-1]) if e_curve else float("nan"),
        "avg_oracle_noise_sq": _safe_mean(oracle_noise_curve),
        "max_oracle_noise_sq": _safe_max(oracle_max_curve),
        "oracle_calls_per_iter": _safe_mean(oracle_calls_curve),
    }


def _run_one_seed_sdbpg(
    task: sr.TaskSpec,
    setting: Dict[str, Any],
    seed: int,
    step_mult: float,
    lambda_reg: float,
    max_iter: int,
    support_tol: float,
) -> Dict[str, Any]:
    problem = _make_problem(setting, seed)
    graph = generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)
    algo = _make_sdbpg_factory(step_mult=step_mult, lambda_reg=lambda_reg)(problem, graph)
    _initialize_sdbpg(algo, problem.dim, seed)
    return _run_one_generic(algo, problem, max_iter=max_iter, success_threshold=task.success_threshold, support_tol=support_tol)


def _run_one_seed_baseline(
    task: sr.TaskSpec,
    setting: Dict[str, Any],
    seed: int,
    algorithm: str,
    params: Dict[str, float],
    max_iter: int,
    support_tol: float,
) -> Dict[str, Any]:
    problem = _make_problem(setting, seed)
    graph = generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)
    algo = _make_baseline_factory(algorithm, params)(problem, graph)
    _initialize_baseline(algo, problem, seed)
    return _run_one_generic(algo, problem, max_iter=max_iter, success_threshold=task.success_threshold, support_tol=support_tol)

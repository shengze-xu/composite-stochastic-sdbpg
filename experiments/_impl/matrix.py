"""Matrix numerical routines."""

from __future__ import annotations

from typing import Dict
from typing import List


import numpy as np


from sdbpg.algorithm import SDBPG
from sdbpg.algorithm import DecentralizedGradientDescent
from sdbpg.algorithm import DecentralizedGradientTracking


from sdbpg.core import exact_quadratic_primal_step
from sdbpg.core import exact_quadratic_smoothing_step
from sdbpg.core import generate_network_topology


N_AGENTS = 6


MAT_DIM = 8


RANK = 2


SEEDS = list(range(10))


INIT_PERTURB_STD = 0.01


SOLVER_MODE = "exact_quadratic"


BASE_PARAMS = dict(rho=2.0, alpha=0.010, lambda_param=0.006, beta=0.80, C=1.5)


SCENARIOS = {
    "easy": dict(samples=120, noise_std=0.00, hetero_strength=0.5),
    "hard": dict(samples=16, noise_std=0.08, hetero_strength=2.0),
}


EPS = 1e-5


HOLD = 20


DIVERGENCE_THRESHOLD = 1e8


class MatrixFactorizationProblem:
    def __init__(
        self,
        n_agents: int,
        mat_dim: int,
        rank: int,
        samples_per_agent: int,
        noise_std: float,
        hetero_strength: float,
        seed: int,
    ):
        self.n_agents = n_agents
        self.mat_dim = mat_dim
        self.rank = rank
        self.dim = mat_dim * rank  # algorithm state dimension (vectorized U)

        rng = np.random.default_rng(seed)

        self.U_star = rng.standard_normal((mat_dim, rank)) / np.sqrt(mat_dim)
        self.X_star = self.U_star @ self.U_star.T

        self.As = []
        self.Ys = []

        for i in range(n_agents):
            scale = 1.0 + hetero_strength * i / max(n_agents - 1, 1)
            a_list = []
            y_list = []
            for _ in range(samples_per_agent):
                a = rng.standard_normal((mat_dim, mat_dim)) / np.sqrt(mat_dim)
                a = 0.5 * (a + a.T) * scale  # symmetric measurement matrix
                y = float(np.sum(a * self.X_star) + noise_std * rng.standard_normal())
                a_list.append(a)
                y_list.append(y)
            self.As.append(a_list)
            self.Ys.append(np.array(y_list))

    def _vec_to_U(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.mat_dim, self.rank)

    def local_gradient(self, i: int, x: np.ndarray) -> np.ndarray:
        U = self._vec_to_U(x)
        grad_U = np.zeros_like(U)

        A_i = self.As[i]
        y_i = self.Ys[i]
        m = len(A_i)

        for A, y in zip(A_i, y_i):
            pred = float(np.sum(A * (U @ U.T)))
            r = pred - y
            # for symmetric A: d pred / dU = 2 A U
            grad_U += 4.0 * r * (A @ U)

        grad_U /= max(m, 1)
        return grad_U.reshape(-1)

    def global_objective(self, x: np.ndarray) -> float:
        U = self._vec_to_U(x)
        X = U @ U.T
        vals = []
        for i in range(self.n_agents):
            A_i = self.As[i]
            y_i = self.Ys[i]
            residuals = [float(np.sum(A * X) - y) for A, y in zip(A_i, y_i)]
            vals.append(float(np.mean(np.square(residuals))))
        return float(np.mean(vals))

    def global_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.mean([self.local_gradient(i, x) for i in range(self.n_agents)], axis=0)

    def recovery_error(self, x: np.ndarray) -> float:
        U = self._vec_to_U(x)
        X = U @ U.T
        return float(np.linalg.norm(X - self.X_star, ord="fro") / max(np.linalg.norm(self.X_star, ord="fro"), 1e-12))

    def spectral_init(self) -> np.ndarray:
        # Lifted spectral initialization for matrix sensing.
        mat = np.zeros((self.mat_dim, self.mat_dim), dtype=float)
        count = 0
        for i in range(self.n_agents):
            for A, y in zip(self.As[i], self.Ys[i]):
                mat += float(y) * A
                count += 1
        mat = 0.5 * (mat + mat.T) / max(count, 1)

        vals, vecs = np.linalg.eigh(mat)
        idx = np.argsort(vals)[::-1][: self.rank]
        vals_top = np.clip(vals[idx], 1e-8, None)
        U0 = vecs[:, idx] @ np.diag(np.sqrt(vals_top))
        return U0.reshape(-1)


class L2SDBPG(SDBPG):
    def __init__(self, *args, lambda_reg: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = float(lambda_reg)
        self.solver_mode = SOLVER_MODE

    def step(self):
        states = {i: ag.get_state() for i, ag in self.agents.items()}
        for i, ag in self.agents.items():
            ag.receive_from_neighbors({j: states[j] for j in ag.neighbors if j in states})
        for ag in self.agents.values():
            ag.dual_update()
        for ag in self.agents.values():
            grad_k = ag.compute_local_surrogate_gradient() + self.lambda_reg * ag.x
            ag.x = exact_quadratic_primal_step(ag.x, grad_k, ag.lambda_param, mu=1.0)
        for ag in self.agents.values():
            ag.z = exact_quadratic_smoothing_step(ag.x, ag.z, ag.beta, mu=1.0)


def _first_sustained_below(curve: np.ndarray, eps: float, hold: int) -> int:
    n = len(curve)
    hold = max(1, min(hold, n))
    mask = (curve <= eps).astype(np.int32)
    csum = np.cumsum(np.concatenate(([0], mask)))
    for i in range(0, n - hold + 1):
        if csum[i + hold] - csum[i] == hold:
            return i + 1
    return -1


def _run_one(algo, problem: MatrixFactorizationProblem, max_iter: int) -> Dict:
    grad_sq = []
    rec_err = []

    is_sdbpg = hasattr(algo, "agents")

    for _ in range(max_iter):
        algo.step()

        if is_sdbpg:
            xs = [ag.x for ag in algo.agents.values()]
        else:
            xs = algo.x

        x_bar = np.mean(xs, axis=0)
        g = problem.global_gradient(x_bar)
        g2 = float(np.dot(g, g))

        if np.isnan(g2) or g2 > DIVERGENCE_THRESHOLD:
            g2 = DIVERGENCE_THRESHOLD

        grad_sq.append(g2)
        rec_err.append(problem.recovery_error(x_bar))

    grad_sq = np.array(grad_sq)
    rec_err = np.array(rec_err)

    return {
        "g2": grad_sq,
        "run_min": np.minimum.accumulate(grad_sq),
        "rec_err": rec_err,
        "final_g2": float(np.minimum.accumulate(grad_sq)[-1]),
        "final_rec": float(rec_err[-1]),
        "hit_t": _first_sustained_below(grad_sq, EPS, HOLD),
        "diverged": bool(np.any(grad_sq >= DIVERGENCE_THRESHOLD)),
    }


def _initialize_algorithm(algo, problem: MatrixFactorizationProblem, x0: np.ndarray, rng: np.random.Generator):
    if hasattr(algo, "agents"):
        for ag in algo.agents.values():
            xi = x0 + INIT_PERTURB_STD * rng.standard_normal(problem.dim)
            ag.x = xi.copy()
            ag.z = xi.copy()
            for j in ag.neighbors:
                ag.y[j] = np.zeros(problem.dim)
            ag.neighbors_x = {}
            ag.neighbors_z = {}
        return

    if hasattr(algo, "x"):
        algo.x = [x0 + INIT_PERTURB_STD * rng.standard_normal(problem.dim) for _ in range(problem.n_agents)]
    if hasattr(algo, "y"):
        algo.y = [problem.local_gradient(i, algo.x[i]) for i in range(problem.n_agents)]


def _make_init_vector(problem: MatrixFactorizationProblem, rng: np.random.Generator, init_mode: str) -> np.ndarray:
    if init_mode == "spectral":
        return problem.spectral_init()
    x0 = rng.standard_normal(problem.dim)
    x0 /= max(np.linalg.norm(x0), 1e-12)
    return x0

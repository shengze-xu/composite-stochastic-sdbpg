"""
Smoothed Decentralized Bregman Proximal Gradient (SDBPG) Algorithm
Core components including Bregman divergence, relative smoothness, and optimization utilities
"""

import warnings
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


class BregmanDivergence:
    """
    Bregman divergence D_φ(x, y) = φ(x) - φ(y) - ⟨∇φ(y), x-y⟩
    """

    def __init__(
        self,
        phi: Callable,
        grad_phi: Callable,
        hess_phi: Optional[Callable] = None,
    ):
        """
        Args:
            phi: strongly convex function φ
            grad_phi: gradient of φ
            hess_phi: Hessian of φ (optional, for numerical stability)
        """
        self.phi = phi
        self.grad_phi = grad_phi
        self.hess_phi = hess_phi

    def divergence(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute D_φ(x, y)"""
        return self.phi(x) - self.phi(y) - np.dot(self.grad_phi(y), x - y)

    def prox_step(
        self,
        x: np.ndarray,
        gradient: np.ndarray,
        stepsize: float,
        constraint_set: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Numerically solve the smooth Bregman proximal subproblem

            argmin_z <gradient, z - x> + (1/stepsize) * D_phi(z, x).

        This BFGS-backed routine is intended only for smooth inner
        subproblems. Nonsmooth composite regularizers require an explicit
        proximal solver, and quadratic-Bregman mainline experiments use the
        exact helpers below instead of this numerical inner solver.
        """

        def objective(z):
            return np.dot(gradient, z - x) + (1.0 / stepsize) * self.divergence(z, x)

        # Use gradient-based optimization
        def grad_objective(z):
            return gradient + (1.0 / stepsize) * (self.grad_phi(z) - self.grad_phi(x))

        # Initial guess
        z0 = x.copy()

        # Optimize
        result = minimize(
            objective,
            z0,
            method="BFGS",
            jac=grad_objective,
            options={"gtol": 1e-6, "maxiter": 100},
        )

        z_opt = result.x

        # Apply constraint projection if provided
        if constraint_set is not None:
            z_opt = constraint_set(z_opt)

        return z_opt


def _as_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _validate_positive(name: str, value: float) -> float:
    val = float(value)
    if val <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return val


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise soft-thresholding operator."""
    arr = _as_float_array(x)
    threshold = max(float(tau), 0.0)
    return np.sign(arr) * np.maximum(np.abs(arr) - threshold, 0.0)


def exact_quadratic_primal_step(
    x: np.ndarray,
    gradient: np.ndarray,
    stepsize: float,
    mu: float = 1.0,
) -> np.ndarray:
    """
    Exact smooth Bregman proximal-gradient step for phi(x)=mu/2||x||^2.
    """
    step = _validate_positive("stepsize", stepsize)
    curvature = _validate_positive("mu", mu)
    return _as_float_array(x) - (step / curvature) * _as_float_array(gradient)


def exact_quadratic_smoothing_step(
    x_new: np.ndarray,
    z_old: np.ndarray,
    beta: float,
    mu: float = 1.0,
) -> np.ndarray:
    """
    Exact solution of min_z 1/2||z-x_new||^2 + beta D_phi(z,z_old)
    for phi(x)=mu/2||x||^2.
    """
    smoothing = _validate_positive("beta", beta)
    curvature = _validate_positive("mu", mu)
    denom = 1.0 + smoothing * curvature
    return (_as_float_array(x_new) + smoothing * curvature * _as_float_array(z_old)) / denom


def exact_l1_quadratic_prox_step(
    x: np.ndarray,
    gradient: np.ndarray,
    stepsize: float,
    lambda_reg: float,
    mu: float = 1.0,
) -> np.ndarray:
    """
    Exact L1 composite proximal step under quadratic Bregman geometry.
    """
    step = _validate_positive("stepsize", stepsize)
    curvature = _validate_positive("mu", mu)
    v = _as_float_array(x) - (step / curvature) * _as_float_array(gradient)
    tau = step * max(float(lambda_reg), 0.0) / curvature
    return soft_threshold(v, tau)


def exact_group_l21_quadratic_prox_step(
    x: np.ndarray,
    gradient: np.ndarray,
    stepsize: float,
    lambda_reg: float,
    feature_dim: int,
    n_tasks: int,
    mu: float = 1.0,
) -> np.ndarray:
    """
    Exact row-wise group L2,1 proximal step under quadratic geometry.
    """
    rows = int(feature_dim)
    cols = int(n_tasks)
    if rows <= 0 or cols <= 0:
        raise ValueError("feature_dim and n_tasks must be positive")
    step = _validate_positive("stepsize", stepsize)
    curvature = _validate_positive("mu", mu)
    v = _as_float_array(x) - (step / curvature) * _as_float_array(gradient)
    matrix = v.reshape(rows, cols)
    tau = step * max(float(lambda_reg), 0.0) / curvature
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    shrink = np.maximum(0.0, 1.0 - tau / np.maximum(row_norms, 1e-12))
    return (shrink * matrix).reshape(-1)


def quadratic_bregman(mu: float = 1.0) -> BregmanDivergence:
    """
    Standard quadratic Bregman divergence: φ(x) = (μ/2)||x||²
    """

    def phi(x):
        return 0.5 * mu * np.dot(x, x)

    def grad_phi(x):
        return mu * x

    def hess_phi(x):
        return mu * np.eye(len(x))

    return BregmanDivergence(phi, grad_phi, hess_phi)


def entropy_bregman(domain_dim: int) -> BregmanDivergence:
    """
    Entropy-based Bregman divergence for simplex constraints
    """

    def phi(x):
        # Add small epsilon for numerical stability
        eps = 1e-12
        x_safe = np.clip(x, eps, 1 - eps)
        return np.sum(x_safe * np.log(x_safe))

    def grad_phi(x):
        eps = 1e-12
        x_safe = np.clip(x, eps, 1 - eps)
        return np.log(x_safe) + 1

    return BregmanDivergence(phi, grad_phi)


class QuadraticRegressionProblem:
    """
    Decentralized quadratic regression: f_i(x) = (1/M) Σ_m (x^T A_i^m x - b_i^m)²
    """

    def __init__(self, n_agents: int, dim: int, samples_per_agent: int, seed: int = 42):
        """
        Args:
            n_agents: number of agents
            dim: problem dimension
            samples_per_agent: number of samples per agent
            seed: random seed
        """
        np.random.seed(seed)

        self.n_agents = n_agents
        self.dim = dim
        self.samples_per_agent = samples_per_agent

        # Generate ground truth
        self.x_star = np.random.randn(dim)
        self.x_star /= np.linalg.norm(self.x_star)  # normalize

        # Generate data for each agent
        self.A_matrices = []  # A_i^m matrices
        self.b_vectors = []  # b_i^m values

        for i in range(n_agents):
            A_i = []
            b_i = []

            for m in range(samples_per_agent):
                # Generate symmetric matrix with uniform[0,1] entries.
                # Normalise by dim so that spectral norm stays O(1) regardless
                # of d — otherwise gradient magnitude grows as O(d²) and causes
                # numerical blow-up for d >= 20.
                A_raw = np.random.uniform(0, 1, (dim, dim))
                A_im = (A_raw + A_raw.T) / (2 * dim)   # normalised symmetric matrix

                # Generate target value
                b_im = np.dot(self.x_star, np.dot(A_im, self.x_star))

                A_i.append(A_im)
                b_i.append(b_im)

            self.A_matrices.append(A_i)
            self.b_vectors.append(b_i)

    def local_objective(self, agent_id: int, x: np.ndarray) -> float:
        """Local objective function f_i(x)"""
        loss = 0.0
        for m in range(self.samples_per_agent):
            A_im = self.A_matrices[agent_id][m]
            b_im = self.b_vectors[agent_id][m]

            quad_form = np.dot(x, np.dot(A_im, x))
            loss += (quad_form - b_im) ** 2

        return loss / self.samples_per_agent

    def local_gradient(self, agent_id: int, x: np.ndarray) -> np.ndarray:
        """Local gradient ∇f_i(x)"""
        grad = np.zeros_like(x)

        for m in range(self.samples_per_agent):
            A_im = self.A_matrices[agent_id][m]
            b_im = self.b_vectors[agent_id][m]

            quad_form = np.dot(x, np.dot(A_im, x))
            # ∇[(x^T A x - b)²] = 4(x^T A x - b) A x
            grad += 4 * (quad_form - b_im) * np.dot(A_im, x)

        return grad / self.samples_per_agent

    def global_objective(self, x: np.ndarray) -> float:
        """Global objective (1/N) Σ_i f_i(x)"""
        return np.mean([self.local_objective(i, x) for i in range(self.n_agents)])


def generate_network_topology(
    n_agents: int,
    topology_type: str = "random",
    connectivity: float = 0.3,
    seed: int = 42,
) -> nx.Graph:
    """
    Generate network topology for decentralized optimization

    Args:
        n_agents: number of agents
        topology_type: 'random', 'ring', 'star', 'complete'
        connectivity: connection probability for random graphs
        seed: random seed
    """
    np.random.seed(seed)

    if topology_type == "random":
        # Ensure connectivity
        G = nx.erdos_renyi_graph(n_agents, connectivity, seed=seed)
        while not nx.is_connected(G):
            connectivity += 0.1
            G = nx.erdos_renyi_graph(n_agents, connectivity, seed=seed)

    elif topology_type == "ring":
        G = nx.cycle_graph(n_agents)

    elif topology_type == "star":
        G = nx.star_graph(n_agents - 1)

    elif topology_type == "complete":
        G = nx.complete_graph(n_agents)

    else:
        raise ValueError(f"Unknown topology type: {topology_type}")

    return G


def compute_consensus_gap(x_list: List[np.ndarray], graph: nx.Graph) -> float:
    """
    Compute consensus gap: (1/|E|) Σ_{(i,j)∈E} ||x_i - x_j||
    """
    total_gap = 0.0
    edge_count = 0

    for i, j in graph.edges():
        total_gap += np.linalg.norm(x_list[i] - x_list[j])
        edge_count += 1

    return total_gap / edge_count if edge_count > 0 else 0.0


def compute_optimality_gap(x_list: List[np.ndarray], problem: QuadraticRegressionProblem) -> float:
    """
    Compute optimality gap: (1/N) Σ_i ||∇f_i(x̄)||, where x̄ = (1/N) Σ_i x_i
    """
    x_bar = np.mean(x_list, axis=0)

    grad_norms = []
    for i in range(problem.n_agents):
        grad_i = problem.local_gradient(i, x_bar)
        grad_norms.append(np.linalg.norm(grad_i))

    return np.mean(grad_norms)

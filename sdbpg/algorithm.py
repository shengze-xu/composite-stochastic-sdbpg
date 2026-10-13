"""
Smoothed Decentralized Bregman Proximal Gradient (SDBPG) Algorithm Implementation
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .core import (
    BregmanDivergence,
    QuadraticRegressionProblem,
    compute_consensus_gap,
    compute_optimality_gap,
    quadratic_bregman,
)


class SDBPGAgent:
    """Single agent in the SDBPG algorithm"""

    def __init__(
        self,
        agent_id: int,
        dim: int,
        neighbors: List[int],
        bregman: BregmanDivergence,
        problem: QuadraticRegressionProblem,
        rho: float = 1.0,
        alpha: float = 0.1,
        lambda_param: float = 0.1,
        beta: float = 0.5,
        C: float = 1.0,
    ):
        """
        Args:
            agent_id: unique agent identifier
            dim: problem dimension
            neighbors: list of neighbor agent IDs
            bregman: Bregman divergence object
            problem: optimization problem
            rho: penalty parameter for consensus
            alpha: dual update stepsize
            lambda_param: primal update stepsize
            beta: smoothing parameter (0 < β ≤ 1)
            C: regularization parameter
        """
        self.agent_id = agent_id
        self.dim = dim
        self.neighbors = neighbors
        self.bregman = bregman
        self.problem = problem

        # Algorithm parameters
        self.rho = rho
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.beta = beta
        self.C = C

        # State variables
        self.x = np.random.randn(dim) * 0.1  # primal variable
        self.z = self.x.copy()  # smoothing variable
        self.y = {j: np.zeros(dim) for j in neighbors}  # dual variables

        # For communication
        self.neighbors_x = {}  # received x from neighbors
        self.neighbors_z = {}  # received z from neighbors

    def receive_from_neighbors(self, neighbors_data: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """Receive (x, z) from neighbors"""
        for neighbor_id, (x_j, z_j) in neighbors_data.items():
            if neighbor_id in self.neighbors:
                self.neighbors_x[neighbor_id] = x_j.copy()
                self.neighbors_z[neighbor_id] = z_j.copy()

    def dual_update(self):
        """Step 1: Dual variable update"""
        for j in self.neighbors:
            if j in self.neighbors_x:
                self.y[j] = self.y[j] + self.alpha * (self.x - self.neighbors_x[j])

    def compute_local_surrogate_gradient(self) -> np.ndarray:
        """
        Compute gradient of local surrogate function K_i(x_i, Y_i, Z_i)
        K_i(x) = f_i(x) + C*D_φ(x, z_i) + ρ*Σ_j D_φ(x, z_j) + Σ_j y_ij^T x
        """
        # Original function gradient
        grad_f = self.problem.local_gradient(self.agent_id, self.x)

        # Bregman divergence gradients
        grad_bregman = self.C * (self.bregman.grad_phi(self.x) - self.bregman.grad_phi(self.z))

        # Consensus Bregman terms
        for j in self.neighbors:
            if j in self.neighbors_z:
                grad_bregman += self.rho * (
                    self.bregman.grad_phi(self.x) - self.bregman.grad_phi(self.neighbors_z[j])
                )

        # Dual variable terms
        grad_dual = np.zeros(self.dim)
        for j in self.neighbors:
            grad_dual += self.y[j]

        return grad_f + grad_bregman + grad_dual

    def primal_update(self):
        """Step 2: Primal variable update (one BPG step)"""
        grad_K = self.compute_local_surrogate_gradient()
        self.x = self.bregman.prox_step(self.x, grad_K, self.lambda_param)

    def smoothing_update(self):
        """
        Step 3: Smoothing variable update
        z^{t+1} = argmin_z (1/2)||z - x^{t+1}||² + β*D_φ(z, z^t)
        """

        def grad_objective(z):
            return (z - self.x) + self.beta * (self.bregman.grad_phi(z) - self.bregman.grad_phi(self.z))

        # Simple gradient descent for smoothing update
        z_new = self.z.copy()
        lr = 0.01
        for _ in range(50):  # few iterations should suffice
            grad = grad_objective(z_new)
            z_new = z_new - lr * grad
            if np.linalg.norm(grad) < 1e-6:
                break

        self.z = z_new

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current (x, z) for broadcasting"""
        return self.x.copy(), self.z.copy()


class SDBPG:
    """Smoothed Decentralized Bregman Proximal Gradient Algorithm"""

    def __init__(
        self,
        problem: QuadraticRegressionProblem,
        graph: nx.Graph,
        bregman: Optional[BregmanDivergence] = None,
        rho: float = 1.0,
        alpha: float = 0.1,
        lambda_param: float = 0.1,
        beta: float = 0.5,
        C: float = 1.0,
    ):
        """
        Args:
            problem: optimization problem
            graph: communication graph
            bregman: Bregman divergence (default: quadratic)
            rho, alpha, lambda_param, beta, C: algorithm parameters
        """
        self.problem = problem
        self.graph = graph
        self.bregman = bregman or quadratic_bregman(mu=1.0)

        # Create agents
        self.agents = {}
        for i in range(problem.n_agents):
            neighbors = list(graph.neighbors(i))
            self.agents[i] = SDBPGAgent(
                agent_id=i,
                dim=problem.dim,
                neighbors=neighbors,
                bregman=self.bregman,
                problem=problem,
                rho=rho,
                alpha=alpha,
                lambda_param=lambda_param,
                beta=beta,
                C=C,
            )

        # History tracking
        self.history = {"consensus_gap": [], "optimality_gap": [], "objective_values": []}

    def step(self):
        """One iteration of SDBPG algorithm"""
        # Step 0: Broadcast current states
        agent_states = {i: agent.get_state() for i, agent in self.agents.items()}

        # Each agent receives from neighbors
        for i, agent in self.agents.items():
            neighbors_data = {j: agent_states[j] for j in agent.neighbors if j in agent_states}
            agent.receive_from_neighbors(neighbors_data)

        # Step 1: Dual update (in parallel)
        for agent in self.agents.values():
            agent.dual_update()

        # Step 2: Primal update (in parallel)
        for agent in self.agents.values():
            agent.primal_update()

        # Step 3: Smoothing update (in parallel)
        for agent in self.agents.values():
            agent.smoothing_update()

    def run(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = True,
        record_history: bool = True,
    ) -> Dict:
        """
        Run SDBPG algorithm

        Returns:
            dict with final states and convergence history
        """
        for iteration in range(max_iterations):
            self.step()

            if record_history:
                # Collect current x values
                x_list = [agent.x for agent in self.agents.values()]

                # Compute metrics
                consensus_gap = compute_consensus_gap(x_list, self.graph)
                optimality_gap = compute_optimality_gap(x_list, self.problem)

                # Average objective value
                x_avg = np.mean(x_list, axis=0)
                obj_val = self.problem.global_objective(x_avg)

                self.history["consensus_gap"].append(consensus_gap)
                self.history["optimality_gap"].append(optimality_gap)
                self.history["objective_values"].append(obj_val)

                # Check convergence
                if consensus_gap < tolerance and optimality_gap < tolerance:
                    if verbose:
                        print(f"SDBPG converged at iteration {iteration}")
                    break

                if verbose and iteration % 50 == 0:
                    print(
                        f"Iter {iteration}: Consensus={consensus_gap:.2e}, "
                        f"Optimality={optimality_gap:.2e}, Obj={obj_val:.2e}"
                    )

        # Return final results
        final_x = [agent.x for agent in self.agents.values()]
        final_z = [agent.z for agent in self.agents.values()]

        return {
            "x_final": final_x,
            "z_final": final_z,
            "x_average": np.mean(final_x, axis=0),
            "history": self.history,
            "iterations": iteration + 1,
        }


# Baseline algorithms for comparison
def _metropolis_mixing_matrix(graph: nx.Graph, n_agents: int) -> np.ndarray:
    """Symmetric doubly-stochastic Metropolis matrix for an undirected graph."""
    W = np.zeros((n_agents, n_agents), dtype=float)
    degrees = dict(graph.degree())
    for i, j in graph.edges():
        weight = 1.0 / (1.0 + max(float(degrees[i]), float(degrees[j])))
        W[i, j] = weight
        W[j, i] = weight
    for i in range(n_agents):
        W[i, i] = 1.0 - float(np.sum(W[i]))
    return W


def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.sign(arr) * np.maximum(np.abs(arr) - float(tau), 0.0)


class DecentralizedGradientDescent:
    """Decentralized Gradient Descent (DGD) baseline"""

    def __init__(
        self,
        problem: QuadraticRegressionProblem,
        graph: nx.Graph,
        stepsize: float = 0.01,
        consensus_weight: float = 0.5,
    ):
        self.problem = problem
        self.graph = graph
        self.stepsize = stepsize
        self.consensus_weight = consensus_weight

        # Initialize agents
        self.x = [np.random.randn(problem.dim) * 0.1 for _ in range(problem.n_agents)]
        self.history = {"consensus_gap": [], "optimality_gap": [], "objective_values": []}

    def step(self):
        """One DGD step"""
        x_new = []

        for i in range(self.problem.n_agents):
            # Gradient step
            grad_i = self.problem.local_gradient(i, self.x[i])
            x_i_new = self.x[i] - self.stepsize * grad_i

            # Consensus averaging with neighbors
            neighbors = list(self.graph.neighbors(i))
            if neighbors:
                neighbor_avg = np.mean([self.x[j] for j in neighbors], axis=0)
                x_i_new = (1 - self.consensus_weight) * x_i_new + self.consensus_weight * neighbor_avg

            x_new.append(x_i_new)

        self.x = x_new

    def run(self, max_iterations: int = 1000, tolerance: float = 1e-6, verbose: bool = True) -> Dict:
        """Run DGD algorithm"""
        for iteration in range(max_iterations):
            self.step()

            # Compute metrics
            consensus_gap = compute_consensus_gap(self.x, self.graph)
            optimality_gap = compute_optimality_gap(self.x, self.problem)

            x_avg = np.mean(self.x, axis=0)
            obj_val = self.problem.global_objective(x_avg)

            self.history["consensus_gap"].append(consensus_gap)
            self.history["optimality_gap"].append(optimality_gap)
            self.history["objective_values"].append(obj_val)

            # Check convergence
            if consensus_gap < tolerance and optimality_gap < tolerance:
                if verbose:
                    print(f"DGD converged at iteration {iteration}")
                break

            if verbose and iteration % 50 == 0:
                print(
                    f"DGD Iter {iteration}: Consensus={consensus_gap:.2e}, "
                    f"Optimality={optimality_gap:.2e}, Obj={obj_val:.2e}"
                )

        return {
            "x_final": self.x,
            "x_average": np.mean(self.x, axis=0),
            "history": self.history,
            "iterations": iteration + 1,
        }


class DecentralizedGradientTracking:
    """Decentralized Gradient Tracking (DGT) baseline"""

    def __init__(self, problem: QuadraticRegressionProblem, graph: nx.Graph, stepsize: float = 0.01):
        self.problem = problem
        self.graph = graph
        self.stepsize = stepsize

        # Initialize variables
        self.x = [np.random.randn(problem.dim) * 0.1 for _ in range(problem.n_agents)]
        self.y = [self.problem.local_gradient(i, self.x[i]) for i in range(problem.n_agents)]

        self.history = {"consensus_gap": [], "optimality_gap": [], "objective_values": []}

    def step(self):
        """One DGT step"""
        W = _metropolis_mixing_matrix(self.graph, self.problem.n_agents)
        x_old = [x.copy() for x in self.x]
        y_old = [y.copy() for y in self.y]

        x_new = []
        for i in range(self.problem.n_agents):
            mixed_x = sum(W[i, j] * x_old[j] for j in range(self.problem.n_agents))
            x_new.append(mixed_x - self.stepsize * y_old[i])

        y_new = []
        for i in range(self.problem.n_agents):
            mixed_y = sum(W[i, j] * y_old[j] for j in range(self.problem.n_agents))
            grad_new = self.problem.local_gradient(i, x_new[i])
            grad_old = self.problem.local_gradient(i, x_old[i])
            y_new.append(mixed_y + grad_new - grad_old)

        self.x = x_new
        self.y = y_new

    def run(self, max_iterations: int = 1000, tolerance: float = 1e-6, verbose: bool = True) -> Dict:
        """Run DGT algorithm"""
        for iteration in range(max_iterations):
            self.step()

            # Compute metrics
            consensus_gap = compute_consensus_gap(self.x, self.graph)
            optimality_gap = compute_optimality_gap(self.x, self.problem)

            x_avg = np.mean(self.x, axis=0)
            obj_val = self.problem.global_objective(x_avg)

            self.history["consensus_gap"].append(consensus_gap)
            self.history["optimality_gap"].append(optimality_gap)
            self.history["objective_values"].append(obj_val)

            # Check convergence
            if consensus_gap < tolerance and optimality_gap < tolerance:
                if verbose:
                    print(f"DGT converged at iteration {iteration}")
                break

            if verbose and iteration % 50 == 0:
                print(
                    f"DGT Iter {iteration}: Consensus={consensus_gap:.2e}, "
                    f"Optimality={optimality_gap:.2e}, Obj={obj_val:.2e}"
                )

        return {
            "x_final": self.x,
            "x_average": np.mean(self.x, axis=0),
            "history": self.history,
            "iterations": iteration + 1,
        }


class ProximalDecentralizedGradientDescent:
    """L1-proximal DGD baseline for composite sparse objectives."""

    def __init__(
        self,
        problem: QuadraticRegressionProblem,
        graph: nx.Graph,
        stepsize: float = 0.01,
        consensus_weight: float = 0.5,
        lambda_reg: float = 0.0,
    ):
        self.problem = problem
        self.graph = graph
        self.stepsize = float(stepsize)
        self.consensus_weight = float(consensus_weight)
        self.lambda_reg = float(lambda_reg)
        self.x = [np.random.randn(problem.dim) * 0.1 for _ in range(problem.n_agents)]
        self.history = {"consensus_gap": [], "optimality_gap": [], "objective_values": []}

    def step(self):
        x_old = [x.copy() for x in self.x]
        x_new = []
        for i in range(self.problem.n_agents):
            grad_i = self.problem.local_gradient(i, x_old[i])
            neighbors = list(self.graph.neighbors(i))
            if neighbors:
                neighbor_avg = np.mean([x_old[j] for j in neighbors], axis=0)
                mixed_x = (1.0 - self.consensus_weight) * x_old[i] + self.consensus_weight * neighbor_avg
            else:
                mixed_x = x_old[i]
            v = mixed_x - self.stepsize * grad_i
            x_new.append(_soft_threshold(v, self.stepsize * self.lambda_reg))
        self.x = x_new

    def run(self, max_iterations: int = 1000, tolerance: float = 1e-6, verbose: bool = True) -> Dict:
        for iteration in range(max_iterations):
            self.step()
            consensus_gap = compute_consensus_gap(self.x, self.graph)
            optimality_gap = compute_optimality_gap(self.x, self.problem)
            x_avg = np.mean(self.x, axis=0)
            obj_val = self.problem.global_objective(x_avg) + self.lambda_reg * float(np.linalg.norm(x_avg, 1))
            self.history["consensus_gap"].append(consensus_gap)
            self.history["optimality_gap"].append(optimality_gap)
            self.history["objective_values"].append(obj_val)
            if consensus_gap < tolerance and optimality_gap < tolerance:
                if verbose:
                    print(f"ProxDGD converged at iteration {iteration}")
                break
        return {
            "x_final": self.x,
            "x_average": np.mean(self.x, axis=0),
            "history": self.history,
            "iterations": iteration + 1,
        }


class ProximalDecentralizedGradientTracking:
    """L1-proximal gradient-tracking baseline for composite sparse objectives."""

    def __init__(
        self,
        problem: QuadraticRegressionProblem,
        graph: nx.Graph,
        stepsize: float = 0.01,
        lambda_reg: float = 0.0,
    ):
        self.problem = problem
        self.graph = graph
        self.stepsize = float(stepsize)
        self.lambda_reg = float(lambda_reg)
        self.x = [np.random.randn(problem.dim) * 0.1 for _ in range(problem.n_agents)]
        self.y = [self.problem.local_gradient(i, self.x[i]) for i in range(problem.n_agents)]
        self.history = {"consensus_gap": [], "optimality_gap": [], "objective_values": []}

    def step(self):
        W = _metropolis_mixing_matrix(self.graph, self.problem.n_agents)
        x_old = [x.copy() for x in self.x]
        y_old = [y.copy() for y in self.y]

        x_new = []
        for i in range(self.problem.n_agents):
            mixed_x = sum(W[i, j] * x_old[j] for j in range(self.problem.n_agents))
            v = mixed_x - self.stepsize * y_old[i]
            x_new.append(_soft_threshold(v, self.stepsize * self.lambda_reg))

        y_new = []
        for i in range(self.problem.n_agents):
            mixed_y = sum(W[i, j] * y_old[j] for j in range(self.problem.n_agents))
            grad_new = self.problem.local_gradient(i, x_new[i])
            grad_old = self.problem.local_gradient(i, x_old[i])
            y_new.append(mixed_y + grad_new - grad_old)

        self.x = x_new
        self.y = y_new

    def run(self, max_iterations: int = 1000, tolerance: float = 1e-6, verbose: bool = True) -> Dict:
        for iteration in range(max_iterations):
            self.step()
            consensus_gap = compute_consensus_gap(self.x, self.graph)
            optimality_gap = compute_optimality_gap(self.x, self.problem)
            x_avg = np.mean(self.x, axis=0)
            obj_val = self.problem.global_objective(x_avg) + self.lambda_reg * float(np.linalg.norm(x_avg, 1))
            self.history["consensus_gap"].append(consensus_gap)
            self.history["optimality_gap"].append(optimality_gap)
            self.history["objective_values"].append(obj_val)
            if consensus_gap < tolerance and optimality_gap < tolerance:
                if verbose:
                    print(f"ProxDGT converged at iteration {iteration}")
                break
        return {
            "x_final": self.x,
            "x_average": np.mean(self.x, axis=0),
            "history": self.history,
            "iterations": iteration + 1,
        }

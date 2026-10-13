"""Fourier numerical routines."""

from __future__ import annotations

import time


from dataclasses import dataclass


from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Tuple


import networkx as nx


import numpy as np


import pandas as pd


import pywt


from scipy.ndimage import gaussian_filter


from skimage import data as skimage_data


from skimage.transform import resize


from sdbpg.algorithm import ProximalDecentralizedGradientDescent


from sdbpg.core import exact_l1_quadratic_prox_step
from sdbpg.core import exact_quadratic_smoothing_step


from experiments._impl import sparse as sr


SDBPG_BASE = {
    "rho": 2.0,
    "alpha": 0.02,
    "lambda_param": 0.015,
    "beta": 0.85,
    "C": 1.5,
}


def modified_shepp_logan(size: int) -> np.ndarray:
    """Generate a normalized modified Shepp--Logan phantom."""
    ellipses = (
        (1.0, 0.6900, 0.9200, 0.0000, 0.0000, 0.0),
        (-0.8, 0.6624, 0.8740, 0.0000, -0.0184, 0.0),
        (-0.2, 0.1100, 0.3100, 0.2200, 0.0000, -18.0),
        (-0.2, 0.1600, 0.4100, -0.2200, 0.0000, 18.0),
        (0.1, 0.2100, 0.2500, 0.0000, 0.3500, 0.0),
        (0.1, 0.0460, 0.0460, 0.0000, 0.1000, 0.0),
        (0.1, 0.0460, 0.0460, 0.0000, -0.1000, 0.0),
        (0.1, 0.0460, 0.0230, -0.0800, -0.6050, 0.0),
        (0.1, 0.0230, 0.0230, 0.0000, -0.6060, 0.0),
        (0.1, 0.0230, 0.0460, 0.0600, -0.6050, 0.0),
    )
    axis = np.linspace(-1.0, 1.0, int(size), dtype=float)
    xx, yy = np.meshgrid(axis, axis)
    image = np.zeros((size, size), dtype=float)
    for amplitude, a, b, x0, y0, angle_deg in ellipses:
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_shift = xx - x0
        y_shift = yy - y0
        x_rot = x_shift * cos_t + y_shift * sin_t
        y_rot = -x_shift * sin_t + y_shift * cos_t
        image[(x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0] += amplitude
    image -= float(image.min())
    peak = float(image.max())
    if peak > 0.0:
        image /= peak
    return image


@dataclass(frozen=True)
class FourierImagingConfig:
    image_name: str = "phantom"
    image_size: int = 64
    n_agents: int = 6
    sampling_rate: float = 0.35
    center_fraction: float = 0.08
    relative_noise: float = 0.01
    wavelet: str = "db2"
    wavelet_level: int = 3

    @property
    def dim(self) -> int:
        return int(self.image_size * self.image_size)


class OrthonormalWavelet2D:
    """Array-backed orthonormal 2-D wavelet analysis and synthesis."""

    def __init__(self, size: int, wavelet: str, level: int):
        self.size = int(size)
        self.wavelet = str(wavelet)
        self.level = int(level)
        zeros = np.zeros((self.size, self.size), dtype=float)
        coeffs = pywt.wavedec2(
            zeros,
            wavelet=self.wavelet,
            mode="periodization",
            level=self.level,
        )
        array, self.slices = pywt.coeffs_to_array(coeffs)
        if array.shape != zeros.shape:
            raise ValueError(f"wavelet coefficient shape {array.shape} does not match image shape {zeros.shape}")

    def analysis(self, image: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec2(
            np.asarray(image, dtype=float),
            wavelet=self.wavelet,
            mode="periodization",
            level=self.level,
        )
        array, _ = pywt.coeffs_to_array(coeffs)
        return np.asarray(array, dtype=float).reshape(-1)

    def synthesis(self, coefficients: np.ndarray) -> np.ndarray:
        array = np.asarray(coefficients, dtype=float).reshape(self.size, self.size)
        coeffs = pywt.array_to_coeffs(array, self.slices, output_format="wavedec2")
        image = pywt.waverec2(coeffs, wavelet=self.wavelet, mode="periodization")
        return np.asarray(image[: self.size, : self.size], dtype=float)


def variable_density_mask(
    size: int,
    sampling_rate: float,
    center_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a fixed-cardinality variable-density Cartesian Fourier mask."""
    n = int(size)
    total = n * n
    target = min(total, max(1, int(round(float(sampling_rate) * total))))
    axis = np.arange(n, dtype=float) - n / 2.0
    xx, yy = np.meshgrid(axis, axis)
    radius = np.sqrt(xx**2 + yy**2) / max(n / np.sqrt(2.0), 1.0)
    weights = np.exp(-5.0 * radius**2).reshape(-1)

    center_width = max(1, int(round(float(center_fraction) * n)))
    if center_width % 2 == 1:
        center_width += 1
    lo = max(0, n // 2 - center_width // 2)
    hi = min(n, lo + center_width)
    centered = np.zeros((n, n), dtype=bool)
    centered[lo:hi, lo:hi] = True
    selected = list(np.flatnonzero(centered.reshape(-1)))

    remaining = np.flatnonzero(~centered.reshape(-1))
    need = max(0, target - len(selected))
    if need > 0:
        probabilities = weights[remaining]
        probabilities /= probabilities.sum()
        selected.extend(rng.choice(remaining, size=need, replace=False, p=probabilities).tolist())

    centered.reshape(-1)[np.asarray(selected, dtype=int)] = True
    return np.fft.ifftshift(centered)


def reference_image(name: str, size: int) -> np.ndarray:
    normalized = str(name).strip().lower()
    if normalized == "phantom":
        return modified_shepp_logan(int(size))
    if normalized == "cameraman":
        image = np.asarray(skimage_data.camera(), dtype=float) / 255.0
        return np.asarray(
            resize(
                image,
                (int(size), int(size)),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            ),
            dtype=float,
        )
    raise ValueError(f"unknown image name: {name!r}")


class DistributedFourierWaveletProblem:
    """Partial-Fourier inverse problem with measurements split across agents."""

    def __init__(self, config: FourierImagingConfig, seed: int):
        self.n_agents = int(config.n_agents)
        self.dim = int(config.dim)
        self.image_size = int(config.image_size)
        self.transform = OrthonormalWavelet2D(
            self.image_size,
            config.wavelet,
            config.wavelet_level,
        )
        self.image_name = str(config.image_name)
        self.image_star = reference_image(self.image_name, self.image_size)
        self.x_star = self.transform.analysis(self.image_star)

        rng = np.random.default_rng(int(seed))
        self.mask = variable_density_mask(
            self.image_size,
            config.sampling_rate,
            config.center_fraction,
            rng,
        )
        selected = np.flatnonzero(self.mask.reshape(-1))
        rng.shuffle(selected)
        partitions = np.array_split(selected, self.n_agents)
        if min(len(part) for part in partitions) == 0:
            raise ValueError("every agent must receive at least one Fourier sample")
        self.indices = [np.asarray(part, dtype=int) for part in partitions]

        kspace = np.fft.fft2(self.image_star, norm="ortho").reshape(-1)
        measured = kspace[selected]
        noise_scale = float(config.relative_noise) * max(float(np.sqrt(np.mean(np.abs(measured) ** 2))), 1e-12)
        noise = noise_scale * (
            rng.standard_normal(selected.size) + 1j * rng.standard_normal(selected.size)
        ) / np.sqrt(2.0)
        noisy = measured + noise
        observation_by_index = {int(index): value for index, value in zip(selected, noisy)}
        self.bs = [np.asarray([observation_by_index[int(index)] for index in part]) for part in self.indices]

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        return self.transform.synthesis(np.asarray(coefficients, dtype=float))

    def zero_filled_reconstruction(self) -> np.ndarray:
        kspace = np.zeros(self.dim, dtype=complex)
        for indices, observations in zip(self.indices, self.bs):
            kspace[indices] = observations
        return np.real(
            np.fft.ifft2(kspace.reshape(self.image_size, self.image_size), norm="ortho")
        )

    def _forward(self, coefficients: np.ndarray, indices: np.ndarray) -> np.ndarray:
        image = self.reconstruct(coefficients)
        return np.fft.fft2(image, norm="ortho").reshape(-1)[indices]

    def local_gradient(self, agent_id: int, x: np.ndarray) -> np.ndarray:
        i = int(agent_id)
        residual = self._forward(x, self.indices[i]) - self.bs[i]
        grid = np.zeros(self.dim, dtype=complex)
        grid[self.indices[i]] = residual
        image_gradient = np.real(
            np.fft.ifft2(grid.reshape(self.image_size, self.image_size), norm="ortho")
        )
        return float(self.n_agents) * self.transform.analysis(image_gradient)

    def global_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.mean([self.local_gradient(i, x) for i in range(self.n_agents)], axis=0)

    def local_objective(self, agent_id: int, x: np.ndarray) -> float:
        i = int(agent_id)
        residual = self._forward(x, self.indices[i]) - self.bs[i]
        return float(0.5 * self.n_agents * np.vdot(residual, residual).real)

    def global_objective(self, x: np.ndarray) -> float:
        return float(np.mean([self.local_objective(i, x) for i in range(self.n_agents)]))

    def lipschitz_upper_bound(self) -> float:
        return 1.0


def imaging_graph(n_agents: int) -> nx.Graph:
    graph = nx.cycle_graph(int(n_agents))
    if n_agents >= 6:
        graph.add_edge(0, n_agents // 2)
        graph.add_edge(1, (n_agents // 2) + 1)
    return graph


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    data_range = max(float(ref.max() - ref.min()), 1e-12)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_ref = gaussian_filter(ref, sigma=1.0)
    mu_est = gaussian_filter(est, sigma=1.0)
    var_ref = gaussian_filter(ref * ref, sigma=1.0) - mu_ref * mu_ref
    var_est = gaussian_filter(est * est, sigma=1.0) - mu_est * mu_est
    cov = gaussian_filter(ref * est, sigma=1.0) - mu_ref * mu_est
    numerator = (2.0 * mu_ref * mu_est + c1) * (2.0 * cov + c2)
    denominator = (mu_ref**2 + mu_est**2 + c1) * (var_ref + var_est + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-15)))


def image_metrics(problem: DistributedFourierWaveletProblem, coefficients: np.ndarray) -> Dict[str, float]:
    reconstruction = problem.reconstruct(coefficients)
    reference = problem.image_star
    error = reconstruction - reference
    nrmse = float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-12))
    rmse = float(np.sqrt(np.mean(error**2)))
    data_range = max(float(reference.max() - reference.min()), 1e-12)
    return {
        "nrmse": nrmse,
        "psnr": float(20.0 * np.log10(data_range / max(rmse, 1e-12))),
        "ssim": ssim(reference, reconstruction),
    }


class L1SDBPG(sr.SparseL1SDBPG):
    """Wavelet L1-SDBPG that retains exact subgradient diagnostics."""

    def step(self) -> None:
        states = {i: agent.get_state() for i, agent in self.agents.items()}
        for agent in self.agents.values():
            agent.receive_from_neighbors({j: states[j] for j in agent.neighbors if j in states})
        for agent in self.agents.values():
            agent.dual_update()

        x_old = {i: agent.x.copy() for i, agent in self.agents.items()}
        z_old = {i: agent.z.copy() for i, agent in self.agents.items()}
        selected: List[np.ndarray] = []
        for i, agent in self.agents.items():
            model_gradient = agent.compute_local_surrogate_gradient()
            step = float(agent.lambda_param)
            x_next = exact_l1_quadratic_prox_step(
                agent.x,
                model_gradient,
                step,
                self.lambda_reg,
                mu=1.0,
            )
            selected.append(-model_gradient - (x_next - agent.x) / step)
            agent.x = x_next
        for agent in self.agents.values():
            agent.z = exact_quadratic_smoothing_step(agent.x, agent.z, agent.beta, mu=1.0)

        self.last_selected_subgradient = np.stack(selected)
        self.last_dx = 0.5 * float(
            sum(np.linalg.norm(agent.x - x_old[i]) ** 2 for i, agent in self.agents.items())
        )
        self.last_dz = 0.5 * float(
            sum(np.linalg.norm(agent.z - z_old[i]) ** 2 for i, agent in self.agents.items())
        )


class CachedProxDGT:
    """L1 ProxDGT with cached old gradients for fair oracle accounting."""

    def __init__(self, problem: Any, graph: nx.Graph, stepsize: float, lambda_reg: float):
        self.problem = problem
        self.graph = graph
        self.stepsize = float(stepsize)
        self.lambda_reg = float(lambda_reg)
        self.W = metropolis_matrix(graph, problem.n_agents)
        self.x = [np.zeros(problem.dim, dtype=float) for _ in range(problem.n_agents)]
        self.grad = [problem.local_gradient(i, self.x[i]) for i in range(problem.n_agents)]
        self.y = [gradient.copy() for gradient in self.grad]

    def step(self) -> None:
        x_old = [x.copy() for x in self.x]
        y_old = [y.copy() for y in self.y]
        x_new = []
        for i in range(self.problem.n_agents):
            mixed = sum(self.W[i, j] * x_old[j] for j in range(self.problem.n_agents))
            x_new.append(sr.soft_threshold(mixed - self.stepsize * y_old[i], self.stepsize * self.lambda_reg))
        grad_new = [self.problem.local_gradient(i, x_new[i]) for i in range(self.problem.n_agents)]
        y_new = []
        for i in range(self.problem.n_agents):
            mixed_y = sum(self.W[i, j] * y_old[j] for j in range(self.problem.n_agents))
            y_new.append(mixed_y + grad_new[i] - self.grad[i])
        self.x = x_new
        self.y = y_new
        self.grad = grad_new


class ProxPDA:
    """Linearized proximal primal-dual augmented-consensus comparator."""

    def __init__(self, problem: Any, graph: nx.Graph, stepsize: float, penalty: float, lambda_reg: float):
        self.problem = problem
        self.graph = graph
        self.stepsize = float(stepsize)
        self.penalty = float(penalty)
        self.lambda_reg = float(lambda_reg)
        self.x = [np.zeros(problem.dim, dtype=float) for _ in range(problem.n_agents)]
        self.y = {
            i: {j: np.zeros(problem.dim, dtype=float) for j in graph.neighbors(i)}
            for i in range(problem.n_agents)
        }

    def step(self) -> None:
        x_old = [x.copy() for x in self.x]
        for i, j in self.graph.edges():
            difference = x_old[i] - x_old[j]
            self.y[i][j] += self.penalty * difference
            self.y[j][i] -= self.penalty * difference
        x_new = []
        for i in range(self.problem.n_agents):
            gradient = self.problem.local_gradient(i, x_old[i])
            dual = sum(self.y[i].values()) if self.y[i] else np.zeros_like(x_old[i])
            augmented = sum((x_old[i] - x_old[j] for j in self.graph.neighbors(i)), np.zeros_like(x_old[i]))
            candidate = x_old[i] - self.stepsize * (gradient + dual + self.penalty * augmented)
            x_new.append(sr.soft_threshold(candidate, self.stepsize * self.lambda_reg))
        self.x = x_new


class ProxExtra:
    """PG-EXTRA-style L1 comparator with one state vector communicated."""

    def __init__(self, problem: Any, graph: nx.Graph, stepsize: float, lambda_reg: float):
        self.problem = problem
        self.graph = graph
        self.stepsize = float(stepsize)
        self.lambda_reg = float(lambda_reg)
        self.W = metropolis_matrix(graph, problem.n_agents)
        self.W_tilde = 0.5 * (np.eye(problem.n_agents) + self.W)
        self.x = [np.zeros(problem.dim, dtype=float) for _ in range(problem.n_agents)]
        self.x_prev: List[np.ndarray] | None = None
        self.grad_prev: List[np.ndarray] | None = None

    def step(self) -> None:
        x_old = [x.copy() for x in self.x]
        grad_old = [self.problem.local_gradient(i, x_old[i]) for i in range(self.problem.n_agents)]
        if self.x_prev is None or self.grad_prev is None:
            x_new = []
            for i in range(self.problem.n_agents):
                mixed = sum(self.W[i, j] * x_old[j] for j in range(self.problem.n_agents))
                x_new.append(sr.soft_threshold(mixed - self.stepsize * grad_old[i], self.stepsize * self.lambda_reg))
            self.x_prev = x_old
            self.grad_prev = grad_old
            self.x = x_new
            return
        x_new = []
        for i in range(self.problem.n_agents):
            mixed_now = x_old[i] + sum(self.W[i, j] * x_old[j] for j in range(self.problem.n_agents))
            mixed_prev = sum(self.W_tilde[i, j] * self.x_prev[j] for j in range(self.problem.n_agents))
            candidate = mixed_now - mixed_prev - self.stepsize * (grad_old[i] - self.grad_prev[i])
            x_new.append(sr.soft_threshold(candidate, self.stepsize * self.lambda_reg))
        self.x_prev = x_old
        self.grad_prev = grad_old
        self.x = x_new


def metropolis_matrix(graph: nx.Graph, n_agents: int) -> np.ndarray:
    matrix = np.zeros((n_agents, n_agents), dtype=float)
    degrees = dict(graph.degree())
    for i, j in graph.edges():
        weight = 1.0 / (1.0 + max(float(degrees[i]), float(degrees[j])))
        matrix[i, j] = weight
        matrix[j, i] = weight
    for i in range(n_agents):
        matrix[i, i] = 1.0 - float(np.sum(matrix[i]))
    return matrix


def initialize_algorithm(algorithm: Any, problem: DistributedFourierWaveletProblem) -> None:
    zero = np.zeros(problem.dim, dtype=float)
    if hasattr(algorithm, "agents"):
        for agent in algorithm.agents.values():
            agent.x = zero.copy()
            agent.z = zero.copy()
            for neighbor in agent.neighbors:
                agent.y[neighbor] = zero.copy()
            agent.neighbors_x = {}
            agent.neighbors_z = {}
    else:
        algorithm.x = [zero.copy() for _ in range(problem.n_agents)]
        if isinstance(algorithm, CachedProxDGT):
            algorithm.grad = [problem.local_gradient(i, zero) for i in range(problem.n_agents)]
            algorithm.y = [gradient.copy() for gradient in algorithm.grad]


def make_algorithm(name: str, config: Mapping[str, float], problem: Any, graph: nx.Graph) -> Any:
    lambda_reg = float(config.get("lambda_reg", 0.0))
    if name in {"SDBPG", "L1-SDBPG"}:
        params = dict(SDBPG_BASE)
        if "step_multiplier" in config:
            multiplier = float(config["step_multiplier"])
            params["alpha"] *= multiplier
            params["lambda_param"] *= multiplier
        parameter_overrides = {
            "dual_stepsize": "alpha",
            "primal_stepsize": "lambda_param",
            "smoothing_weight": "beta",
            "neighbor_penalty": "rho",
            "anchor_weight": "C",
        }
        for config_name, parameter_name in parameter_overrides.items():
            if config_name in config:
                params[parameter_name] = float(config[config_name])
        algorithm = L1SDBPG(problem=problem, graph=graph, lambda_reg=lambda_reg, **params)
    elif name == "ProxDGT":
        algorithm = CachedProxDGT(problem, graph, float(config["stepsize"]), lambda_reg)
    elif name == "ProxDGD":
        algorithm = ProximalDecentralizedGradientDescent(
            problem,
            graph,
            stepsize=float(config["stepsize"]),
            consensus_weight=float(config["consensus_weight"]),
            lambda_reg=lambda_reg,
        )
    elif name == "ProxPDA":
        algorithm = ProxPDA(problem, graph, float(config["stepsize"]), float(config["penalty"]), lambda_reg)
    elif name == "ProxEXTRA":
        algorithm = ProxExtra(problem, graph, float(config["stepsize"]), lambda_reg)
    else:
        raise ValueError(f"unknown decentralized method: {name}")
    initialize_algorithm(algorithm, problem)
    return algorithm


def algorithm_states(algorithm: Any) -> List[np.ndarray]:
    if hasattr(algorithm, "agents"):
        return [np.asarray(agent.x, dtype=float) for agent in algorithm.agents.values()]
    return [np.asarray(x, dtype=float) for x in algorithm.x]


def vectors_per_edge(name: str) -> int:
    return 2 if name in {"SDBPG", "L1-SDBPG", "ProxDGT"} else 1


def communication_scalars(name: str, graph: nx.Graph, dim: int, iterations: int) -> int:
    return int(2 * graph.number_of_edges() * dim * vectors_per_edge(name) * iterations)


def run_decentralized(
    name: str,
    algorithm: Any,
    problem: DistributedFourierWaveletProblem,
    graph: nx.Graph,
    iterations: int,
    lambda_reg: float,
    checkpoint: int,
    track: bool,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    history: Dict[str, List[float]] = {
        "iteration": [],
        "communication_mib": [],
        "nrmse": [],
        "psnr": [],
        "ssim": [],
        "consensus": [],
        "defect_iteration": [],
        "motion_defect": [],
        "aggregation_defect": [],
        "cumulative_defect": [],
        "support_size": [],
    }
    start = time.perf_counter()
    diverged = False
    completed = 0
    selected_previous: np.ndarray | None = None
    dx_previous = 0.0
    dz_previous = 0.0
    cumulative_defect = 0.0
    support_patterns: List[np.ndarray] = []
    all_defects: List[float] = []
    for iteration in range(1, int(iterations) + 1):
        algorithm.step()
        completed = iteration
        states = algorithm_states(algorithm)
        if name == "L1-SDBPG" and hasattr(algorithm, "last_selected_subgradient"):
            selected = np.asarray(algorithm.last_selected_subgradient, dtype=float)
            dx = float(algorithm.last_dx)
            dz = float(algorithm.last_dz)
            motion_raw = 0.0 if selected_previous is None else float(np.sum((selected - selected_previous) ** 2))
            motion_budget = 3.0 * (dx + dx_previous + dz_previous)
            motion_defect = max(motion_raw - motion_budget, 0.0) if selected_previous is not None else 0.0
            average_for_defect = np.mean(states, axis=0)
            selected_sum = np.sum(selected, axis=0)
            radius = problem.n_agents * float(lambda_reg)
            active = np.abs(average_for_defect) > 1e-7
            target = radius * np.sign(average_for_defect)
            active_distance = np.where(active, selected_sum - target, 0.0)
            inactive_distance = np.where(active, 0.0, selected_sum - np.clip(selected_sum, -radius, radius))
            aggregation_raw = float(np.sum(active_distance**2 + inactive_distance**2))
            consensus_sum = float(
                sum(np.linalg.norm(states[i] - states[j]) ** 2 for i, j in graph.edges())
            )
            aggregation_defect = max(aggregation_raw - 3.0 * consensus_sum, 0.0)
            cumulative_defect += motion_defect + aggregation_defect
            pattern = np.sign(np.where(active, average_for_defect, 0.0))
            support_patterns.append(pattern)
            all_defects.append(motion_defect + aggregation_defect)
            history["defect_iteration"].append(float(iteration))
            history["motion_defect"].append(float(motion_defect))
            history["aggregation_defect"].append(float(aggregation_defect))
            history["cumulative_defect"].append(float(cumulative_defect))
            history["support_size"].append(float(np.count_nonzero(pattern)))
            selected_previous = selected.copy()
            dx_previous = dx
            dz_previous = dz
        if iteration % 20 == 0:
            norms = np.asarray([np.linalg.norm(x) for x in states], dtype=float)
            if not np.all(np.isfinite(norms)) or float(norms.max()) > 1e6:
                diverged = True
                break
        if track and (iteration == 1 or iteration % int(checkpoint) == 0 or iteration == iterations):
            average = np.mean(states, axis=0)
            metrics = image_metrics(problem, average)
            consensus = float(np.mean([np.linalg.norm(x - average) ** 2 for x in states]))
            history["iteration"].append(float(iteration))
            history["communication_mib"].append(
                communication_scalars(name, graph, problem.dim, iteration) * 8.0 / (1024.0**2)
            )
            for metric in ("nrmse", "psnr", "ssim"):
                history[metric].append(float(metrics[metric]))
            history["consensus"].append(consensus)
    runtime = float(time.perf_counter() - start)
    states = algorithm_states(algorithm)
    average = np.mean(states, axis=0)
    if diverged or not np.all(np.isfinite(average)):
        metrics = {"nrmse": 1e6, "psnr": -1e6, "ssim": -1e6}
        consensus = 1e12
        stationarity = 1e12
        objective = 1e12
    else:
        metrics = image_metrics(problem, average)
        consensus = float(np.mean([np.linalg.norm(x - average) ** 2 for x in states]))
        gradient = problem.global_gradient(average)
        mapping = sr.l1_prox_gradient_mapping(average, gradient, lambda_reg, eta=0.25)
        stationarity = float(np.dot(mapping, mapping))
        objective = float(problem.global_objective(average) + lambda_reg * np.linalg.norm(average, 1))
    row: Dict[str, Any] = {
        "algorithm": name,
        **metrics,
        "consensus_mse": consensus,
        "composite_stationarity_g2": stationarity,
        "composite_objective": objective,
        "runtime_sec": runtime,
        "iterations": int(completed),
        "local_gradient_calls": int(problem.n_agents * completed),
        "communication_scalars": communication_scalars(name, graph, problem.dim, completed),
        "communication_mib": communication_scalars(name, graph, problem.dim, completed) * 8.0 / (1024.0**2),
        "diverged": int(diverged),
        "lambda_reg": lambda_reg,
        "reconstruction": problem.reconstruct(average),
    }
    if support_patterns:
        final_pattern = support_patterns[-1]
        identification_time = 0
        for index, pattern in enumerate(support_patterns):
            if not np.array_equal(pattern, final_pattern):
                identification_time = index + 1
        tail_start = min(identification_time + 50, len(all_defects) - 1)
        row.update(
            {
                "identification_time": int(identification_time),
                "final_support_size": int(np.count_nonzero(final_pattern)),
                "total_defect": float(cumulative_defect),
                "post_identification_tail": float(max(all_defects[tail_start:])),
            }
        )
    else:
        row.update(
            {
                "identification_time": float("nan"),
                "final_support_size": float("nan"),
                "total_defect": float("nan"),
                "post_identification_tail": float("nan"),
            }
        )
    return row, history


def run_fista(
    problem: DistributedFourierWaveletProblem,
    iterations: int,
    lambda_reg: float,
    checkpoint: int,
    track: bool,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    x = np.zeros(problem.dim, dtype=float)
    y = x.copy()
    momentum = 1.0
    step = 0.99 / problem.lipschitz_upper_bound()
    history: Dict[str, List[float]] = {
        "iteration": [],
        "communication_mib": [],
        "nrmse": [],
        "psnr": [],
        "ssim": [],
        "consensus": [],
    }
    start = time.perf_counter()
    for iteration in range(1, int(iterations) + 1):
        x_next = sr.soft_threshold(y - step * problem.global_gradient(y), step * lambda_reg)
        momentum_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        y = x_next + ((momentum - 1.0) / momentum_next) * (x_next - x)
        x, momentum = x_next, momentum_next
        if track and (iteration == 1 or iteration % int(checkpoint) == 0 or iteration == iterations):
            metrics = image_metrics(problem, x)
            history["iteration"].append(float(iteration))
            history["communication_mib"].append(0.0)
            for metric in ("nrmse", "psnr", "ssim"):
                history[metric].append(float(metrics[metric]))
            history["consensus"].append(0.0)
    runtime = float(time.perf_counter() - start)
    metrics = image_metrics(problem, x)
    gradient = problem.global_gradient(x)
    mapping = sr.l1_prox_gradient_mapping(x, gradient, lambda_reg, eta=step)
    row: Dict[str, Any] = {
        "algorithm": "Centralized FISTA",
        **metrics,
        "consensus_mse": 0.0,
        "composite_stationarity_g2": float(np.dot(mapping, mapping)),
        "composite_objective": float(problem.global_objective(x) + lambda_reg * np.linalg.norm(x, 1)),
        "runtime_sec": runtime,
        "iterations": int(iterations),
        "local_gradient_calls": int(problem.n_agents * iterations),
        "communication_scalars": 0,
        "communication_mib": 0.0,
        "diverged": 0,
        "lambda_reg": lambda_reg,
        "reconstruction": problem.reconstruct(x),
        "identification_time": float("nan"),
        "final_support_size": float("nan"),
        "total_defect": float("nan"),
        "post_identification_tail": float("nan"),
    }
    return row, history


def run_config(
    name: str,
    config: Mapping[str, float],
    problem: DistributedFourierWaveletProblem,
    graph: nx.Graph,
    iterations: int,
    checkpoint: int,
    track: bool,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    if name == "Centralized FISTA":
        return run_fista(problem, iterations, float(config["lambda_reg"]), checkpoint, track)
    algorithm = make_algorithm(name, config, problem, graph)
    return run_decentralized(
        name,
        algorithm,
        problem,
        graph,
        iterations,
        float(config.get("lambda_reg", 0.0)),
        checkpoint,
        track,
    )


def bootstrap_ci(values: np.ndarray, seed: int = 20260804, rounds: int = 5000) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(rounds), array.size))
    means = array[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        frame.groupby("algorithm", as_index=False)
        .agg(
            nrmse_mean=("nrmse", "mean"),
            nrmse_std=("nrmse", "std"),
            psnr_mean=("psnr", "mean"),
            ssim_mean=("ssim", "mean"),
            consensus_mean=("consensus_mse", "mean"),
            stationarity_mean=("composite_stationarity_g2", "mean"),
            runtime_mean=("runtime_sec", "mean"),
            communication_mib=("communication_mib", "mean"),
            diverged=("diverged", "sum"),
        )
        .sort_values("nrmse_mean")
    )
    comparisons: List[Dict[str, Any]] = []
    pivot = frame.pivot(index="seed", columns="algorithm", values="nrmse")
    if "L1-SDBPG" in pivot.columns:
        for comparator in pivot.columns:
            if comparator == "L1-SDBPG":
                continue
            differences = pivot["L1-SDBPG"].to_numpy() - pivot[comparator].to_numpy()
            low, high = bootstrap_ci(differences)
            baseline_mean = float(pivot[comparator].mean())
            comparisons.append(
                {
                    "comparison": f"L1-SDBPG minus {comparator}",
                    "mean_nrmse_difference": float(differences.mean()),
                    "relative_gain_percent": float(-100.0 * differences.mean() / max(baseline_mean, 1e-12)),
                    "ci_low": low,
                    "ci_high": high,
                    "wins": int(np.sum(differences < 0.0)),
                    "seeds": int(differences.size),
                }
            )
    return summary, pd.DataFrame(comparisons)


def communication_targets(
    frame: pd.DataFrame,
    histories: Mapping[Tuple[int, str], Mapping[str, List[float]]],
    slack: float = 1.10,
) -> pd.DataFrame:
    decentralized = frame[frame["algorithm"] != "Centralized FISTA"]
    target_by_seed = decentralized.groupby("seed")["nrmse"].min() * float(slack)
    rows: List[Dict[str, Any]] = []
    for (seed, name), history in histories.items():
        if name == "Centralized FISTA":
            continue
        target = float(target_by_seed.loc[int(seed)])
        nrmse = np.asarray(history["nrmse"], dtype=float)
        communication = np.asarray(history["communication_mib"], dtype=float)
        indices = np.flatnonzero(nrmse <= target)
        reached = bool(indices.size)
        rows.append(
            {
                "seed": int(seed),
                "algorithm": name,
                "target_nrmse": target,
                "reached": int(reached),
                "communication_to_target_mib": float(communication[indices[0]]) if reached else float("nan"),
            }
        )
    detail = pd.DataFrame(rows)
    return (
        detail.groupby("algorithm", as_index=False)
        .agg(
            target_success_rate=("reached", "mean"),
            communication_to_target_mib=("communication_to_target_mib", "mean"),
        )
        .sort_values(["target_success_rate", "communication_to_target_mib"], ascending=[False, True])
    )


def defect_summary(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["algorithm"] == "L1-SDBPG"]
    if subset.empty or "total_defect" not in subset:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "algorithm": "L1-SDBPG",
                "seeds": int(len(subset)),
                "median_identification_time": float(subset["identification_time"].median()),
                "q1_identification_time": float(subset["identification_time"].quantile(0.25)),
                "q3_identification_time": float(subset["identification_time"].quantile(0.75)),
                "median_final_support_size": float(subset["final_support_size"].median()),
                "median_total_defect": float(subset["total_defect"].median()),
                "max_post_identification_tail": float(subset["post_identification_tail"].max()),
            }
        ]
    )

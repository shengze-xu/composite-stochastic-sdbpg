"""Stochastic trajectories numerical routines."""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple


import numpy as np


from sdbpg.core import generate_network_topology


from experiments._impl import stochastic as st


from experiments._impl.noise_protocol import LAMBDA_REG, MAX_ITER, STEP_MULT


Case = Dict[str, Any]


def _setting(
    scenario: str,
    oracle_mode: str,
    batch_size: int | None = None,
    noise_sigma: float = 0.0,
) -> Dict[str, Any]:
    if oracle_mode == "mini_batch":
        return st.build_setting_manifest([scenario], "mini_batch", [int(batch_size or 4)], [])[0]
    return st.build_setting_manifest([scenario], "additive", [], [float(noise_sigma)])[0]


def _run_case(case: Case, seed: int, max_iter: int = MAX_ITER) -> Dict[str, np.ndarray]:
    setting = _setting(
        scenario=str(case["scenario"]),
        oracle_mode=str(case["oracle_mode"]),
        batch_size=case.get("batch_size"),
        noise_sigma=float(case.get("noise_sigma", 0.0)),
    )
    problem = st._make_problem(setting, seed)
    graph = generate_network_topology(problem.n_agents, "random", connectivity=0.5, seed=seed)
    algo = st._make_sdbpg_factory(step_mult=STEP_MULT, lambda_reg=LAMBDA_REG)(problem, graph)
    st._initialize_sdbpg(algo, problem.dim, seed)

    schedule: Callable[[int, int], int] | None = case.get("schedule")
    e_curve: List[float] = []
    noise_curve: List[float] = []
    g2_curve: List[float] = []

    for t in range(max_iter):
        if schedule is not None:
            problem.batch_size = schedule(t, problem.samples_per_agent)
        problem.begin_step()
        state_before = st._snapshot_sdbpg_state(algo)
        algo.step()
        theory = st._compute_theory_step_metrics(algo, state_before)
        oracle = problem.consume_step_oracle_summary()

        x_bar = np.mean([agent.x for agent in algo.agents.values()], axis=0)
        grad = problem.full_global_gradient(x_bar)
        e_curve.append(float(theory["e"]))
        noise_curve.append(float(oracle["avg_oracle_noise_sq"]))
        g2_curve.append(float(np.dot(grad, grad)))

    return {
        "e": np.asarray(e_curve, dtype=float),
        "noise": np.asarray(noise_curve, dtype=float),
        "g2": np.asarray(g2_curve, dtype=float),
    }

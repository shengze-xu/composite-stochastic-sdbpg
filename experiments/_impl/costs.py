"""Costs numerical routines."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Tuple


import numpy as np


import pandas as pd


def communication_target_table(
    final: pd.DataFrame,
    history: pd.DataFrame,
    relative_factor: float = 1.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for task, metric in (
        ("fourier_imaging", "nrmse"),
        ("stochastic_sparse_recovery", "recovery_error"),
    ):
        task_final = final[final["task"] == task]
        task_history = history[history["task"] == task]
        for seed, seed_final in task_final.groupby("seed"):
            target = float(relative_factor) * float(seed_final[metric].min())
            for method, trajectory in task_history[task_history["seed"] == seed].groupby("algorithm"):
                ordered = trajectory.sort_values("iterations")
                reached = ordered[ordered[metric] <= target]
                if reached.empty:
                    rows.append(
                        {
                            "task": task,
                            "seed": int(seed),
                            "algorithm": method,
                            "metric": metric,
                            "target": target,
                            "reached": 0,
                            "communication_rounds_to_target": np.nan,
                            "communication_mib_to_target": np.nan,
                            "oracle_samples_to_target": np.nan,
                            "wall_time_sec_to_target": np.nan,
                        }
                    )
                else:
                    first = reached.iloc[0]
                    rows.append(
                        {
                            "task": task,
                            "seed": int(seed),
                            "algorithm": method,
                            "metric": metric,
                            "target": target,
                            "reached": 1,
                            "communication_rounds_to_target": float(first["communication_rounds"]),
                            "communication_mib_to_target": float(first["communication_mib"]),
                            "oracle_samples_to_target": float(first["oracle_samples"]),
                            "wall_time_sec_to_target": float(first["wall_time_sec"]),
                        }
                    )
    seed_level = pd.DataFrame(rows)
    aggregate = (
        seed_level.groupby(["task", "algorithm"], as_index=False)
        .agg(
            target_success_rate=("reached", "mean"),
            communication_rounds_to_target=("communication_rounds_to_target", "mean"),
            communication_mib_to_target=("communication_mib_to_target", "mean"),
            oracle_samples_to_target=("oracle_samples_to_target", "mean"),
            wall_time_sec_to_target=("wall_time_sec_to_target", "mean"),
        )
    )
    return seed_level, aggregate

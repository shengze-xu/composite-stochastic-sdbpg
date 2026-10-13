"""Configuration, output, and paired statistics shared by the experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text())


def write(output, name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(output / f"{name}.csv", index=False)
    return frame


def summarize(output, rows, groups, reference=None, bootstrap_seed=20260923):
    """Write seed-level values, means, and paired percentile-bootstrap intervals.

    Failed/non-finite observations remain in the seed-level file. Paired rows
    report their total and finite pair counts, so exclusions remain visible.
    Differences are reference minus baseline; their direction follows the metric.
    """
    frame = write(output, "seed_level", rows)
    numeric = [c for c in frame.select_dtypes(include="number") if c not in {*groups, "seed"}]
    write(output, "summary", frame.groupby(groups, dropna=False)[numeric].mean().reset_index())
    paired = []
    if reference is not None and "algorithm" in groups:
        strata = [c for c in groups if c != "algorithm"]
        subsets = frame.groupby(strata, dropna=False) if strata else [((), frame)]
        rng = np.random.default_rng(bootstrap_seed)
        for key, subset in subsets:
            key = key if isinstance(key, tuple) else (key,)
            ours = subset[subset.algorithm == reference].set_index("seed")
            if ours.empty:
                continue
            for method in subset.algorithm.unique():
                if method == reference:
                    continue
                other = subset[subset.algorithm == method].set_index("seed")
                seeds = ours.index.intersection(other.index)
                for metric in numeric:
                    delta = (ours.loc[seeds, metric] - other.loc[seeds, metric]).to_numpy(float)
                    finite = delta[np.isfinite(delta)]
                    if len(finite):
                        samples = finite[rng.integers(len(finite), size=(10000, len(finite)))].mean(axis=1)
                        low, high = np.quantile(samples, [0.025, 0.975])
                    else:
                        low = high = np.nan
                    paired.append({
                        **dict(zip(strata, key)), "reference": reference,
                        "baseline": method, "metric": metric, "pairs": len(delta),
                        "finite_pairs": len(finite),
                        "mean_difference": float(finite.mean()) if len(finite) else np.nan,
                        "ci95_low": low, "ci95_high": high,
                    })
    if paired:
        write(output, "paired", paired)
    return frame

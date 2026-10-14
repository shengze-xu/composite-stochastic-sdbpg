# Composite and stochastic SDBPG

[![arXiv](https://img.shields.io/badge/arXiv-2610.18231-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.18231)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-5A96CD)](https://arxiv.org/pdf/2610.18231)

Official numerical implementation for [**Composite and Stochastic Smoothed Decentralized
Bregman Proximal-Gradient Methods**](https://arxiv.org/abs/2610.18231).

The paper extends the smooth deterministic SDBPG method of Pu et al.,
[A Smoothed Bregman Proximal Gradient Algorithm for Decentralized Nonconvex
Optimization](https://doi.org/10.1109/ICASSP48485.2024.10448285), ICASSP 2024,
to composite regularization and stochastic local gradients.

The experiments cover sparse and group-sparse recovery, phase retrieval,
structured Fourier imaging, stochastic gradients, Bregman geometry, and
communication costs. Each entry point generates its data, runs the algorithms
with fixed parameters, and saves numerical results.

## Setup

Use Python 3.10 or newer. Clone the repository and run:

```bash
git clone https://github.com/shengze-xu/composite-stochastic-sdbpg.git
cd composite-stochastic-sdbpg
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m experiments.run --list
python -m experiments.run sparse_recovery --smoke
```

`--smoke` uses one seed and five iterations to check execution. For the full
experiment, omit this option:

```bash
python -m experiments.run sparse_recovery
python -m experiments.run modern_fourier
python -m experiments.run stochastic_trajectories
```

By default, results go into a new timestamped subdirectory of `results/`.
To choose a budget or output location:

```bash
python -m experiments.run quartic --seeds 30,31 --iterations 600 --output results/quartic_two_seeds
```

An explicitly supplied output directory must be empty. `--checkpoint` controls
the recording interval for experiments that use checkpoints. Runs may be made
with one BLAS thread using `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

## Paper experiments

Defaults below use the paper's evaluation seeds and iteration budgets. Parameters
selected on validation instances are stored in `configs/`; model definitions,
initialization rules, and fixed diagnostic settings are in `experiments/_impl/`.
Each row is run with `python -m experiments.run <entry>`.

| Entry | Numerical results | Iterations | Seeds |
|---|---|---:|---|
| `defect_identification` | Temporal and aggregation defects; coordinate/group support stabilization | 3000 | 0–9 |
| `sparse_recovery` | Sparse recovery and support accuracy, including proximal baselines | 900 | 10–19 |
| `group_sparse` | Shared-support recovery with a group penalty | 950 | 10–19 |
| `sparse_coverage` | Sparse recovery across sparsity, noise, and graph settings | 900 | 10–19 |
| `sparse_phase` | Sparse phase retrieval and support accuracy | 1200 | 10–19 |
| `extra_baselines` | PG-EXTRA and Prox-EXTRA sparse-recovery comparisons | 900 | 10–19 |
| `recovery_trajectories` | Sparse recovery, support, stationarity, and consensus trajectories | 900 | 10–19 |
| `smooth_stationarity` | Smooth quadratic stationarity and consensus comparison | 2000 | 0–4 |
| `smooth_benchmarks` | Unregularized phase retrieval and matrix-factorization recovery | 1200 | 10–19 |
| `imaging_phantom` | Extended Fourier-imaging comparison on Phantom | 1000 | 30–39 |
| `imaging_cameraman` | Extended Fourier-imaging comparison on Cameraman | 1000 | 40–49 |
| `modern_fourier` | Fourier-imaging accuracy and costs for SDBPG, ProxDGT, and ProxGT | 1000 | 50–59 |
| `modern_stochastic` | Stochastic accuracy and costs for SDBPG, ProxGT-SA, and Prox-DASA-GT | 900 | 50–59 |
| `quartic` | Quadratic/quartic kernels and ProxGT across three heterogeneity levels | 600 | 30–39 |
| `weighted_quadratic` | Diagonal quadratic kernels across condition numbers | 700 | 0–9 |
| `noise_sweep` | Mini-batch and additive-noise recovery comparisons | 900 | 10–29 |
| `stochastic_trajectories` | Fixed/increasing-batch smoothing residual and oracle-noise trajectories | 900 | 10–29 |
| `lifted_residuals` | Smooth, composite, and stochastic lifted-residual diagnostics | 700 | 10–19 |
| `sensitivity` | Network size, smoothing weight, and problem dimension | 2000 | 0–4 |
| `hitting_times` | Consensus proximal-point threshold crossings on four graph regimes | 1200 | 0–4 |

The increasing-batch schedule uses batches of 2 for the first 300 updates, 4 for
the next 300, and all 8 local samples thereafter. The final stage therefore uses
the full local gradient. These transition points remain fixed when the iteration
budget is overridden. Hitting times use the first of two consecutive successful
checkpoints, with a default spacing of 10 iterations after the initial checkpoint.

The defect-identification experiment uses its own zero initialization and signal
construction. Its strict-complementarity margin is measured using the aggregate
gradient and aggregate regularization weight. The remaining recovery and imaging
experiments retain their respective data and initialization protocols.

## Outputs

Depending on the experiment, the output directory contains:

- `run.json`: seeds, budgets, package versions, source/configuration hashes, and completion status.
- `seed_level.csv` and `summary.csv`: individual outcomes and aggregate metrics.
- `paired.csv`: paired differences and bootstrap confidence intervals.
- `trajectories.csv`: iteration-level or checkpoint-level numerical diagnostics.
- Cost-to-target, hitting-time, or defect-summary CSV files for the corresponding experiments.
- Compressed NumPy arrays (`.npz`) with ground truth and reconstructed images for the extended imaging experiments.

Recovery errors, support F1, support sizes, squared residuals, and communication
costs are recorded separately. Squared gradient and stationarity quantities use
`g2`, `gradient_sq`, or `stationarity_sq` in their field names. Support size is a
descriptive quantity; recovery accuracy is evaluated against the generated truth.

The shared statistics helper uses 10,000 paired percentile-bootstrap resamples
with a fixed random seed. Its difference is reference minus baseline, with
`pairs` and `finite_pairs` recording the sample counts. The Fourier, modern-cost,
and quartic runners retain their experiment-specific statistics routines and
field conventions. Numerical summaries recomputed with different bootstrap
settings may differ slightly from rounded manuscript intervals. Non-finite
observations and divergence indicators remain in the individual outputs.

Timing fields measure the current execution environment. Communication and oracle
counts follow each algorithm's payload and sampling rules. The run metadata
records the numerical environment for comparisons across machines.

## Code layout

```text
sdbpg/
  core.py          Bregman operators, exact proximal updates, graph utilities
  algorithm.py     SDBPG and decentralized gradient/proximal baselines
  baselines.py     ProxGT and Prox-DASA-GT with cost accounting
  geometry.py      Quartic geometry and geometry-dependent SDBPG updates
experiments/
  run.py           Command-line entry point and default budgets
  recovery.py      Recovery experiments
  imaging.py       Fourier imaging and modern-method comparisons
  stochastic.py    Stochastic-oracle experiments
  geometry.py      Kernel comparisons
  diagnostics.py   Defects, residuals, sensitivity, and hitting times
  common.py        Configuration, CSV output, and paired statistics
  _impl/           Data models, experiment-specific solvers, and metric routines
configs/           Fixed numerical configurations
requirements.txt   Runtime dependencies
```

The directory can be copied to a separate checkout and run on its own. It produces
CSV and NumPy outputs suitable for analysis; manuscript sources, figure rendering,
and publication table formatting are maintained in the manuscript project.

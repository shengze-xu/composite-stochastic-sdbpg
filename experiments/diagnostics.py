"""Defect accumulation, residual diagnostics, and sensitivity experiments."""

from dataclasses import asdict
import numpy as np

from experiments.common import summarize, write
from experiments._impl import defects, diagnostics, agents, beta, dimension, complexity
from sdbpg.core import QuadraticRegressionProblem, generate_network_topology
from sdbpg.algorithm import DecentralizedGradientDescent, DecentralizedGradientTracking
from experiments._impl.stationarity import L2SDBPG, _run_and_record


def smooth_stationarity(args):
    rows, curves = [], []
    for seed in args.seeds:
        problem = QuadraticRegressionProblem(6, 10, 20, seed=seed)
        graph = generate_network_topology(6, "random", connectivity=0.5, seed=seed)
        factories = {
            "SDBPG": lambda: L2SDBPG(problem, graph, lambda_reg=0.0, rho=2.0,
                             alpha=0.02, lambda_param=0.025, beta=0.8, C=1.5),
            "DGD": lambda: DecentralizedGradientDescent(problem, graph, stepsize=0.008, consensus_weight=0.4),
            "DGT": lambda: DecentralizedGradientTracking(problem, graph, stepsize=0.008),
        }
        for method, factory in factories.items():
            np.random.seed(seed)
            algorithm = factory()
            result = _run_and_record(algorithm, problem, 0.0, args.iterations)
            identity = {"algorithm": method, "seed": seed}
            rows.append({**identity, "stationarity_sq": result["running_min"][-1],
                         "consensus_sq": result["consensus_err"][-1]})
            curves.extend({**identity, "iteration": t + 1,
                           "gradient_sq": result["grad_norm_sq"][t],
                           "running_min_gradient_sq": result["running_min"][t],
                           "consensus_sq": result["consensus_err"][t]}
                          for t in range(args.iterations))
    summarize(args.output, rows, ["algorithm"], "SDBPG")
    write(args.output, "trajectories", curves)


def defect_identification(args):
    rows, curves = [], []
    for penalty in ("l1", "l21"):
        for seed in args.seeds:
            result = defects._run(seed, penalty, args.iterations)
            rows.append({k: v for k, v in asdict(result).items() if np.isscalar(v)})
            for t in range(args.iterations):
                curves.append({"penalty": penalty, "seed": seed, "iteration": t,
                    **{k: getattr(result, k)[t] for k in ("eta", "zeta", "cumulative", "support_size")}})
    summarize(args.output, rows, ["penalty"])
    write(args.output, "trajectories", curves)


def lifted_residuals(args):
    rows = []
    for seed in args.seeds:
        rows.extend(diagnostics._run_deterministic_case("smooth", "SDBPG", 0.0, seed, args.iterations))
        rows.extend(diagnostics._run_deterministic_case("l1", "L1-SDBPG", 0.02, seed, args.iterations))
        rows.extend(diagnostics._run_stochastic_case(seed, args.iterations))
    write(args.output, "trajectories", rows)


def sensitivity(args):
    rows, histories = [], []
    for kind, values in (("agents", agents.N_LIST), ("beta", [p[0] for p in beta.BETA_CONFIGS]),
                         ("dimension", dimension.DIM_LIST)):
        for value in values:
            for seed in args.seeds:
                if kind == "agents":
                    curve = agents._run_one(value, 0.0, seed, args.iterations)
                elif kind == "beta":
                    curve = beta._run_one(value, seed, args.iterations)
                else:
                    problem = QuadraticRegressionProblem(dimension.N_AGENTS, value, dimension.SAMPLES, seed=seed)
                    graph = generate_network_topology(dimension.N_AGENTS, "random", connectivity=0.5, seed=seed)
                    curve = dimension._fast_sdbpg_run(problem, graph, 0.0, args.iterations, seed)
                identity = {"parameter": kind, "value": value, "seed": seed}
                rows.append({**identity, "final_running_min_gradient_sq": curve[-1]})
                histories.extend({**identity, "iteration": t, "running_min_gradient_sq": g}
                                 for t, g in enumerate(curve, 1))
    summarize(args.output, rows, ["parameter", "value"])
    write(args.output, "trajectories", histories)


def hitting_times(args):
    rows = []
    for scenario in complexity.SCENARIOS:
        for seed in args.seeds:
            rows.extend(complexity._run_seed(scenario, seed, args.iterations, args.checkpoint))
    summary, slopes = complexity.build_complexity_summary(rows,
        [0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001], 2)
    write(args.output, "trajectories", rows)
    write(args.output, "hitting_times", summary)
    write(args.output, "empirical_slopes", slopes)

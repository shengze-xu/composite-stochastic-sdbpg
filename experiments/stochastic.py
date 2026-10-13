"""Fixed-oracle and increasing-batch sparse recovery experiments."""

from experiments.common import config, summarize, write
from experiments._impl import stochastic as st, sparse, noise_protocol
from experiments._impl import stochastic_trajectories as trajectories


def noise_sweep(args):
    rows = []
    baseline_settings = {r["setting_id"]: r["parameters"] for r in config("stochastic_baselines")}
    for setting in noise_protocol.settings():
        methods = {"L1-SDBPG": {"step_mult": 1.5, "lambda_reg": 0.02},
                   "SDBPG": {"step_mult": 1.5, "lambda_reg": 0.0}}
        if setting["setting_id"] in baseline_settings:
            methods["ProxDGD"] = baseline_settings[setting["setting_id"]]
        for method, parameters in methods.items():
            for seed in args.seeds:
                if method == "ProxDGD":
                    metrics = st._run_one_seed_baseline(sparse.TASKS["sparse_recovery"], setting,
                        seed, method, parameters, args.iterations, sparse.SUPPORT_TOL)
                else:
                    metrics = st._run_one_seed_sdbpg(sparse.TASKS["sparse_recovery"], setting,
                        seed, parameters["step_mult"], parameters["lambda_reg"],
                        args.iterations, sparse.SUPPORT_TOL)
                rows.append({**setting, "algorithm": method, "seed": seed, **metrics})
        print(f"Completed {setting['setting_id']}", flush=True)
    summarize(args.output, rows, ["setting_id", "algorithm"], "L1-SDBPG")


def residual_trajectories(args):
    cases = [
        {"case_id": "batch4", "oracle_mode": "mini_batch", "batch_size": 4},
        {"case_id": "batch7", "oracle_mode": "mini_batch", "batch_size": 7},
        {"case_id": "additive003", "oracle_mode": "additive", "noise_sigma": 0.03},
        {"case_id": "additive010", "oracle_mode": "additive", "noise_sigma": 0.10},
        {"case_id": "increasing", "oracle_mode": "mini_batch", "batch_size": 2,
         "schedule": lambda t, samples: 2 if t < 300 else (4 if t < 600 else samples)},
    ]
    rows = []
    for case in cases:
        case["scenario"] = "sparse_noisy_moderate"
        for seed in args.seeds:
            curves = trajectories._run_case(case, seed, args.iterations)
            for t in range(args.iterations):
                rows.append({"case_id": case["case_id"], "seed": seed, "iteration": t + 1,
                             **{metric: curve[t] for metric, curve in curves.items()}})
    frame = write(args.output, "trajectories", rows)
    summary = frame.groupby(["case_id", "iteration"])[["e", "noise", "g2"]].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    write(args.output, "trajectory_summary", summary.reset_index())

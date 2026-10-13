"""Structured Fourier imaging and communication-cost experiments."""

from experiments.common import config, write
from experiments._impl import fourier, modern, costs
import pandas as pd
import numpy as np


def imaging(args):
    name = args.experiment.removeprefix("imaging_")
    selected = config(args.experiment)
    rows, histories, defect_histories, original_histories = [], [], [], {}
    for seed in args.seeds:
        problem = fourier.DistributedFourierWaveletProblem(
            fourier.FourierImagingConfig(image_name=name, image_size=64), seed=seed)
        graph = fourier.imaging_graph(problem.n_agents)
        np.savez_compressed(args.output / f"reference_seed{seed}.npz",
                            ground_truth=problem.image_star,
                            zero_filled=problem.zero_filled_reconstruction())
        for method, parameters in selected.items():
            metrics, history = fourier.run_config(
                method, parameters, problem, graph, args.iterations, args.checkpoint, True)
            reconstruction = metrics.pop("reconstruction")
            np.savez_compressed(args.output / f"reconstruction_seed{seed}_{method.replace(' ', '_')}.npz",
                                reconstruction=reconstruction)
            rows.append({"seed": seed, "algorithm": method, **metrics})
            original_histories[(seed, method)] = history
            if history:
                frame = pd.DataFrame({k: history[k] for k in
                    ("iteration", "communication_mib", "nrmse", "psnr", "ssim", "consensus")})
                frame["seed"], frame["algorithm"] = seed, method
                histories.extend(frame.to_dict("records"))
                if history.get("defect_iteration"):
                    defect = pd.DataFrame({k: history[k] for k in
                        ("defect_iteration", "motion_defect", "aggregation_defect", "cumulative_defect", "support_size")})
                    defect["seed"], defect["algorithm"] = seed, method
                    defect_histories.extend(defect.to_dict("records"))
        print(f"Completed {name} seed {seed}", flush=True)
    frame = write(args.output, "seed_level", rows)
    summary, paired = fourier.summarize(frame)
    write(args.output, "summary", summary)
    write(args.output, "paired", paired)
    write(args.output, "trajectories", histories)
    write(args.output, "defect_trajectories", defect_histories)
    write(args.output, "communication_to_target", fourier.communication_targets(frame, original_histories))
    write(args.output, "defect_summary", fourier.defect_summary(frame))


def modern_comparison(args):
    selected = config("modern")
    if args.experiment == "modern_fourier":
        legacy = config("modern_fourier")
        methods = {"L1-SDBPG": legacy["L1-SDBPG"], "ProxDGT": legacy["ProxDGT"],
                   "ProxGT": selected["fourier/ProxGT"]}
        task, metric = "fourier_imaging", "nrmse"
    else:
        methods = {k.split("/", 1)[1]: v for k, v in selected.items() if k.startswith("stochastic/")}
        task, metric = "stochastic_sparse_recovery", "recovery_error"
    rows, histories = [], []
    for seed in args.seeds:
        for method, parameters in methods.items():
            call = (seed, method, parameters, args.iterations, args.checkpoint)
            if task == "fourier_imaging":
                row, history = modern._run_fourier_method(*call)
            else:
                row, history = modern._run_stochastic_method(*call, 4)
            rows.append(row)
            histories.extend(history)
        print(f"Completed {task} seed {seed}", flush=True)
    frame = write(args.output, "seed_level", rows)
    history = write(args.output, "trajectories", histories)
    write(args.output, "summary", modern._aggregate(frame))
    paired = [modern._paired_bootstrap(frame, task, "L1-SDBPG", method, metric, True)
              for method in methods if method != "L1-SDBPG"]
    if task == "stochastic_sparse_recovery":
        paired.extend(modern._paired_bootstrap(frame, task, "L1-SDBPG", method, "support_f1", False)
                      for method in methods if method != "L1-SDBPG")
    write(args.output, "paired", paired)
    target, summary = costs.communication_target_table(frame, history)
    write(args.output, "cost_to_target", target)
    write(args.output, "cost_to_target_summary", summary)

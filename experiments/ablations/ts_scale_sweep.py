#!/usr/bin/env python
"""Thompson-sampling scale sweep for tuned Bayesian last-layer agents."""

import argparse
import os
import sys

sys.path.insert(0, ".")

from experiments.ablations.common import (  # noqa: E402
    get_quick_params,
    filter_results_for_run,
    load_config,
    load_results,
    make_dataset_config,
    run_ablation_sweep,
    save_results,
    select_ablation_seeds,
)
from experiments.ablations.posterior_strategy_ablation import (  # noqa: E402
    STRATEGY_LABELS,
    load_tuned_params,
    resolve_base_hp,
)
from src.agents.contextual import BayesianMSEAgent, BayesianSPOAgent  # noqa: E402


def agent_name(method, multiplier):
    return f"{method}-TS-x{multiplier:g}"


def make_agents(env, seed, spo_hp, mse_hp, multipliers):
    minimize = getattr(env, "minimize", False)
    solver = env.solve

    base_common = dict(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        solver=solver,
        seed=seed,
        minimize=minimize,
        exploration_strategy="ts",
    )

    spo_common = dict(
        **base_common,
        hidden_dim=spo_hp["hidden_dim"],
        lambda_reg=spo_hp["lambda_reg"],
        lr=spo_hp["lr"],
        beta=spo_hp["beta"],
        mask_grad=True,
        imputation_samples=int(spo_hp.get("imputation_samples", 1)),
        imputation_scale=spo_hp.get("imputation_scale", 1.0),
        use_posterior_predictive=spo_hp.get("use_posterior_predictive", False),
        noise_var=spo_hp.get("noise_var", 1.0),
    )
    mse_common = dict(
        **base_common,
        hidden_dim=mse_hp["hidden_dim"],
        lambda_reg=mse_hp["lambda_reg"],
        lr=mse_hp["lr"],
        beta=mse_hp["beta"],
    )

    agents = {}
    for multiplier in multipliers:
        multiplier = float(multiplier)
        agents[agent_name("BayesianSPO", multiplier)] = BayesianSPOAgent(
            **spo_common,
            ts_scale=float(spo_hp["beta"]) * multiplier,
        )
        agents[agent_name("BayesianMSE", multiplier)] = BayesianMSEAgent(
            **mse_common,
            ts_scale=float(mse_hp["beta"]) * multiplier,
        )
    return agents


def main():
    parser = argparse.ArgumentParser(description="TS-scale sweep")
    parser.add_argument("--config", default="experiments/configs/experiment.yaml")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--ts-scale-multipliers",
        nargs="+",
        type=float,
        default=None,
        help="Override TS scale multipliers of each method's tuned beta.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--use-tuned-params", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing dataset results and recompute from scratch.",
    )
    parser.add_argument("--tuning-dir", default=None)
    parser.add_argument(
        "--plot-format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Format used when rendering LetsPlot figures.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config["ablations"]["ts_scale_sweep"]
    if args.mode == "quick":
        abl_cfg = get_quick_params(abl_cfg)
        abl_cfg["ts_scale_multipliers"] = abl_cfg.get(
            "quick_ts_scale_multipliers", [0.25, 1.0]
        )

    datasets = args.datasets or abl_cfg.get("datasets", ["knapsack_32"])
    n_rounds = abl_cfg["n_rounds"]
    n_seeds = abl_cfg["n_seeds"]
    seeds = select_ablation_seeds(config, n_seeds, quick=args.mode == "quick")
    default_multipliers = args.ts_scale_multipliers or abl_cfg["ts_scale_multipliers"]
    dataset_multipliers = abl_cfg.get("dataset_ts_scale_multipliers", {})
    output_dir = args.output or abl_cfg["output_dir"]
    tuning_dir = args.tuning_dir or abl_cfg.get("tuning_dir", "results/FINAL/tuning")
    defaults = config["agent_defaults"]

    print("\n=== TS-scale sweep ===")
    print(f"Datasets: {datasets}")
    print(f"n_rounds={n_rounds}, n_seeds={n_seeds}")
    print(f"Seeds: {seeds}")
    print(f"Strategy: {STRATEGY_LABELS['ts']}")
    print(f"Default scale multipliers of tuned beta: {[float(v) for v in default_multipliers]}")
    print(f"Use tuned params: {args.use_tuned_params} (tuning_dir={tuning_dir})")

    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        dc = make_dataset_config(dataset, config)
        multipliers = [
            float(v)
            for v in dataset_multipliers.get(dataset, default_multipliers)
        ]
        variant_labels = [
            agent_name(method, multiplier)
            for multiplier in multipliers
            for method in ("BayesianSPO", "BayesianMSE")
        ]
        print(f"  scale multipliers: {multipliers}")
        tuned_spo = (
            load_tuned_params(tuning_dir, dataset, "BayesianSPO")
            if args.use_tuned_params
            else None
        )
        tuned_mse = (
            load_tuned_params(tuning_dir, dataset, "BayesianMSE")
            if args.use_tuned_params
            else None
        )
        if args.use_tuned_params and tuned_spo is None:
            print(f"  warning: no tuned BayesianSPO params for {dataset}; using defaults")
        if args.use_tuned_params and tuned_mse is None:
            print(f"  warning: no tuned BayesianMSE params for {dataset}; using defaults")

        spo_hp = resolve_base_hp(defaults, tuned_spo, include_imputation=True)
        mse_hp = resolve_base_hp(defaults, tuned_mse, include_imputation=False)
        print(
            "  BayesianSPO beta="
            f"{spo_hp['beta']}; actual ts_scales="
            f"{[round(float(spo_hp['beta']) * m, 6) for m in multipliers]}"
        )
        print(
            "  BayesianMSE beta="
            f"{mse_hp['beta']}; actual ts_scales="
            f"{[round(float(mse_hp['beta']) * m, 6) for m in multipliers]}"
        )

        ds_dir = os.path.join(output_dir, dataset)
        result_path = os.path.join(ds_dir, "results.pkl")
        existing_results = {}
        if os.path.exists(result_path) and not args.no_resume:
            loaded_results = load_results(result_path)
            existing_results = filter_results_for_run(
                loaded_results, variant_labels, seeds, n_rounds=n_rounds
            )
            print(
                f"  resuming from {result_path} "
                f"({len(existing_results)}/{len(loaded_results)} matching entries)"
            )

        results = run_ablation_sweep(
            make_agents_fn=lambda env, seed: make_agents(
                env, seed, spo_hp, mse_hp, multipliers
            ),
            dataset=dataset,
            dataset_config=dc,
            seeds=seeds,
            n_rounds=n_rounds,
            diagnostics=False,
            output_dir=output_dir,
            compact=True,
            keep_trajectory=False,
            initial_results=existing_results,
            save_path=result_path,
        )

        save_results(results, result_path)

    if not args.no_plot:
        from experiments.plot_ablation_results import plot_ts_scale_sweep

        plot_ts_scale_sweep(output_dir, output_dir, args.plot_format)


if __name__ == "__main__":
    main()

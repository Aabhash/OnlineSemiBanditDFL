#!/usr/bin/env python
"""Exploration-policy ablation for tuned Bayesian last-layer agents.

This isolates action selection while holding each method's tuned learning
hyperparameters fixed. BayesianSPO and BayesianMSE are each evaluated with
greedy posterior means, tuned UCB, and Thompson sampling.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

from experiments.ablations.common import (  # noqa: E402
    get_quick_params,
    load_config,
    make_dataset_config,
    run_ablation_sweep,
    save_results,
)
from src.agents.contextual import BayesianMSEAgent, BayesianSPOAgent  # noqa: E402


STRATEGIES = ("greedy", "ucb", "ts")
STRATEGY_LABELS = {
    "greedy": "Greedy",
    "ucb": "UCB",
    "ts": "TS",
}


def load_tuned_params(tuning_dir, dataset, agent_name, feedback_type="semi_bandit"):
    path = os.path.join(tuning_dir, f"best_hyperparams_{dataset}_{feedback_type}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    entry = data.get(agent_name)
    if entry is None:
        return None
    if isinstance(entry, dict) and "params" in entry:
        return dict(entry["params"])
    return dict(entry)


def resolve_base_hp(defaults, tuned_params, include_imputation=False):
    tuned_params = tuned_params or {}
    hp = {
        "lr": tuned_params.get("init_lr", tuned_params.get("lr", defaults["lr"])),
        "beta": tuned_params.get("beta", defaults["beta"]),
        "hidden_dim": tuned_params.get("hidden_dim", defaults["hidden_dim"]),
        "lambda_reg": tuned_params.get("lambda_reg", defaults["lambda_reg"]),
    }
    if include_imputation:
        hp.update(
            {
                "imputation_samples": tuned_params.get("imputation_samples", 1),
                "imputation_scale": tuned_params.get("imputation_scale", 1.0),
                "use_posterior_predictive": tuned_params.get(
                    "use_posterior_predictive", False
                ),
                "noise_var": tuned_params.get("noise_var", 1.0),
            }
        )
    return hp


def make_agents(env, seed, spo_hp, mse_hp):
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
    for strategy in STRATEGIES:
        label = STRATEGY_LABELS[strategy]
        agents[f"BayesianSPO-{label}"] = BayesianSPOAgent(
            **spo_common,
            exploration_strategy=strategy,
        )
        agents[f"BayesianMSE-{label}"] = BayesianMSEAgent(
            **mse_common,
            exploration_strategy=strategy,
        )
    return agents


def main():
    parser = argparse.ArgumentParser(description="Exploration-policy ablation")
    parser.add_argument("--config", default="experiments/configs/experiment.yaml")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run (default: config datasets)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Save data only, skip plotting")
    parser.add_argument(
        "--use-tuned-params",
        action="store_true",
        help="Use tuned BayesianSPO and BayesianMSE hyperparameters from <tuning_dir>.",
    )
    parser.add_argument(
        "--tuning-dir",
        default=None,
        help="Override tuning dir (default: ablations.exploration_strategy.tuning_dir).",
    )
    parser.add_argument(
        "--plot-format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Format used when rendering LetsPlot figures.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config["ablations"]["exploration_strategy"]

    if args.mode == "quick":
        abl_cfg = get_quick_params(abl_cfg)

    datasets = args.datasets or abl_cfg.get("datasets", ["knapsack_32"])
    n_rounds = abl_cfg["n_rounds"]
    n_seeds = abl_cfg["n_seeds"]
    seeds = list(range(n_seeds))
    output_dir = args.output or abl_cfg["output_dir"]
    tuning_dir = args.tuning_dir or abl_cfg.get("tuning_dir", "results/FINAL/tuning")
    defaults = config["agent_defaults"]

    print("\n=== Exploration-policy ablation ===")
    print(f"Datasets: {datasets}")
    print(f"n_rounds={n_rounds}, n_seeds={n_seeds}")
    print(f"Strategies: {', '.join(STRATEGY_LABELS[s] for s in STRATEGIES)}")
    print(f"Use tuned params: {args.use_tuned_params} (tuning_dir={tuning_dir})")

    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        dc = make_dataset_config(dataset, config)
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
            "  BayesianSPO hp: "
            f"hidden_dim={spo_hp['hidden_dim']}, lr={spo_hp['lr']}, "
            f"lambda={spo_hp['lambda_reg']}, beta={spo_hp['beta']}, "
            f"M={spo_hp.get('imputation_samples', 1)}"
        )
        print(
            "  BayesianMSE hp: "
            f"hidden_dim={mse_hp['hidden_dim']}, lr={mse_hp['lr']}, "
            f"lambda={mse_hp['lambda_reg']}, beta={mse_hp['beta']}"
        )

        results = run_ablation_sweep(
            make_agents_fn=lambda env, seed: make_agents(env, seed, spo_hp, mse_hp),
            dataset=dataset,
            dataset_config=dc,
            seeds=seeds,
            n_rounds=n_rounds,
            diagnostics=False,
            output_dir=output_dir,
        )

        ds_dir = os.path.join(output_dir, dataset)
        save_results(results, os.path.join(ds_dir, "results.pkl"))

    if not args.no_plot:
        from experiments.plot_ablation_results import plot_exploration

        plot_exploration(output_dir, output_dir, args.plot_format)


if __name__ == "__main__":
    main()

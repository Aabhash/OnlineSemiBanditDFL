#!/usr/bin/env python
"""Architecture ablation: Bayesian last-layer vs deterministic jointly trained head."""

import argparse
import os
import sys

sys.path.insert(0, ".")

from experiments.ablations.common import (  # noqa: E402
    compact_result,
    get_quick_params,
    load_config,
    make_dataset_config,
    save_results,
)
from experiments.ablations.posterior_strategy_ablation import (  # noqa: E402
    load_tuned_params,
    resolve_base_hp,
)
from src.agents.base import pregenerate_episode, run_agent_on_episode  # noqa: E402
from src.agents.contextual import (  # noqa: E402
    BayesianMSEAgent,
    BayesianSPOAgent,
    DeterministicMSEAgent,
    DeterministicSPOAgent,
)
from src.envs import create_env  # noqa: E402


def make_agents(env, seed, spo_hp, mse_hp, defaults, n_rounds):
    minimize = getattr(env, "minimize", False)
    solver = env.solve
    epsilon = float(defaults.get("epsilon", 0.1))
    decay_epsilon = bool(defaults.get("decay_epsilon", True))

    base_common = dict(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        solver=solver,
        seed=seed,
        minimize=minimize,
    )

    return {
        "BayesianSPO": BayesianSPOAgent(
            **base_common,
            hidden_dim=spo_hp["hidden_dim"],
            lambda_reg=spo_hp["lambda_reg"],
            lr=spo_hp["lr"],
            beta=spo_hp["beta"],
            mask_grad=True,
            exploration_strategy="ucb",
            imputation_samples=int(spo_hp.get("imputation_samples", 1)),
            imputation_scale=spo_hp.get("imputation_scale", 1.0),
            use_posterior_predictive=spo_hp.get("use_posterior_predictive", False),
            noise_var=spo_hp.get("noise_var", 1.0),
        ),
        "BayesianMSE": BayesianMSEAgent(
            **base_common,
            hidden_dim=mse_hp["hidden_dim"],
            lambda_reg=mse_hp["lambda_reg"],
            lr=mse_hp["lr"],
            beta=mse_hp["beta"],
            exploration_strategy="ucb",
        ),
        "GreedySPO": DeterministicSPOAgent(
            **base_common,
            hidden_dim=spo_hp["hidden_dim"],
            lr=spo_hp["lr"],
            epsilon=epsilon,
            decay_epsilon=decay_epsilon,
            decay_rounds=n_rounds,
            mask_grad=True,
        ),
        "GreedyMSE": DeterministicMSEAgent(
            **base_common,
            hidden_dim=mse_hp["hidden_dim"],
            lr=mse_hp["lr"],
            epsilon=epsilon,
            decay_epsilon=decay_epsilon,
            decay_rounds=n_rounds,
            mask_grad=True,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Architecture ablation")
    parser.add_argument("--config", default="experiments/configs/experiment.yaml")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--use-tuned-params", action="store_true")
    parser.add_argument("--tuning-dir", default=None)
    parser.add_argument("--keep-trajectory", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config["ablations"].get(
        "architecture_ablation",
        {
            "datasets": ["knapsack_32", "knapsack_100", "shortest_path_8x8", "shortest_path_20x20", "portfolio_32"],
            "n_rounds": 10000,
            "n_seeds": 10,
            "tuning_dir": "results/FINAL/tuning",
            "output_dir": "results/ablations/architecture",
        },
    )
    if args.mode == "quick":
        abl_cfg = get_quick_params(abl_cfg)

    datasets = args.datasets or abl_cfg["datasets"]
    n_rounds = int(abl_cfg["n_rounds"])
    seeds = list(range(int(abl_cfg["n_seeds"])))
    output_dir = args.output or abl_cfg["output_dir"]
    tuning_dir = args.tuning_dir or abl_cfg.get("tuning_dir", "results/FINAL/tuning")
    defaults = config["agent_defaults"]

    print("\n=== Architecture ablation ===")
    print(f"Datasets: {datasets}")
    print(f"n_rounds={n_rounds}, n_seeds={len(seeds)}")
    print("Variants: BayesianSPO, BayesianMSE, GreedySPO, GreedyMSE")
    print(f"Use tuned params: {args.use_tuned_params} (tuning_dir={tuning_dir})")

    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        dc = make_dataset_config(dataset, config)
        tuned_spo = load_tuned_params(tuning_dir, dataset, "BayesianSPO") if args.use_tuned_params else None
        tuned_mse = load_tuned_params(tuning_dir, dataset, "BayesianMSE") if args.use_tuned_params else None
        spo_hp = resolve_base_hp(defaults, tuned_spo, include_imputation=True)
        mse_hp = resolve_base_hp(defaults, tuned_mse, include_imputation=False)

        results = {}
        for seed_idx, seed in enumerate(seeds):
            print(f"  Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):")
            env = create_env(dataset, seed=seed, **dc)
            episode = pregenerate_episode(env, n_rounds)
            agents = make_agents(env, seed, spo_hp, mse_hp, defaults, n_rounds)
            for agent_name, agent in agents.items():
                print(f"    {agent_name}...", end=" ", flush=True)
                result = run_agent_on_episode(
                    agent,
                    episode,
                    n_rounds,
                    lr=defaults["lr"],
                    full_feedback=False,
                    diagnostics=False,
                )
                results[(agent_name, seed)] = compact_result(
                    result, keep_trajectory=args.keep_trajectory
                )
                print(f"regret={result['final_regret']:.0f}", flush=True)

        ds_dir = os.path.join(output_dir, dataset)
        save_results(results, os.path.join(ds_dir, "results.pkl"))


if __name__ == "__main__":
    main()

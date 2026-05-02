#!/usr/bin/env python
"""Empirical observation-coverage diagnostic for the masking theorem.

For each tuned BayesianSPO trajectory, this script logs the fraction of
high-uncertainty arm-rounds that are actually selected. The output is compact:
per-arm eligible/selected counts and per-seed summaries, not full trajectories.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from experiments.ablations.common import (
    get_quick_params,
    load_config,
    load_results,
    make_agent_solver,
    make_dataset_config,
    save_results,
    select_ablation_seeds,
)
from experiments.ablations.imputation_samples import (
    load_tuned_bayesian_spo_params,
    resolve_base_hp,
)
from src.agents.base import pregenerate_episode
from src.agents.contextual import BayesianSPOAgent
from src.envs import create_env


def _quantile_label(q):
    return f"q{float(q):g}"


def _make_agent(env, seed, base_hp):
    minimize = getattr(env, "minimize", False)
    return BayesianSPOAgent(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        hidden_dim=base_hp["hidden_dim"],
        lambda_reg=base_hp["lambda_reg"],
        lr=base_hp["lr"],
        solver=make_agent_solver(env),
        seed=seed,
        beta=base_hp["beta"],
        mask_grad=True,
        minimize=minimize,
        imputation_samples=int(base_hp.get("imputation_samples", 1)),
        imputation_scale=base_hp.get("imputation_scale", 1.0),
        use_posterior_predictive=base_hp.get("use_posterior_predictive", False),
        noise_var=base_hp.get("noise_var", 1.0),
        exploration_strategy="ucb",
    )


def _run_coverage_episode(dataset, dataset_config, seed, n_rounds, base_hp, threshold_quantiles):
    env = create_env(dataset, seed=seed, **dataset_config)
    episode = pregenerate_episode(env, n_rounds)
    agent = _make_agent(env, seed, base_hp)
    minimize = bool(episode["minimize"])

    sigma_rows = []
    selected_rows = []
    total_regret = 0.0
    total_reward = 0.0
    total_optimal = 0.0

    for t, rd in enumerate(episode["rounds_data"]):
        X = rd["context"]
        true_vals = np.asarray(rd["true_values"], dtype=float).reshape(-1)
        optimal_val = float(rd["optimal_value"])

        _, sigmas = agent.predict_mean_and_uncertainty(X)
        sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
        selection = np.asarray(agent.act(X), dtype=float).reshape(-1)
        selected = selection > 0.5

        agent_val = float(np.sum(true_vals * selection))
        if minimize:
            instant_regret = agent_val - optimal_val
            reward = -agent_val
        else:
            instant_regret = optimal_val - agent_val
            reward = agent_val

        total_regret += instant_regret
        total_reward += reward
        total_optimal += optimal_val

        observed = np.zeros_like(true_vals)
        observed[selected] = true_vals[selected]
        agent.update(X, selection, observed, lr=base_hp["lr"], full_feedback=False)

        sigma_rows.append(sigmas)
        selected_rows.append(selected)

        if (t + 1) % 1000 == 0:
            print(f"      round {t + 1}/{n_rounds}", flush=True)

    sigma_matrix = np.vstack(sigma_rows)
    selected_matrix = np.vstack(selected_rows)
    obs_counts = selected_matrix.sum(axis=0).astype(int)
    observation_rate = float(selected_matrix.mean())

    thresholds = {}
    for q in threshold_quantiles:
        label = _quantile_label(q)
        epsilon = float(np.quantile(sigma_matrix, float(q)))
        eligible = sigma_matrix >= epsilon
        selected_eligible = eligible & selected_matrix

        eligible_counts = eligible.sum(axis=0).astype(int)
        selected_eligible_counts = selected_eligible.sum(axis=0).astype(int)
        coverage = np.divide(
            selected_eligible_counts,
            eligible_counts,
            out=np.full_like(selected_eligible_counts, np.nan, dtype=float),
            where=eligible_counts > 0,
        )
        valid = np.isfinite(coverage)
        if valid.any():
            min_idx = int(np.nanargmin(coverage))
            p_min = float(np.nanmin(coverage))
            p10 = float(np.nanpercentile(coverage, 10))
            p25 = float(np.nanpercentile(coverage, 25))
            p50 = float(np.nanmedian(coverage))
            p_mean = float(np.nanmean(coverage))
        else:
            min_idx = -1
            p_min = p10 = p25 = p50 = p_mean = np.nan

        thresholds[label] = {
            "quantile": float(q),
            "epsilon": epsilon,
            "p_min": p_min,
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p_mean": p_mean,
            "min_arm": min_idx,
            "eligible_arms": int(valid.sum()),
            "eligible_rate": float(eligible.mean()),
            "eligible_counts": eligible_counts.tolist(),
            "selected_eligible_counts": selected_eligible_counts.tolist(),
            "coverage_by_arm": coverage.tolist(),
        }

    return {
        "n_rounds": int(n_rounds),
        "final_regret": float(total_regret),
        "final_reward": float(total_reward),
        "final_optimal": float(total_optimal),
        "observation_rate": observation_rate,
        "obs_counts": obs_counts.tolist(),
        "threshold_source": "within-run quantile over all pre-action arm posterior widths",
        "thresholds": thresholds,
    }


def _result_is_compatible(result, n_rounds, threshold_quantiles):
    if not isinstance(result, dict):
        return False
    if int(result.get("n_rounds", -1)) != int(n_rounds):
        return False
    thresholds = result.get("thresholds", {})
    expected = {_quantile_label(q) for q in threshold_quantiles}
    return expected.issubset(set(thresholds.keys()))


def write_summary(output_dir, datasets):
    rows = []
    for dataset in datasets:
        path = os.path.join(output_dir, dataset, "results.pkl")
        if not os.path.exists(path):
            continue
        data = load_results(path)
        for key, result in data.items():
            if not (isinstance(key, tuple) and len(key) == 2):
                continue
            agent_name, seed = key
            for threshold_label, threshold in result.get("thresholds", {}).items():
                rows.append(
                    {
                        "dataset": dataset,
                        "agent": agent_name,
                        "seed": seed,
                        "threshold": threshold_label,
                        "quantile": threshold["quantile"],
                        "epsilon": threshold["epsilon"],
                        "p_min": threshold["p_min"],
                        "p10": threshold["p10"],
                        "p25": threshold["p25"],
                        "p50": threshold["p50"],
                        "p_mean": threshold["p_mean"],
                        "eligible_rate": threshold["eligible_rate"],
                        "observation_rate": result["observation_rate"],
                        "final_regret": result["final_regret"],
                    }
                )

    if not rows:
        print("No p_min coverage rows to summarize yet")
        return

    os.makedirs(output_dir, exist_ok=True)
    long_df = pd.DataFrame(rows)
    long_path = os.path.join(output_dir, "pmin_coverage_long.csv")
    long_df.to_csv(long_path, index=False)

    summary = (
        long_df.groupby(["dataset", "threshold"], as_index=False)
        .agg(
            n=("p_min", "count"),
            p_min_mean=("p_min", "mean"),
            p_min_std=("p_min", "std"),
            p10_mean=("p10", "mean"),
            p50_mean=("p50", "mean"),
            eligible_rate_mean=("eligible_rate", "mean"),
            observation_rate_mean=("observation_rate", "mean"),
            regret_mean=("final_regret", "mean"),
        )
    )
    summary["p_min_se"] = summary["p_min_std"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))
    summary_path = os.path.join(output_dir, "pmin_coverage_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Summary saved to {summary_path}")
    print(summary[["dataset", "threshold", "n", "p_min_mean", "p10_mean", "p50_mean"]])


def main():
    parser = argparse.ArgumentParser(description="Empirical p_min coverage diagnostic")
    parser.add_argument("--config", default="experiments/configs/experiment.yaml")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--tuning-dir", default=None)
    parser.add_argument("--threshold-quantiles", nargs="+", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config["ablations"].get("pmin_coverage", {})
    quick = args.mode == "quick"
    if quick:
        abl_cfg = get_quick_params(abl_cfg)

    datasets = args.datasets or abl_cfg.get(
        "datasets",
        ["knapsack_32", "knapsack_100", "shortest_path_8x8", "shortest_path_20x20", "portfolio_32"],
    )
    threshold_quantiles = args.threshold_quantiles or abl_cfg.get("threshold_quantiles", [0.5, 0.75])
    threshold_quantiles = [float(q) for q in threshold_quantiles]
    n_rounds = int(abl_cfg.get("n_rounds", 10000))
    n_seeds = int(abl_cfg.get("n_seeds", 10))
    seeds = select_ablation_seeds(config, n_seeds, quick=quick)
    output_dir = args.output or abl_cfg.get("output_dir", "results/ablations/pmin_coverage")
    tuning_dir = args.tuning_dir or abl_cfg.get("tuning_dir", "results/FINAL/tuning")
    defaults = config["agent_defaults"]

    print("\n=== Empirical p_min coverage diagnostic ===")
    print(f"datasets={datasets}")
    print(f"threshold_quantiles={threshold_quantiles}")
    print(f"n_rounds={n_rounds}, seeds={seeds}")
    print(f"tuning_dir={tuning_dir}")

    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        dataset_config = make_dataset_config(dataset, config)
        tuned = load_tuned_bayesian_spo_params(tuning_dir, dataset)
        if tuned is None:
            print(f"  warning: no tuned BayesianSPO params for {dataset}; using agent defaults")
        base_hp = resolve_base_hp(None, defaults, tuned)

        ds_dir = os.path.join(output_dir, dataset)
        result_path = os.path.join(ds_dir, "results.pkl")
        results = {}
        if os.path.exists(result_path) and not args.no_resume:
            loaded = load_results(result_path)
            for key, value in loaded.items():
                if (
                    isinstance(key, tuple)
                    and len(key) == 2
                    and key[0] == "BayesianSPO"
                    and key[1] in seeds
                    and _result_is_compatible(value, n_rounds, threshold_quantiles)
                ):
                    results[key] = value
            print(f"  resuming from {result_path} ({len(results)}/{len(loaded)} matching entries)")

        for seed_idx, seed in enumerate(seeds):
            key = ("BayesianSPO", seed)
            if key in results:
                print(f"  Seed {seed_idx + 1}/{len(seeds)} (seed={seed})... skipped")
                continue

            print(f"  Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):")
            result = _run_coverage_episode(
                dataset=dataset,
                dataset_config=dataset_config,
                seed=seed,
                n_rounds=n_rounds,
                base_hp=base_hp,
                threshold_quantiles=threshold_quantiles,
            )
            results[key] = result
            save_results(results, result_path)
            med = result["thresholds"].get("q0.5", {})
            print(
                f"    regret={result['final_regret']:.1f}, "
                f"p_min(q0.5)={med.get('p_min', np.nan):.4f}, "
                f"p50(q0.5)={med.get('p50', np.nan):.4f}",
                flush=True,
            )

        save_results(results, result_path)
        write_summary(output_dir, datasets)

    write_summary(output_dir, datasets)


if __name__ == "__main__":
    main()

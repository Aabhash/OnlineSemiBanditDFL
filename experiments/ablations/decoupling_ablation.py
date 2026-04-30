#!/usr/bin/env python
"""Coupled-vs-decoupled feature/head training ablation."""

import argparse
import os
import pickle
import sys

import numpy as np

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
from src.agents.contextual import BayesianSPOAgent, DeterministicSPOAgent  # noqa: E402
from src.envs import create_env  # noqa: E402


def load_existing_results(path):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


class EpsilonGreedyWrapper:
    """Apply the same epsilon-greedy exploration rule to an arbitrary greedy agent."""

    def __init__(self, agent, solver, n_arms, epsilon=0.1, decay_epsilon=True, decay_rounds=10000, seed=0):
        self.agent = agent
        self.solver = solver
        self.n_arms = n_arms
        self.epsilon = float(epsilon)
        self.decay_epsilon = bool(decay_epsilon)
        self.decay_rounds = int(decay_rounds)
        self.rng = np.random.default_rng(seed + 35507)
        self.t = 0

    def _current_epsilon(self):
        if not self.decay_epsilon:
            return self.epsilon
        return self.epsilon * max(0.0, 1.0 - self.t / max(self.decay_rounds, 1))

    def act(self, X):
        self.t += 1
        if self.rng.random() < self._current_epsilon():
            return self.solver(self.rng.standard_normal(self.n_arms))
        return self.agent.act(X)

    def update(self, *args, **kwargs):
        return self.agent.update(*args, **kwargs)

    def predict_mean(self, X):
        return self.agent.predict_mean(X)

    def predict_uncertainty(self, X):
        if hasattr(self.agent, "predict_uncertainty"):
            return self.agent.predict_uncertainty(X)
        return np.zeros(self.n_arms)

    def predict_mean_and_uncertainty(self, X):
        if hasattr(self.agent, "predict_mean_and_uncertainty"):
            return self.agent.predict_mean_and_uncertainty(X)
        return self.predict_mean(X), self.predict_uncertainty(X)


def make_agents(env, seed, spo_hp, defaults, n_rounds, lr_multipliers):
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

    decoupled_ucb = BayesianSPOAgent(
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
    )
    decoupled_greedy = BayesianSPOAgent(
        **base_common,
        hidden_dim=spo_hp["hidden_dim"],
        lambda_reg=spo_hp["lambda_reg"],
        lr=spo_hp["lr"],
        beta=spo_hp["beta"],
        mask_grad=True,
        exploration_strategy="greedy",
        imputation_samples=int(spo_hp.get("imputation_samples", 1)),
        imputation_scale=spo_hp.get("imputation_scale", 1.0),
        use_posterior_predictive=spo_hp.get("use_posterior_predictive", False),
        noise_var=spo_hp.get("noise_var", 1.0),
    )

    agents = {
        "DecoupledSPO-UCB": decoupled_ucb,
        "DecoupledSPO-Eps": EpsilonGreedyWrapper(
            decoupled_greedy,
            solver=solver,
            n_arms=env.n_arms,
            epsilon=epsilon,
            decay_epsilon=decay_epsilon,
            decay_rounds=n_rounds,
            seed=seed,
        ),
    }

    for multiplier in lr_multipliers:
        multiplier = float(multiplier)
        agents[f"CoupledSPO-Eps-x{multiplier:g}"] = DeterministicSPOAgent(
            **base_common,
            hidden_dim=spo_hp["hidden_dim"],
            lr=spo_hp["lr"] * multiplier,
            epsilon=epsilon,
            decay_epsilon=decay_epsilon,
            decay_rounds=n_rounds,
            mask_grad=True,
        )
    return agents


def main():
    parser = argparse.ArgumentParser(description="Coupled-vs-decoupled training ablation")
    parser.add_argument("--config", default="experiments/configs/experiment.yaml")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--use-tuned-params", action="store_true")
    parser.add_argument("--tuning-dir", default=None)
    parser.add_argument("--keep-trajectory", action="store_true", default=True)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config["ablations"].get(
        "decoupling_ablation",
        {
            "datasets": ["knapsack_32", "knapsack_100", "shortest_path_8x8", "shortest_path_20x20", "portfolio_32"],
            "lr_multipliers": [0.5, 1.0, 2.0],
            "n_rounds": 10000,
            "n_seeds": 10,
            "tuning_dir": "results/FINAL/tuning",
            "output_dir": "results/ablations/decoupling",
        },
    )
    if args.mode == "quick":
        abl_cfg = get_quick_params(abl_cfg)

    datasets = args.datasets or abl_cfg["datasets"]
    n_rounds = int(abl_cfg["n_rounds"])
    seeds = list(range(int(abl_cfg["n_seeds"])))
    lr_multipliers = [float(v) for v in abl_cfg.get("lr_multipliers", [0.5, 1.0, 2.0])]
    output_dir = args.output or abl_cfg["output_dir"]
    tuning_dir = args.tuning_dir or abl_cfg.get("tuning_dir", "results/FINAL/tuning")
    defaults = config["agent_defaults"]
    keep_trajectory = args.keep_trajectory and not args.summary_only

    print("\n=== Coupled-vs-decoupled ablation ===")
    print(f"Datasets: {datasets}")
    print(f"n_rounds={n_rounds}, n_seeds={len(seeds)}")
    print(f"Coupled lr multipliers: {lr_multipliers}")
    print(f"Use tuned params: {args.use_tuned_params} (tuning_dir={tuning_dir})")

    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        dc = make_dataset_config(dataset, config)
        tuned_spo = load_tuned_params(tuning_dir, dataset, "BayesianSPO") if args.use_tuned_params else None
        spo_hp = resolve_base_hp(defaults, tuned_spo, include_imputation=True)

        ds_dir = os.path.join(output_dir, dataset)
        result_path = os.path.join(ds_dir, "results.pkl")
        results = {} if args.no_resume else load_existing_results(result_path)
        for seed_idx, seed in enumerate(seeds):
            print(f"  Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):")
            env = create_env(dataset, seed=seed, **dc)
            episode = pregenerate_episode(env, n_rounds)
            agents = make_agents(env, seed, spo_hp, defaults, n_rounds, lr_multipliers)
            for agent_name, agent in agents.items():
                key = (agent_name, seed)
                if key in results:
                    print(f"    {agent_name}... skip", flush=True)
                    continue
                print(f"    {agent_name}...", end=" ", flush=True)
                result = run_agent_on_episode(
                    agent,
                    episode,
                    n_rounds,
                    lr=defaults["lr"],
                    full_feedback=False,
                    diagnostics=False,
                )
                results[key] = compact_result(result, keep_trajectory=keep_trajectory)
                print(f"regret={result['final_regret']:.0f}", flush=True)
                save_results(results, result_path)

        save_results(results, result_path)


if __name__ == "__main__":
    main()

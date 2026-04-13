#!/usr/bin/env python
import argparse
import importlib
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, '.')

plt = importlib.import_module('matplotlib.pyplot') if importlib.util.find_spec('matplotlib.pyplot') is not None else None

from experiments.ablations.common import (
    get_quick_params,
    load_config,
    make_co_solver,
    make_dataset_config,
    run_ablation_sweep,
    save_results,
)
from src.agents.contextual import (
    BayesianMSEAgent,
    BayesianSPOAgent,
    BayesianSPO_DFE_Agent,
)


AGENT_ORDER = [
    'MSE+UCB',
    'SPO+Greedy',
    'SPO+UCB',
    'SPO-unmasked',
    'SPO+DFE',
    'SPO+DI',
]


def make_agents(env, seed, config):
    defaults = config['agent_defaults']
    hidden_dim = defaults['hidden_dim_small']
    beta = defaults['beta']
    lambda_reg = defaults['lambda_reg']
    minimize = getattr(env, 'minimize', False)
    bayes_beta = -beta if minimize else beta
    solver = env.solve
    co_solver = make_co_solver(env)

    common_base = dict(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        hidden_dim=hidden_dim,
        lambda_reg=lambda_reg,
        solver=solver,
        seed=seed,
    )
    common_spo = dict(**common_base, minimize=minimize)

    return {
        'MSE+UCB': BayesianMSEAgent(
            **common_base,
            beta=bayes_beta,
        ),
        'SPO+Greedy': BayesianSPOAgent(
            **common_spo,
            beta=0.0,
            mask_grad=True,
        ),
        'SPO+UCB': BayesianSPOAgent(
            **common_spo,
            beta=bayes_beta,
            mask_grad=True,
        ),
        'SPO-unmasked': BayesianSPOAgent(
            **common_spo,
            beta=bayes_beta,
            mask_grad=False,
        ),
        'SPO+DFE': BayesianSPO_DFE_Agent(
            **common_spo,
            beta=bayes_beta,
            imputation='disc_explore',
            co_solver=co_solver,
            decision_focused_explore=True,
            adaptive_dfe=True,
            mask_grad=True,
        ),
        'SPO+DI': BayesianSPO_DFE_Agent(
            **common_spo,
            beta=bayes_beta,
            imputation='disc_impute',
            co_solver=co_solver,
        ),
    }


def _tail_metric_mean(results, agent_name, seeds, metric_name, window=100):
    vals_all = []
    for seed in seeds:
        key = (agent_name, seed)
        if key not in results or metric_name not in results[key]:
            continue
        vals = results[key][metric_name]
        vals = [v for v in vals[-window:] if np.isfinite(v)]
        if vals:
            vals_all.append(np.mean(vals))
    return np.mean(vals_all) if vals_all else np.nan


def plot_results(all_results, seeds, output_dir, dataset):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    agent_names = [name for name in AGENT_ORDER if any((name, s) in all_results for s in seeds)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    for name in agent_names:
        trajs = [
            all_results[(name, s)]['cumulative_regret']
            for s in seeds
            if (name, s) in all_results
        ]
        if trajs:
            mean_traj = np.mean(trajs, axis=0)
            ax.plot(mean_traj, label=name)
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Regret Trajectories')
    ax.legend(fontsize=8)

    x = np.arange(len(agent_names))

    ax = axes[0, 1]
    mean_regrets = []
    std_regrets = []
    for name in agent_names:
        regrets = [
            all_results[(name, s)]['final_regret']
            for s in seeds
            if (name, s) in all_results
        ]
        mean_regrets.append(np.mean(regrets) if regrets else np.nan)
        std_regrets.append(np.std(regrets) if regrets else 0.0)
    ax.bar(x, mean_regrets, yerr=std_regrets, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=35, ha='right')
    ax.set_ylabel('Final Regret')
    ax.set_title('Final Regret')

    ax = axes[1, 0]
    flip_mae = [
        _tail_metric_mean(all_results, name, seeds, 'flip_contested_mae')
        for name in agent_names
    ]
    ax.bar(x, flip_mae, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=35, ha='right')
    ax.set_ylabel('Flip-Set MAE (last 100)')
    ax.set_title('Target / Imputation Quality')

    ax = axes[1, 1]
    contested_sigma = [
        _tail_metric_mean(all_results, name, seeds, 'contested_sigma_sum')
        for name in agent_names
    ]
    ax.bar(x, contested_sigma, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=35, ha='right')
    ax.set_ylabel('Contested Sigma Sum (last 100)')
    ax.set_title('Observation Allocation Pressure')

    plt.suptitle(f'Mechanism Ablation: {dataset}', fontsize=14)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'mechanism_ablation.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'mechanism_ablation.png'), dpi=150, bbox_inches='tight')
    print(f'Plots saved to {output_dir}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Mechanism ablation')
    parser.add_argument('--config', default='experiments/configs/experiment.yaml')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--output', default=None)
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: config datasets)')
    parser.add_argument('--no-plot', action='store_true', help='Save data only, skip plotting')
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config['ablations']['mechanism']

    if args.mode == 'quick':
        abl_cfg = get_quick_params(abl_cfg)

    datasets = args.datasets or abl_cfg.get('datasets', ['knapsack_32'])
    n_rounds = abl_cfg['n_rounds']
    n_seeds = abl_cfg['n_seeds']
    seeds = list(range(n_seeds))
    output_dir = args.output or abl_cfg['output_dir']
    lr = config['agent_defaults']['lr']

    for dataset in datasets:
        print(f'\n=== Mechanism Ablation: {dataset} ===')
        dc = make_dataset_config(dataset, config)

        results = run_ablation_sweep(
            make_agents_fn=lambda env, seed: make_agents(env, seed, config),
            dataset=dataset,
            dataset_config=dc,
            seeds=seeds,
            n_rounds=n_rounds,
            lr=lr,
            diagnostics=True,
            output_dir=output_dir,
        )

        ds_dir = os.path.join(output_dir, dataset)
        save_results(results, os.path.join(ds_dir, 'results.pkl'))
        if not args.no_plot:
            plot_results(results, seeds, ds_dir, dataset)


if __name__ == '__main__':
    main()

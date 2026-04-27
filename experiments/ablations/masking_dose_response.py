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
    make_dataset_config,
    run_ablation_sweep,
    save_results,
)
from experiments.ablations.imputation_samples import (
    load_tuned_bayesian_spo_params,
    resolve_base_hp,
)
from src.agents.contextual import BayesianSPOAgent


def agent_name(weight):
    return f'w{weight:g}'


def make_agents(env, seed, base_hp, weights):
    minimize = getattr(env, 'minimize', False)
    solver = env.solve

    common = dict(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        hidden_dim=base_hp['hidden_dim'],
        lambda_reg=base_hp['lambda_reg'],
        lr=base_hp['lr'],
        solver=solver,
        seed=seed,
        beta=base_hp['beta'],
        mask_grad=True,
        minimize=minimize,
        imputation_samples=int(base_hp.get('imputation_samples', 1)),
        imputation_scale=base_hp.get('imputation_scale', 1.0),
        use_posterior_predictive=base_hp.get('use_posterior_predictive', False),
        noise_var=base_hp.get('noise_var', 1.0),
    )

    return {
        agent_name(weight): BayesianSPOAgent(
            **common,
            missing_grad_weight=float(weight),
        )
        for weight in weights
    }


def plot_results(results, weights, seeds, output_dir, dataset):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    means, stds = [], []
    for weight in weights:
        name = agent_name(weight)
        regrets = [
            results[(name, seed)]['final_regret']
            for seed in seeds
            if (name, seed) in results
        ]
        means.append(np.mean(regrets) if regrets else np.nan)
        stds.append(np.std(regrets) if regrets else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(weights, means, yerr=stds, marker='o', capsize=3)
    ax.set_xlabel('Missing-arm gradient weight')
    ax.set_ylabel('Final cumulative regret')
    ax.set_title(f'Masking dose-response: {dataset}')
    ax.set_xscale('symlog', linthresh=0.05)
    ax.set_xticks(weights)
    ax.set_xticklabels([str(w) for w in weights])
    ax.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'masking_dose_response.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'masking_dose_response.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plots saved to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Masking dose-response ablation')
    parser.add_argument('--config', default='experiments/configs/experiment.yaml')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--output', default=None)
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: config datasets)')
    parser.add_argument('--no-plot', action='store_true', help='Save data only, skip plotting')
    parser.add_argument('--use-tuned-params', action='store_true',
                        help='Use tuned BayesianSPO hyperparameters from <tuning_dir>.')
    parser.add_argument('--tuning-dir', default=None,
                        help='Override tuning dir (default: ablations.masking_dose_response.tuning_dir).')
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config['ablations']['masking_dose_response']

    if args.mode == 'quick':
        abl_cfg = get_quick_params(abl_cfg)
        abl_cfg['missing_grad_weights'] = [0.0, 0.25, 1.0]

    datasets = args.datasets or abl_cfg.get('datasets', ['knapsack_32'])
    weights = [float(w) for w in abl_cfg['missing_grad_weights']]
    n_rounds = abl_cfg['n_rounds']
    n_seeds = abl_cfg['n_seeds']
    seeds = list(range(n_seeds))
    output_dir = args.output or abl_cfg['output_dir']
    tuning_dir = args.tuning_dir or abl_cfg.get('tuning_dir', 'results/FINAL/tuning')
    defaults = config['agent_defaults']

    print(f'\n=== Masking dose-response ablation ===')
    print(f'Missing gradient weights: {weights}')
    print(f'Use tuned params: {args.use_tuned_params} (tuning_dir={tuning_dir})')

    for dataset in datasets:
        print(f'\n--- {dataset} ---')
        dc = make_dataset_config(dataset, config)
        tuned = load_tuned_bayesian_spo_params(tuning_dir, dataset) if args.use_tuned_params else None
        if args.use_tuned_params and tuned is None:
            print(f'  warning: no tuned BayesianSPO params for {dataset}, falling back to agent_defaults')
        base_hp = resolve_base_hp(None, defaults, tuned)

        results = run_ablation_sweep(
            make_agents_fn=lambda env, seed: make_agents(env, seed, base_hp, weights),
            dataset=dataset,
            dataset_config=dc,
            seeds=seeds,
            n_rounds=n_rounds,
            lr=base_hp['lr'],
            diagnostics=True,
            output_dir=output_dir,
        )

        ds_dir = os.path.join(output_dir, dataset)
        save_results(results, os.path.join(ds_dir, 'results.pkl'))
        if not args.no_plot:
            plot_results(results, weights, seeds, ds_dir, dataset)


if __name__ == '__main__':
    main()

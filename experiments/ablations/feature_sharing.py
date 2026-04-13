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
    load_config, make_co_solver, make_dataset_config,
    run_ablation_sweep, save_results, get_quick_params,
)
from src.agents.contextual import (
    BayesianSPOAgent, BayesianMSEAgent, LinUCBAgent,
)
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.envs import create_env


def make_agents_for_config(env, seed, config, hidden_dim, nonlinear):
    defaults = config['agent_defaults']
    beta = defaults['beta']
    lambda_reg = defaults['lambda_reg']
    minimize = getattr(env, 'minimize', False)
    bayes_beta = -beta if minimize else beta
    solver = env.solve

    common_base = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=env.weights, capacity=env.capacity,
        hidden_dim=hidden_dim, lambda_reg=lambda_reg,
        solver=solver, seed=seed, nonlinear=nonlinear,
    )

    nl_tag = 'NL' if nonlinear else 'L'
    agents = {
        f'SPO-{nl_tag}-h{hidden_dim}': BayesianSPOAgent(
            **common_base, beta=bayes_beta, mask_grad=True, minimize=minimize,
        ),
        f'MSE-{nl_tag}-h{hidden_dim}': BayesianMSEAgent(
            **common_base, beta=bayes_beta,
        ),
    }
    return agents


def plot_results(all_results, hidden_dims, seeds, output_dir):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, loss in enumerate(['SPO', 'MSE']):
        for col, (nl, nl_tag) in enumerate([(False, 'L'), (True, 'NL')]):
            ax = axes[row, col]

            means = []
            stds = []
            for hd in hidden_dims:
                name = f'{loss}-{nl_tag}-h{hd}'
                regrets = [all_results[(name, s)]['final_regret']
                           for s in seeds if (name, s) in all_results]
                means.append(np.mean(regrets) if regrets else np.nan)
                stds.append(np.std(regrets) if regrets else 0)

            ax.errorbar(hidden_dims, means, yerr=stds, marker='o', capsize=3)
            ax.set_xlabel('Hidden Dimension')
            ax.set_ylabel('Final Regret')
            ax.set_title(f'{loss} - {"Nonlinear" if nl else "Linear"}')
            ax.set_xscale('log', base=2)

                                      
    linucb_regrets = [all_results[('LinUCB', s)]['final_regret']
                       for s in seeds if ('LinUCB', s) in all_results]
    if linucb_regrets:
        linucb_mean = np.mean(linucb_regrets)
        for ax in axes.flat:
            ax.axhline(linucb_mean, color='gray', linestyle='--', label='LinUCB')
            ax.legend()

    plt.suptitle('Feature Sharing: Effect of Hidden Dimension')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'feature_sharing.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'feature_sharing.png'), dpi=150, bbox_inches='tight')
    print(f'Plots saved to {output_dir}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Feature Sharing Ablation')
    parser.add_argument('--config', default='experiments/configs/experiment.yaml')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--output', default=None)
    parser.add_argument('--no-plot', action='store_true', help='Save data only, skip plotting')
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config['ablations']['feature_sharing']

    if args.mode == 'quick':
        abl_cfg = get_quick_params(abl_cfg)
        abl_cfg['hidden_dims'] = [4, 16, 64]

    datasets = abl_cfg.get('datasets', ['knapsack_32'])
    hidden_dims = abl_cfg['hidden_dims']
    n_rounds = abl_cfg['n_rounds']
    n_seeds = abl_cfg['n_seeds']
    seeds = list(range(n_seeds))
    output_dir = args.output or abl_cfg['output_dir']
    lr = config['agent_defaults']['lr']

    for dataset in datasets:
        print(f'\n=== Feature Sharing: {dataset} ===')
        dc = make_dataset_config(dataset, config)

        all_results = {}

        for seed_idx, seed in enumerate(seeds):
            print(f'  Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):')

            env = create_env(dataset, seed=seed, **dc)
            co_solver = make_co_solver(env)
            episode = pregenerate_episode(env, n_rounds)

                                                  
            linucb = LinUCBAgent(
                env.context_dim, env.n_arms, env.weights, env.capacity,
                alpha=config['agent_defaults']['alpha'],
                lambda_reg=config['agent_defaults']['lambda_reg'],
                minimize=getattr(env, 'minimize', False),
                solver=env.solve,
            )
            print(f'    LinUCB...', end=' ', flush=True)
            result = run_agent_on_episode(
                linucb, episode, n_rounds,
                lr=lr, full_feedback=False,
                diagnostics=True, co_solver=co_solver,
            )
            all_results[('LinUCB', seed)] = result
            print(f'regret={result["final_regret"]:.0f}', flush=True)

                                                  
            for hd in hidden_dims:
                for nonlinear in [False, True]:
                    agents = make_agents_for_config(env, seed, config, hd, nonlinear)

                    for agent_name, agent in agents.items():
                        print(f'    {agent_name}...', end=' ', flush=True)
                        result = run_agent_on_episode(
                            agent, episode, n_rounds,
                            lr=lr, full_feedback=False,
                            diagnostics=True, co_solver=co_solver,
                        )
                        all_results[(agent_name, seed)] = result
                        print(f'regret={result["final_regret"]:.0f}', flush=True)

        ds_dir = os.path.join(output_dir, dataset)
        save_results(all_results, os.path.join(ds_dir, 'results.pkl'))
        if not args.no_plot:
            plot_results(all_results, hidden_dims, seeds, ds_dir)


if __name__ == '__main__':
    main()

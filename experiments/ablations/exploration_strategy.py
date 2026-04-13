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
    BayesianSPOAgent, BayesianMSEAgent, BayesianSPO_DFE_Agent,
)


def make_agents(env, seed, config):
    defaults = config['agent_defaults']
    hidden_dim = defaults['hidden_dim_small']
    lambda_reg = defaults['lambda_reg']
    minimize = getattr(env, 'minimize', False)
    solver = env.solve
    co_solver = make_co_solver(env)

    common_base = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=env.weights, capacity=env.capacity,
        hidden_dim=hidden_dim, lambda_reg=lambda_reg,
        solver=solver, seed=seed,
    )
    common_spo = dict(**common_base, minimize=minimize)

    beta_sign = -1.0 if minimize else 1.0

    agents = {
        'SPO+Greedy': BayesianSPOAgent(
            **common_spo, beta=0.0, mask_grad=True,
        ),
        'SPO+UCB': BayesianSPOAgent(
            **common_spo, beta=1.0 * beta_sign, mask_grad=True,
        ),
        'SPO+UCB-strong': BayesianSPOAgent(
            **common_spo, beta=3.0 * beta_sign, mask_grad=True,
        ),
        'SPO+DFE': BayesianSPO_DFE_Agent(
            **common_spo, beta=1.0 * beta_sign, imputation='disc_explore',
            co_solver=co_solver, decision_focused_explore=True,
            adaptive_dfe=True, mask_grad=True,
        ),
        'SPO+DI': BayesianSPO_DFE_Agent(
            **common_spo, beta=1.0 * beta_sign, imputation='disc_impute',
            co_solver=co_solver,
        ),
        'MSE+Greedy': BayesianMSEAgent(
            **common_base, beta=0.0,
        ),
        'MSE+UCB': BayesianMSEAgent(
            **common_base, beta=1.0 * beta_sign,
        ),
    }

    return agents


def plot_results(all_results, seeds, n_rounds, output_dir):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    agent_names = sorted(set(name for name, _ in all_results.keys()))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

                                                    
    ax = axes[0, 0]
    for name in agent_names:
        trajs = [all_results[(name, s)]['cumulative_regret'] for s in seeds
                 if (name, s) in all_results]
        if trajs:
            mean_traj = np.mean(trajs, axis=0)
            ax.plot(mean_traj, label=name)
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Regret Trajectories')
    ax.legend(fontsize=7)

                                                
    ax = axes[0, 1]
    for name in agent_names:
        trajs = []
        for s in seeds:
            key = (name, s)
            if key in all_results and 'flip_set_size' in all_results[key]:
                fs = all_results[key]['flip_set_size']
                fs = [v if np.isfinite(v) else 0 for v in fs]
                trajs.append(fs)
        if trajs:
            mean_traj = np.mean(trajs, axis=0)
                                        
            window = min(50, len(mean_traj) // 5) or 1
            smoothed = np.convolve(mean_traj, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=name)
    ax.set_xlabel('Round')
    ax.set_ylabel('Flip Set Size |F_t|')
    ax.set_title('Flip Set Dynamics')
    ax.legend(fontsize=7)

                                                    
    ax = axes[0, 2]
    for name in agent_names:
        trajs = []
        for s in seeds:
            key = (name, s)
            if key in all_results and 'decision_correct' in all_results[key]:
                dc = all_results[key]['decision_correct']
                dc = [v if np.isfinite(v) else 0 for v in dc]
                trajs.append(dc)
        if trajs:
            mean_traj = np.mean(trajs, axis=0)
            window = min(50, len(mean_traj) // 5) or 1
            smoothed = np.convolve(mean_traj, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=name)
    ax.set_xlabel('Round')
    ax.set_ylabel('Decision Accuracy (rolling)')
    ax.set_title('Decision Correctness')
    ax.legend(fontsize=7)

                                            
    ax = axes[1, 0]
    mean_regrets = {}
    std_regrets = {}
    for name in agent_names:
        regrets = [all_results[(name, s)]['final_regret'] for s in seeds
                    if (name, s) in all_results]
        mean_regrets[name] = np.mean(regrets)
        std_regrets[name] = np.std(regrets)

    x = np.arange(len(agent_names))
    ax.bar(x, [mean_regrets[n] for n in agent_names],
           yerr=[std_regrets[n] for n in agent_names], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha='right')
    ax.set_ylabel('Final Regret')
    ax.set_title('Final Regret')

                                                              
    ax = axes[1, 1]
    mean_fcmae = {}
    for name in agent_names:
        vals_all = []
        for s in seeds:
            key = (name, s)
            if key in all_results and 'flip_contested_mae' in all_results[key]:
                vals = all_results[key]['flip_contested_mae'][-100:]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    vals_all.append(np.mean(vals))
        mean_fcmae[name] = np.mean(vals_all) if vals_all else np.nan

    ax.bar(x, [mean_fcmae[n] for n in agent_names], color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha='right')
    ax.set_ylabel('Flip Set MAE (last 100)')
    ax.set_title('Flip Set Prediction Quality')

                                                           
    ax = axes[1, 2]
    mean_css = {}
    for name in agent_names:
        vals_all = []
        for s in seeds:
            key = (name, s)
            if key in all_results and 'contested_sigma_sum' in all_results[key]:
                vals = all_results[key]['contested_sigma_sum'][-100:]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    vals_all.append(np.mean(vals))
        mean_css[name] = np.mean(vals_all) if vals_all else np.nan

    ax.bar(x, [mean_css[n] for n in agent_names], color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha='right')
    ax.set_ylabel('Contested Sigma Sum (last 100)')
    ax.set_title('Flip Set Uncertainty')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'exploration_strategy.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'exploration_strategy.png'), dpi=150, bbox_inches='tight')
    print(f'Plots saved to {output_dir}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Exploration Strategy Ablation')
    parser.add_argument('--config', default='experiments/configs/experiment.yaml')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--output', default=None)
    parser.add_argument('--no-plot', action='store_true', help='Save data only, skip plotting')
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config['ablations']['exploration_strategy']

    if args.mode == 'quick':
        abl_cfg = get_quick_params(abl_cfg)

    datasets = abl_cfg.get('datasets', ['knapsack_32'])
    n_rounds = abl_cfg['n_rounds']
    n_seeds = abl_cfg['n_seeds']
    seeds = list(range(n_seeds))
    output_dir = args.output or abl_cfg['output_dir']

    for dataset in datasets:
        print(f'\n=== Exploration Strategy: {dataset} ===')
        dc = make_dataset_config(dataset, config)

        results = run_ablation_sweep(
            make_agents_fn=lambda env, seed: make_agents(env, seed, config),
            dataset=dataset, dataset_config=dc,
            seeds=seeds, n_rounds=n_rounds,
            lr=config['agent_defaults']['lr'],
            diagnostics=True, output_dir=output_dir,
        )

        ds_dir = os.path.join(output_dir, dataset)
        save_results(results, os.path.join(ds_dir, 'results.pkl'))
        if not args.no_plot:
            plot_results(results, seeds, n_rounds, ds_dir)


if __name__ == '__main__':
    main()

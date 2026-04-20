#!/usr/bin/env python
import argparse
import importlib
import importlib.util
import sys
import os
import numpy as np

sys.path.insert(0, '.')

plt = importlib.import_module('matplotlib.pyplot') if importlib.util.find_spec('matplotlib.pyplot') is not None else None

from experiments.ablations.common import (
    load_config, make_co_solver, make_dataset_config,
    save_results, get_quick_params,
)
from src.envs import create_env
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.agents.contextual import (
    BayesianSPOAgent, BayesianMSEAgent, LinUCBAgent,
)


def make_agents(env, seed, config):
    defaults = config['agent_defaults']
    hidden_dim = defaults['hidden_dim_small']
    beta = defaults['beta']
    lambda_reg = defaults['lambda_reg']
    minimize = getattr(env, 'minimize', False)
    solver = env.solve

    common_base = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=env.weights, capacity=env.capacity,
        hidden_dim=hidden_dim, lambda_reg=lambda_reg,
        solver=solver, seed=seed,
    )

    agents = {
        'BayesianSPO': BayesianSPOAgent(
            **common_base, beta=beta, mask_grad=True, minimize=minimize,
        ),
        'BayesianMSE': BayesianMSEAgent(
            **common_base, beta=beta, minimize=minimize,
        ),
        'LinUCB': LinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=defaults['alpha'], lambda_reg=lambda_reg,
            minimize=minimize, solver=solver,
        ),
    }
    return agents


def plot_results(all_results, noise_widths, seeds, output_dir):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    agent_names = ['BayesianSPO', 'BayesianMSE', 'LinUCB']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    ax = axes[0]
    for name in agent_names:
        means = []
        stds = []
        for nw in noise_widths:
            regrets = [all_results[(name, nw, s)]['final_regret']
                       for s in seeds if (name, nw, s) in all_results]
            means.append(np.mean(regrets) if regrets else np.nan)
            stds.append(np.std(regrets) if regrets else 0)
        ax.errorbar(noise_widths, means, yerr=stds, marker='o', label=name, capsize=3)

    ax.set_xlabel('Noise Width')
    ax.set_ylabel('Final Cumulative Regret')
    ax.set_title('Context Variation: Regret vs Noise')
    ax.legend()


    ax = axes[1]
    gaps = []
    for nw in noise_widths:
        spo = [all_results[('BayesianSPO', nw, s)]['final_regret']
               for s in seeds if ('BayesianSPO', nw, s) in all_results]
        mse = [all_results[('BayesianMSE', nw, s)]['final_regret']
               for s in seeds if ('BayesianMSE', nw, s) in all_results]
        if spo and mse:
            gaps.append(np.mean(mse) - np.mean(spo))
        else:
            gaps.append(np.nan)

    ax.bar(range(len(noise_widths)), gaps, tick_label=[str(nw) for nw in noise_widths],
           color='purple', alpha=0.7)
    ax.set_xlabel('Noise Width')
    ax.set_ylabel('MSE Regret - SPO Regret')
    ax.set_title('SPO+ Advantage (MSE - SPO)')
    ax.axhline(0, color='gray', linestyle='--')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'context_variation.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'context_variation.png'), dpi=150, bbox_inches='tight')
    print(f'Plots saved to {output_dir}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Context Variation Ablation')
    parser.add_argument('--config', default='experiments/configs/experiment.yaml')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--output', default=None)
    parser.add_argument('--no-plot', action='store_true', help='Save data only, skip plotting')
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config['ablations']['context_variation']

    if args.mode == 'quick':
        abl_cfg = get_quick_params(abl_cfg)
        abl_cfg['noise_widths'] = [0.0, 0.25, 1.0]

    dataset = abl_cfg['dataset']
    noise_widths = abl_cfg['noise_widths']
    n_rounds = abl_cfg['n_rounds']
    n_seeds = abl_cfg['n_seeds']
    seeds = list(range(n_seeds))
    output_dir = args.output or abl_cfg['output_dir']
    lr = config['agent_defaults']['lr']

    print(f'\n=== Context Variation: {dataset} ===')
    print(f'noise_widths={noise_widths}')

    all_results = {}

    for nw in noise_widths:
        print(f'\n  noise_width={nw}:')

        for seed_idx, seed in enumerate(seeds):
            print(f'    Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):')

            dc = make_dataset_config(dataset, config, noise_width=nw)
            env = create_env(dataset, seed=seed, **dc)
            co_solver = make_co_solver(env)
            episode = pregenerate_episode(env, n_rounds)

            agents = make_agents(env, seed, config)

            for agent_name, agent in agents.items():
                print(f'      {agent_name}...', end=' ', flush=True)
                result = run_agent_on_episode(
                    agent, episode, n_rounds,
                    lr=lr, full_feedback=False,
                    diagnostics=True, co_solver=co_solver,
                )
                all_results[(agent_name, nw, seed)] = result
                print(f'regret={result["final_regret"]:.0f}', flush=True)

    save_results(all_results, os.path.join(output_dir, 'results.pkl'))
    if not args.no_plot:
        plot_results(all_results, noise_widths, seeds, output_dir)


if __name__ == '__main__':
    main()

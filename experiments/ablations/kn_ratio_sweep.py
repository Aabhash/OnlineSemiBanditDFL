#!/usr/bin/env python
"""Ablation: k/n Ratio Sweep.

Hypothesis: SPO+ advantage grows as k/n decreases (tighter capacity).

Fix n_items=32, unit_weights=True. Sweep capacity [4,8,12,16,20,24,28].
Pregenerate episode once (costs unchanged), recompute_optimals per capacity.

Plot: X=k/n, Y=final regret per agent + SPO advantage % on secondary axis.
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, '.')

from experiments.ablations.common import (
    load_config, make_dataset_config, recompute_optimals,
    save_results, get_quick_params,
)
from src.envs import create_env
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.solvers.knapsack import KnapsackSolver
from src.agents.shared_context import (
    BayesianSPOAgent, BayesianMSEAgent, RandomAgent,
)
from src.models import BayesianLinearModel
from src.agents.shared_context.ucb_agents import UCBAgent


def make_agents(env, seed, config, capacity):
    defaults = config['agent_defaults']
    hidden_dim = defaults['hidden_dim_small']
    beta = defaults['beta']
    lambda_reg = defaults['lambda_reg']

    weights = np.ones(env.n_arms)
    solver_fn = KnapsackSolver(weights, capacity).solve

    common = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=weights, capacity=capacity,
        hidden_dim=hidden_dim, lambda_reg=lambda_reg,
        solver=solver_fn, seed=seed,
    )

    agents = {
        'BayesianSPO': BayesianSPOAgent(
            **common, beta=beta, mask_grad=True, minimize=False,
        ),
        'BayesianMSE': BayesianMSEAgent(
            **common, beta=beta,
        ),
        'UCB-Bayesian': UCBAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            weights, capacity, beta=beta, minimize=False, solver=solver_fn,
        ),
        'Random': RandomAgent(
            env.n_arms, weights, capacity, seed=seed,
        ),
    }
    return agents


def plot_results(all_results, capacities, n_items, seeds, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plots')
        return

    ratios = [cap / n_items for cap in capacities]
    agent_names = ['BayesianSPO', 'BayesianMSE', 'UCB-Bayesian', 'Random']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for name in agent_names:
        means = []
        for cap in capacities:
            regrets = [all_results[(name, cap, s)]['final_regret']
                       for s in seeds if (name, cap, s) in all_results]
            means.append(np.mean(regrets) if regrets else np.nan)
        ax1.plot(ratios, means, 'o-', label=name)

    ax1.set_xlabel('k/n (capacity ratio)')
    ax1.set_ylabel('Final Cumulative Regret')
    ax1.set_title('k/n Ratio Sweep: SPO+ Advantage vs Capacity Tightness')
    ax1.legend(loc='upper left')

    # Secondary axis: SPO advantage %
    ax2 = ax1.twinx()
    spo_advantage = []
    for cap in capacities:
        spo_regrets = [all_results[('BayesianSPO', cap, s)]['final_regret']
                       for s in seeds if ('BayesianSPO', cap, s) in all_results]
        mse_regrets = [all_results[('BayesianMSE', cap, s)]['final_regret']
                       for s in seeds if ('BayesianMSE', cap, s) in all_results]
        if spo_regrets and mse_regrets:
            mse_mean = np.mean(mse_regrets)
            spo_mean = np.mean(spo_regrets)
            adv = (mse_mean - spo_mean) / max(mse_mean, 1e-8) * 100
            spo_advantage.append(adv)
        else:
            spo_advantage.append(np.nan)

    ax2.bar(ratios, spo_advantage, width=0.03, alpha=0.3, color='purple', label='SPO+ advantage %')
    ax2.set_ylabel('SPO+ Advantage (%)')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'kn_ratio_sweep.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'kn_ratio_sweep.png'), dpi=150, bbox_inches='tight')
    print(f'Plots saved to {output_dir}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='k/n Ratio Sweep Ablation')
    parser.add_argument('--config', default='experiments/configs/experiment.yaml')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config['ablations']['kn_ratio']

    if args.mode == 'quick':
        abl_cfg = get_quick_params(abl_cfg)
        abl_cfg['capacities'] = [4, 16, 28]  # fewer capacities in quick mode

    n_items = abl_cfg['n_items']
    capacities = abl_cfg['capacities']
    n_rounds = abl_cfg['n_rounds']
    n_seeds = abl_cfg['n_seeds']
    seeds = list(range(n_seeds))
    output_dir = args.output or abl_cfg['output_dir']
    lr = config['agent_defaults']['lr']

    print(f'\n=== k/n Ratio Sweep ===')
    print(f'n_items={n_items}, capacities={capacities}')

    all_results = {}

    for seed_idx, seed in enumerate(seeds):
        print(f'\n  Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):')

        # Create env with unit weights and generous capacity for episode generation
        dc = make_dataset_config('knapsack_32', config,
                                 unit_weights=True, capacity=float(max(capacities)))
        env = create_env('knapsack_32', seed=seed, **dc)

        # Pregenerate episode once with max capacity
        base_episode = pregenerate_episode(env, n_rounds)

        for cap in capacities:
            print(f'    capacity={cap} (k/n={cap/n_items:.3f}):')

            solver = KnapsackSolver(np.ones(n_items), float(cap))
            episode = recompute_optimals(base_episode, solver.solve)
            co_solver = solver

            agents = make_agents(env, seed, config, float(cap))

            for agent_name, agent in agents.items():
                print(f'      {agent_name}...', end=' ', flush=True)

                result = run_agent_on_episode(
                    agent, episode, n_rounds,
                    lr=lr, full_feedback=False,
                    diagnostics=True, co_solver=co_solver,
                )
                all_results[(agent_name, cap, seed)] = result
                print(f'regret={result["final_regret"]:.0f}', flush=True)

    save_results(all_results, os.path.join(output_dir, 'results.pkl'))
    plot_results(all_results, capacities, n_items, seeds, output_dir)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
import argparse
import importlib
import importlib.util
import sys
import os
import copy
import warnings
import numpy as np

sys.path.insert(0, '.')

plt = importlib.import_module('matplotlib.pyplot') if importlib.util.find_spec('matplotlib.pyplot') is not None else None

from experiments.ablations.common import (
    load_config, make_co_solver, make_dataset_config,
    save_results, get_quick_params,
)
from src.envs import create_env
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.agents.contextual import BayesianSPOAgent, BayesianMSEAgent


def make_fixed_context_episode(episode, n_rounds):
    ep = copy.deepcopy(episode)
    fixed_ctx = ep['rounds_data'][0]['context'].copy()
    for rd in ep['rounds_data'][:n_rounds]:
        rd['context'] = fixed_ctx.copy()
    return ep


def make_agents(env, seed, config, n_rounds):
    defaults = config['agent_defaults']
    hidden_dim = defaults['hidden_dim_small']
    beta = defaults['beta']
    lambda_reg = defaults['lambda_reg']
    minimize = getattr(env, 'minimize', False)
    solver = env.solve

    common = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=env.weights, capacity=env.capacity,
        hidden_dim=hidden_dim, lambda_reg=lambda_reg,
        solver=solver, seed=seed,
    )

    agents = {
        'BayesianSPO': BayesianSPOAgent(
            **common, beta=beta, mask_grad=True,
            minimize=minimize, decay_rounds=n_rounds,
        ),
        'BayesianMSE': BayesianMSEAgent(
            **common, beta=beta, decay_rounds=n_rounds, minimize=minimize,
        ),
    }
    return agents


def estimate_T0(flip_sizes, n_arms, threshold_frac=0.1, window=50):
    threshold = threshold_frac * n_arms
    if len(flip_sizes) < window:
        return len(flip_sizes)
    smoothed = np.convolve(flip_sizes, np.ones(window) / window, mode='valid')
    below = np.where(smoothed < threshold)[0]
    if len(below) == 0:
        return len(flip_sizes)
    return int(below[0] + window // 2)


def plot_results(all_results, datasets, seeds, n_rounds, output_dir):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    agent_names = ['BayesianSPO', 'BayesianMSE']
    regimes = ['changing', 'fixed']
    colors = {'BayesianSPO': 'purple', 'BayesianMSE': 'orange'}
    styles = {'changing': '-', 'fixed': '--'}

    for ds_name in datasets:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        metrics = [
            ('flip_set_size', 'Flip Set Size |F_t|'),
            ('contested_mae', 'Contested MAE'),
            ('noncontested_mae', 'Non-Contested MAE'),
            ('decision_correct', 'Decision Accuracy'),
            ('contested_sigma_sum', 'Contested Sigma Sum'),
            ('rank_corr', 'Rank Correlation'),
        ]

        for ax_idx, (metric, title) in enumerate(metrics):
            ax = axes[ax_idx // 3, ax_idx % 3]

            for regime in regimes:
                for agent_name in agent_names:
                    traces = []
                    for s in seeds:
                        key = (agent_name, regime, ds_name, s)
                        if key in all_results and metric in all_results[key]:
                            vals = all_results[key][metric]
                            if len(vals) > 0 and not all(
                                v is None or (isinstance(v, float) and np.isnan(v))
                                for v in vals
                            ):
                                trace = np.array(
                                    [v if v is not None and np.isfinite(v) else np.nan for v in vals]
                                )
                                traces.append(trace)

                    if not traces:
                        continue

                    min_len = min(len(t) for t in traces)
                    traces = [t[:min_len] for t in traces]
                    arr = np.array(traces)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        mean_v = np.nanmean(arr, axis=0)
                        std_v = np.nanstd(arr, axis=0)
                    x = np.arange(1, min_len + 1)

                    label = f'{agent_name} ({regime})'
                    ls = styles[regime]
                    color = colors[agent_name]
                    ax.plot(x, mean_v, label=label, color=color, linestyle=ls, linewidth=1.5)
                    ax.fill_between(x, mean_v - std_v, mean_v + std_v, alpha=0.1, color=color)

            ax.set_title(title)
            ax.set_xlabel('Round')
            ax.grid(True, alpha=0.3)
            if ax_idx == 0:
                ax.legend(fontsize=7)

        plt.suptitle(f'Flip Set Analysis: {ds_name} ({n_rounds} rounds, {len(seeds)} seeds)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'flip_set_collapse_{ds_name}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'flip_set_collapse_{ds_name}.png'), dpi=150, bbox_inches='tight')
        print(f'Plot saved for {ds_name}')
        plt.close()


def print_summary(all_results, datasets, seeds, n_rounds):
    agent_names = ['BayesianSPO', 'BayesianMSE']
    regimes = ['changing', 'fixed']

    for ds_name in datasets:
        print(f'\n=== {ds_name} ===')
        header = f'{"Agent":<15s} {"Regime":<10s} {"Regret":>8s} {"FlipEnd":>8s} {"T0":>6s} {"DecAcc":>7s} {"CntMAE":>7s} {"RankCorr":>9s}'
        print(header)
        print('-' * len(header))

        for regime in regimes:
            for agent_name in agent_names:
                regrets, flip_ends, t0s, dec_accs, cnt_maes, rank_corrs = [], [], [], [], [], []

                for s in seeds:
                    key = (agent_name, regime, ds_name, s)
                    if key not in all_results:
                        continue
                    r = all_results[key]
                    regrets.append(r['final_regret'])

                    fs = r.get('flip_set_size', [])
                    if fs:
                        valid_fs = [v for v in fs if v is not None and np.isfinite(v)]
                        if valid_fs:
                            flip_ends.append(valid_fs[-1])

                            n_arms_est = max(valid_fs) if max(valid_fs) > 0 else 24
                            t0s.append(estimate_T0(valid_fs, n_arms_est))

                    dc = r.get('decision_correct', [])
                    if dc:
                        valid_dc = [v for v in dc if v is not None and np.isfinite(v)]
                        if valid_dc:

                            tail = valid_dc[int(0.8 * len(valid_dc)):]
                            dec_accs.append(np.mean(tail))

                    cm = r.get('contested_mae', [])
                    if cm:
                        valid_cm = [v for v in cm if v is not None and np.isfinite(v)]
                        if valid_cm:
                            tail = valid_cm[int(0.8 * len(valid_cm)):]
                            cnt_maes.append(np.mean(tail))

                    rc = r.get('rank_corr', [])
                    if rc:
                        valid_rc = [v for v in rc if v is not None and np.isfinite(v)]
                        if valid_rc:
                            tail = valid_rc[int(0.8 * len(valid_rc)):]
                            rank_corrs.append(np.mean(tail))

                def fmt(vals, decimals=1):
                    if not vals:
                        return 'N/A'
                    return f'{np.mean(vals):.{decimals}f}'

                print(f'{agent_name:<15s} {regime:<10s} {fmt(regrets):>8s} {fmt(flip_ends):>8s} '
                      f'{fmt(t0s, 0):>6s} {fmt(dec_accs, 3):>7s} {fmt(cnt_maes, 4):>7s} {fmt(rank_corrs, 3):>9s}')


def main():
    parser = argparse.ArgumentParser(description='Flip Set Collapse Ablation')
    parser.add_argument('--config', default='experiments/configs/experiment.yaml')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--output', default=None)
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: knapsack_32, shortest_path_8x8)')
    parser.add_argument('--no-plot', action='store_true', help='Save data only, skip plotting')
    args = parser.parse_args()

    config = load_config(args.config)


    default_datasets = ['knapsack_32', 'shortest_path_8x8']
    datasets = args.datasets or default_datasets

    n_rounds = 2500
    n_seeds = 5
    if args.mode == 'quick':
        n_rounds = 500
        n_seeds = 2

    seeds = list(range(n_seeds))
    output_dir = args.output or 'results/ablations/flip_set'
    lr = config['agent_defaults']['lr']

    print(f'\n=== Flip Set Collapse Ablation ===')
    print(f'Datasets: {datasets}, rounds={n_rounds}, seeds={n_seeds}')

    all_results = {}

    for ds_name in datasets:
        print(f'\n--- {ds_name} ---')
        dc = make_dataset_config(ds_name, config)

        for seed_idx, seed in enumerate(seeds):
            print(f'  Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):')

            env = create_env(ds_name, seed=seed, **dc)
            co_solver = make_co_solver(env)
            episode = pregenerate_episode(env, n_rounds)


            fixed_episode = make_fixed_context_episode(episode, n_rounds)

            for regime, ep in [('changing', episode), ('fixed', fixed_episode)]:
                print(f'    [{regime}]')
                agents = make_agents(env, seed, config, n_rounds)

                for agent_name, agent in agents.items():
                    print(f'      {agent_name}...', end=' ', flush=True)
                    result = run_agent_on_episode(
                        agent, ep, n_rounds,
                        lr=lr, full_feedback=False,
                        diagnostics=True, co_solver=co_solver,
                    )
                    all_results[(agent_name, regime, ds_name, seed)] = result
                    fs = result.get('flip_set_size', [])
                    fs_end = fs[-1] if fs else 'N/A'
                    print(f'regret={result["final_regret"]:.0f}, flip_end={fs_end}', flush=True)

    save_results(all_results, os.path.join(output_dir, 'results.pkl'))
    print_summary(all_results, datasets, seeds, n_rounds)
    if not args.no_plot:
        plot_results(all_results, datasets, seeds, n_rounds, output_dir)


if __name__ == '__main__':
    main()

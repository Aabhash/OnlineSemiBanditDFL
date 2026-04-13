#!/usr/bin/env python
import argparse
import importlib
import importlib.util
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, '.')

plt = importlib.import_module('matplotlib.pyplot') if importlib.util.find_spec('matplotlib.pyplot') is not None else None


ABLATION_DIR = 'results/ablations'
OUTPUT_DIR = 'results/figures/ablations'


def load_pkl(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_kn_ratio(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    data = load_pkl(os.path.join(results_dir, 'kn_ratio', 'results.pkl'))
    if data is None:
        print('  kn_ratio: no data found, skipping')
        return


    caps = sorted(set(cap for _, cap, _ in data.keys()))
    seeds = sorted(set(s for _, _, s in data.keys()))
    agents = sorted(set(name for name, _, _ in data.keys()))
    n_items = 32

    ratios = [cap / n_items for cap in caps]

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in agents:
        means = []
        stds = []
        for cap in caps:
            regs = [data[(name, cap, s)]['final_regret']
                    for s in seeds if (name, cap, s) in data]
            means.append(np.mean(regs) if regs else np.nan)
            stds.append(np.std(regs) if regs else 0)
        ax.errorbar(ratios, means, yerr=stds, fmt='o-', label=name, capsize=3)

    ax.set_xlabel('k/n (capacity ratio)', fontsize=12)
    ax.set_ylabel('Final Cumulative Regret', fontsize=12)
    ax.set_title('k/n Ratio Sweep')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kn_ratio_sweep.{fmt}'), bbox_inches='tight')
    plt.close()
    print(f'  Saved kn_ratio_sweep.{fmt}')


def plot_exploration(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    data = load_pkl(os.path.join(results_dir, 'exploration', 'results.pkl'))
    if data is None:
        print('  exploration: no data found, skipping')
        return

    agents = sorted(set(name for (name, _) in data.keys()))
    seeds = sorted(set(s for (_, s) in data.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))
    means = []
    stds = []
    for name in agents:
        regs = [data[(name, s)]['final_regret'] for s in seeds if (name, s) in data]
        means.append(np.mean(regs) if regs else 0)
        stds.append(np.std(regs) if regs else 0)

    x = np.arange(len(agents))
    ax.bar(x, means, yerr=stds, capsize=4, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Final Cumulative Regret')
    ax.set_title('Exploration Strategy Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'exploration_strategy.{fmt}'), bbox_inches='tight')
    plt.close()
    print(f'  Saved exploration_strategy.{fmt}')


def plot_imputation(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    data = load_pkl(os.path.join(results_dir, 'imputation', 'results.pkl'))
    if data is None:
        print('  imputation: no data found, skipping')
        return

    agents = sorted(set(name for (name, _) in data.keys()))
    seeds = sorted(set(s for (_, s) in data.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))
    means = []
    stds = []
    for name in agents:
        regs = [data[(name, s)]['final_regret'] for s in seeds if (name, s) in data]
        means.append(np.mean(regs) if regs else 0)
        stds.append(np.std(regs) if regs else 0)

    x = np.arange(len(agents))
    ax.bar(x, means, yerr=stds, capsize=4, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Final Cumulative Regret')
    ax.set_title('Imputation Strategy Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'imputation_strategy.{fmt}'), bbox_inches='tight')
    plt.close()
    print(f'  Saved imputation_strategy.{fmt}')


def plot_feature_sharing(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    data = load_pkl(os.path.join(results_dir, 'feature_sharing', 'results.pkl'))
    if data is None:
        print('  feature_sharing: no data found, skipping')
        return

    agents = sorted(set(name for (name, _) in data.keys()))
    seeds = sorted(set(s for (_, s) in data.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in agents:
        regs = [data[(name, s)]['final_regret'] for s in seeds if (name, s) in data]
        if regs:
            ax.bar(name, np.mean(regs), yerr=np.std(regs), capsize=4)

    ax.set_ylabel('Final Cumulative Regret')
    ax.set_title('Feature Sharing (Hidden Dim Sweep)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_sharing.{fmt}'), bbox_inches='tight')
    plt.close()
    print(f'  Saved feature_sharing.{fmt}')


def plot_context_variation(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    data = load_pkl(os.path.join(results_dir, 'context_variation', 'results.pkl'))
    if data is None:
        print('  context_variation: no data found, skipping')
        return

    agents = sorted(set(name for (name, _) in data.keys()))
    seeds = sorted(set(s for (_, s) in data.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in agents:
        regs = [data[(name, s)]['final_regret'] for s in seeds if (name, s) in data]
        if regs:
            ax.bar(name, np.mean(regs), yerr=np.std(regs), capsize=4)

    ax.set_ylabel('Final Cumulative Regret')
    ax.set_title('Context Variation (Noise Sensitivity)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'context_variation.{fmt}'), bbox_inches='tight')
    plt.close()
    print(f'  Saved context_variation.{fmt}')


def plot_flip_set(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    data = load_pkl(os.path.join(results_dir, 'flip_set', 'results.pkl'))
    if data is None:
        print('  flip_set: no data found, skipping')
        return

    agents = sorted(set(name for (name, _) in data.keys()))
    seeds = sorted(set(s for (_, s) in data.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    ax = axes[0]
    for name in agents:
        flip_sizes = [data[(name, s)].get('flip_set_size', [])
                      for s in seeds if (name, s) in data]
        flip_sizes = [f for f in flip_sizes if len(f) > 0]
        if flip_sizes:
            mean_flip = np.mean(flip_sizes, axis=0)
            ax.plot(mean_flip, label=name, alpha=0.8)
    ax.set_xlabel('Round')
    ax.set_ylabel('Flip Set Size')
    ax.set_title('Flip Set Size Over Time')
    ax.legend(fontsize=8)


    ax = axes[1]
    means = []
    names_plot = []
    for name in agents:
        regs = [data[(name, s)]['final_regret'] for s in seeds if (name, s) in data]
        if regs:
            means.append(np.mean(regs))
            names_plot.append(name)
    ax.bar(range(len(names_plot)), means)
    ax.set_xticks(range(len(names_plot)))
    ax.set_xticklabels(names_plot, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Final Regret')
    ax.set_title('Final Regret')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'flip_set_collapse.{fmt}'), bbox_inches='tight')
    plt.close()
    print(f'  Saved flip_set_collapse.{fmt}')


PLOT_FUNCTIONS = {
    'kn_ratio': plot_kn_ratio,
    'exploration': plot_exploration,
    'imputation': plot_imputation,
    'feature_sharing': plot_feature_sharing,
    'context_variation': plot_context_variation,
    'flip_set': plot_flip_set,
}


def main():
    parser = argparse.ArgumentParser(description='Plot ablation results')
    parser.add_argument('--results-dir', type=str, default=ABLATION_DIR)
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    parser.add_argument('--format', type=str, default='pdf', choices=['pdf', 'png', 'svg'])
    parser.add_argument('--ablations', type=str, default=None,
                        help='Comma-separated ablation names (default: all)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.ablations:
        ablations = [a.strip() for a in args.ablations.split(',')]
    else:
        ablations = list(PLOT_FUNCTIONS.keys())

    print(f'Plotting ablations: {ablations}')
    print(f'Reading from: {args.results_dir}')
    print(f'Saving to: {args.output}')
    print()

    for name in ablations:
        if name in PLOT_FUNCTIONS:
            PLOT_FUNCTIONS[name](args.results_dir, args.output, args.format)
        else:
            print(f'  Unknown ablation: {name}')

    print(f'\nDone. Figures in {args.output}/')


if __name__ == '__main__':
    main()

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


def resolve_ablation_dir(results_dir, ablation_name):
    candidate = os.path.join(results_dir, ablation_name)
    if os.path.isdir(candidate) or os.path.exists(os.path.join(candidate, 'results.pkl')):
        return candidate
    return results_dir


def list_result_entries(ablation_dir):
    direct_path = os.path.join(ablation_dir, 'results.pkl')
    if os.path.exists(direct_path):
        return [('.', direct_path)]

    entries = []
    if not os.path.isdir(ablation_dir):
        return entries

    for name in sorted(os.listdir(ablation_dir)):
        path = os.path.join(ablation_dir, name, 'results.pkl')
        if os.path.exists(path):
            entries.append((name, path))
    return entries


def plot_kn_ratio(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    ablation_dir = resolve_ablation_dir(results_dir, 'kn_ratio')
    data = load_pkl(os.path.join(ablation_dir, 'results.pkl'))
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

    ablation_dir = resolve_ablation_dir(results_dir, 'exploration')
    entries = list_result_entries(ablation_dir)
    if not entries:
        print('  exploration: no data found, skipping')
        return

    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue

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
        ax.set_title(f'Exploration Strategy Comparison: {dataset_name}')
        plt.tight_layout()
        title = 'exploration_strategy' if dataset_name == '.' else f'exploration_strategy_{dataset_name}'
        plt.savefig(os.path.join(output_dir, f'{title}.{fmt}'), bbox_inches='tight')
        plt.close()
        print(f'  Saved {title}.{fmt}')


def plot_imputation(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    ablation_dir = resolve_ablation_dir(results_dir, 'imputation')
    entries = list_result_entries(ablation_dir)
    if not entries:
        print('  imputation: no data found, skipping')
        return

    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue

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
        ax.set_title(f'Imputation Strategy Comparison: {dataset_name}')
        plt.tight_layout()
        title = 'imputation_strategy' if dataset_name == '.' else f'imputation_strategy_{dataset_name}'
        plt.savefig(os.path.join(output_dir, f'{title}.{fmt}'), bbox_inches='tight')
        plt.close()
        print(f'  Saved {title}.{fmt}')


def plot_feature_sharing(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    ablation_dir = resolve_ablation_dir(results_dir, 'feature_sharing')
    entries = list_result_entries(ablation_dir)
    if not entries:
        print('  feature_sharing: no data found, skipping')
        return

    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue

        agents = sorted(set(name for (name, _) in data.keys()))
        seeds = sorted(set(s for (_, s) in data.keys()))

        fig, ax = plt.subplots(figsize=(10, 6))
        for name in agents:
            regs = [data[(name, s)]['final_regret'] for s in seeds if (name, s) in data]
            if regs:
                ax.bar(name, np.mean(regs), yerr=np.std(regs), capsize=4)

        ax.set_ylabel('Final Cumulative Regret')
        ax.set_title(f'Feature Sharing (Hidden Dim Sweep): {dataset_name}')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        title = 'feature_sharing' if dataset_name == '.' else f'feature_sharing_{dataset_name}'
        plt.savefig(os.path.join(output_dir, f'{title}.{fmt}'), bbox_inches='tight')
        plt.close()
        print(f'  Saved {title}.{fmt}')


def plot_context_variation(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    ablation_dir = resolve_ablation_dir(results_dir, 'context_variation')
    entries = list_result_entries(ablation_dir)
    data = load_pkl(entries[0][1]) if entries else None
    if data is None:
        print('  context_variation: no data found, skipping')
        return

    agents = sorted(set(name for (name, _, _) in data.keys()))
    noise_widths = sorted(set(noise for (_, noise, _) in data.keys()))
    seeds = sorted(set(seed for (_, _, seed) in data.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name in agents:
        means = []
        stds = []
        for noise_width in noise_widths:
            regs = [
                data[(name, noise_width, seed)]['final_regret']
                for seed in seeds if (name, noise_width, seed) in data
            ]
            means.append(np.mean(regs) if regs else np.nan)
            stds.append(np.std(regs) if regs else 0.0)
        ax.errorbar(noise_widths, means, yerr=stds, marker='o', label=name, capsize=3)
    ax.set_xlabel('Noise Width')
    ax.set_ylabel('Final Cumulative Regret')
    ax.set_title('Context Variation: Regret vs Noise')
    ax.legend()

    ax = axes[1]
    gaps = []
    for noise_width in noise_widths:
        spo = [
            data[('BayesianSPO', noise_width, seed)]['final_regret']
            for seed in seeds if ('BayesianSPO', noise_width, seed) in data
        ]
        mse = [
            data[('BayesianMSE', noise_width, seed)]['final_regret']
            for seed in seeds if ('BayesianMSE', noise_width, seed) in data
        ]
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
    plt.savefig(os.path.join(output_dir, f'context_variation.{fmt}'), bbox_inches='tight')
    plt.close()
    print(f'  Saved context_variation.{fmt}')


def plot_flip_set(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    ablation_dir = resolve_ablation_dir(results_dir, 'flip_set')
    entries = list_result_entries(ablation_dir)
    data = load_pkl(entries[0][1]) if entries else None
    if data is None:
        print('  flip_set: no data found, skipping')
        return

    agents = sorted(set(name for (name, _, _, _) in data.keys()))
    regimes = sorted(set(regime for (_, regime, _, _) in data.keys()))
    datasets = sorted(set(dataset for (_, _, dataset, _) in data.keys()))
    seeds = sorted(set(seed for (_, _, _, seed) in data.keys()))

    for dataset_name in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for regime in regimes:
            for name in agents:
                flip_sizes = [
                    data[(name, regime, dataset_name, seed)].get('flip_set_size', [])
                    for seed in seeds if (name, regime, dataset_name, seed) in data
                ]
                flip_sizes = [f for f in flip_sizes if len(f) > 0]
                if flip_sizes:
                    min_len = min(len(f) for f in flip_sizes)
                    trimmed = np.array([f[:min_len] for f in flip_sizes])
                    mean_flip = np.mean(trimmed, axis=0)
                    ax.plot(mean_flip, label=f'{name} ({regime})', alpha=0.8)
        ax.set_xlabel('Round')
        ax.set_ylabel('Flip Set Size')
        ax.set_title(f'Flip Set Size Over Time: {dataset_name}')
        ax.legend(fontsize=8)

        ax = axes[1]
        means = []
        names_plot = []
        for regime in regimes:
            for name in agents:
                regs = [
                    data[(name, regime, dataset_name, seed)]['final_regret']
                    for seed in seeds if (name, regime, dataset_name, seed) in data
                ]
                if regs:
                    means.append(np.mean(regs))
                    names_plot.append(f'{name} ({regime})')
        ax.bar(range(len(names_plot)), means)
        ax.set_xticks(range(len(names_plot)))
        ax.set_xticklabels(names_plot, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Final Regret')
        ax.set_title(f'Final Regret: {dataset_name}')

        plt.tight_layout()
        title = 'flip_set_collapse' if dataset_name == '.' else f'flip_set_collapse_{dataset_name}'
        plt.savefig(os.path.join(output_dir, f'{title}.{fmt}'), bbox_inches='tight')
        plt.close()
        print(f'  Saved {title}.{fmt}')


def plot_mechanism(results_dir, output_dir, fmt='pdf'):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    mechanism_dir = resolve_ablation_dir(results_dir, 'mechanism')
    if not os.path.isdir(mechanism_dir):
        print('  mechanism: no data found, skipping')
        return

    dataset_dirs = sorted(
        d for d in os.listdir(mechanism_dir)
        if os.path.isdir(os.path.join(mechanism_dir, d))
    )
    if not dataset_dirs and os.path.exists(os.path.join(mechanism_dir, 'results.pkl')):
        dataset_dirs = ['.']

    if not dataset_dirs:
        print('  mechanism: no data found, skipping')
        return

    for dataset_dir in dataset_dirs:
        data_path = (
            os.path.join(mechanism_dir, 'results.pkl')
            if dataset_dir == '.'
            else os.path.join(mechanism_dir, dataset_dir, 'results.pkl')
        )
        data = load_pkl(data_path)
        if data is None:
            continue

        agent_names = sorted(set(name for (name, _) in data.keys()))
        seeds = sorted(set(s for (_, s) in data.keys()))
        x = np.arange(len(agent_names))

        mean_regrets = []
        std_regrets = []
        flip_mae = []
        contested_sigma = []
        for name in agent_names:
            regrets = [data[(name, s)]['final_regret'] for s in seeds if (name, s) in data]
            mean_regrets.append(np.mean(regrets) if regrets else np.nan)
            std_regrets.append(np.std(regrets) if regrets else 0.0)

            flip_vals = []
            sigma_vals = []
            for seed in seeds:
                key = (name, seed)
                if key not in data:
                    continue
                flip_trace = [v for v in data[key].get('flip_contested_mae', [])[-100:] if np.isfinite(v)]
                sigma_trace = [v for v in data[key].get('contested_sigma_sum', [])[-100:] if np.isfinite(v)]
                if flip_trace:
                    flip_vals.append(np.mean(flip_trace))
                if sigma_trace:
                    sigma_vals.append(np.mean(sigma_trace))
            flip_mae.append(np.mean(flip_vals) if flip_vals else np.nan)
            contested_sigma.append(np.mean(sigma_vals) if sigma_vals else np.nan)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].bar(x, mean_regrets, yerr=std_regrets, capsize=3)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(agent_names, rotation=35, ha='right')
        axes[0].set_ylabel('Final Regret')
        axes[0].set_title('Final Regret')

        axes[1].bar(x, flip_mae, color='coral')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(agent_names, rotation=35, ha='right')
        axes[1].set_ylabel('Flip-Set MAE (last 100)')
        axes[1].set_title('Target / Imputation Quality')

        axes[2].bar(x, contested_sigma, color='steelblue')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(agent_names, rotation=35, ha='right')
        axes[2].set_ylabel('Contested Sigma Sum (last 100)')
        axes[2].set_title('Observation Allocation Pressure')

        title = 'mechanism_ablation' if dataset_dir == '.' else f'mechanism_ablation_{dataset_dir}'
        fig.suptitle(f'Mechanism Ablation: {dataset_dir}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{title}.{fmt}'), bbox_inches='tight')
        plt.close()
        print(f'  Saved {title}.{fmt}')


PLOT_FUNCTIONS = {
    'kn_ratio': plot_kn_ratio,
    'kn_ratio_sweep': plot_kn_ratio,
    'exploration': plot_exploration,
    'exploration_strategy': plot_exploration,
    'imputation': plot_imputation,
    'imputation_strategy': plot_imputation,
    'mechanism': plot_mechanism,
    'mechanism_ablation': plot_mechanism,
    'feature_sharing': plot_feature_sharing,
    'context_variation': plot_context_variation,
    'flip_set': plot_flip_set,
    'flip_set_collapse': plot_flip_set,
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

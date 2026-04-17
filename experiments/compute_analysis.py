#!/usr/bin/env python
import argparse
import importlib
import importlib.util
import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, '.')

if importlib.util.find_spec('matplotlib') is not None:
    matplotlib = importlib.import_module('matplotlib')
    matplotlib.use('Agg')
    plt = importlib.import_module('matplotlib.pyplot')
    Line2D = importlib.import_module('matplotlib.lines').Line2D
else:
    matplotlib = None
    plt = None
    Line2D = None


OUTPUT_DIR = 'results/compute'


DEFAULT_GRADIENT_MULTIPLIER = {
    'Random': 0, 'EpsilonGreedy': 1, 'TS': 0, 'LinUCB': 0,
    'NeuralUCB': 1, 'BayesianMSE': 1, 'BayesianSPO': 1,
    'BayesianPFY': 1,
    'BayesianSPO-Phased': 5,
    'BayesianPFY-Phased': 5,
    'BayesianMSE-Phased': 5,
    'WeberBLL-Phased': 5, 'WeberBLL-Online': 1,
    'NeuralUCB-Phased': 5,
}


def get_gradient_multiplier(agent_name, data):
    hp = data.get('hyperparameters', {})
    if 'phase_epochs' in hp:
        return int(hp['phase_epochs'])
    return DEFAULT_GRADIENT_MULTIPLIER.get(agent_name, 1)

AGENT_CATEGORY = {
    'Random': 'baseline', 'EpsilonGreedy': 'baseline', 'TS': 'baseline',
    'LinUCB': 'baseline',
    'NeuralUCB': 'online', 'BayesianMSE': 'online', 'BayesianSPO': 'online',
    'BayesianPFY': 'online',
    'BayesianSPO-Phased': 'phased',
    'BayesianMSE-Phased': 'phased',
    'WeberBLL-Phased': 'phased', 'WeberBLL-Online': 'online',
    'NeuralUCB-Phased': 'phased',
}

CATEGORY_COLORS = {'baseline': '#95a5a6', 'online': '#3498db', 'phased': '#2ecc71'}


def load_results_summary(results_dirs):
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]

    merged = {}
    for results_dir in results_dirs:
        path = os.path.join(results_dir, 'results_summary.json')
        if not os.path.exists(path):
            print(f'  Skipping {path} (not found)')
            continue
        with open(path) as f:
            data = json.load(f)
        merged.update(data)
        print(f'  Loaded {len(data)} entries from {path}')

    if not merged:
        print('No results found. Run first:')
        print('  python experiments/evaluate.py --output_dir results/eval')
        print('  or')
        print('  python experiments/evaluate_online.py --output_dir results/eval_online')
        print('  python experiments/evaluate_phased.py --output_dir results/eval_phased')
        return None

    return merged


def generate_wallclock_table(summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for key, data in sorted(summary.items()):
        agent = data.get('agent', key)
        dataset = data.get('dataset', '')
        mean_reg = data.get('mean_cumulative_regret', 0)
        std_reg = data.get('std_cumulative_regret', 0)
        mean_time = data.get('mean_wall_time_seconds', 0)

        rows.append({
            'Dataset': dataset,
            'Agent': agent,
            'Mean Regret': f'{mean_reg:.1f}',
            'Std Regret': f'{std_reg:.1f}',
            'Mean Time (s)': f'{mean_time:.1f}',
            'Regret/Second': f'{mean_reg / max(mean_time, 0.01):.1f}',
        })

    print('\n=== Wallclock Table ===')
    print(f'{"Agent":30s} {"Dataset":25s} {"Regret":>12s} {"Time (s)":>10s}')
    print('-' * 80)
    for r in rows:
        print(f'{r["Agent"]:30s} {r["Dataset"]:25s} '
              f'{r["Mean Regret"]:>6s} +/- {r["Std Regret"]:>4s} '
              f'{r["Mean Time (s)"]:>8s}')

    csv_path = os.path.join(output_dir, 'wallclock_table.csv')
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved to {csv_path}')


def generate_pareto_plot(summary, output_dir):
    if plt is None or Line2D is None:
        print('matplotlib not available, skipping Pareto plot')
        return

    os.makedirs(output_dir, exist_ok=True)


    datasets = {}
    for key, data in summary.items():
        ds = data.get('dataset', 'unknown')
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(data)

    for dataset, entries in datasets.items():
        fig, ax = plt.subplots(figsize=(10, 7))

        points = []
        for data in entries:
            agent = data.get('agent', '')
            mean_reg = data.get('mean_cumulative_regret', 0)
            std_reg = data.get('std_cumulative_regret', 0)
            mean_time = data.get('mean_wall_time_seconds', 0)

            cat = AGENT_CATEGORY.get(agent, 'online')
            color = CATEGORY_COLORS[cat]

            ax.errorbar(mean_time, mean_reg, yerr=std_reg,
                        fmt='o', color=color, markersize=8,
                        capsize=3, linewidth=1.5, zorder=5)
            ax.annotate(agent, (mean_time, mean_reg),
                        textcoords='offset points',
                        xytext=(8, 5), fontsize=7, alpha=0.8)
            points.append((mean_time, mean_reg))


        points.sort(key=lambda p: p[0])
        pareto = []
        best_reg = float('inf')
        for t, r in points:
            if r < best_reg:
                pareto.append((t, r))
                best_reg = r
        if len(pareto) > 1:
            px, py = zip(*pareto)
            ax.plot(px, py, '--', color='black', alpha=0.3, linewidth=1)

        ax.set_xlabel('Wallclock Time (seconds)', fontsize=12)
        ax.set_ylabel('Cumulative Regret', fontsize=12)
        ax.set_title(f'Regret vs Compute: {dataset}', fontsize=14)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=CATEGORY_COLORS[c], markersize=8, label=c.title())
            for c in ['baseline', 'online', 'phased']
        ]
        ax.legend(handles=legend_elements, fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pareto_{dataset}.pdf'),
                    bbox_inches='tight')
        plt.close()
        print(f'  Saved pareto_{dataset}.pdf')


def generate_compute_normalized(summary, n_rounds, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print('\n=== Compute-Normalized Comparison ===')
    print(f'{"Agent":30s} {"Dataset":25s} {"Regret":>8s} '
          f'{"Steps":>8s} {"Reg/Step":>10s}')
    print('-' * 85)

    rows = []
    for key, data in sorted(summary.items()):
        agent = data.get('agent', key)
        dataset = data.get('dataset', '')
        mean_reg = data.get('mean_cumulative_regret', 0)

        multiplier = get_gradient_multiplier(agent, data)
        total_steps = multiplier * n_rounds if multiplier > 0 else n_rounds
        reg_per_step = mean_reg / max(total_steps, 1)

        rows.append({
            'Dataset': dataset, 'Agent': agent,
            'Mean Regret': f'{mean_reg:.1f}',
            'Gradient Steps': str(total_steps),
            'Regret/Step': f'{reg_per_step:.4f}',
        })
        print(f'{agent:30s} {dataset:25s} {mean_reg:8.1f} '
              f'{total_steps:8d} {reg_per_step:10.4f}')

    csv_path = os.path.join(output_dir, 'compute_normalized.csv')
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved to {csv_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Compute analysis: wallclock + Pareto plots')
    parser.add_argument('--results-dir', type=str, nargs='+',
                        default=[
                            'results/eval/metrics',
                            'results/eval_online/metrics',
                            'results/eval_phased/metrics',
                        ],
                        help='One or more directories with results_summary.json')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    parser.add_argument('--n-rounds', type=int, default=5000,
                        help='Number of rounds (for gradient step calculation)')
    args = parser.parse_args()

    summary = load_results_summary(args.results_dir)
    if summary is None:
        sys.exit(1)

    print(f'Loaded {len(summary)} agent-dataset entries')

    generate_wallclock_table(summary, args.output)
    generate_pareto_plot(summary, args.output)
    generate_compute_normalized(summary, args.n_rounds, args.output)

    print(f'\nAll outputs saved to {args.output}/')


if __name__ == '__main__':
    main()

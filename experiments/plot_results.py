import argparse
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional


COLOR_MAP = {
    'Random': 'gray',
    'UCB-Bayesian': 'blue',
    'ThompsonSampling': 'green',
    'LinUCB': 'orange',
    'NeuralUCB': 'red',
    'OptimisticSPO': 'purple',
    'EndToEndSPO': 'brown',
    'FullSPO': 'pink',
}


def load_data(input_dir):
    input_path = Path(input_dir)

    traj_file = input_path / 'metrics' / 'trajectories.pkl'
    with open(traj_file, 'rb') as f:
        trajectories = pickle.load(f)

    results_file = input_path / 'metrics' / 'results.json'
    results = None
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

    return trajectories, results


def extract_curves(trajectories, dataset, feedback_mode, metric='cumulative_regret'):
    agents = {}

    for key, data in trajectories.items():
        agent_name, ds, mode, seed = key
        if ds == dataset and mode == feedback_mode:
            if agent_name not in agents:
                agents[agent_name] = []
            if metric in data and len(data[metric]) > 0:
                agents[agent_name].append(data[metric])

    return agents


def plot_with_ci(ax, agents, title, ylabel):
    for agent_name, curves in sorted(agents.items()):
        if len(curves) == 0:
            continue

        max_len = max(len(c) for c in curves)
        padded = []
        for c in curves:
            if len(c) < max_len:
                c = list(c) + [c[-1]] * (max_len - len(c))
            padded.append(c)

        curves_arr = np.array(padded)
        mean = np.mean(curves_arr, axis=0)
        std = np.std(curves_arr, axis=0)

        x = np.arange(len(mean))
        color = COLOR_MAP.get(agent_name, 'black')

        ax.plot(x, mean, label=agent_name, color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Round', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)


def plot_learning_curves(trajectories, dataset, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    agents = extract_curves(trajectories, dataset, 'semi_bandit', 'cumulative_regret')
    if agents:
        plot_with_ci(axes[0, 0], agents, f'{dataset.title()}: Semi-Bandit Cumulative Regret', 'Cumulative Regret')

    agents = extract_curves(trajectories, dataset, 'full', 'cumulative_regret')
    if agents:
        plot_with_ci(axes[0, 1], agents, f'{dataset.title()}: Full Feedback Cumulative Regret', 'Cumulative Regret')

    agents = extract_curves(trajectories, dataset, 'full', 'regrets')
    if agents:
        smoothed = {}
        for name, curves in agents.items():
            smoothed[name] = []
            for c in curves:
                arr = np.array(c)
                window = min(50, len(arr) // 10)
                if window > 1:
                    smoothed_c = np.convolve(arr, np.ones(window)/window, mode='valid')
                else:
                    smoothed_c = arr
                smoothed[name].append(smoothed_c)
        plot_with_ci(axes[0, 2], smoothed, f'{dataset.title()}: Per-Round Regret (smoothed)', 'Per-Round Regret')

    agents = extract_curves(trajectories, dataset, 'full', 'mse_losses')
    neural_agents = {k: v for k, v in agents.items() if k in ['NeuralUCB', 'OptimisticSPO', 'EndToEndSPO', 'FullSPO']}
    if neural_agents:
        plot_with_ci(axes[1, 0], neural_agents, f'{dataset.title()}: MSE Loss', 'MSE Loss')

    agents = extract_curves(trajectories, dataset, 'full', 'spo_losses')
    neural_agents = {k: v for k, v in agents.items() if k in ['NeuralUCB', 'OptimisticSPO', 'EndToEndSPO', 'FullSPO']}
    if neural_agents:
        plot_with_ci(axes[1, 1], neural_agents, f'{dataset.title()}: SPO+ Loss (Full)', 'SPO+ Loss')

    agents = extract_curves(trajectories, dataset, 'semi_bandit', 'spo_losses')
    neural_agents = {k: v for k, v in agents.items() if k in ['NeuralUCB', 'OptimisticSPO', 'EndToEndSPO', 'FullSPO']}
    if neural_agents:
        plot_with_ci(axes[1, 2], neural_agents, f'{dataset.title()}: SPO+ Loss (Semi-Bandit)', 'SPO+ Loss')

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f'learning_curves_{dataset}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: learning_curves_{dataset}.png")
    plt.close()


def plot_final_regret_bars(results, output_dir):
    if results is None:
        return

    datasets = list(results.keys())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        agent_means = {}
        for feedback in ['semi_bandit', 'full']:
            if feedback not in results[dataset]:
                continue
            for agent, regrets in results[dataset][feedback].items():
                key = f"{agent} ({feedback.replace('_', '-')})"
                agent_means[key] = np.mean(regrets)

        agents = list(agent_means.keys())
        means = list(agent_means.values())

        colors = [COLOR_MAP.get(a.split(' (')[0], 'gray') for a in agents]

        x = np.arange(len(agents))
        ax.bar(x, means, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Final Regret', fontsize=10)
        ax.set_title(f'{dataset.title()}: Final Regret by Agent', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'final_regret_bars.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: final_regret_bars.png")
    plt.close()


def plot_feedback_comparison(results, dataset, output_dir):
    if results is None or dataset not in results:
        return

    agent_data = {}
    for feedback in ['semi_bandit', 'full']:
        if feedback not in results[dataset]:
            continue
        for agent, regrets in results[dataset][feedback].items():
            if agent not in agent_data:
                agent_data[agent] = {}
            agent_data[agent][feedback] = np.mean(regrets)

    agents = sorted(agent_data.keys())
    semi_bandit_vals = [agent_data[a].get('semi_bandit', 0) for a in agents]
    full_vals = [agent_data[a].get('full', 0) for a in agents]

    x = np.arange(len(agents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, semi_bandit_vals, width, label='Semi-Bandit', alpha=0.7)
    ax.bar(x + width/2, full_vals, width, label='Full Feedback', alpha=0.7)

    ax.set_xlabel('Agent', fontsize=10)
    ax.set_ylabel('Final Regret', fontsize=10)
    ax.set_title(f'{dataset.title()}: Semi-Bandit vs Full Feedback', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f'feedback_comparison_{dataset}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: feedback_comparison_{dataset}.png")
    plt.close()


def plot_ranking_heatmap(results, output_dir):
    if results is None:
        return

    datasets = list(results.keys())
    feedbacks = ['semi_bandit', 'full']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for fb_idx, feedback in enumerate(feedbacks):
        ax = axes[fb_idx]

        agent_ranks = {}
        for dataset in datasets:
            if feedback not in results[dataset]:
                continue

            agent_regrets = {agent: np.mean(regrets)
                           for agent, regrets in results[dataset][feedback].items()}
            sorted_agents = sorted(agent_regrets.items(), key=lambda x: x[1])

            for rank, (agent, _) in enumerate(sorted_agents, 1):
                if agent not in agent_ranks:
                    agent_ranks[agent] = []
                agent_ranks[agent].append(rank)

        agents = sorted(agent_ranks.keys())
        rank_matrix = np.array([agent_ranks[a] for a in agents])

        im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(agents))

        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(len(agents)))
        ax.set_xticklabels([d.title() for d in datasets])
        ax.set_yticklabels(agents)

        for i in range(len(agents)):
            for j in range(len(datasets)):
                text = ax.text(j, i, f'{int(rank_matrix[i, j])}',
                             ha="center", va="center", color="black", fontsize=9)

        title = 'Semi-Bandit' if feedback == 'semi_bandit' else 'Full Feedback'
        ax.set_title(f'{title} Rankings', fontsize=11)

        plt.colorbar(im, ax=ax, label='Rank')

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'ranking_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: ranking_heatmap.png")
    plt.close()


def plot_all(trajectories, results, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = set()
    for key in trajectories.keys():
        _, dataset, _, _ = key
        datasets.add(dataset)

    print("\nGenerating plots:")

    for dataset in sorted(datasets):
        print(f"\n{dataset.title()}:")
        plot_learning_curves(trajectories, dataset, output_dir)
        plot_feedback_comparison(results, dataset, output_dir)

    print(f"\nAll datasets:")
    plot_final_regret_bars(results, output_dir)
    plot_ranking_heatmap(results, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive plots from experiment results')
    parser.add_argument('--input', type=str, default='results/eval',
                        help='Input directory with metrics/ folder')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots (default: input/plots)')
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'curves', 'bars', 'comparison', 'heatmap'],
                        help='Type of plot to generate')

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(args.input) / 'plots'

    print(f"Loading data from {args.input}")
    trajectories, results = load_data(args.input)

    datasets = set()
    for key in trajectories.keys():
        _, dataset, _, _ = key
        datasets.add(dataset)

    print(f"Found datasets: {sorted(datasets)}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.type == 'all':
        plot_all(trajectories, results, args.output)
    elif args.type == 'curves':
        print("\nGenerating learning curves:")
        for dataset in sorted(datasets):
            print(f"  {dataset.title()}:")
            plot_learning_curves(trajectories, dataset, args.output)
    elif args.type == 'bars':
        print("\nGenerating bar charts:")
        plot_final_regret_bars(results, args.output)
    elif args.type == 'comparison':
        print("\nGenerating feedback comparisons:")
        for dataset in sorted(datasets):
            print(f"  {dataset.title()}:")
            plot_feedback_comparison(results, dataset, args.output)
    elif args.type == 'heatmap':
        print("\nGenerating ranking heatmap:")
        plot_ranking_heatmap(results, args.output)

    print(f"\nDone! Plots saved to {args.output}")


if __name__ == '__main__':
    main()

import argparse
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from scipy import stats


COLOR_MAP = {
    'Random': 'gray',
    'UCB-Bayesian': 'blue',
    'ThompsonSampling': 'green',
    'LinUCB': 'orange',
    'NeuralUCB': 'red',
    'BayesianSPO': 'purple',
    'FullSPO': 'pink',
    'DecisionUCB': 'cyan',
    'SimplePKL': 'brown',
    'IRM': 'olive',
    'ConservativeUnified': 'teal',
    'CN-UCB': 'magenta',
    'CN-TS': 'darkblue',
    'CombLinUCB': 'darkgreen',
    'CombLinTS': 'darkred',
    'EpsilonGreedy': 'gold',
    'MSENeural': 'coral',
    # Per-arm context agents
    'PerArmBayesianSPO': 'purple',
    'PerArmFullSPO': 'pink',
    'PerArmDecisionUCB': 'cyan',
    'PerArmNTKSPO': 'darkviolet',
    'PerArmNTKDecisionUCB': 'mediumorchid',
    'PerArmLinUCB': 'orange',
    'PerArmUCB': 'blue',
    'PerArmTS': 'green',
    'PerArmNeuralUCB': 'red',
    'PerArmMSENeural': 'coral',
    'PerArmEpsilonGreedy': 'gold',
    'PerArmBayesianIERM-DM': 'forestgreen',
    'PerArmBayesianIERM-DR': 'limegreen',
}


def compute_significance(results_dict, alpha=0.05):
    agents = list(results_dict.keys())
    n_agents = len(agents)

    p_values = {}
    significant_pairs = []

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            a1, a2 = agents[i], agents[j]
            r1, r2 = results_dict[a1], results_dict[a2]

            min_len = min(len(r1), len(r2))
            if min_len < 3:
                continue

            r1, r2 = r1[:min_len], r2[:min_len]

            try:
                stat, p = stats.wilcoxon(r1, r2)
                p_values[(a1, a2)] = p

                if p < alpha:
                    if np.mean(r1) < np.mean(r2):
                        significant_pairs.append((a1, a2, p, 'better'))
                    else:
                        significant_pairs.append((a2, a1, p, 'better'))
            except Exception:
                p_values[(a1, a2)] = 1.0

    summary_lines = ["Pairwise Significance Tests (Wilcoxon, alpha=0.05):", "-" * 50]

    if significant_pairs:
        summary_lines.append("\nSignificant differences:")
        for winner, loser, p, _ in sorted(significant_pairs, key=lambda x: x[2]):
            summary_lines.append(f"  {winner} < {loser} (p={p:.4f})")
    else:
        summary_lines.append("\nNo significant differences found.")

    return {
        'p_values': p_values,
        'significant_pairs': significant_pairs,
        'summary': '\n'.join(summary_lines),
    }


def compute_confidence_intervals(results_dict, confidence=0.95):
    z = stats.norm.ppf((1 + confidence) / 2)  # z-score for confidence level

    intervals = {}
    for agent, regrets in results_dict.items():
        mean = np.mean(regrets)
        std = np.std(regrets)
        n = len(regrets)
        ci = z * std / np.sqrt(n)
        intervals[agent] = {
            'mean': mean,
            'std': std,
            'ci': ci,
            'ci_lower': mean - ci,
            'ci_upper': mean + ci,
            'n': n,
        }
    return intervals


def print_significance_report(results, dataset, feedback_mode, output_file=None):
    if results is None or dataset not in results:
        return

    if feedback_mode not in results[dataset]:
        return

    agent_regrets = results[dataset][feedback_mode]

    sig_results = compute_significance(agent_regrets)
    ci_results = compute_confidence_intervals(agent_regrets)

    lines = [
        f"\n{'=' * 60}",
        f"STATISTICAL ANALYSIS: {dataset.upper()} - {feedback_mode.replace('_', ' ').title()}",
        f"{'=' * 60}",
        "\nFinal Regret (Mean +/- 95% CI):",
        "-" * 40,
    ]

    sorted_agents = sorted(ci_results.items(), key=lambda x: x[1]['mean'])

    for rank, (agent, ci) in enumerate(sorted_agents, 1):
        lines.append(
            f"  {rank}. {agent:<20} {ci['mean']:>8.1f} +/- {ci['ci']:>6.1f} "
            f"(n={ci['n']}, std={ci['std']:.1f})"
        )

    lines.append("")
    lines.append(sig_results['summary'])

    report = '\n'.join(lines)
    print(report)

    if output_file:
        with open(output_file, 'a') as f:
            f.write(report + '\n')


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


def extract_holdout_curves(trajectories, dataset, feedback_mode):
    agents = {}

    for key, data in trajectories.items():
        agent_name, ds, mode, seed = key
        if ds == dataset and mode == feedback_mode:
            holdout_evals = data.get('holdout_evals', [])
            if holdout_evals:
                if agent_name not in agents:
                    agents[agent_name] = []
                # Extract (round, regret) pairs
                rounds = [e[0] for e in holdout_evals]
                regrets = [e[1]['regret'] for e in holdout_evals]
                agents[agent_name].append({'rounds': rounds, 'regrets': regrets})

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


def plot_holdout_curves(ax, agents, title):
    for agent_name, curves in sorted(agents.items(), key=lambda x: np.mean([c['regrets'][-1] for c in x[1]])):
        if len(curves) == 0:
            continue

        # All seeds should have same rounds
        rounds = curves[0]['rounds']
        all_regrets = np.array([c['regrets'] for c in curves])

        mean_regret = np.mean(all_regrets, axis=0)
        std_regret = np.std(all_regrets, axis=0)

        color = COLOR_MAP.get(agent_name, 'black')

        ax.plot(rounds, mean_regret, 'o-', label=agent_name, color=color, linewidth=1.5, markersize=4)
        ax.fill_between(rounds, mean_regret - std_regret, mean_regret + std_regret, color=color, alpha=0.2)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Training Round', fontsize=9)
    ax.set_ylabel('Holdout Regret', fontsize=9)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)


def smooth_curves(agents, window_fraction=10, min_window=5):
    smoothed = {}
    for name, curves in agents.items():
        smoothed[name] = []
        for c in curves:
            arr = np.array(c)
            window = max(min_window, min(50, len(arr) // window_fraction))
            if window > 1 and len(arr) > window:
                smoothed_c = np.convolve(arr, np.ones(window)/window, mode='valid')
            else:
                smoothed_c = arr
            smoothed[name].append(smoothed_c)
    return smoothed


def plot_learning_curves(trajectories, dataset, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top row: Cumulative regret
    agents = extract_curves(trajectories, dataset, 'semi_bandit', 'cumulative_regret')
    if agents:
        plot_with_ci(axes[0, 0], agents, 'Semi-Bandit: Cumulative Regret', 'Cumulative Regret')

    agents = extract_curves(trajectories, dataset, 'full', 'cumulative_regret')
    if agents:
        plot_with_ci(axes[0, 1], agents, 'Full Feedback: Cumulative Regret', 'Cumulative Regret')

    # Bottom row: Per-round regret (smoothed)
    agents = extract_curves(trajectories, dataset, 'semi_bandit', 'regrets')
    if agents:
        smoothed = smooth_curves(agents)
        plot_with_ci(axes[1, 0], smoothed, 'Semi-Bandit: Per-Round Regret', 'Per-Round Regret')

    agents = extract_curves(trajectories, dataset, 'full', 'regrets')
    if agents:
        smoothed = smooth_curves(agents)
        plot_with_ci(axes[1, 1], smoothed, 'Full Feedback: Per-Round Regret', 'Per-Round Regret')

    fig.suptitle(f'{dataset.title()} - Learning Curves', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path / f'learning_curves_{dataset}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: learning_curves_{dataset}.png")
    plt.close()


def plot_holdout_performance(trajectories, dataset, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if holdout data exists
    holdout_agents = extract_holdout_curves(trajectories, dataset, 'semi_bandit')
    if not holdout_agents:
        return  # No holdout data for this dataset

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # Compute cumulative holdout regret
    holdout_cumulative = {}
    for agent_name, curves in holdout_agents.items():
        all_regrets = np.array([c['regrets'] for c in curves])
        cumulative = np.cumsum(all_regrets, axis=1)
        holdout_cumulative[agent_name] = {
            'rounds': curves[0]['rounds'],
            'regrets': cumulative
        }

    # Top-left: Cumulative Holdout Regret
    ax1 = fig.add_subplot(gs[0, 0])
    for agent_name in sorted(holdout_cumulative.keys()):
        data = holdout_cumulative[agent_name]
        mean_regret = data['regrets'].mean(axis=0)
        std_regret = data['regrets'].std(axis=0)
        color = COLOR_MAP.get(agent_name, 'black')
        ax1.plot(data['rounds'], mean_regret, 'o-', label=agent_name, color=color, markersize=4, linewidth=1.5)
        ax1.fill_between(data['rounds'], mean_regret - std_regret, mean_regret + std_regret, color=color, alpha=0.2)
    ax1.set_xlabel('Training Round', fontsize=9)
    ax1.set_ylabel('Cumulative Holdout Regret', fontsize=9)
    ax1.set_title('Cumulative Holdout Regret', fontsize=10)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Top-right: Per-Week Holdout Regret
    ax2 = fig.add_subplot(gs[0, 1])
    plot_holdout_curves(ax2, holdout_agents, 'Per-Week Holdout Regret')

    # Bottom: Points per Holdout Week (average across holdout weeks)
    ax3 = fig.add_subplot(gs[1, :])

    # Extract reward data from holdout_evals
    reward_data = {}
    for key, data in trajectories.items():
        agent_name, ds, mode, seed = key
        if ds == dataset and mode == 'semi_bandit':
            holdout_evals = data.get('holdout_evals', [])
            if holdout_evals:
                if agent_name not in reward_data:
                    reward_data[agent_name] = {'rounds': [], 'rewards': [], 'optimal': []}

                rounds = [e[0] for e in holdout_evals]
                # Use avg_reward (per week) instead of total_reward
                rewards = [e[1].get('avg_reward', e[1].get('total_reward', 0)) for e in holdout_evals]
                optimal = [e[1].get('avg_optimal', e[1].get('total_optimal', 0)) for e in holdout_evals]

                reward_data[agent_name]['rounds'] = rounds
                reward_data[agent_name]['rewards'].append(rewards)
                reward_data[agent_name]['optimal'].append(optimal)

    if reward_data:
        for agent_name, data in sorted(reward_data.items(), key=lambda x: -np.mean([np.mean(r) for r in x[1]['rewards']])):
            rounds = data['rounds']
            rewards = np.array(data['rewards'])
            optimal = np.array(data['optimal'])

            mean_rewards = rewards.mean(axis=0)
            std_rewards = rewards.std(axis=0)

            color = COLOR_MAP.get(agent_name, 'black')
            ax3.plot(rounds, mean_rewards, 'o-', label=agent_name, color=color, markersize=4, linewidth=1.5)
            ax3.fill_between(rounds, mean_rewards - std_rewards, mean_rewards + std_rewards, color=color, alpha=0.2)

        # Plot optimal as reference
        if len(optimal) > 0:
            mean_optimal = optimal.mean(axis=0)
            ax3.plot(rounds, mean_optimal, 'k--', label='Optimal', alpha=0.7, linewidth=2, marker='o', markersize=4)

    ax3.set_xlabel('Training Round', fontsize=9)
    ax3.set_ylabel('Points per Holdout Week', fontsize=9)
    ax3.set_title('Points per Holdout Week', fontsize=10)
    ax3.legend(fontsize=7, loc='lower right')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset.title()} - Holdout Performance', fontsize=12, fontweight='bold')

    plt.savefig(output_path / f'holdout_{dataset}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: holdout_{dataset}.png")
    plt.close()


def plot_trajectories(trajectories, dataset, output_dir, feedback_mode='semi_bandit'):
    output_path = Path(output_dir) / 'plots'
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect data per agent
    agent_data = {}
    for (agent_name, ds, mode, seed), data in trajectories.items():
        if ds != dataset or mode != feedback_mode:
            continue
        if agent_name not in agent_data:
            agent_data[agent_name] = {
                'cumulative_regret': [],
                'regrets': [],
                'rewards': [],
                'optimal_rewards': [],
                'holdout_evals': []
            }
        agent_data[agent_name]['cumulative_regret'].append(data['cumulative_regret'])
        agent_data[agent_name]['regrets'].append(data.get('regrets', []))
        agent_data[agent_name]['rewards'].append(data['rewards'])
        agent_data[agent_name]['optimal_rewards'].append(data['optimal_rewards'])
        agent_data[agent_name]['holdout_evals'].append(data.get('holdout_evals', []))

    if not agent_data:
        return

    # Smoothing window for per-round plots
    sample_data = list(agent_data.values())[0]
    n_rounds = len(sample_data['rewards'][0])
    window = min(50, max(1, n_rounds // 10))

    # === TRAINING PLOT ===
    fig = plt.figure(figsize=(14, 10))

    # Create GridSpec: 2 rows, 2 cols. Top row has 2 subplots, bottom spans both columns
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # Top-left: Cumulative Regret
    ax1 = fig.add_subplot(gs[0, 0])
    for agent_name, data in sorted(agent_data.items(), key=lambda x: np.mean([r[-1] for r in x[1]['cumulative_regret']])):
        regrets = np.array(data['cumulative_regret'])
        mean_regret = regrets.mean(axis=0)
        std_regret = regrets.std(axis=0)
        x = np.arange(len(mean_regret))
        color = COLOR_MAP.get(agent_name, 'black')
        ax1.plot(x, mean_regret, label=agent_name, color=color, linewidth=1.5)
        ax1.fill_between(x, mean_regret - std_regret, mean_regret + std_regret, color=color, alpha=0.2)
    ax1.set_xlabel('Round', fontsize=9)
    ax1.set_ylabel('Cumulative Regret', fontsize=9)
    ax1.set_title('Cumulative Regret', fontsize=10)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Top-right: Per-Round Regret (smoothed)
    ax2 = fig.add_subplot(gs[0, 1])
    for agent_name, data in sorted(agent_data.items(), key=lambda x: np.mean([np.mean(r) for r in x[1]['regrets']])):
        regrets = np.array(data['regrets'])
        if window > 1 and regrets.shape[1] > window:
            kernel = np.ones(window) / window
            smoothed = np.array([np.convolve(r, kernel, mode='valid') for r in regrets])
        else:
            smoothed = regrets
        mean_regret = smoothed.mean(axis=0)
        std_regret = smoothed.std(axis=0)
        x = np.arange(len(mean_regret))
        color = COLOR_MAP.get(agent_name, 'black')
        ax2.plot(x, mean_regret, label=agent_name, color=color, linewidth=1.5)
        ax2.fill_between(x, mean_regret - std_regret, mean_regret + std_regret, color=color, alpha=0.2)
    ax2.set_xlabel(f'Round (smoothed, window={window})', fontsize=9)
    ax2.set_ylabel('Per-Round Regret', fontsize=9)
    ax2.set_title('Per-Round Regret', fontsize=10)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Bottom: Points per Week (spans both columns)
    ax3 = fig.add_subplot(gs[1, :])
    smoothed_optimal = None
    for agent_name, data in sorted(agent_data.items(), key=lambda x: -np.mean([sum(r) for r in x[1]['rewards']])):
        rewards = np.array(data['rewards'])
        optimal = np.array(data['optimal_rewards'])
        if window > 1:
            kernel = np.ones(window) / window
            smoothed_rewards = np.array([np.convolve(r, kernel, mode='valid') for r in rewards])
            smoothed_optimal = np.array([np.convolve(o, kernel, mode='valid') for o in optimal])
        else:
            smoothed_rewards = rewards
            smoothed_optimal = optimal
        mean_rewards = smoothed_rewards.mean(axis=0)
        std_rewards = smoothed_rewards.std(axis=0)
        x = np.arange(len(mean_rewards))
        color = COLOR_MAP.get(agent_name, 'black')
        ax3.plot(x, mean_rewards, label=agent_name, color=color, linewidth=1.5)
        ax3.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, color=color, alpha=0.2)
    if smoothed_optimal is not None:
        mean_optimal = smoothed_optimal.mean(axis=0)
        ax3.plot(np.arange(len(mean_optimal)), mean_optimal, 'k--', label='Optimal', alpha=0.7, linewidth=2)
    ax3.set_xlabel(f'Round (smoothed, window={window})', fontsize=9)
    ax3.set_ylabel('Points per Week', fontsize=9)
    ax3.set_title('Points per Week', fontsize=10)
    ax3.legend(fontsize=7, loc='lower right')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset.upper()} - Training Performance ({feedback_mode})', fontsize=12, fontweight='bold')
    plt.savefig(output_path / f'{dataset}_training_{feedback_mode}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {dataset}_training_{feedback_mode}.png")
    plt.close()

    # === HOLDOUT PLOT (if available) ===
    has_holdout = any(data.get('holdout_evals', []) and data['holdout_evals'][0] for data in agent_data.values())
    if has_holdout:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

        # Compute cumulative holdout regret for each agent
        holdout_cumulative = {}
        holdout_per_week = {}

        for agent_name, data in agent_data.items():
            holdout_evals = data.get('holdout_evals', [])
            if not holdout_evals or not holdout_evals[0]:
                continue

            all_rounds = []
            all_regrets = []
            for seed_evals in holdout_evals:
                if seed_evals:
                    rounds = [e[0] for e in seed_evals]
                    regrets = [e[1]['regret'] for e in seed_evals]
                    all_rounds.append(rounds)
                    all_regrets.append(regrets)

            if all_rounds:
                rounds = all_rounds[0]
                regrets = np.array(all_regrets)
                # Cumulative regret over holdout evaluations
                cumulative = np.cumsum(regrets, axis=1)
                holdout_cumulative[agent_name] = {'rounds': rounds, 'regrets': cumulative}
                holdout_per_week[agent_name] = {'rounds': rounds, 'regrets': regrets}

        # Top-left: Cumulative Holdout Regret
        ax1 = fig.add_subplot(gs[0, 0])
        for agent_name in sorted(holdout_cumulative.keys()):
            data = holdout_cumulative[agent_name]
            mean_regret = data['regrets'].mean(axis=0)
            std_regret = data['regrets'].std(axis=0)
            color = COLOR_MAP.get(agent_name, 'black')
            ax1.plot(data['rounds'], mean_regret, 'o-', label=agent_name, color=color, markersize=4, linewidth=1.5)
            ax1.fill_between(data['rounds'], mean_regret - std_regret, mean_regret + std_regret, color=color, alpha=0.2)
        ax1.set_xlabel('Training Round', fontsize=9)
        ax1.set_ylabel('Cumulative Holdout Regret', fontsize=9)
        ax1.set_title('Cumulative Holdout Regret', fontsize=10)
        ax1.legend(fontsize=7, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Top-right: Per-Week Holdout Regret
        ax2 = fig.add_subplot(gs[0, 1])
        for agent_name in sorted(holdout_per_week.keys()):
            data = holdout_per_week[agent_name]
            mean_regret = data['regrets'].mean(axis=0)
            std_regret = data['regrets'].std(axis=0)
            color = COLOR_MAP.get(agent_name, 'black')
            ax2.plot(data['rounds'], mean_regret, 'o-', label=agent_name, color=color, markersize=4, linewidth=1.5)
            ax2.fill_between(data['rounds'], mean_regret - std_regret, mean_regret + std_regret, color=color, alpha=0.2)
        ax2.set_xlabel('Training Round', fontsize=9)
        ax2.set_ylabel('Per-Week Holdout Regret', fontsize=9)
        ax2.set_title('Per-Week Holdout Regret', fontsize=10)
        ax2.legend(fontsize=7, loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Bottom: Points per Holdout Week (average across holdout weeks)
        ax3 = fig.add_subplot(gs[1, :])
        optimal_data = None
        for agent_name, data in agent_data.items():
            holdout_evals = data.get('holdout_evals', [])
            if not holdout_evals or not holdout_evals[0]:
                continue

            all_rounds = []
            all_rewards = []
            all_optimal = []
            for seed_evals in holdout_evals:
                if seed_evals:
                    rounds = [e[0] for e in seed_evals]
                    # Use avg_reward (per week) instead of total_reward
                    rewards = [e[1].get('avg_reward', e[1].get('total_reward', 0)) for e in seed_evals]
                    optimal = [e[1].get('avg_optimal', e[1].get('total_optimal', 0)) for e in seed_evals]
                    all_rounds.append(rounds)
                    all_rewards.append(rewards)
                    all_optimal.append(optimal)

            if all_rounds:
                rounds = all_rounds[0]
                rewards = np.array(all_rewards)
                optimal = np.array(all_optimal)
                mean_rewards = rewards.mean(axis=0)
                std_rewards = rewards.std(axis=0)
                color = COLOR_MAP.get(agent_name, 'black')
                ax3.plot(rounds, mean_rewards, 'o-', label=agent_name, color=color, markersize=4, linewidth=1.5)
                ax3.fill_between(rounds, mean_rewards - std_rewards, mean_rewards + std_rewards, color=color, alpha=0.2)
                optimal_data = {'rounds': rounds, 'optimal': optimal}

        # Plot optimal as reference
        if optimal_data is not None:
            mean_optimal = optimal_data['optimal'].mean(axis=0)
            ax3.plot(optimal_data['rounds'], mean_optimal, 'k--', label='Optimal', alpha=0.7, linewidth=2, marker='o', markersize=4)

        ax3.set_xlabel('Training Round', fontsize=9)
        ax3.set_ylabel('Points per Holdout Week', fontsize=9)
        ax3.set_title('Points per Holdout Week', fontsize=10)
        ax3.legend(fontsize=7, loc='lower right')
        ax3.grid(True, alpha=0.3)

        fig.suptitle(f'{dataset.upper()} - Holdout Performance ({feedback_mode})', fontsize=12, fontweight='bold')
        plt.savefig(output_path / f'{dataset}_holdout_{feedback_mode}.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {dataset}_holdout_{feedback_mode}.png")
        plt.close()


def plot_generalization_comparison(trajectories, dataset, output_dir):
    """Plot training vs holdout regret per week to assess generalizability.

    Both metrics normalized to regret-per-week for fair comparison:
    - Training: cumulative_regret / n_rounds (average regret per round)
    - Holdout: holdout_regret / n_holdout_weeks (average regret per holdout week)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect data per agent
    agent_data = {}
    for (agent_name, ds, mode, seed), data in trajectories.items():
        if ds != dataset or mode != 'semi_bandit':
            continue
        if agent_name not in agent_data:
            agent_data[agent_name] = {
                'cumulative_regret': [],
                'holdout_evals': []
            }
        agent_data[agent_name]['cumulative_regret'].append(data['cumulative_regret'])
        agent_data[agent_name]['holdout_evals'].append(data.get('holdout_evals', []))

    if not agent_data:
        return

    # Check if holdout data exists
    has_holdout = any(data.get('holdout_evals', []) and data['holdout_evals'][0] for data in agent_data.values())
    if not has_holdout:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for agent_name, data in sorted(agent_data.items()):
        holdout_evals = data.get('holdout_evals', [])
        cum_regrets = data.get('cumulative_regret', [])

        if not holdout_evals or not holdout_evals[0] or not cum_regrets:
            continue

        # Get evaluation rounds from holdout
        eval_rounds = [e[0] for e in holdout_evals[0]]

        # Compute training regret per round at each eval point
        train_regret_per_round = []
        for seed_regrets in cum_regrets:
            seed_vals = [seed_regrets[r] / (r + 1) for r in eval_rounds]
            train_regret_per_round.append(seed_vals)
        train_regret_per_round = np.array(train_regret_per_round)

        # Compute holdout regret per week at each eval point
        holdout_regret_per_week = []
        for seed_evals in holdout_evals:
            if seed_evals:
                # avg_regret is already per-week
                seed_vals = [e[1].get('avg_regret', e[1]['regret']) for e in seed_evals]
                holdout_regret_per_week.append(seed_vals)
        holdout_regret_per_week = np.array(holdout_regret_per_week)

        color = COLOR_MAP.get(agent_name, 'black')

        # Plot training regret (solid line)
        mean_train = train_regret_per_round.mean(axis=0)
        std_train = train_regret_per_round.std(axis=0)
        ax.plot(eval_rounds, mean_train, '-', label=f'{agent_name} (train)',
                color=color, linewidth=1.5, alpha=0.8)
        ax.fill_between(eval_rounds, mean_train - std_train, mean_train + std_train,
                       color=color, alpha=0.1)

        # Plot holdout regret (dashed line)
        mean_holdout = holdout_regret_per_week.mean(axis=0)
        std_holdout = holdout_regret_per_week.std(axis=0)
        ax.plot(eval_rounds, mean_holdout, '--', label=f'{agent_name} (holdout)',
                color=color, linewidth=1.5, alpha=0.8)
        ax.fill_between(eval_rounds, mean_holdout - std_holdout, mean_holdout + std_holdout,
                       color=color, alpha=0.1)

    ax.set_xlabel('Training Round', fontsize=10)
    ax.set_ylabel('Regret per Week', fontsize=10)
    ax.set_title(f'{dataset.title()} - Training vs Holdout Regret (per week)', fontsize=12)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f'generalization_{dataset}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: generalization_{dataset}.png")
    plt.close()


def plot_points_per_round(trajectories, dataset, output_dir):
    """Plot actual points per round (smoothed) for fantasy-type datasets. (Legacy - now integrated into training plots)"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract rewards and optimal rewards
    agents = {}
    for key, data in trajectories.items():
        agent_name, ds, mode, seed = key
        if ds == dataset and mode == 'semi_bandit':
            rewards = data.get('rewards', [])
            optimal_rewards = data.get('optimal_rewards', [])
            if rewards and optimal_rewards:
                if agent_name not in agents:
                    agents[agent_name] = {'rewards': [], 'optimal_rewards': []}
                agents[agent_name]['rewards'].append(rewards)
                agents[agent_name]['optimal_rewards'].append(optimal_rewards)

    if not agents:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine smoothing window
    sample_rewards = list(agents.values())[0]['rewards'][0]
    window = max(5, min(50, len(sample_rewards) // 10))

    for agent_name, data in sorted(agents.items(), key=lambda x: -np.mean([sum(r) for r in x[1]['rewards']])):
        rewards = np.array(data['rewards'])
        optimal = np.array(data['optimal_rewards'])

        # Smooth
        if window > 1:
            kernel = np.ones(window) / window
            smoothed_rewards = np.array([np.convolve(r, kernel, mode='valid') for r in rewards])
            smoothed_optimal = np.array([np.convolve(o, kernel, mode='valid') for o in optimal])
        else:
            smoothed_rewards = rewards
            smoothed_optimal = optimal

        mean_rewards = smoothed_rewards.mean(axis=0)
        std_rewards = smoothed_rewards.std(axis=0)

        x = np.arange(len(mean_rewards))
        color = COLOR_MAP.get(agent_name, 'black')
        ax.plot(x, mean_rewards, label=agent_name, color=color, linewidth=1.5)
        ax.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, color=color, alpha=0.2)

    # Plot optimal as reference
    mean_optimal = smoothed_optimal.mean(axis=0)
    ax.plot(x, mean_optimal, 'k--', label='Optimal', alpha=0.7, linewidth=2)

    ax.set_xlabel(f'Round (smoothed, window={window})', fontsize=10)
    ax.set_ylabel('Points per Week', fontsize=10)
    ax.set_title(f'{dataset.title()}: Points per Week', fontsize=12)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f'points_per_round_{dataset}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: points_per_round_{dataset}.png")
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


def plot_all(trajectories, results, output_dir, include_stats=True):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = set()
    for key in trajectories.keys():
        _, dataset, _, _ = key
        datasets.add(dataset)

    print("\nGenerating plots:")

    for dataset in sorted(datasets):
        print(f"\n{dataset.title()}:")
        # Learning curves: semi-bandit vs full feedback comparison (2x2)
        plot_learning_curves(trajectories, dataset, output_dir)
        # Training and holdout plots with detailed subplots (cumulative, per-round, points)
        # Note: plot_trajectories generates both training and holdout plots
        plot_trajectories(trajectories, dataset, output_dir, 'semi_bandit')
        # Generalization comparison: training vs holdout regret per week
        plot_generalization_comparison(trajectories, dataset, output_dir)
        # Feedback comparison bar chart
        plot_feedback_comparison(results, dataset, output_dir)

    print(f"\nAll datasets:")
    plot_final_regret_bars(results, output_dir)
    plot_ranking_heatmap(results, output_dir)

    if include_stats:
        stats_file = output_path / 'statistical_analysis.txt'
        # Clear existing file
        if stats_file.exists():
            stats_file.unlink()

        print("\nStatistical Analysis:")
        for dataset in sorted(datasets):
            for feedback in ['semi_bandit', 'full']:
                print_significance_report(results, dataset, feedback, stats_file)

        print(f"\n  Saved: statistical_analysis.txt")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive plots from experiment results')
    parser.add_argument('--input', type=str, default='results/eval',
                        help='Input directory with metrics/ folder')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots (default: input/plots)')
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'curves', 'bars', 'comparison', 'heatmap', 'stats'],
                        help='Type of plot to generate')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip statistical analysis when using --type all')

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
        plot_all(trajectories, results, args.output, include_stats=not args.no_stats)
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
    elif args.type == 'stats':
        print("\nStatistical Analysis:")
        stats_file = output_path / 'statistical_analysis.txt'
        if stats_file.exists():
            stats_file.unlink()
        for dataset in sorted(datasets):
            for feedback in ['semi_bandit', 'full']:
                print_significance_report(results, dataset, feedback, stats_file)
        print(f"\n  Saved: statistical_analysis.txt")

    print(f"\nDone! Plots saved to {args.output}")


if __name__ == '__main__':
    main()

import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, '.')

from experiments.tuning.results_store import ResultsStore


def plot_sensitivity(
    results_store: ResultsStore,
    agent_name: str,
    dataset: str,
    hyperparameter: str,
    output_path: str,
    feedback_type: str,
    figsize: Tuple[int, int] = (10, 6),
):
    hp_values = results_store.get_trials_for_sensitivity(
        agent_name, dataset, hyperparameter, feedback_type
    )

    if not hp_values:
        raise ValueError(f"No data found for {agent_name}/{hyperparameter}")

    sorted_values = sorted(hp_values.keys())
    means = [np.mean(hp_values[v]) for v in sorted_values]
    stds = [np.std(hp_values[v]) for v in sorted_values]
    n_samples = [len(hp_values[v]) for v in sorted_values]

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        range(len(sorted_values)), means, yerr=stds,
        marker='o', capsize=5, capthick=2, linewidth=2, markersize=8
    )

    ax.set_xticks(range(len(sorted_values)))
    ax.set_xticklabels([f"{v:.3g}" if isinstance(v, float) else str(v) for v in sorted_values],
                       rotation=45, ha='right')

    ax.set_xlabel(hyperparameter, fontsize=12)
    ax.set_ylabel('Train Cumulative Regret', fontsize=12)
    ax.set_title(f'{agent_name} on {dataset}: Sensitivity to {hyperparameter}', fontsize=14)
    ax.grid(True, alpha=0.3)

    for i, n in enumerate(n_samples):
        ax.annotate(f'n={n}', (i, means[i] + stds[i] + 0.02 * max(means)),
                   ha='center', fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved sensitivity plot: {output_path}")


def plot_sensitivity_heatmap(
    results_store: ResultsStore,
    agent_name: str,
    dataset: str,
    hp1: str,
    hp2: str,
    output_path: str,
    feedback_type: str,
    figsize: Tuple[int, int] = (10, 8),
):
    trials = results_store.load_all_trials(agent_name, dataset, feedback_type)

    if not trials:
        raise ValueError(f"No trials found for {agent_name}")

    pair_values: Dict[Tuple, List[float]] = {}
    for trial in trials:
        if hp1 not in trial.params or hp2 not in trial.params:
            continue
        key = (trial.params[hp1], trial.params[hp2])
        if key not in pair_values:
            pair_values[key] = []
        pair_values[key].append(trial.final_train_regret)

    if not pair_values:
        raise ValueError(f"No data found for {hp1} vs {hp2}")

    hp1_values = sorted(set(k[0] for k in pair_values.keys()))
    hp2_values = sorted(set(k[1] for k in pair_values.keys()))

    heatmap = np.full((len(hp2_values), len(hp1_values)), np.nan)
    for (v1, v2), regrets in pair_values.items():
        i = hp1_values.index(v1)
        j = hp2_values.index(v2)
        heatmap[j, i] = np.mean(regrets)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(heatmap, cmap='viridis_r', aspect='auto')

    ax.set_xticks(range(len(hp1_values)))
    ax.set_xticklabels([f"{v:.3g}" if isinstance(v, float) else str(v) for v in hp1_values],
                       rotation=45, ha='right')
    ax.set_yticks(range(len(hp2_values)))
    ax.set_yticklabels([f"{v:.3g}" if isinstance(v, float) else str(v) for v in hp2_values])

    ax.set_xlabel(hp1, fontsize=12)
    ax.set_ylabel(hp2, fontsize=12)
    ax.set_title(f'{agent_name}: {hp1} vs {hp2}', fontsize=14)

    plt.colorbar(im, label='Mean Train Cumulative Regret')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_all_sensitivity_plots(
    results_store: ResultsStore,
    search_spaces: Dict[str, Dict[str, Any]],
    dataset: str,
    output_dir: str,
    feedback_type: str,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for agent_name, hp_space in search_spaces.items():
        # Skip agents that don't have trials for this dataset
        trials = results_store.load_all_trials(agent_name=agent_name, dataset=dataset, feedback_type=feedback_type)
        if not trials:
            continue

        agent_dir = output_path / agent_name
        agent_dir.mkdir(exist_ok=True)

        hps = [k for k in hp_space.keys() if k not in ('tune_lr',) and isinstance(hp_space[k], dict)]

        for hp_name in hps:
            try:
                plot_path = agent_dir / f"sensitivity_{hp_name}.png"
                plot_sensitivity(
                    results_store, agent_name, dataset, hp_name,
                    str(plot_path), feedback_type
                )
            except ValueError as e:
                print(f"  Skipping {agent_name}/{hp_name}: {e}")


def compute_sensitivity_scores(
    results_store: ResultsStore,
    agent_name: str,
    dataset: str,
    hyperparameter: str,
    feedback_type: str,
) -> Dict[str, float]:
    hp_values = results_store.get_trials_for_sensitivity(
        agent_name, dataset, hyperparameter, feedback_type
    )

    if not hp_values or len(hp_values) < 2:
        raise ValueError(f"Not enough data for sensitivity analysis of {hyperparameter}")

    means = [np.mean(v) for v in hp_values.values()]
    stds = [np.std(v) for v in hp_values.values()]

    return {
        'range': float(max(means) - min(means)),
        'cv': float(np.std(means) / np.mean(means)) if np.mean(means) > 0 else 0,
        'mean_std': float(np.mean(stds)),
        'best_value': list(hp_values.keys())[np.argmin(means)],
        'worst_value': list(hp_values.keys())[np.argmax(means)],
        'n_values': len(hp_values),
    }


def generate_sensitivity_report(
    results_store: ResultsStore,
    agents: List[str],
    search_spaces: Dict[str, Dict[str, Any]],
    dataset: str,
    output_path: str,
    feedback_type: str,
):
    report = {
        'dataset': dataset,
        'feedback_type': feedback_type,
        'agents': {},
    }

    for agent_name in agents:
        if agent_name not in search_spaces:
            continue

        # Skip agents that don't have trials for this dataset
        trials = results_store.load_all_trials(agent_name=agent_name, dataset=dataset, feedback_type=feedback_type)
        if not trials:
            continue

        hp_space = search_spaces[agent_name]
        hps = [k for k in hp_space.keys() if k not in ('tune_lr',) and isinstance(hp_space[k], dict)]

        agent_report = {}
        for hp_name in hps:
            try:
                scores = compute_sensitivity_scores(
                    results_store, agent_name, dataset, hp_name, feedback_type
                )
                agent_report[hp_name] = scores
            except ValueError:
                # Skip HPs without enough data
                continue

        if agent_report:
            report['agents'][agent_name] = agent_report

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Saved sensitivity report: {output_path}")

    return report


def print_most_sensitive_hps(
    report: Dict[str, Any],
    top_k: int = 5,
):
    print(f"\n{'='*70}")
    print("MOST SENSITIVE HYPERPARAMETERS")
    print(f"{'='*70}")

    sensitivities = []
    for agent_name, hps in report['agents'].items():
        for hp_name, scores in hps.items():
            if 'range' in scores:
                sensitivities.append((agent_name, hp_name, scores['range'], scores['cv']))

    by_range = sorted(sensitivities, key=lambda x: x[2], reverse=True)[:top_k]

    print(f"\nBy absolute range (regret difference between best/worst HP value):")
    print(f"{'Agent':<25} {'Hyperparameter':<20} {'Range':>10}")
    print("-" * 60)
    for agent, hp, range_val, _ in by_range:
        print(f"{agent:<25} {hp:<20} {range_val:>10.1f}")

    by_cv = sorted(sensitivities, key=lambda x: x[3], reverse=True)[:top_k]

    print(f"\nBy coefficient of variation (relative sensitivity):")
    print(f"{'Agent':<25} {'Hyperparameter':<20} {'CV':>10}")
    print("-" * 60)
    for agent, hp, _, cv in by_cv:
        print(f"{agent:<25} {hp:<20} {cv:>10.3f}")

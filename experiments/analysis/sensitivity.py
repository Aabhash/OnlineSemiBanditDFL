import numpy as np
from typing import Dict, Any, List
import json

import sys
sys.path.insert(0, '.')

from experiments.tuning.results_store import ResultsStore


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

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, '.')

from experiments.tuning.results_store import ResultsStore


def wilcoxon_test_agents(
    results_store: ResultsStore,
    agent1: str,
    agent2: str,
    dataset: str,
    feedback_type: str,
    alpha: float,
    alternative: str = 'two-sided',
    use_holdout: bool = False,
) -> Dict[str, Any]:
    params1 = results_store.get_best_params(agent1, dataset, feedback_type, use_holdout=use_holdout)
    params2 = results_store.get_best_params(agent2, dataset, feedback_type, use_holdout=use_holdout)

    trials1 = results_store.get_results_for_params(agent1, dataset, params1, feedback_type)
    trials2 = results_store.get_results_for_params(agent2, dataset, params2, feedback_type)

    if not trials1:
        raise ValueError(f"No trials found for {agent1}")
    if not trials2:
        raise ValueError(f"No trials found for {agent2}")

    seed_to_regret1 = {t.seed: t.final_validation_regret for t in trials1}
    seed_to_regret2 = {t.seed: t.final_validation_regret for t in trials2}

    common_seeds = sorted(set(seed_to_regret1.keys()) & set(seed_to_regret2.keys()))

    if len(common_seeds) < 2:
        raise ValueError(f"Not enough common seeds for comparison (found {len(common_seeds)})")

    regrets1 = np.array([seed_to_regret1[s] for s in common_seeds])
    regrets2 = np.array([seed_to_regret2[s] for s in common_seeds])

    stat, p_value = stats.wilcoxon(regrets1, regrets2, alternative=alternative)

    mean1 = np.mean(regrets1)
    mean2 = np.mean(regrets2)
    winner = agent1 if mean1 < mean2 else agent2

    pooled_var = (np.var(regrets1) + np.var(regrets2)) / 2
    effect_size = float((mean1 - mean2) / np.sqrt(pooled_var)) if pooled_var > 0 else 0

    return {
        'agent1': agent1,
        'agent2': agent2,
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'winner': winner if p_value < alpha else None,
        'mean1': float(mean1),
        'mean2': float(mean2),
        'std1': float(np.std(regrets1)),
        'std2': float(np.std(regrets2)),
        'n_pairs': len(common_seeds),
        'effect_size': effect_size,
    }


def pairwise_significance_matrix(
    results_store: ResultsStore,
    agents: List[str],
    dataset: str,
    feedback_type: str,
    alpha: float,
    use_holdout: bool = False,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], 'pd.DataFrame']:
    results = {}
    n_agents = len(agents)

    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i >= j:
                continue

            result = wilcoxon_test_agents(
                results_store, agent1, agent2, dataset, feedback_type, alpha,
                use_holdout=use_holdout
            )
            results[(agent1, agent2)] = result

    matrix_data = np.ones((n_agents, n_agents))
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i == j:
                continue
            key = (agent1, agent2) if (agent1, agent2) in results else (agent2, agent1)
            if key in results:
                matrix_data[i, j] = results[key]['p_value']
                matrix_data[j, i] = results[key]['p_value']

    p_matrix = pd.DataFrame(matrix_data, index=agents, columns=agents)

    return results, p_matrix


def save_significance_results(
    results: Dict[Tuple[str, str], Dict[str, Any]],
    p_matrix: 'pd.DataFrame',
    output_dir: str,
    dataset: str,
    feedback_type: str,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_results = {}
    for (a1, a2), result in results.items():
        key = f"{a1}_vs_{a2}"
        json_results[key] = result

    json_path = output_path / f"significance_{dataset}_{feedback_type}.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    csv_path = output_path / f"pvalue_matrix_{dataset}_{feedback_type}.csv"
    p_matrix.to_csv(csv_path)

    print(f"Saved significance results to {output_path}")


def print_significance_summary(
    results: Dict[Tuple[str, str], Dict[str, Any]],
    alpha: float,
):
    print(f"\n{'='*70}")
    print("SIGNIFICANCE SUMMARY (Wilcoxon signed-rank test)")
    print(f"{'='*70}")

    significant_pairs = []
    for (a1, a2), result in results.items():
        if result['significant']:
            significant_pairs.append((a1, a2, result))

    if not significant_pairs:
        print(f"No significant differences at alpha={alpha}")
        return

    print(f"\nSignificant differences (alpha={alpha}):\n")
    print(f"{'Agent 1':<20} {'Agent 2':<20} {'p-value':>10} {'Winner':<20}")
    print("-" * 70)

    for a1, a2, result in sorted(significant_pairs, key=lambda x: x[2]['p_value']):
        p_str = f"{result['p_value']:.4f}"
        winner = result['winner']
        print(f"{a1:<20} {a2:<20} {p_str:>10} {winner:<20}")


def rank_agents(
    results_store: ResultsStore,
    agents: List[str],
    dataset: str,
    feedback_type: str,
    alpha: float,
    use_holdout: bool = False,
) -> List[Tuple[str, float, int]]:
    results, _ = pairwise_significance_matrix(
        results_store, agents, dataset, feedback_type, alpha, use_holdout=use_holdout
    )

    win_counts = {agent: 0 for agent in agents}
    for (a1, a2), result in results.items():
        winner = result['winner']
        if winner:
            win_counts[winner] += 1

    mean_regrets = {}
    for agent in agents:
        summary = results_store.compute_agent_summary(agent, dataset, feedback_type, use_holdout=use_holdout)
        mean_regrets[agent] = summary.mean_validation_regret

    ranking = [
        (agent, mean_regrets[agent], win_counts[agent])
        for agent in agents
    ]

    return sorted(ranking, key=lambda x: x[1])


def generate_significance_report(
    results_store: ResultsStore,
    agents: List[str],
    dataset: str,
    feedback_type: str,
    alpha: float,
    output_dir: Optional[str] = None,
    use_holdout: bool = False,
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("STATISTICAL SIGNIFICANCE ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Dataset: {dataset}")
    lines.append(f"Feedback: {feedback_type}")
    lines.append(f"Agents: {len(agents)}")
    lines.append(f"Selection metric: {'holdout regret' if use_holdout else 'training regret'}")
    lines.append("")

    results, p_matrix = pairwise_significance_matrix(
        results_store, agents, dataset, feedback_type, alpha, use_holdout=use_holdout
    )

    lines.append("AGENT RANKING (by validation regret)")
    lines.append("-" * 70)
    ranking = rank_agents(results_store, agents, dataset, feedback_type, alpha, use_holdout=use_holdout)

    lines.append(f"{'Rank':<6} {'Agent':<25} {'Mean Regret':>12} {'Wins':>8}")
    for i, (agent, regret, wins) in enumerate(ranking):
        lines.append(f"{i+1:<6} {agent:<25} {regret:>12.1f} {wins:>8}")

    lines.append("")

    lines.append(f"SIGNIFICANT DIFFERENCES (p < {alpha})")
    lines.append("-" * 70)

    significant = [(k, v) for k, v in results.items() if v['significant']]
    if significant:
        for (a1, a2), result in sorted(significant, key=lambda x: x[1]['p_value']):
            winner = result['winner']
            p = result['p_value']
            lines.append(f"  {a1} vs {a2}: p={p:.4f}, winner={winner}")
    else:
        lines.append("  No significant differences found")

    report = "\n".join(lines)

    if output_dir:
        save_significance_results(results, p_matrix, output_dir, dataset, feedback_type)

        report_path = Path(output_dir) / f"significance_report_{dataset}_{feedback_type}.txt"
        with open(report_path, 'w') as f:
            f.write(report)

    return report

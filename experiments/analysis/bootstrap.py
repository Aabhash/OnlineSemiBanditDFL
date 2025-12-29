import numpy as np
from typing import Tuple, Dict, Any, List, Callable, Optional
import json
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, '.')

from experiments.tuning.results_store import ResultsStore


def bootstrap_confidence_interval(
    data: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    statistic: Callable = np.mean,
) -> Tuple[float, float, float]:
    if len(data) == 0:
        raise ValueError("Empty data array")

    rng = np.random.default_rng(seed)
    n = len(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    point_estimate = statistic(data)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return (float(point_estimate), float(ci_lower), float(ci_upper))


def bootstrap_ci_bca(
    data: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    statistic: Callable = np.mean,
) -> Tuple[float, float, float]:
    if len(data) < 2:
        raise ValueError("BCa requires at least 2 data points")

    rng = np.random.default_rng(seed)
    n = len(data)
    point_estimate = statistic(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    bootstrap_stats = np.array(bootstrap_stats)

    z0 = stats.norm.ppf(np.mean(bootstrap_stats < point_estimate))

    jackknife_stats = []
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_stats.append(statistic(jackknife_sample))
    jackknife_stats = np.array(jackknife_stats)
    jackknife_mean = np.mean(jackknife_stats)
    num = np.sum((jackknife_mean - jackknife_stats) ** 3)
    denom = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0

    alpha = 1 - confidence
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    def adjusted_percentile(z_alpha):
        return stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))

    p_lower = adjusted_percentile(z_alpha_lower)
    p_upper = adjusted_percentile(z_alpha_upper)

    ci_lower = np.percentile(bootstrap_stats, 100 * p_lower)
    ci_upper = np.percentile(bootstrap_stats, 100 * p_upper)

    return (float(point_estimate), float(ci_lower), float(ci_upper))


def compute_agent_ci(
    results_store: ResultsStore,
    agent_name: str,
    dataset: str,
    feedback_type: str,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    use_bca: bool = False,
) -> Dict[str, Tuple[float, float, float]]:
    best_params = results_store.get_best_params(agent_name, dataset, feedback_type)

    trials = results_store.get_results_for_params(
        agent_name, dataset, best_params, feedback_type
    )

    if not trials:
        raise ValueError(f"No trials found for {agent_name} with best params")

    ci_func = bootstrap_ci_bca if use_bca else bootstrap_confidence_interval

    val_regrets = np.array([t.final_validation_regret for t in trials])
    train_regrets = np.array([t.final_train_regret for t in trials])
    holdout_regrets = np.array([t.holdout_regret for t in trials if t.holdout_regret is not None])

    results = {
        'validation_regret': ci_func(val_regrets, n_bootstrap, confidence, seed),
        'train_regret': ci_func(train_regrets, n_bootstrap, confidence, seed),
    }

    if len(holdout_regrets) > 0:
        results['holdout_regret'] = ci_func(holdout_regrets, n_bootstrap, confidence, seed)

    return results


def compute_all_confidence_intervals(
    results_store: ResultsStore,
    agents: List[str],
    dataset: str,
    feedback_type: str,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    all_cis = {}

    for agent_name in agents:
        cis = compute_agent_ci(
            results_store, agent_name, dataset, feedback_type,
            n_bootstrap, confidence, seed
        )
        all_cis[agent_name] = cis

    if output_path:
        json_cis = {}
        for agent, metrics in all_cis.items():
            json_cis[agent] = {
                metric: {
                    'mean': ci[0],
                    'ci_lower': ci[1],
                    'ci_upper': ci[2],
                    'ci_width': ci[2] - ci[1],
                }
                for metric, ci in metrics.items()
            }

        with open(output_path, 'w') as f:
            json.dump({
                'dataset': dataset,
                'feedback_type': feedback_type,
                'confidence': confidence,
                'n_bootstrap': n_bootstrap,
                'agents': json_cis,
            }, f, indent=2)

    return all_cis


def print_ci_table(
    all_cis: Dict[str, Dict[str, Tuple[float, float, float]]],
    metric: str = 'validation_regret',
):
    print(f"\n{'Agent':<25} {'Mean':>10} {'95% CI':>20}")
    print("-" * 60)

    sorted_agents = sorted(
        all_cis.items(),
        key=lambda x: x[1][metric][0]
    )

    for agent_name, cis in sorted_agents:
        mean, ci_lower, ci_upper = cis[metric]
        ci_str = f"[{ci_lower:.1f}, {ci_upper:.1f}]"
        print(f"{agent_name:<25} {mean:>10.1f} {ci_str:>20}")

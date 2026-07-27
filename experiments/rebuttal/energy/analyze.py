import argparse
import json
from pathlib import Path

import numpy as np


METHODS = (
    "BayesianSPO",
    "BayesianMSE",
    "BLL-Online",
    "LinUCB",
    "TS",
    "EpsilonGreedy",
    "Random",
)


def load_rows(path):
    return json.loads((Path(path) / "per_seed.json").read_text())


def load_selected_params(path, method):
    selected = json.loads(Path(path).read_text())
    if method not in selected:
        raise ValueError(f"{path} does not contain {method}")
    return selected[method]


def validate_rows(rows, method, expected_seeds, expected_params):
    method_rows = [row for row in rows if row["agent"] == method]
    seeds = [row["seed"] for row in method_rows]
    if sorted(seeds) != list(expected_seeds):
        raise ValueError(f"{method} seeds differ from expected set: {sorted(seeds)}")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{method} has duplicate seed rows")
    for row in method_rows:
        if row["feedback"] != "semi_bandit":
            raise ValueError(f"{method} contains non-semi-bandit rows")
        if row["rounds"] != 395:
            raise ValueError(f"{method} contains a non-395-round row")
        if row["observed_per_round"] != 20:
            raise ValueError(f"{method} does not observe 20 components per round")
        if not np.isfinite(row["final_cumulative_regret_raw"]):
            raise ValueError(f"{method} contains a non-finite regret")
        if row.get("hyperparameters") != expected_params:
            raise ValueError(f"{method} row uses unexpected hyperparameters")
    return method_rows


def paired_comparison(spo_rows, baseline_rows, bootstrap_seed=2026):
    spo = {row["seed"]: row["final_cumulative_regret_raw"] for row in spo_rows}
    baseline = {
        row["seed"]: row["final_cumulative_regret_raw"] for row in baseline_rows
    }
    seeds = sorted(set(spo) & set(baseline))
    spo_values = np.asarray([spo[seed] for seed in seeds], dtype=float)
    baseline_values = np.asarray(
        [baseline[seed] for seed in seeds],
        dtype=float,
    )
    difference = baseline_values - spo_values
    relative = 100 * difference / baseline_values
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(seeds), size=(100_000, len(seeds)))
    boot_relative = relative[indices].mean(axis=1)
    return {
        "n_pairs": len(seeds),
        "ratio_of_means_reduction_percent": float(
            100 * (baseline_values.mean() - spo_values.mean()) / baseline_values.mean()
        ),
        "mean_paired_reduction_ci95_percent": [
            float(relative.mean()),
            *np.percentile(boot_relative, [2.5, 97.5]).tolist(),
        ],
        "BayesianSPO_wins": int((difference > 0).sum()),
        "baseline_wins": int((difference < 0).sum()),
    }


def analyze(args):
    expected_seeds = tuple(range(30))
    task_root = Path(args.bayesian_dir).parent
    bayesian_rows = load_rows(args.bayesian_dir)
    bayesian_tuning_path = task_root / "semi_bandit_tuning" / "best_params.json"
    selected_params = {
        method: load_selected_params(bayesian_tuning_path, method)
        for method in ("BayesianSPO", "BayesianMSE")
    }
    rows_by_method = {
        method: validate_rows(
            bayesian_rows,
            method,
            expected_seeds,
            selected_params[method],
        )
        for method in ("BayesianSPO", "BayesianMSE")
    }

    for method in METHODS[2:]:
        result_dir = Path(args.baseline_root) / method
        rows = load_rows(result_dir)
        if method == "BLL-Online":
            tuning_path = task_root / "semi_bandit_bll_tuning" / "best_params.json"
        else:
            tuning_path = (
                task_root / "semi_bandit_baseline_tuning" / method / "best_params.json"
            )
        selected_params[method] = load_selected_params(tuning_path, method)
        rows_by_method[method] = validate_rows(
            rows,
            method,
            expected_seeds,
            selected_params[method],
        )

    summary = {}
    comparisons = {}
    spo_rows = rows_by_method["BayesianSPO"]
    for method in METHODS:
        method_rows = rows_by_method[method]
        regrets = np.asarray(
            [row["final_cumulative_regret_raw"] for row in method_rows]
        )
        runtimes = np.asarray([row["wall_time_seconds"] for row in method_rows])
        summary[method] = {
            "mean_cumulative_regret_raw": float(regrets.mean()),
            "std_cumulative_regret_raw": float(regrets.std(ddof=1)),
            "mean_wall_time_seconds": float(runtimes.mean()),
            "n_seeds": len(method_rows),
        }
        if method != "BayesianSPO":
            comparison = paired_comparison(spo_rows, method_rows)
            comparisons[method] = comparison
            summary[method]["ratio_of_means_reduction_percent"] = comparison[
                "ratio_of_means_reduction_percent"
            ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "paired_vs_bayesian_spo.json").write_text(
        json.dumps(comparisons, indent=2)
    )
    print(json.dumps({"summary": summary, "comparisons": comparisons}, indent=2))
    return summary, comparisons


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bayesian-dir",
        default=(
            "results/rebuttal/energy_scheduling_task_start/" "semi_bandit_final_30seed"
        ),
    )
    parser.add_argument(
        "--baseline-root",
        default=(
            "results/rebuttal/energy_scheduling_task_start/"
            "semi_bandit_baseline_final_30seed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/rebuttal/energy_scheduling_task_start/"
            "semi_bandit_all_methods_30seed"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

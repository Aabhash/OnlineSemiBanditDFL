from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRUCTURE_SUMMARY = (
    REPO_ROOT / "results/rebuttal/pretraining_structure_validation/summary.csv"
)
DEFAULT_OUTCOME_JSON = (
    REPO_ROOT / "results/paper/eval_online_10k/metrics/final_eval_regrets.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/rebuttal/pretraining_structure_validation"
DATASET_ORDER = [
    "knapsack_32",
    "knapsack_100",
    "shortest_path_8x8",
    "shortest_path_20x20",
    "portfolio_32",
]
DATASET_LABELS = {
    "knapsack_32": "Knapsack 32",
    "knapsack_100": "Knapsack 100",
    "shortest_path_8x8": "Shortest path 8x8",
    "shortest_path_20x20": "Shortest path 20x20",
    "portfolio_32": "Portfolio 32",
}
STRUCTURAL_METRICS = [
    "normalized_per_change_margin_median",
    "strict_gap_noise_snr_median",
    "global_invariant_predictable_variance_fraction",
    "variance_weighted_noncontest_fraction",
    "component_contest_probability_mean",
    "q0_directional_effective_exposure_mean",
    "q0_directional_observation_coverage_mean",
    "near_alternative_hamming_mean",
    "strict_alternative_new_arm_burden_mean",
    "strict_alternative_one_exchange_rate",
    "local_near_alternative_predictable_variance_fraction",
    "equivalent_optimal_action_rate",
]


def _validate_structure_summary(frame: pd.DataFrame) -> None:
    required = {
        "dataset",
        "environment_seed",
        *STRUCTURAL_METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Structural summary is missing required columns: " + ", ".join(missing)
        )
    unexpected = sorted(set(frame["dataset"]) - set(DATASET_ORDER))
    missing_datasets = sorted(set(DATASET_ORDER) - set(frame["dataset"]))
    if unexpected or missing_datasets:
        raise ValueError(
            "Unexpected benchmark set: "
            f"unexpected={unexpected}, missing={missing_datasets}."
        )
    duplicate = frame.duplicated(["dataset", "environment_seed"])
    if bool(duplicate.any()):
        rows = frame.loc[duplicate, ["dataset", "environment_seed"]]
        raise ValueError(f"Duplicate dataset/seed rows:\n{rows}")
    counts = frame.groupby("dataset")["environment_seed"].nunique()
    if not bool((counts == 10).all()):
        raise ValueError(
            "Expected ten independent environment instances per benchmark; "
            f"observed {counts.to_dict()}."
        )
    values = frame[STRUCTURAL_METRICS].to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        raise ValueError("Structural metrics contain non-finite values.")


def aggregate_structure(frame: pd.DataFrame) -> pd.DataFrame:
    _validate_structure_summary(frame)
    rows: list[dict[str, float | int | str]] = []
    for dataset in DATASET_ORDER:
        group = frame.loc[frame["dataset"] == dataset]
        row: dict[str, float | int | str] = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "n_environment_instances": int(group["environment_seed"].nunique()),
        }
        for metric in STRUCTURAL_METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"design_{metric}_seed_mean"] = float(values.mean())
            row[f"design_{metric}_seed_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_outcomes(
    outcome_payload: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dataset in DATASET_ORDER:
        if dataset not in outcome_payload:
            raise ValueError(f"Missing outcome dataset {dataset!r}.")
        methods = outcome_payload[dataset]
        try:
            mse = np.asarray(methods["BayesianMSE"], dtype=float)
            spo = np.asarray(methods["BayesianSPO"], dtype=float)
        except KeyError as error:
            raise ValueError(
                f"{dataset} is missing {error.args[0]!r} outcomes."
            ) from error
        if mse.ndim != 1 or spo.ndim != 1 or len(mse) != len(spo):
            raise ValueError(
                f"Malformed endpoint arrays for {dataset}: "
                f"MSE={mse.shape}, SPO={spo.shape}."
            )
        if len(mse) != 10:
            raise ValueError(
                f"Expected ten endpoints for {dataset}; observed {len(mse)}."
            )
        if not bool(np.isfinite(mse).all() and np.isfinite(spo).all()):
            raise ValueError(f"Non-finite endpoint for {dataset}.")
        mse_mean = float(mse.mean())
        spo_mean = float(spo.mean())
        if mse_mean <= 0.0 or spo_mean <= 0.0:
            raise ValueError(
                f"Regret means must be positive for the log ratio: {dataset}."
            )
        rows.append(
            {
                "dataset": dataset,
                "n_archived_endpoints_per_method": int(len(mse)),
                "bayesian_mse_realized_regret_mean": mse_mean,
                "bayesian_spo_realized_regret_mean": spo_mean,
                "bayesian_spo_regret_reduction_percent": float(
                    100.0 * (mse_mean - spo_mean) / mse_mean
                ),
                "log_mean_realized_regret_ratio_mse_over_spo": float(
                    np.log(mse_mean / spo_mean)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_scorecard(
    structure_frame: pd.DataFrame,
    outcome_payload: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    structure = aggregate_structure(structure_frame)
    outcomes = aggregate_outcomes(outcome_payload)
    scorecard = structure.merge(
        outcomes,
        on="dataset",
        how="left",
        validate="one_to_one",
    )
    ordering = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    scorecard["_order"] = scorecard["dataset"].map(ordering)
    scorecard = scorecard.sort_values("_order").drop(columns="_order")
    return scorecard.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structure-summary", type=Path, default=DEFAULT_STRUCTURE_SUMMARY
    )
    parser.add_argument("--outcome-json", type=Path, default=DEFAULT_OUTCOME_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structure_path = args.structure_summary.resolve()
    outcome_path = args.outcome_json.resolve()
    output_dir = args.output_dir.resolve()

    structure_frame = pd.read_csv(structure_path)
    outcome_payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    scorecard = build_scorecard(structure_frame, outcome_payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "dataset_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)
    print(scorecard.to_string(index=False))
    print(f"\nWrote {scorecard_path}")


if __name__ == "__main__":
    main()

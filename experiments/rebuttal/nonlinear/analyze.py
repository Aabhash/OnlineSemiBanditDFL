import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np


METHODS = ("BayesianSPO", "BayesianMSE")
EXTRACTORS = ("affine", "nonlinear")
METRICS = ("realized", "pseudo")
DATASET_LABELS = {
    "knapsack_32": "Knapsack 32",
    "knapsack_100": "Knapsack 100",
    "shortest_path_8x8": "SP 8x8",
    "shortest_path_20x20": "SP 20x20",
    "portfolio_32": "Portfolio",
}
ENDPOINT_COLUMNS = (
    "dataset",
    "seed",
    "method",
    "extractor",
    "metric",
    "endpoint",
)
DEFAULT_ENDPOINTS = Path(__file__).with_name("artifacts") / "endpoints.csv"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_rows(paths):
    rows = []
    seen = set()
    for path in map(Path, paths):
        require(path.is_file(), f"Missing endpoint file: {path}")
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            require(
                tuple(reader.fieldnames or ()) == ENDPOINT_COLUMNS,
                f"Unexpected columns in {path}",
            )
            for raw in reader:
                row = {
                    "dataset": raw["dataset"],
                    "seed": int(raw["seed"]),
                    "method": raw["method"],
                    "extractor": raw["extractor"],
                    "metric": raw["metric"],
                    "endpoint": float(raw["endpoint"]),
                }
                require(
                    row["dataset"] in DATASET_LABELS,
                    f"Unknown dataset in {path}",
                )
                require(row["method"] in METHODS, f"Unknown method in {path}")
                require(
                    row["extractor"] in EXTRACTORS,
                    f"Unknown extractor in {path}",
                )
                require(row["metric"] in METRICS, f"Unknown metric in {path}")
                require(
                    math.isfinite(row["endpoint"]),
                    f"Non-finite endpoint in {path}",
                )
                identity = tuple(row[column] for column in ENDPOINT_COLUMNS[:-1])
                require(identity not in seen, f"Duplicate endpoint {identity}")
                seen.add(identity)
                rows.append(row)
    require(rows, "No endpoints found")

    expected_cells = {
        (method, extractor, metric)
        for method in METHODS
        for extractor in EXTRACTORS
        for metric in METRICS
    }
    for dataset in {row["dataset"] for row in rows}:
        seeds = {row["seed"] for row in rows if row["dataset"] == dataset}
        for seed in seeds:
            cells = {
                (row["method"], row["extractor"], row["metric"])
                for row in rows
                if row["dataset"] == dataset and row["seed"] == seed
            }
            require(
                cells == expected_cells, f"Incomplete endpoints for {dataset}/{seed}"
            )
    return rows


def exact_sign_test(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def cell_rng(seed, dataset, method, metric):
    key = f"{seed}:{dataset}:{method}:{metric}".encode()
    cell_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(cell_seed)


def sample_std(values):
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def summarize(rows, bootstrap_samples=10_000, bootstrap_seed=20_260_727):
    require(bootstrap_samples > 0, "bootstrap_samples must be positive")
    summary = []
    for dataset, label in DATASET_LABELS.items():
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        seeds = sorted({row["seed"] for row in dataset_rows})
        for method in METHODS:
            for metric in METRICS:
                endpoints = {
                    (row["seed"], row["extractor"]): row["endpoint"]
                    for row in dataset_rows
                    if row["method"] == method and row["metric"] == metric
                }
                affine = np.asarray(
                    [endpoints[(seed, "affine")] for seed in seeds],
                    dtype=float,
                )
                nonlinear = np.asarray(
                    [endpoints[(seed, "nonlinear")] for seed in seeds],
                    dtype=float,
                )
                require(
                    abs(float(affine.mean())) > 1e-12,
                    f"Zero affine mean for {dataset}/{method}/{metric}",
                )

                indices = cell_rng(
                    bootstrap_seed,
                    dataset,
                    method,
                    metric,
                ).integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
                affine_boot = affine[indices].mean(axis=1)
                nonlinear_boot = nonlinear[indices].mean(axis=1)
                require(
                    bool(np.all(np.abs(affine_boot) > 1e-12)),
                    f"Zero bootstrap denominator for {dataset}/{method}/{metric}",
                )
                difference = affine - nonlinear
                difference_boot = difference[indices].mean(axis=1)
                reduction_boot = 100.0 * (affine_boot - nonlinear_boot) / affine_boot
                wins = int(np.sum(difference > 0))
                losses = int(np.sum(difference < 0))
                summary.append(
                    {
                        "dataset": dataset,
                        "dataset_label": label,
                        "method": method,
                        "metric": metric,
                        "n_seeds": len(seeds),
                        "affine_mean": float(affine.mean()),
                        "affine_std": sample_std(affine),
                        "nonlinear_mean": float(nonlinear.mean()),
                        "nonlinear_std": sample_std(nonlinear),
                        "paired_mean_difference_affine_minus_nonlinear": float(
                            difference.mean()
                        ),
                        "paired_mean_difference_ci95_low": float(
                            np.quantile(difference_boot, 0.025)
                        ),
                        "paired_mean_difference_ci95_high": float(
                            np.quantile(difference_boot, 0.975)
                        ),
                        "ratio_of_means_reduction_percent": float(
                            100.0 * (affine.mean() - nonlinear.mean()) / affine.mean()
                        ),
                        "ratio_of_means_reduction_ci95_low": float(
                            np.quantile(reduction_boot, 0.025)
                        ),
                        "ratio_of_means_reduction_ci95_high": float(
                            np.quantile(reduction_boot, 0.975)
                        ),
                        "nonlinear_seed_wins": wins,
                        "nonlinear_seed_losses": losses,
                        "ties": int(np.sum(difference == 0)),
                        "exact_sign_test_pvalue_two_sided": exact_sign_test(
                            wins,
                            losses,
                        ),
                    }
                )
    return summary


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "endpoints",
        nargs="*",
        type=Path,
        default=[DEFAULT_ENDPOINTS],
    )
    parser.add_argument("--output", type=Path, default=Path("summary.csv"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_727)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize(
        load_rows(args.endpoints),
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    write_csv(args.output, summary)
    print(args.output)


if __name__ == "__main__":
    main()

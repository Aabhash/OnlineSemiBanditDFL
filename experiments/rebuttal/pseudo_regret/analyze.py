import argparse
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS = (
    "knapsack_32",
    "knapsack_100",
    "shortest_path_8x8",
    "shortest_path_20x20",
    "portfolio_32",
)
DATASET_LABELS = {
    "knapsack_32": "Knapsack 32",
    "knapsack_100": "Knapsack 100",
    "shortest_path_8x8": "SP 8x8",
    "shortest_path_20x20": "SP 20x20",
    "portfolio_32": "Portfolio",
}
MINIMIZE = {
    "knapsack_32": False,
    "knapsack_100": False,
    "shortest_path_8x8": True,
    "shortest_path_20x20": True,
    "portfolio_32": False,
}
AGENTS = ("BayesianSPO", "BayesianMSE")
SEEDS = tuple(range(100, 110))
N_ROUNDS = 10_000
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_726


def paired_reduction_percent(bayesian_spo, bayesian_mse):
    spo = np.asarray(bayesian_spo, dtype=float)
    mse = np.asarray(bayesian_mse, dtype=float)
    if spo.ndim != 1 or spo.shape != mse.shape:
        raise ValueError(f"Expected paired vectors, got {spo.shape} and {mse.shape}")
    denominator = float(mse.mean())
    if denominator <= 0:
        raise ValueError("BayesianMSE mean must be positive")
    return 100.0 * (denominator - float(spo.mean())) / denominator


def paired_bootstrap(bayesian_spo, bayesian_mse, *, draws, seed):
    spo = np.asarray(bayesian_spo, dtype=float)
    mse = np.asarray(bayesian_mse, dtype=float)
    if spo.ndim != 1 or spo.shape != mse.shape:
        raise ValueError(f"Expected paired vectors, got {spo.shape} and {mse.shape}")
    if draws <= 0:
        raise ValueError("draws must be positive")

    indices = np.random.default_rng(seed).integers(
        0,
        len(spo),
        size=(draws, len(spo)),
    )
    boot_spo = spo[indices].mean(axis=1)
    boot_mse = mse[indices].mean(axis=1)
    if np.any(boot_mse <= 0):
        raise ValueError("A bootstrap resample has a non-positive MSE mean")
    relative = 100.0 * (boot_mse - boot_spo) / boot_mse
    absolute = boot_mse - boot_spo
    return {
        "relative_low": float(np.quantile(relative, 0.025)),
        "relative_high": float(np.quantile(relative, 0.975)),
        "absolute_low": float(np.quantile(absolute, 0.025)),
        "absolute_high": float(np.quantile(absolute, 0.975)),
    }


def exact_paired_randomization_pvalue(bayesian_spo, bayesian_mse):
    differences = np.asarray(bayesian_mse, dtype=float) - np.asarray(
        bayesian_spo, dtype=float
    )
    observed = abs(float(differences.mean()))
    permuted = (
        abs(float(np.mean(differences * signs)))
        for signs in product((-1.0, 1.0), repeat=len(differences))
    )
    return float(np.mean([value >= observed - 1e-12 for value in permuted]))


def expected_pseudo_regret(dataset, agent_values, optimal_values):
    if MINIMIZE[dataset]:
        return agent_values - optimal_values
    return optimal_values - agent_values


def load_endpoints(results_dir):
    rows = []
    for dataset in DATASETS:
        path = Path(results_dir) / dataset / "metrics" / "trajectories.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            trajectories = pickle.load(handle)

        expected_keys = {
            (agent, dataset, "semi_bandit", seed) for agent in AGENTS for seed in SEEDS
        }
        if set(trajectories) != expected_keys:
            raise ValueError(f"Unexpected trajectory inventory in {path}")

        for agent in AGENTS:
            for seed in SEEDS:
                trajectory = trajectories[(agent, dataset, "semi_bandit", seed)]
                arrays = {
                    name: np.asarray(trajectory[name], dtype=float)
                    for name in (
                        "regrets",
                        "cumulative_regret",
                        "pseudo_regrets",
                        "cumulative_pseudo_regret",
                        "mean_agent_values",
                        "mean_optimal_values",
                    )
                }
                for name, values in arrays.items():
                    if values.shape != (N_ROUNDS,):
                        raise ValueError(
                            f"{dataset}/{agent}/{seed}/{name}: "
                            f"expected {(N_ROUNDS,)}, got {values.shape}"
                        )
                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"{dataset}/{agent}/{seed}/{name} is non-finite"
                        )

                reconstructed = expected_pseudo_regret(
                    dataset,
                    arrays["mean_agent_values"],
                    arrays["mean_optimal_values"],
                )
                if not np.allclose(
                    reconstructed,
                    arrays["pseudo_regrets"],
                    rtol=0,
                    atol=1e-10,
                ):
                    raise ValueError(
                        f"Pseudo-regret identity failed for {dataset}/{agent}/{seed}"
                    )
                for per_round, cumulative, name in (
                    (
                        arrays["regrets"],
                        arrays["cumulative_regret"],
                        "realized regret",
                    ),
                    (
                        arrays["pseudo_regrets"],
                        arrays["cumulative_pseudo_regret"],
                        "pseudo-regret",
                    ),
                ):
                    if not np.allclose(
                        np.cumsum(per_round),
                        cumulative,
                        rtol=0,
                        atol=1e-7,
                    ):
                        raise ValueError(
                            f"{name} sum failed for {dataset}/{agent}/{seed}"
                        )

                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "seed": seed,
                        "agent": agent,
                        "final_realized_regret": float(arrays["cumulative_regret"][-1]),
                        "final_pseudo_regret": float(
                            arrays["cumulative_pseudo_regret"][-1]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_endpoints(endpoints):
    rows = []
    for dataset_index, dataset in enumerate(DATASETS):
        subset = endpoints[endpoints["dataset"] == dataset]
        by_agent = {
            agent: subset[subset["agent"] == agent]
            .sort_values("seed")
            .reset_index(drop=True)
            for agent in AGENTS
        }
        if not np.array_equal(
            by_agent["BayesianSPO"]["seed"],
            by_agent["BayesianMSE"]["seed"],
        ):
            raise ValueError(f"Seed pairing failed for {dataset}")

        row = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "n_paired_seeds": len(SEEDS),
        }
        for metric_index, (metric, short) in enumerate(
            (
                ("final_realized_regret", "realized"),
                ("final_pseudo_regret", "pseudo"),
            )
        ):
            spo = by_agent["BayesianSPO"][metric].to_numpy(dtype=float)
            mse = by_agent["BayesianMSE"][metric].to_numpy(dtype=float)
            bootstrap = paired_bootstrap(
                spo,
                mse,
                draws=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED + 10 * dataset_index + metric_index,
            )
            row.update(
                {
                    f"bayesian_spo_{short}_mean": float(spo.mean()),
                    f"bayesian_spo_{short}_sd": float(spo.std(ddof=1)),
                    f"bayesian_mse_{short}_mean": float(mse.mean()),
                    f"bayesian_mse_{short}_sd": float(mse.std(ddof=1)),
                    f"bayesian_spo_{short}_reduction_percent": (
                        paired_reduction_percent(spo, mse)
                    ),
                    f"bayesian_spo_{short}_reduction_ci_low": bootstrap["relative_low"],
                    f"bayesian_spo_{short}_reduction_ci_high": bootstrap[
                        "relative_high"
                    ],
                    f"bayesian_spo_{short}_absolute_reduction": float(
                        (mse - spo).mean()
                    ),
                    f"bayesian_spo_{short}_absolute_reduction_ci_low": (
                        bootstrap["absolute_low"]
                    ),
                    f"bayesian_spo_{short}_absolute_reduction_ci_high": (
                        bootstrap["absolute_high"]
                    ),
                    f"{short}_paired_randomization_pvalue": (
                        exact_paired_randomization_pvalue(spo, mse)
                    ),
                    f"{short}_ci_excludes_zero": bool(
                        bootstrap["relative_low"] > 0 or bootstrap["relative_high"] < 0
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results/rebuttal/direct_pseudo_regret_10k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paired_endpoint_summary.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize_endpoints(load_endpoints(args.results_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

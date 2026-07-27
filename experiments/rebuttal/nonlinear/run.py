import argparse
import csv
import json
from pathlib import Path

import yaml

from experiments.ablations.common import make_agent_solver
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.agents.contextual.bayesian_spo import BayesianSPOAgent
from src.agents.contextual.neural_agents import BayesianMSEAgent
from src.envs import create_env


METHODS = {
    "BayesianSPO": BayesianSPOAgent,
    "BayesianMSE": BayesianMSEAgent,
}
DATASETS = (
    "knapsack_32",
    "knapsack_100",
    "shortest_path_8x8",
    "shortest_path_20x20",
    "portfolio_32",
)
DEFAULT_FROZEN_CONFIG = Path(__file__).with_name("frozen_config.json")
SETTING_FIELDS = {
    "evaluation_seeds",
    "learning_rate_multiplier",
    "beta_multiplier",
    "nonlinear_depth",
    "residual_gate_scale",
    "residual_activation",
    "residual_weight_init",
}


def load_dataset_config(path, dataset):
    with Path(path).open() as handle:
        config = dict(yaml.safe_load(handle)["datasets"][dataset])
    for key in ("env_type", "skip_tuning", "tuning_year"):
        config.pop(key, None)
    return config


def load_tuned_params(tuning_dir, dataset):
    path = Path(tuning_dir) / f"best_hyperparams_{dataset}_semi_bandit.json"
    with path.open() as handle:
        stored = json.load(handle)
    missing = METHODS.keys() - stored.keys()
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return {name: dict(stored[name]["params"]) for name in METHODS}


def load_frozen_config(path):
    with Path(path).open() as handle:
        config = json.load(handle)
    if set(config["datasets"]) != set(DATASETS):
        raise ValueError("Frozen config must contain all five datasets")
    if int(config["rounds"]) <= 0:
        raise ValueError("Frozen round count must be positive")
    for dataset, settings in config["datasets"].items():
        if set(settings) != SETTING_FIELDS:
            raise ValueError(f"Unexpected frozen settings for {dataset}")
        if settings["learning_rate_multiplier"] <= 0:
            raise ValueError(f"Invalid learning-rate multiplier for {dataset}")
        if settings["beta_multiplier"] < 0:
            raise ValueError(f"Invalid beta multiplier for {dataset}")
        if settings["nonlinear_depth"] <= 0:
            raise ValueError(f"Invalid nonlinear depth for {dataset}")
        if settings["residual_gate_scale"] <= 0:
            raise ValueError(f"Invalid residual gate scale for {dataset}")
        seeds = settings["evaluation_seeds"]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError(f"Invalid evaluation seeds for {dataset}")
    return config


def apply_multipliers(params, settings):
    effective = {}
    for method, method_params in params.items():
        method_params = dict(method_params)
        lr_key = "init_lr" if "init_lr" in method_params else "lr"
        method_params[lr_key] = (
            method_params.get(lr_key, 0.001) * settings["learning_rate_multiplier"]
        )
        method_params["beta"] = (
            method_params.get("beta", 1.0) * settings["beta_multiplier"]
        )
        effective[method] = method_params
    return effective


def make_agent(method, env, seed, rounds, params, nonlinear, settings):
    lr = params.get("init_lr", params.get("lr", 0.001))
    common = {
        "context_dim": env.context_dim,
        "n_arms": env.n_arms,
        "weights": env.weights,
        "capacity": env.capacity,
        "hidden_dim": params.get("hidden_dim", 16),
        "beta": params.get("beta", 1.0),
        "lambda_reg": params.get("lambda_reg", 1.0),
        "lr": lr,
        "decay_rounds": rounds,
        "max_grad_norm": params.get("max_grad_norm", 5.0),
        "nonlinear": nonlinear,
        "nonlinear_depth": settings["nonlinear_depth"],
        "residual_gate_scale": settings["residual_gate_scale"],
        "residual_activation": settings["residual_activation"],
        "residual_weight_init": settings["residual_weight_init"],
        "minimize": bool(getattr(env, "minimize", False)),
        "solver": make_agent_solver(env),
        "seed": seed,
        "exploration_strategy": params.get("exploration_strategy", "ucb"),
        "ts_scale": params.get("ts_scale", 1.0),
    }
    if method == "BayesianSPO":
        return BayesianSPOAgent(
            **common,
            mask_grad=True,
            imputation_samples=params.get("imputation_samples", 1),
            imputation_scale=params.get("imputation_scale", 1.0),
            use_posterior_predictive=params.get(
                "use_posterior_predictive",
                False,
            ),
            noise_var=params.get("noise_var", 1.0),
            missing_grad_weight=params.get("missing_grad_weight", 0.0),
        )
    return BayesianMSEAgent(
        **common,
        mask_grad=False,
        missing_grad_weight=params.get("missing_grad_weight", 0.0),
    )


def write_endpoints(path, summary):
    rows = []
    for dataset, dataset_payload in summary["datasets"].items():
        for run in dataset_payload["runs"].values():
            for metric, key in (
                ("realized", "final_regret"),
                ("pseudo", "final_pseudo_regret"),
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": summary["seed"],
                        "method": run["agent"],
                        "extractor": run["extractor"],
                        "metric": metric,
                        "endpoint": run[key],
                    }
                )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    frozen = load_frozen_config(args.frozen_config)
    rounds = frozen["rounds"] if args.rounds is None else args.rounds
    if rounds <= 0:
        raise ValueError("--rounds must be positive")

    summary = {
        "seed": args.seed,
        "rounds": rounds,
        "feedback": "semi_bandit",
        "extractors": ["affine", "nonlinear"],
        "agents": list(METHODS),
        "frozen_config": str(args.frozen_config),
        "datasets": {},
    }

    for dataset in args.datasets:
        settings = frozen["datasets"][dataset]
        params = apply_multipliers(
            load_tuned_params(frozen["tuning_dir"], dataset),
            settings,
        )

        env = create_env(
            dataset,
            seed=args.seed,
            **load_dataset_config(frozen["environment_config"], dataset),
        )
        episode = pregenerate_episode(env, rounds, full_feedback=False)
        runs = {}

        for extractor in ("affine", "nonlinear"):
            for method in METHODS:
                agent = make_agent(
                    method,
                    env,
                    args.seed,
                    rounds,
                    params[method],
                    nonlinear=extractor == "nonlinear",
                    settings=settings,
                )
                result = run_agent_on_episode(
                    agent,
                    episode,
                    rounds,
                    lr=params[method].get(
                        "init_lr",
                        params[method].get("lr", 0.001),
                    ),
                    full_feedback=False,
                    diagnostics=False,
                )
                runs[f"{method}_{extractor}"] = {
                    "agent": method,
                    "extractor": extractor,
                    "completed_rounds": len(result["regrets"]),
                    "final_regret": float(result["final_regret"]),
                    "final_pseudo_regret": float(result["final_pseudo_regret"]),
                    "hyperparameters": params[method],
                }
                print(
                    f"{dataset} {method} {extractor}: " f'{result["final_regret"]:.3f}',
                    flush=True,
                )

        summary["datasets"][dataset] = {
            "configuration": settings,
            "runs": runs,
        }

    output = Path(args.output_dir) / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_endpoints(output / "endpoints.csv", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=DATASETS,
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--rounds", type=int)
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=DEFAULT_FROZEN_CONFIG,
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

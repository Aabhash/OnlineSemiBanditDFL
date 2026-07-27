import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.agents.contextual.baselines import RandomAgent
from src.agents.contextual.bayesian_spo import BayesianSPOAgent
from src.agents.contextual.greedy_agents import EpsilonGreedyAgent
from src.agents.contextual.neural_agents import BayesianMSEAgent
from src.agents.contextual.ts_agents import ThompsonSamplingAgent
from src.agents.contextual.ucb_agents import LinUCBAgent
from src.agents.contextual.weber_bll import WeberBLLAgent
from src.envs.energy_scheduling import EnergySchedulingEnv


PAPER_METHODS = (
    "Random",
    "EpsilonGreedy",
    "LinUCB",
    "TS",
    "BLL-Online",
    "BayesianMSE",
    "BayesianSPO",
)


def write_progress(rows, per_seed_path):
    ordered_rows = sorted(rows, key=lambda row: (row["seed"], row["agent"]))
    per_seed_path.write_text(json.dumps(ordered_rows, indent=2))


def constrain_params_for_feedback(params, full_feedback):
    constrained = dict(params)
    if full_feedback:
        constrained["beta"] = 0.0
        constrained["imputation_samples"] = 1
    return constrained


def params_for_agent(name, params, full_feedback):
    keys = {
        "Random": ("lr",),
        "EpsilonGreedy": (
            "hidden_dim",
            "epsilon",
            "lr",
            "max_grad_norm",
        ),
        "LinUCB": ("alpha", "lambda_reg", "lr"),
        "TS": ("prior_var", "noise_var", "lr"),
        "BLL-Online": (
            "hidden_dim",
            "beta",
            "lambda_reg",
            "lr",
            "max_grad_norm",
            "noise_var",
        ),
        "BayesianMSE": (
            "hidden_dim",
            "beta",
            "lambda_reg",
            "lr",
            "max_grad_norm",
        ),
        "BayesianSPO": (
            "hidden_dim",
            "beta",
            "lambda_reg",
            "lr",
            "max_grad_norm",
            "imputation_samples",
        ),
    }
    constrained = constrain_params_for_feedback(params, full_feedback)
    return {key: constrained[key] for key in keys[name]}


def make_agent(name, env, seed, rounds, params):
    if name in ("BayesianSPO", "BayesianMSE", "BLL-Online"):
        common = dict(
            context_dim=env.context_dim,
            n_arms=env.n_arms,
            weights=env.weights,
            capacity=env.capacity,
            hidden_dim=params["hidden_dim"],
            beta=params["beta"],
            lambda_reg=params["lambda_reg"],
            lr=params["lr"],
            decay_rounds=rounds,
            max_grad_norm=params["max_grad_norm"],
            mask_grad=True,
            nonlinear=False,
            minimize=True,
            solver=env.solve,
            seed=seed,
        )
        if name == "BayesianSPO":
            return BayesianSPOAgent(
                **common,
                imputation_samples=params["imputation_samples"],
            )
        if name == "BayesianMSE":
            return BayesianMSEAgent(**common)
        bll_common = {**common, "mask_grad": False}
        return WeberBLLAgent(
            **bll_common,
            posterior_complete_missing=True,
            noise_var=params["noise_var"],
        )
    if name == "LinUCB":
        return LinUCBAgent(
            context_dim=env.context_dim,
            n_arms=env.n_arms,
            weights=env.weights,
            capacity=env.capacity,
            alpha=params["alpha"],
            lambda_reg=params["lambda_reg"],
            minimize=True,
            solver=env.solve,
        )
    if name == "TS":
        return ThompsonSamplingAgent(
            context_dim=env.context_dim,
            n_arms=env.n_arms,
            weights=env.weights,
            capacity=env.capacity,
            prior_var=params["prior_var"],
            noise_var=params["noise_var"],
            minimize=True,
            solver=env.solve,
            seed=seed,
        )
    if name == "EpsilonGreedy":
        return EpsilonGreedyAgent(
            context_dim=env.context_dim,
            n_arms=env.n_arms,
            weights=env.weights,
            capacity=env.capacity,
            hidden_dim=params["hidden_dim"],
            epsilon=params["epsilon"],
            decay_epsilon=True,
            decay_rounds=rounds,
            lr=params["lr"],
            max_grad_norm=params["max_grad_norm"],
            solver=env.solve,
            seed=seed,
            random_action_fn=env.solve,
        )
    if name == "Random":
        return RandomAgent(
            n_arms=env.n_arms,
            weights=env.weights,
            capacity=env.capacity,
            seed=seed,
            random_action_fn=env.solve,
        )
    raise ValueError(f"unknown agent: {name}")


def run_energy_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds = args.rounds
    seeds = args.seeds
    full_feedback = args.feedback == "full"
    params = {
        "hidden_dim": args.hidden_dim,
        "beta": args.beta,
        "lambda_reg": args.lambda_reg,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "imputation_samples": args.imputation_samples,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "prior_var": args.prior_var,
        "noise_var": args.noise_var,
    }
    params_by_agent = {
        name: params_for_agent(name, params, full_feedback) for name in args.agents
    }
    if args.params_json:
        tuned = json.loads(Path(args.params_json).read_text())
        for name in args.agents:
            if name in tuned:
                params_by_agent[name] = params_for_agent(
                    name,
                    {**params, **tuned[name]},
                    full_feedback,
                )

    per_seed_path = output_dir / "per_seed.json"
    rows = (
        json.loads(per_seed_path.read_text())
        if args.resume and per_seed_path.exists()
        else []
    )
    for seed in seeds:
        pending_agents = []
        for agent_name in args.agents:
            agent_params = params_by_agent[agent_name]
            matching = [
                row
                for row in rows
                if row["agent"] == agent_name
                and row["seed"] == seed
                and row["rounds"] == rounds
                and row.get("feedback") == args.feedback
                and row.get("hyperparameters") == agent_params
            ]
            if matching:
                print(f"{agent_name} seed={seed}: already complete", flush=True)
                continue
            rows = [
                row
                for row in rows
                if not (row["agent"] == agent_name and row["seed"] == seed)
            ]
            pending_agents.append(agent_name)
        if not pending_agents:
            continue

        episode_env = EnergySchedulingEnv(
            data_root=args.data_root,
            energy_instance=args.energy_instance,
            dataset_split=args.dataset_split,
            validation_fraction=args.validation_fraction,
            seed=seed,
        )
        if rounds > episode_env.n_days:
            raise ValueError(
                f"requested {rounds} rounds from a " f"{episode_env.n_days}-day split"
            )

        objective_errors = [
            episode_env.objective_error(i) for i in range(min(5, episode_env.n_days))
        ]
        if max(objective_errors) > 1e-6:
            raise RuntimeError(
                f"task-start objective mismatch: {max(objective_errors)}"
            )
        episode = pregenerate_episode(episode_env, rounds)

        for agent_name in pending_agents:
            agent_params = params_by_agent[agent_name]
            agent_env = EnergySchedulingEnv(
                data_root=args.data_root,
                energy_instance=args.energy_instance,
                dataset_split=args.dataset_split,
                validation_fraction=args.validation_fraction,
                cost_scale=episode_env.cost_scale,
                seed=seed,
            )
            agent = make_agent(agent_name, agent_env, seed, rounds, agent_params)
            started = time.perf_counter()
            result = run_agent_on_episode(
                agent,
                episode,
                rounds,
                lr=agent_params["lr"],
                full_feedback=full_feedback,
                diagnostics=False,
            )
            wall_time = time.perf_counter() - started
            selected_counts = np.asarray(result["actions"]).sum(axis=1)
            if not np.allclose(selected_counts, episode_env.instance.n_tasks):
                raise RuntimeError(
                    f"{agent_name} returned invalid selected counts: {selected_counts}"
                )

            observed_components = (
                rounds * episode_env.n_arms
                if full_feedback
                else int(result["obs_counts"].sum())
            )
            rows.append(
                {
                    "agent": agent_name,
                    "seed": seed,
                    "rounds": rounds,
                    "feedback": args.feedback,
                    "final_cumulative_regret_raw": float(
                        result["final_regret"] * episode_env.cost_scale
                    ),
                    "observed_per_round": float(observed_components / rounds),
                    "wall_time_seconds": wall_time,
                    "hyperparameters": agent_params,
                }
            )
            write_progress(rows, per_seed_path)
            print(
                f"{agent_name} seed={seed}: "
                f'regret={result["final_regret"]:.4f} '
                f'raw={result["final_regret"] * episode_env.cost_scale:.1f} '
                f"time={wall_time:.1f}s",
                flush=True,
            )

    rows.sort(key=lambda row: (row["seed"], row["agent"]))
    write_progress(rows, per_seed_path)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/Energy/SchedulingInstances",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rebuttal/energy_scheduling_task_start",
    )
    parser.add_argument("--energy-instance", default="load3")
    parser.add_argument("--dataset-split", default="experiment")
    parser.add_argument(
        "--feedback",
        choices=("semi_bandit", "full"),
        default="semi_bandit",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=395)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["BayesianSPO", "BayesianMSE"],
        choices=PAPER_METHODS,
    )
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--imputation-samples", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--prior-var", type=float, default=1.0)
    parser.add_argument("--noise-var", type=float, default=0.1)
    parser.add_argument("--params-json")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_energy_experiment(parse_args())

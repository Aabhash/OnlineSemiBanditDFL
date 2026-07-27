import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.rebuttal.energy.run import (
    constrain_params_for_feedback,
    make_agent,
    params_for_agent,
)
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.envs.energy_scheduling import EnergySchedulingEnv


BASE = {
    "hidden_dim": 8,
    "beta": 1.0,
    "lambda_reg": 1.0,
    "lr": 1e-3,
    "max_grad_norm": 5.0,
    "imputation_samples": 1,
    "alpha": 1.0,
    "epsilon": 0.1,
    "prior_var": 1.0,
    "noise_var": 0.1,
}


def candidate(name, **changes):
    params = dict(BASE)
    params.update(changes)
    return {"name": name, "params": params}


def architecture_candidates():
    return [
        candidate(
            f"hidden_{hidden_dim}_lambda_{lambda_reg:g}_lr_{lr:g}",
            hidden_dim=hidden_dim,
            beta=0.0,
            lambda_reg=lambda_reg,
            lr=lr,
            imputation_samples=1,
        )
        for hidden_dim in (8, 16, 32)
        for lambda_reg in (0.1, 1.0, 4.0)
        for lr in (3e-4, 1e-3, 3e-3)
    ]


def baseline_candidates(agent_name):
    if agent_name == "Random":
        return [candidate("random")]
    if agent_name == "EpsilonGreedy":
        return [
            candidate(
                f"hidden_{hidden_dim}_epsilon_{epsilon:g}_lr_{lr:g}",
                hidden_dim=hidden_dim,
                epsilon=epsilon,
                lr=lr,
            )
            for hidden_dim in (8, 16, 32)
            for epsilon in (0.01, 0.05, 0.1, 0.2)
            for lr in (3e-4, 1e-3, 3e-3)
        ]
    if agent_name == "LinUCB":
        return [
            candidate(
                f"alpha_{alpha:g}_lambda_{lambda_reg:g}",
                alpha=alpha,
                lambda_reg=lambda_reg,
            )
            for alpha in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
            for lambda_reg in (0.1, 1.0, 4.0)
        ]
    if agent_name == "TS":
        return [
            candidate(
                f"prior_{prior_var:g}_noise_{noise_var:g}",
                prior_var=prior_var,
                noise_var=noise_var,
            )
            for prior_var in (0.1, 0.5, 1.0, 4.0)
            for noise_var in (0.01, 0.1, 1.0, 4.0)
        ]
    raise ValueError(f"no baseline grid for agent: {agent_name}")


def bll_architecture_candidates():
    return [
        candidate(
            f"hidden_{hidden_dim}_lambda_{lambda_reg:g}_lr_{lr:g}",
            hidden_dim=hidden_dim,
            beta=0.0,
            lambda_reg=lambda_reg,
            lr=lr,
            noise_var=1.0,
        )
        for hidden_dim in (8, 16, 32)
        for lambda_reg in (0.1, 1.0, 4.0)
        for lr in (3e-4, 1e-3, 3e-3)
    ]


def bll_feedback_candidates(base_params=None):
    base = base_params or {
        "hidden_dim": 8,
        "lambda_reg": 1.0,
        "lr": 1e-3,
        "max_grad_norm": 5.0,
    }
    shared = {
        key: base[key] for key in ("hidden_dim", "lambda_reg", "lr", "max_grad_norm")
    }
    return [
        candidate(
            f"beta_{beta:g}_noise_{noise_var:g}",
            **shared,
            beta=beta,
            noise_var=noise_var,
        )
        for beta in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
        for noise_var in (0.1, 1.0, 4.0)
    ]


def task_start_semi_candidates(agent_name, base_params=None):
    betas = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    base = base_params or {
        "hidden_dim": 16,
        "lambda_reg": 4.0,
        "lr": 1e-3,
        "max_grad_norm": 5.0,
    }
    shared = {
        key: base[key] for key in ("hidden_dim", "lambda_reg", "lr", "max_grad_norm")
    }
    if agent_name == "BayesianSPO":
        return [
            candidate(
                f"beta_{beta:g}_samples_{samples}",
                **shared,
                beta=beta,
                imputation_samples=samples,
            )
            for beta in betas
            for samples in (1, 2, 4, 8)
        ]
    if agent_name == "BayesianMSE":
        return [
            candidate(
                f"beta_{beta:g}",
                **shared,
                beta=beta,
                imputation_samples=1,
            )
            for beta in betas
        ]
    raise ValueError(f"unknown agent: {agent_name}")


def candidates_for(agent_name, candidate_grid, base_params=None):
    if candidate_grid in {"architecture", "full-feedback"}:
        return architecture_candidates()
    if candidate_grid == "task-start-semi":
        return task_start_semi_candidates(agent_name, base_params=base_params)
    if candidate_grid == "baselines":
        return baseline_candidates(agent_name)
    if candidate_grid == "bll-architecture":
        if agent_name != "BLL-Online":
            raise ValueError("bll-architecture only supports BLL-Online")
        return bll_architecture_candidates()
    if candidate_grid == "bll-feedback":
        if agent_name != "BLL-Online":
            raise ValueError("bll-feedback only supports BLL-Online")
        return bll_feedback_candidates(base_params=base_params)
    raise ValueError(f"unknown candidate grid: {candidate_grid}")


def evaluate_candidates(
    args,
    agent_name,
    candidates,
    rounds,
    seeds,
    existing_rows=None,
):
    rows = list(existing_rows or [])
    full_feedback = args.feedback == "full"
    completed = {
        (row["candidate"], row["seed"], row["rounds"], row.get("feedback"))
        for row in rows
    }
    for seed in seeds:
        pending = [
            spec
            for spec in candidates
            if (spec["name"], seed, rounds, args.feedback) not in completed
        ]
        if not pending:
            continue
        episode_env = EnergySchedulingEnv(
            data_root=args.data_root,
            energy_instance=args.energy_instance,
            dataset_split="validation",
            validation_fraction=args.validation_fraction,
            seed=seed,
        )
        if rounds > episode_env.n_days:
            raise ValueError(
                f"validation split has {episode_env.n_days} days, fewer than {rounds} rounds"
            )
        episode = pregenerate_episode(episode_env, rounds)
        for spec in pending:
            params = constrain_params_for_feedback(
                spec["params"],
                full_feedback,
            )
            agent_env = EnergySchedulingEnv(
                data_root=args.data_root,
                energy_instance=args.energy_instance,
                dataset_split="validation",
                validation_fraction=args.validation_fraction,
                cost_scale=episode_env.cost_scale,
                seed=seed,
            )
            agent = make_agent(agent_name, agent_env, seed, rounds, params)
            started = time.perf_counter()
            result = run_agent_on_episode(
                agent,
                episode,
                rounds,
                lr=params["lr"],
                full_feedback=full_feedback,
                diagnostics=False,
            )
            wall_time = time.perf_counter() - started
            raw_regret = float(result["final_regret"] * episode_env.cost_scale)
            rows.append(
                {
                    "agent": agent_name,
                    "candidate": spec["name"],
                    "params": params,
                    "seed": seed,
                    "rounds": rounds,
                    "feedback": args.feedback,
                    "cumulative_regret_raw": raw_regret,
                    "wall_time_seconds": wall_time,
                }
            )
            print(
                f'{agent_name} {spec["name"]} seed={seed}: '
                f"raw_regret={raw_regret:.1f} time={wall_time:.1f}s",
                flush=True,
            )
    return rows


def rank_candidates(rows):
    names = sorted({row["candidate"] for row in rows})
    ranking = []
    for name in names:
        candidate_rows = [row for row in rows if row["candidate"] == name]
        regrets = np.asarray(
            [row["cumulative_regret_raw"] for row in candidate_rows],
            dtype=float,
        )
        ranking.append(
            {
                "candidate": name,
                "mean_cumulative_regret_raw": float(regrets.mean()),
                "std_cumulative_regret_raw": (
                    float(regrets.std(ddof=1)) if len(regrets) > 1 else 0.0
                ),
                "params": candidate_rows[0]["params"],
                "n_seeds": len(candidate_rows),
            }
        )
    return sorted(ranking, key=lambda row: row["mean_cumulative_regret_raw"])


def run_tuning(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results, best_params = {}, {}
    full_feedback = args.feedback == "full"

    candidate_grid = args.candidate_grid
    if candidate_grid == "auto":
        candidate_grid = (
            "full-feedback" if args.feedback == "full" else "task-start-semi"
        )

    base_params_by_agent = {}
    if args.base_params_json:
        base_params_by_agent = json.loads(Path(args.base_params_json).read_text())
        missing = sorted(set(args.agents) - set(base_params_by_agent))
        if missing:
            raise ValueError(f"base parameter file is missing agents: {missing}")

    for agent_name in args.agents:
        candidates = candidates_for(
            agent_name,
            candidate_grid,
            base_params=base_params_by_agent.get(agent_name),
        )
        result_path = output_dir / f"{agent_name}.json"
        previous = (
            json.loads(result_path.read_text())
            if args.resume and result_path.exists()
            else {}
        )
        screen_rows = evaluate_candidates(
            args,
            agent_name,
            candidates,
            args.screen_rounds,
            args.screen_seeds,
            existing_rows=previous.get("screen_rows"),
        )
        screen_ranking = rank_candidates(screen_rows)
        finalist_names = {
            row["candidate"] for row in screen_ranking[: args.n_finalists]
        }
        finalists = [spec for spec in candidates if spec["name"] in finalist_names]
        final_rows = evaluate_candidates(
            args,
            agent_name,
            finalists,
            args.validation_rounds,
            args.validation_seeds,
            existing_rows=previous.get("final_rows"),
        )
        final_ranking = rank_candidates(final_rows)
        best_params[agent_name] = params_for_agent(
            agent_name,
            final_ranking[0]["params"],
            full_feedback=full_feedback,
        )
        all_results[agent_name] = {
            "feedback": args.feedback,
            "candidate_grid": candidate_grid,
            "screen_rows": screen_rows,
            "screen_ranking": screen_ranking,
            "final_rows": final_rows,
            "final_ranking": final_ranking,
        }
        result_path.write_text(json.dumps(all_results[agent_name], indent=2))

    (output_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    return best_params, all_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/Energy/SchedulingInstances",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rebuttal/energy_scheduling_task_start/tuning",
    )
    parser.add_argument("--energy-instance", default="load3")
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument(
        "--feedback",
        choices=("semi_bandit", "full"),
        default="semi_bandit",
    )
    parser.add_argument(
        "--candidate-grid",
        choices=(
            "auto",
            "architecture",
            "full-feedback",
            "task-start-semi",
            "baselines",
            "bll-architecture",
            "bll-feedback",
        ),
        default="auto",
    )
    parser.add_argument("--base-params-json")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["BayesianSPO", "BayesianMSE"],
    )
    parser.add_argument("--screen-rounds", type=int, default=75)
    parser.add_argument("--screen-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n-finalists", type=int, default=3)
    parser.add_argument("--validation-rounds", type=int, default=394)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_tuning(parse_args())

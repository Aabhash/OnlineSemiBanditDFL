from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.agents.bayesian_utils import (
    compute_bayesian_predictions,
    compute_ucb_values,
    compute_uncertainties,
    update_bayesian_stats,
)
from src.agents.contextual.bayesian_spo import BayesianSPOAgent
from src.agents.contextual.neural_agents import BayesianMSEAgent
from src.losses import make_spo_grad_fn
from src.solvers.shortest_path import ShortestPathSolver


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "rebuttal" / "sequential_imputation_degeneracy"
)

PATH_A = np.array([1.0, 0.0, 0.0, 1.0])
PATH_B = np.array([0.0, 1.0, 1.0, 0.0])
TRUE_COSTS = -0.5 * PATH_A - PATH_B
INITIAL_THETA = 0.5 * PATH_A + PATH_B
CONTEXT = np.array([0.0])
SOLVER = ShortestPathSolver(2)
SPO_GRAD = make_spo_grad_fn(SOLVER.solve, minimize=True)
STRICT_MARGIN_TOL = 1e-8

PRIMARY_Z0 = 0.43
DEFAULT_Z0_VALUES = (0.33, PRIMARY_Z0, 0.53)
ROUNDS = 50
GAMMA = 10.0
BETA = 0.2
LEARNING_RATE = 0.05
SEED = 0
RHO_VALUES = (0.0, 0.5, 1.0)

REAL_METHODS = (
    "Full-feedback SPO+",
    "Masked BayesianSPO",
    "Unmasked BayesianSPO",
    "BayesianMSE",
)
REAL_METHOD_CONFIG = {
    "Full-feedback SPO+": {
        "agent": "spo",
        "full_feedback": True,
        "missing_grad_weight": 0.0,
    },
    "Masked BayesianSPO": {
        "agent": "spo",
        "full_feedback": False,
        "missing_grad_weight": 0.0,
    },
    "Unmasked BayesianSPO": {
        "agent": "spo",
        "full_feedback": False,
        "missing_grad_weight": 1.0,
    },
    "BayesianMSE": {
        "agent": "mse",
        "full_feedback": False,
        "missing_grad_weight": 0.0,
    },
}


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _path_totals(values: np.ndarray) -> dict[str, float]:
    values = _as_float_array(values)
    return {
        "A": float(values @ PATH_A),
        "B": float(values @ PATH_B),
    }


def identify_path(action: np.ndarray) -> str:
    action = _as_float_array(action)
    if np.allclose(action, PATH_A):
        return "A"
    if np.allclose(action, PATH_B):
        return "B"
    raise AssertionError(f"Unexpected 2x2 path vector: {action.tolist()}")


def solve_with_gap(values: np.ndarray) -> dict[str, Any]:
    values = _as_float_array(values)
    action = _as_float_array(SOLVER.solve(values))
    path = identify_path(action)
    totals = _path_totals(values)
    other = "B" if path == "A" else "A"
    return {
        "action": action,
        "path": path,
        "path_a_value": totals["A"],
        "path_b_value": totals["B"],
        "margin": float(totals[other] - totals[path]),
    }


def assert_native_path_ordering() -> None:
    path_a = SOLVER.solve(np.array([0.0, 10.0, 10.0, 0.0]))
    path_b = SOLVER.solve(np.array([10.0, 0.0, 0.0, 10.0]))
    np.testing.assert_array_equal(path_a, PATH_A)
    np.testing.assert_array_equal(path_b, PATH_B)
    true_solution = SOLVER.solve(TRUE_COSTS)
    np.testing.assert_array_equal(true_solution, PATH_B)


def component_columns(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_{index}": float(value)
        for index, value in enumerate(_as_float_array(values))
    }


def fixed_posterior_step(
    z: float,
    missing_grad_weight: float,
    learning_rate: float,
    *,
    full_feedback: bool = False,
) -> dict[str, Any]:
    predictions = INITIAL_THETA * float(z)
    selection_info = solve_with_gap(predictions)
    selection = selection_info["action"]
    selected = selection > 0.5
    target = (
        TRUE_COSTS.copy()
        if full_feedback
        else np.where(selected, TRUE_COSTS, predictions)
    )
    reflected = 2.0 * predictions - target
    raw_gradient = _as_float_array(SPO_GRAD(predictions, target))
    gradient_weights = (
        np.ones_like(raw_gradient)
        if full_feedback
        else np.where(selected, 1.0, float(missing_grad_weight))
    )
    masked_gradient = raw_gradient * gradient_weights
    scalar_gradient = float(masked_gradient @ INITIAL_THETA)
    next_z = float(z - learning_rate * scalar_gradient)

    return {
        "z_before": float(z),
        "z_after": next_z,
        "selected_path": selection_info["path"],
        "selection": selection,
        "predictions": predictions,
        "target": target,
        "reflected": reflected,
        "raw_gradient": raw_gradient,
        "gradient_weights": gradient_weights,
        "masked_gradient": masked_gradient,
        "scalar_gradient": scalar_gradient,
        "target_solve": solve_with_gap(target),
        "reflected_solve": solve_with_gap(reflected),
        "pseudo_regret": float(TRUE_COSTS @ selection - TRUE_COSTS @ PATH_B),
        "path_b_prediction_error_after": float(
            abs((INITIAL_THETA * next_z) @ PATH_B - TRUE_COSTS @ PATH_B)
        ),
    }


def run_fixed_posterior_trace(
    z0: float,
    missing_grad_weight: float,
    rounds: int,
    learning_rate: float,
) -> list[dict[str, Any]]:
    z = float(z0)
    cumulative_regret = 0.0
    records: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        step = fixed_posterior_step(
            z,
            missing_grad_weight,
            learning_rate,
            full_feedback=False,
        )
        cumulative_regret += step["pseudo_regret"]
        record = {
            "kind": "fixed_posterior",
            "method": f"Detached rho={missing_grad_weight:g}",
            "z0": float(z0),
            "round": round_index,
            "rho": float(missing_grad_weight),
            "z_before": step["z_before"],
            "z_after": step["z_after"],
            "selected_path": step["selected_path"],
            "scalar_gradient": step["scalar_gradient"],
            "target_path": step["target_solve"]["path"],
            "target_margin": step["target_solve"]["margin"],
            "reflected_path": step["reflected_solve"]["path"],
            "reflected_margin": step["reflected_solve"]["margin"],
            "pseudo_regret": step["pseudo_regret"],
            "cumulative_pseudo_regret": cumulative_regret,
            "path_b_prediction_error_after": step["path_b_prediction_error_after"],
        }
        record.update(component_columns("prediction", step["predictions"]))
        record.update(component_columns("target", step["target"]))
        record.update(component_columns("reflected", step["reflected"]))
        record.update(component_columns("raw_gradient", step["raw_gradient"]))
        record.update(component_columns("masked_gradient", step["masked_gradient"]))
        records.append(record)
        z = step["z_after"]
    return records


def make_agent(
    agent_kind: str,
    z0: float,
    *,
    gamma: float,
    beta: float,
    learning_rate: float,
    rounds: int,
    missing_grad_weight: float,
    seed: int = 0,
):
    common = {
        "context_dim": 1,
        "n_arms": 4,
        "weights": np.ones(4),
        "capacity": 2.0,
        "hidden_dim": 1,
        "beta": beta,
        "lambda_reg": gamma,
        "lr": learning_rate,
        "decay_rounds": rounds,
        "max_grad_norm": 5.0,
        "nonlinear": False,
        "minimize": True,
        "solver": SOLVER.solve,
        "seed": seed,
        "exploration_strategy": "ucb",
    }
    if agent_kind == "spo":
        agent = BayesianSPOAgent(
            **common,
            mask_grad=True,
            missing_grad_weight=missing_grad_weight,
            imputation_samples=1,
        )
    elif agent_kind == "mse":
        agent = BayesianMSEAgent(
            **common,
            mask_grad=False,
            missing_grad_weight=missing_grad_weight,
        )
    else:
        raise ValueError(f"Unknown agent kind: {agent_kind}")

    agent._set_params(
        {
            "W1": jnp.zeros((1, 1)),
            "b1": jnp.asarray([float(z0)]),
            "blocks": (),
        }
    )
    agent.opt_state = agent.optimizer.init(agent._get_params())

    eye = jnp.eye(1)
    agent.A = jnp.broadcast_to(gamma * eye, (4, 1, 1)).copy()
    agent.A_inv = jnp.broadcast_to(eye / gamma, (4, 1, 1)).copy()
    agent.b_bayes = jnp.asarray(
        gamma * INITIAL_THETA[:, None],
        dtype=agent.A.dtype,
    )
    return agent


def _posterior_theta(agent) -> np.ndarray:
    return _as_float_array(jnp.einsum("nij,nj->ni", agent.A_inv, agent.b_bayes))


def trace_agent_round(
    agent,
    *,
    method: str,
    z0: float,
    round_index: int,
    full_feedback: bool,
    cumulative_regret: float,
) -> dict[str, Any]:
    x = jnp.asarray(CONTEXT)
    h_pre = agent._features(x)
    z_before = float(_as_float_array(h_pre)[0])
    theta_pre = _posterior_theta(agent)
    predictions_pre = _as_float_array(
        compute_bayesian_predictions(
            h_pre,
            agent.A_inv,
            agent.b_bayes,
        )
    )
    widths_pre = _as_float_array(compute_uncertainties(h_pre, agent.A_inv))
    ucb_values_pre = _as_float_array(
        compute_ucb_values(
            h_pre,
            agent.A_inv,
            agent.b_bayes,
            agent.beta,
            minimize=True,
        )
    )

    selection = _as_float_array(agent.act(CONTEXT))
    expected_selection = _as_float_array(SOLVER.solve(ucb_values_pre))
    np.testing.assert_array_equal(selection, expected_selection)
    selected_path = identify_path(selection)
    selection_mask = selection > 0.5
    pseudo_regret = float(TRUE_COSTS @ selection - TRUE_COSTS @ PATH_B)
    cumulative_regret += pseudo_regret

    observation_mask = np.ones(4, dtype=bool) if full_feedback else selection_mask
    agent.A, agent.b_bayes, agent.A_inv = update_bayesian_stats(
        agent.A,
        agent.b_bayes,
        agent.A_inv,
        h_pre,
        jnp.asarray(TRUE_COSTS),
        jnp.asarray(observation_mask),
    )
    theta_post = _posterior_theta(agent)
    predictions_post = _as_float_array(
        compute_bayesian_predictions(
            h_pre,
            agent.A_inv,
            agent.b_bayes,
        )
    )
    widths_post = _as_float_array(compute_uncertainties(h_pre, agent.A_inv))
    target = (
        TRUE_COSTS.copy()
        if full_feedback
        else np.where(selection_mask, TRUE_COSTS, predictions_post)
    )
    reflected = 2.0 * predictions_post - target

    if isinstance(agent, BayesianSPOAgent):
        raw_gradient = _as_float_array(agent.loss_grad_fn(predictions_post, target))
        gradient_weights = (
            np.ones(4)
            if full_feedback
            else np.where(
                selection_mask,
                1.0,
                agent.missing_grad_weight,
            )
        )
        component_gradient = raw_gradient * gradient_weights
        grad_h = agent._compute_spo_grad_h(
            x,
            h_pre,
            jnp.asarray(target),
            jnp.asarray(selection_mask),
            full_feedback,
        )
        objective = "SPO+"
    else:
        raw_gradient = 2.0 * (predictions_post - target) / 4.0
        gradient_weights = np.ones(4)
        component_gradient = raw_gradient
        grad_h = agent._compute_mse_grad_h(
            x,
            h_pre,
            jnp.asarray(target),
            jnp.asarray(selection_mask),
            full_feedback,
        )
        objective = "MSE"

    scalar_gradient = float(_as_float_array(grad_h)[0])
    reconstructed_gradient = float(component_gradient @ theta_post)
    np.testing.assert_allclose(
        scalar_gradient,
        reconstructed_gradient,
        rtol=2e-6,
        atol=2e-7,
    )
    target_solve = solve_with_gap(target)
    reflected_solve = solve_with_gap(reflected)

    agent._backprop_and_step(x, grad_h)
    h_after = agent._features(x)
    z_after = float(_as_float_array(h_after)[0])
    predictions_after = _as_float_array(
        compute_bayesian_predictions(
            h_after,
            agent.A_inv,
            agent.b_bayes,
        )
    )
    widths_after = _as_float_array(compute_uncertainties(h_after, agent.A_inv))
    path_b_error_after = float(abs(predictions_after @ PATH_B - TRUE_COSTS @ PATH_B))

    record: dict[str, Any] = {
        "kind": "real_agent",
        "method": method,
        "objective": objective,
        "z0": float(z0),
        "round": int(round_index),
        "full_feedback": bool(full_feedback),
        "beta": float(agent.beta),
        "z_before": z_before,
        "z_after": z_after,
        "selected_path": selected_path,
        "target_path": target_solve["path"],
        "target_margin": target_solve["margin"],
        "reflected_path": reflected_solve["path"],
        "reflected_margin": reflected_solve["margin"],
        "scalar_gradient": scalar_gradient,
        "pseudo_regret": pseudo_regret,
        "cumulative_pseudo_regret": cumulative_regret,
        "path_b_prediction_error_after": path_b_error_after,
    }
    for prefix, vector in (
        ("selection", selection),
        ("theta_pre", theta_pre),
        ("prediction_pre", predictions_pre),
        ("width_pre", widths_pre),
        ("ucb_pre", ucb_values_pre),
        ("theta_post", theta_post),
        ("prediction_post", predictions_post),
        ("width_post", widths_post),
        ("target", target),
        ("reflected", reflected),
        ("raw_gradient", raw_gradient),
        ("gradient_weight", gradient_weights),
        ("masked_gradient", component_gradient),
        ("prediction_after", predictions_after),
        ("width_after", widths_after),
    ):
        record.update(component_columns(prefix, vector))
    return record


def run_agent_trace(
    method: str,
    z0: float,
    *,
    rounds: int,
    gamma: float,
    beta: float,
    learning_rate: float,
    seed: int,
) -> list[dict[str, Any]]:
    config = REAL_METHOD_CONFIG[method]
    agent = make_agent(
        config["agent"],
        z0,
        gamma=gamma,
        beta=beta,
        learning_rate=learning_rate,
        rounds=rounds,
        missing_grad_weight=config["missing_grad_weight"],
        seed=seed,
    )
    cumulative_regret = 0.0
    records = []
    for round_index in range(1, rounds + 1):
        record = trace_agent_round(
            agent,
            method=method,
            z0=z0,
            round_index=round_index,
            full_feedback=config["full_feedback"],
            cumulative_regret=cumulative_regret,
        )
        cumulative_regret = record["cumulative_pseudo_regret"]
        records.append(record)
    return records


def scalar_alignment(gradient: float, reference: float) -> float:
    if abs(gradient) <= 1e-12 or abs(reference) <= 1e-12:
        return 0.0
    return float(np.sign(gradient * reference))


def summarize_real_traces(trajectories: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for z0, z_frame in trajectories.groupby("z0", sort=True):
        full_first = z_frame[
            z_frame["method"].eq("Full-feedback SPO+") & z_frame["round"].eq(1)
        ].iloc[0]
        reference_gradient = float(full_first["scalar_gradient"])
        for method in REAL_METHODS:
            method_frame = (
                z_frame[z_frame["method"].eq(method)]
                .sort_values("round")
                .reset_index(drop=True)
            )
            first = method_frame.iloc[0]
            last = method_frame.iloc[-1]
            optimal_rounds = method_frame.loc[
                method_frame["selected_path"].eq("B"),
                "round",
            ]
            first_optimal_round = (
                int(optimal_rounds.iloc[0]) if len(optimal_rounds) else None
            )
            summaries.append(
                {
                    "kind": "real_agent",
                    "method": method,
                    "z0": float(z0),
                    "initial_gradient": float(first["scalar_gradient"]),
                    "initial_gradient_alignment": scalar_alignment(
                        float(first["scalar_gradient"]),
                        reference_gradient,
                    ),
                    "first_optimal_path_round": first_optimal_round,
                    "final_z": float(last["z_after"]),
                    "final_path_b_prediction_error": float(
                        last["path_b_prediction_error_after"]
                    ),
                    "cumulative_pseudo_regret": float(last["cumulative_pseudo_regret"]),
                    "initial_target_path": first["target_path"],
                    "initial_target_margin": float(first["target_margin"]),
                    "initial_reflected_path": first["reflected_path"],
                    "initial_reflected_margin": float(first["reflected_margin"]),
                }
            )
    return pd.DataFrame(summaries)


def summarize_fixed_traces(trajectories: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (z0, method), frame in trajectories.groupby(
        ["z0", "method"],
        sort=True,
    ):
        frame = frame.sort_values("round").reset_index(drop=True)
        first = frame.iloc[0]
        last = frame.iloc[-1]
        full_reference = fixed_posterior_step(
            float(z0),
            0.0,
            learning_rate=0.0,
            full_feedback=True,
        )["scalar_gradient"]
        optimal_rounds = frame.loc[
            frame["selected_path"].eq("B"),
            "round",
        ]
        summaries.append(
            {
                "kind": "fixed_posterior",
                "method": method,
                "z0": float(z0),
                "initial_gradient": float(first["scalar_gradient"]),
                "initial_gradient_alignment": scalar_alignment(
                    float(first["scalar_gradient"]),
                    float(full_reference),
                ),
                "first_optimal_path_round": (
                    int(optimal_rounds.iloc[0]) if len(optimal_rounds) else None
                ),
                "final_z": float(last["z_after"]),
                "final_path_b_prediction_error": float(
                    last["path_b_prediction_error_after"]
                ),
                "cumulative_pseudo_regret": float(last["cumulative_pseudo_regret"]),
                "initial_target_path": first["target_path"],
                "initial_target_margin": float(first["target_margin"]),
                "initial_reflected_path": first["reflected_path"],
                "initial_reflected_margin": float(first["reflected_margin"]),
            }
        )
    return pd.DataFrame(summaries)


def validate_results(
    real_trajectories: pd.DataFrame,
    fixed_trajectories: pd.DataFrame,
    real_summary: pd.DataFrame,
) -> None:
    assert_native_path_ordering()
    first_rounds = real_trajectories[
        real_trajectories["round"].eq(1)
        & real_trajectories["method"].isin(
            ["Masked BayesianSPO", "Unmasked BayesianSPO"]
        )
    ]
    if not bool(
        (first_rounds["target_margin"] > STRICT_MARGIN_TOL).all()
        and (first_rounds["reflected_margin"] > STRICT_MARGIN_TOL).all()
    ):
        raise ValueError("The target or reflected solve has no strict margin.")

    primary = real_summary[np.isclose(real_summary["z0"], PRIMARY_Z0)].set_index(
        "method"
    )
    expected_alignment = {
        "Full-feedback SPO+": 1.0,
        "Masked BayesianSPO": 1.0,
        "Unmasked BayesianSPO": -1.0,
    }
    observed_alignment = primary["initial_gradient_alignment"].to_dict()
    if any(
        observed_alignment.get(method) != expected
        for method, expected in expected_alignment.items()
    ):
        raise ValueError("Unexpected initial gradient alignment.")

    for z0 in DEFAULT_Z0_VALUES:
        frame = real_summary[np.isclose(real_summary["z0"], z0)].set_index("method")
        if not (
            frame.loc["Full-feedback SPO+", "final_z"] < z0
            and frame.loc["Masked BayesianSPO", "final_z"] < z0
            and frame.loc["Unmasked BayesianSPO", "final_z"] > z0
            and pd.isna(
                frame.loc[
                    "Unmasked BayesianSPO",
                    "first_optimal_path_round",
                ]
            )
        ):
            raise ValueError(f"Unexpected feature-state transition at z0={z0}.")

    if not (
        pd.notna(
            primary.loc[
                "Full-feedback SPO+",
                "first_optimal_path_round",
            ]
        )
        and pd.notna(
            primary.loc[
                "Masked BayesianSPO",
                "first_optimal_path_round",
            ]
        )
        and pd.isna(
            primary.loc[
                "Unmasked BayesianSPO",
                "first_optimal_path_round",
            ]
        )
    ):
        raise ValueError("Unexpected primary action transition.")

    unmasked_primary = real_trajectories[
        np.isclose(real_trajectories["z0"], PRIMARY_Z0)
        & real_trajectories["method"].eq("Unmasked BayesianSPO")
    ].sort_values("round")
    if not bool(
        unmasked_primary.iloc[-1]["path_b_prediction_error_after"]
        > unmasked_primary.iloc[0]["path_b_prediction_error_after"]
    ):
        raise ValueError("Unmasked imputation error did not increase.")

    fixed_first = fixed_trajectories[
        fixed_trajectories["round"].eq(1)
        & np.isclose(fixed_trajectories["z0"], PRIMARY_Z0)
    ].set_index("method")
    exact_expected = {
        "Detached rho=0": 0.5,
        "Detached rho=0.5": 0.0,
        "Detached rho=1": -0.5,
    }
    if not all(
        np.isclose(
            fixed_first.loc[method, "scalar_gradient"],
            expected,
            atol=1e-8,
        )
        for method, expected in exact_expected.items()
    ):
        raise ValueError("The detached recurrence is incorrect.")


def run_experiment(output_dir: Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_native_path_ordering()

    real_records = []
    fixed_records = []
    for z0 in DEFAULT_Z0_VALUES:
        for method in REAL_METHODS:
            real_records.extend(
                run_agent_trace(
                    method,
                    z0,
                    rounds=ROUNDS,
                    gamma=GAMMA,
                    beta=BETA,
                    learning_rate=LEARNING_RATE,
                    seed=SEED,
                )
            )
        for rho in RHO_VALUES:
            fixed_records.extend(
                run_fixed_posterior_trace(
                    z0,
                    rho,
                    ROUNDS,
                    LEARNING_RATE,
                )
            )

    real_trajectories = pd.DataFrame(real_records)
    fixed_trajectories = pd.DataFrame(fixed_records)
    real_summary = summarize_real_traces(real_trajectories)
    fixed_summary = summarize_fixed_traces(fixed_trajectories)
    summary = pd.concat(
        [real_summary, fixed_summary],
        ignore_index=True,
    )
    validate_results(real_trajectories, fixed_trajectories, real_summary)

    real_trajectories.to_csv(output_dir / "agent_trajectories.csv", index=False)
    fixed_trajectories.to_csv(output_dir / "detached_recurrence.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.output_dir)
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

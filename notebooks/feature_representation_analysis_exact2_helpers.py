from pathlib import Path
import re
from typing import Dict

import numpy as np
import pandas as pd

from src.agents.base import pregenerate_episode

from notebooks.feature_representation_analysis_claude_helpers import (
    DecisionFocusedDGP,
    create_agent_pair,
    extract_features,
    generate_test_contexts,
    train_with_checkpoints,
)
from notebooks.feature_representation_analysis_helpers import (
    _mechanism_summary_table,
    build_regime_dashboard,
    plot_cumulative_regret,
    plot_decoded_latents,
    plot_feedback_feature_grid,
    plot_prediction_error_trajectory,
    plot_regret_per_round,
    plot_action_timeline,
    plot_feature_directions,
)
from lets_plot.export import ggsave


class Exact2Solver:
    """Select exactly the top-2 items by value."""

    def solve(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        action = np.zeros_like(values)
        top2 = np.argsort(values)[-2:]
        action[top2] = 1.0
        return action


class DecisionFocusedExact2Env:
    """Exact-2 version of the decision-focused toy problem."""

    def __init__(self, seed: int = 42, noise_width: float = 0.0):
        self.data = DecisionFocusedDGP(seed=seed, noise_width=noise_width)
        self.n_arms = 4
        self.context_dim = 5
        self.weights = np.ones(4, dtype=float)
        self.capacity = 2.0
        self.knapsack_solver = Exact2Solver()
        self.current_context, self.current_values = self._generate_sample()

    @property
    def minimize(self) -> bool:
        return False

    def _generate_sample(self):
        return self.data.sample()

    def reset(self):
        self.data.reset()
        self.current_context, self.current_values = self._generate_sample()
        return self.current_context

    def _advance(self):
        self.data.step()
        self.current_context, self.current_values = self._generate_sample()

    def get_context(self):
        return self.current_context

    def solve(self, values: np.ndarray) -> np.ndarray:
        return self.knapsack_solver.solve(values)


def compute_oracle_decision_exact2(x: np.ndarray) -> np.ndarray:
    """In the exact-2 toy, the oracle picks one winner from each pair."""
    sel = np.zeros(4)
    sel[0 if x[0] > 0 else 1] = 1.0
    sel[2 if x[1] > 0 else 3] = 1.0
    return sel


def oracle_agreement_summary_exact2(
    n_verify: int = 200,
    seed: int = 42,
    context_seed: int = 123,
    noise_width: float = 0.0,
) -> pd.DataFrame:
    env = DecisionFocusedExact2Env(seed=seed, noise_width=noise_width)
    rng = np.random.default_rng(context_seed)
    matches = 0
    for _ in range(n_verify):
        x = rng.standard_normal(env.context_dim)
        v = env.data._generate_values(x)
        oracle_sel = compute_oracle_decision_exact2(x)
        solver_sel = env.solve(v)
        if np.array_equal(oracle_sel.astype(int), solver_sel.astype(int)):
            matches += 1
    return pd.DataFrame(
        [
            {
                "noise_width": noise_width,
                "oracle_matches": matches,
                "n_verify": n_verify,
                "agreement_rate": matches / n_verify,
            }
        ]
    )


def _slugify_regime(regime: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", regime.lower()).strip("_")
    return slug or "regime"


def save_exact2_dashboard_plot(dashboard_plot, regime: str) -> str:
    output_dir = Path("results") / "feature_representation_analysis_exact2"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify_regime(regime)}_dashboard.svg"
    return ggsave(dashboard_plot, filename, path=str(output_dir), iframe=False)


def run_exact2_feedback_regime_experiment(
    regime: str,
    full_feedback: bool,
    n_rounds: int = 10000,
    checkpoint_interval: int = 200,
    noise_width: float = 0.0,
    hidden_dim: int = 2,
    beta: float = 1.0,
    lambda_reg: float = 0.1,
    lr: float = 0.03,
    seed: int = 0,
    n_test: int = 500,
    test_seed: int = 99,
    mask_spo: bool = False,
    mask_mse: bool = False,
    centered_spo: bool = False,
    phased_spo: bool = False,
    phased_mse: bool = False,
    phase_length: int = 50,
    phase_epochs: int = 5,
    posterior_imputed_spo: bool = False,
    adaptive_posterior_imputed_spo: bool = False,
    quantile_posterior_imputed_spo: bool = False,
    leaky_mask_posterior_imputed_spo: bool = False,
    residual_projected_posterior_imputed_spo: bool = False,
    clipped_posterior_imputed_spo: bool = False,
    scenario_posterior_imputed_spo: bool = False,
    orthogonalized_posterior_imputed_spo: bool = False,
    antithetic_posterior_imputed_spo: bool = False,
    local_posterior_imputed_spo: bool = False,
    target_posterior_imputed_spo: bool = False,
    action_gap_spo: bool = False,
    posterior_imputed_action_gap_spo: bool = False,
    action_gap_samples: int = 16,
    action_gap_top_k: int = 3,
    action_gap_imputation_samples: int = 8,
    action_gap_imputation_scale: float = 1.0,
    posterior_missing_grad_weight: float = 0.25,
    posterior_projection_strength: float = 0.5,
    comparison_balanced_replay_spo: bool = False,
    comparison_replay_samples: int = 16,
    pair_balanced_spo: bool = False,
    pair_balanced_mse: bool = False,
    disagreement_coverage_spo: bool = False,
    disagreement_coverage_mse: bool = False,
    disagreement_samples: int = 16,
    disagreement_lambda: float = 1.0,
) -> Dict[str, object]:
    env = DecisionFocusedExact2Env(seed=42, noise_width=noise_width)
    episode = pregenerate_episode(env, n_rounds=n_rounds)
    env_test = DecisionFocusedExact2Env(seed=42, noise_width=noise_width)
    X_test, decisions_test, values_test = generate_test_contexts(
        env_test, n_samples=n_test, seed=test_seed
    )

    spo_agent, _ = create_agent_pair(
        env,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        seed=seed,
        mask_spo=mask_spo,
        mask_mse=mask_mse,
        centered_spo=centered_spo,
        phased_spo=phased_spo,
        phased_mse=phased_mse,
        phase_length=phase_length,
        phase_epochs=phase_epochs,
        posterior_imputed_spo=posterior_imputed_spo,
        adaptive_posterior_imputed_spo=adaptive_posterior_imputed_spo,
        quantile_posterior_imputed_spo=quantile_posterior_imputed_spo,
        leaky_mask_posterior_imputed_spo=leaky_mask_posterior_imputed_spo,
        residual_projected_posterior_imputed_spo=residual_projected_posterior_imputed_spo,
        clipped_posterior_imputed_spo=clipped_posterior_imputed_spo,
        scenario_posterior_imputed_spo=scenario_posterior_imputed_spo,
        orthogonalized_posterior_imputed_spo=orthogonalized_posterior_imputed_spo,
        antithetic_posterior_imputed_spo=antithetic_posterior_imputed_spo,
        local_posterior_imputed_spo=local_posterior_imputed_spo,
        target_posterior_imputed_spo=target_posterior_imputed_spo,
        action_gap_spo=action_gap_spo,
        posterior_imputed_action_gap_spo=posterior_imputed_action_gap_spo,
        action_gap_samples=action_gap_samples,
        action_gap_top_k=action_gap_top_k,
        action_gap_imputation_samples=action_gap_imputation_samples,
        action_gap_imputation_scale=action_gap_imputation_scale,
        posterior_missing_grad_weight=posterior_missing_grad_weight,
        posterior_projection_strength=posterior_projection_strength,
        comparison_balanced_replay_spo=comparison_balanced_replay_spo,
        comparison_replay_samples=comparison_replay_samples,
        pair_balanced_spo=pair_balanced_spo,
        pair_balanced_mse=pair_balanced_mse,
        disagreement_coverage_spo=disagreement_coverage_spo,
        disagreement_coverage_mse=disagreement_coverage_mse,
        disagreement_samples=disagreement_samples,
        disagreement_lambda=disagreement_lambda,
    )
    res_spo = train_with_checkpoints(
        spo_agent,
        episode,
        n_rounds,
        checkpoint_interval=checkpoint_interval,
        full_feedback=full_feedback,
        eval_contexts=X_test,
        eval_values=values_test,
    )

    _, mse_agent = create_agent_pair(
        env,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        seed=seed,
        mask_spo=mask_spo,
        mask_mse=mask_mse,
        centered_spo=centered_spo,
        phased_spo=phased_spo,
        phased_mse=phased_mse,
        phase_length=phase_length,
        phase_epochs=phase_epochs,
        posterior_imputed_spo=posterior_imputed_spo,
        adaptive_posterior_imputed_spo=adaptive_posterior_imputed_spo,
        quantile_posterior_imputed_spo=quantile_posterior_imputed_spo,
        leaky_mask_posterior_imputed_spo=leaky_mask_posterior_imputed_spo,
        residual_projected_posterior_imputed_spo=residual_projected_posterior_imputed_spo,
        clipped_posterior_imputed_spo=clipped_posterior_imputed_spo,
        scenario_posterior_imputed_spo=scenario_posterior_imputed_spo,
        orthogonalized_posterior_imputed_spo=orthogonalized_posterior_imputed_spo,
        antithetic_posterior_imputed_spo=antithetic_posterior_imputed_spo,
        local_posterior_imputed_spo=local_posterior_imputed_spo,
        target_posterior_imputed_spo=target_posterior_imputed_spo,
        action_gap_spo=action_gap_spo,
        posterior_imputed_action_gap_spo=posterior_imputed_action_gap_spo,
        action_gap_samples=action_gap_samples,
        action_gap_top_k=action_gap_top_k,
        action_gap_imputation_samples=action_gap_imputation_samples,
        action_gap_imputation_scale=action_gap_imputation_scale,
        posterior_missing_grad_weight=posterior_missing_grad_weight,
        posterior_projection_strength=posterior_projection_strength,
        comparison_balanced_replay_spo=comparison_balanced_replay_spo,
        comparison_replay_samples=comparison_replay_samples,
        pair_balanced_spo=pair_balanced_spo,
        pair_balanced_mse=pair_balanced_mse,
        disagreement_coverage_spo=disagreement_coverage_spo,
        disagreement_coverage_mse=disagreement_coverage_mse,
        disagreement_samples=disagreement_samples,
        disagreement_lambda=disagreement_lambda,
    )
    res_mse = train_with_checkpoints(
        mse_agent,
        episode,
        n_rounds,
        checkpoint_interval=checkpoint_interval,
        full_feedback=full_feedback,
        eval_contexts=X_test,
        eval_values=values_test,
    )

    h_spo = extract_features(spo_agent, X_test)
    h_mse = extract_features(mse_agent, X_test)

    summary = _mechanism_summary_table(
        regime,
        res_spo,
        res_mse,
        spo_agent,
        mse_agent,
        env_test.solve,
        h_spo,
        h_mse,
        X_test,
        decisions_test,
        values_test,
    )
    feature_plot = plot_feedback_feature_grid(
        h_spo,
        h_mse,
        X_test,
        decisions_test,
        values_test,
        title_prefix=f"{regime}: decision structure vs nuisance prediction",
    )
    decoded_latents_plot = plot_decoded_latents(
        h_spo,
        h_mse,
        values_test,
        decisions_test,
    )
    regret_plot = plot_regret_per_round(
        res_spo,
        res_mse,
        title_prefix=f"{regime}: regret per round",
    )
    cumulative_regret_plot = plot_cumulative_regret(
        res_spo,
        res_mse,
        title_prefix=f"{regime}: cumulative regret",
    )
    prediction_error_plot = plot_prediction_error_trajectory(
        res_spo,
        res_mse,
        title_prefix=f"{regime}: prediction error over training",
    )
    loading_plot = plot_feature_directions(
        res_spo["W1_checkpoints"][-1][1],
        res_mse["W1_checkpoints"][-1][1],
    )
    oracle_actions = [rd["optimal_action"] for rd in episode["rounds_data"]]
    action_plot = plot_action_timeline(
        oracle_actions,
        res_spo["actions"],
        res_mse["actions"],
        title_prefix=f"{regime}: action-space match vs oracle",
    )
    dashboard_plot = build_regime_dashboard(
        feature_plot,
        decoded_latents_plot,
        regret_plot,
        cumulative_regret_plot,
        prediction_error_plot,
        loading_plot,
        action_plot,
        title_prefix=f"{regime}: dashboard",
    )
    dashboard_path = save_exact2_dashboard_plot(dashboard_plot, regime)

    return {
        "regime": regime,
        "summary": summary,
        "dashboard_plot": dashboard_plot,
        "dashboard_path": dashboard_path,
        "feature_plot": feature_plot,
        "decoded_latents_plot": decoded_latents_plot,
        "regret_plot": regret_plot,
        "cumulative_regret_plot": cumulative_regret_plot,
        "prediction_error_plot": prediction_error_plot,
        "loading_plot": loading_plot,
        "action_plot": action_plot,
        "res_spo": res_spo,
        "res_mse": res_mse,
        "h_spo": h_spo,
        "h_mse": h_mse,
        "X_test": X_test,
        "decisions_test": decisions_test,
        "values_test": values_test,
    }

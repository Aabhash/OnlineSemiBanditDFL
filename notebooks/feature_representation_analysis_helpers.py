"""Helpers for feature representation analysis notebook (Claude version)."""

import numpy as np
import jax.numpy as jnp
import pandas as pd
from typing import Tuple, Dict, List, Optional
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from lets_plot import (
    aes,
    geom_point,
    geom_line,
    geom_hline,
    geom_bar,
    ggplot,
    ggsize,
    labs,
    scale_color_manual,
    scale_fill_manual,
    coord_fixed,
    theme,
    element_text,
    theme_minimal,
    facet_wrap,
)
from lets_plot.mapping import as_discrete

from src.data.knapsack import KnapsackData
from src.envs.knapsack import KnapsackEnv
from src.solvers.knapsack import KnapsackSolver
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.agents.contextual.bayesian_spo import BayesianSPOAgent
from src.agents.contextual.bayesian_spo_pi import BayesianSPOPIAgent
from src.agents.contextual.weber_bll import WeberBLLAgent
from notebooks.archive.ablations import (
    AdaptivePosteriorImputedBayesianSPOAgent,
    AntitheticPosteriorImputedBayesianSPOAgent,
    ClippedPosteriorImputedBayesianSPOAgent,
    LocalPosteriorImputedBayesianSPOAgent,
    OrthogonalizedPosteriorImputedBayesianSPOAgent,
    QuantilePosteriorImputedBayesianSPOAgent,
    ResidualProjectedPosteriorImputedBayesianSPOAgent,
    ScenarioPosteriorImputedBayesianSPOAgent,
    TargetNetworkPosteriorImputedBayesianSPOAgent,
    LeakyMaskPosteriorImputedBayesianSPOAgent,
    WeberBlendedSPOAgent,
    WeberDecisionPFYAgent,
)
from src.agents.contextual.neural_agents import BayesianMSEAgent
from src.agents.bayesian_utils import (
    compute_bayesian_predictions,
    compute_ucb_values,
    compute_uncertainties,
    init_per_arm_regression,
    update_bayesian_stats,
)


# ── Task 1: Custom DGP and Oracle ──────────────────────────────────────────


class DecisionFocusedDGP(KnapsackData):
    """Custom DGP where decision-relevant and nuisance dimensions are known.

    4-arm unit-weight knapsack, capacity=2 (select exactly 2 of 4).
    Context x in R^5.

    B matrix design:
    - Items 0,1: costs depend on x[0] (decision dim 1)
    - Items 2,3: costs depend on x[1] (decision dim 2)
    - All items: costs shift by x[2:5] (nuisance dims) equally,
      so nuisance dims change magnitudes but never flip the ranking
      within {0,1} or within {2,3}.
    """

    def __init__(self, seed: int = 42, noise_width: float = 0.1):
        self.n_items = 4
        self.n_features = 5
        self.context_dim = 5
        self.capacity = 2.0
        self.deg = 1
        self.noise_width = noise_width
        self.rng = np.random.default_rng(seed)

        self.weights = np.ones(4)

        # Decision dims (0-1): opposite signs for paired items -> flip rankings
        # Nuisance dims (2-4): large equal coefficients -> high MSE impact but
        # never change relative ordering within pairs, so zero decision impact.
        # This creates a strong incentive for MSE to learn nuisance dims
        # (they explain a lot of variance) while SPO should ignore them.
        self.B = np.array([
            [ 1.0,  0.0,  2.0,  2.0,  2.0],
            [-1.0,  0.0,  2.0,  2.0,  2.0],
            [ 0.0,  1.0,  2.0,  2.0,  2.0],
            [ 0.0, -1.0,  2.0,  2.0,  2.0],
        ])

    def _generate_values(self, x: np.ndarray) -> np.ndarray:
        """Linear costs: v_i = B_i @ x / sqrt(d) + offset, with small noise."""
        linear = self.B @ x / np.sqrt(self.n_features)
        values = linear + 3.0
        noise = self.rng.uniform(
            1 - self.noise_width,
            1 + self.noise_width,
            self.n_items,
        )
        return values * noise


class DecisionFocusedEnv(KnapsackEnv):
    """Knapsack environment using the custom DGP."""

    def __init__(self, seed: int = 42, noise_width: float = 0.1):
        from src.envs.base import ConstrainedBanditEnv

        self.data = DecisionFocusedDGP(seed=seed, noise_width=noise_width)

        ConstrainedBanditEnv.__init__(
            self,
            n_arms=4,
            context_dim=5,
            weights=self.data.weights,
            capacity=2.0,
            seed=seed,
        )
        self.knapsack_solver = KnapsackSolver(self.weights, self.capacity)
        self.current_context, self.current_values = self._generate_sample()


def compute_oracle_decision(x: np.ndarray) -> np.ndarray:
    """Return the optimal 2-of-4 selection for context x under the noiseless DGP.

    Decision logic (maximization, unit weights, capacity=2):
    - Within pair (0,1): item 0 wins when x[0] > 0, item 1 when x[0] < 0
    - Within pair (2,3): item 2 wins when x[1] > 0, item 3 when x[1] < 0
    - Pick the winner from each pair -> always exactly 2 items.
    """
    sel = np.zeros(4)
    sel[0 if x[0] > 0 else 1] = 1.0
    sel[2 if x[1] > 0 else 3] = 1.0
    return sel


def compute_oracle_decisions_batch(X: np.ndarray) -> np.ndarray:
    """Batch version: X is (n_samples, 5), returns (n_samples, 4)."""
    return np.array([compute_oracle_decision(x) for x in X])


def decision_label(sel: np.ndarray) -> str:
    """Convert binary selection to a readable label like '0,2'."""
    indices = np.where(np.asarray(sel) > 0.5)[0]
    return ",".join(str(i) for i in indices)


ORACLE_W1 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
])
PAIR_INDICES = ((0, 1), (2, 3))


# ── Task 2: Training Utilities and Feature Extraction ──────────────────────


def create_agent_pair(
    env,
    hidden_dim: int = 2,
    beta: float = 1.0,
    lambda_reg: float = 1.0,
    lr: float = 0.001,
    seed: int = 0,
    mask_spo: bool = False,
    mask_mse: bool = False,
    centered_spo: bool = False,
    phased_spo: bool = False,
    phased_mse: bool = False,
    phase_length: int = 50,
    phase_epochs: int = 5,
    replay_batch_size: int = 1,
    posterior_imputed_spo: bool = False,
    posterior_imputed_predictive_spo: bool = False,
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
    weber_blended_spo: bool = False,
    weber_pfy: bool = False,
    weber_bll: bool = False,
    weber_spo_blend_weight: float = 0.5,
) -> Tuple[BayesianSPOAgent, BayesianMSEAgent]:
    """Create matched SPO and MSE agents with identical initialization."""
    common = dict(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        minimize=getattr(env, "minimize", False),
        nonlinear=False,
        seed=seed,
        solver=env.knapsack_solver.solve,
    )
    if weber_bll:
        spo_cls = WeberBLLAgent
    elif weber_pfy:
        spo_cls = WeberDecisionPFYAgent
    elif weber_blended_spo:
        spo_cls = WeberBlendedSPOAgent
    elif comparison_balanced_replay_spo:
        spo_cls = ComparisonBalancedReplayBayesianSPOAgent
        phased_spo = True
    elif local_posterior_imputed_spo:
        spo_cls = LocalPosteriorImputedBayesianSPOAgent
    elif adaptive_posterior_imputed_spo:
        spo_cls = AdaptivePosteriorImputedBayesianSPOAgent
    elif quantile_posterior_imputed_spo:
        spo_cls = QuantilePosteriorImputedBayesianSPOAgent
    elif leaky_mask_posterior_imputed_spo:
        spo_cls = LeakyMaskPosteriorImputedBayesianSPOAgent
    elif residual_projected_posterior_imputed_spo:
        spo_cls = ResidualProjectedPosteriorImputedBayesianSPOAgent
    elif scenario_posterior_imputed_spo:
        spo_cls = ScenarioPosteriorImputedBayesianSPOAgent
    elif clipped_posterior_imputed_spo:
        spo_cls = ClippedPosteriorImputedBayesianSPOAgent
    elif orthogonalized_posterior_imputed_spo:
        spo_cls = OrthogonalizedPosteriorImputedBayesianSPOAgent
    elif antithetic_posterior_imputed_spo:
        spo_cls = AntitheticPosteriorImputedBayesianSPOAgent
    elif target_posterior_imputed_spo:
        spo_cls = TargetNetworkPosteriorImputedBayesianSPOAgent
    elif posterior_imputed_spo:
        spo_cls = BayesianSPOPIAgent
    elif posterior_imputed_predictive_spo:
        spo_cls = BayesianSPOPIAgent
    elif posterior_imputed_action_gap_spo:
        spo_cls = PosteriorImputedActionGapBayesianSPOAgent
    elif action_gap_spo:
        spo_cls = ActionGapBayesianSPOAgent
    elif disagreement_coverage_spo and centered_spo:
        spo_cls = CenteredDisagreementCoverageBayesianSPOAgent
    elif disagreement_coverage_spo:
        spo_cls = DisagreementCoverageBayesianSPOAgent
    elif centered_spo:
        spo_cls = CenteredBayesianSPOAgent
    elif pair_balanced_spo:
        spo_cls = PairBalancedBayesianSPOAgent
    else:
        spo_cls = BayesianSPOAgent

    if disagreement_coverage_mse:
        mse_cls = DisagreementCoverageBayesianMSEAgent
    elif pair_balanced_mse:
        mse_cls = PairBalancedBayesianMSEAgent
    else:
        mse_cls = BayesianMSEAgent

    extra = dict(
        disagreement_samples=disagreement_samples,
        disagreement_lambda=disagreement_lambda,
    )
    action_gap_extra = dict(
        action_gap_samples=action_gap_samples,
        action_gap_top_k=action_gap_top_k,
        action_gap_imputation_samples=action_gap_imputation_samples,
        action_gap_imputation_scale=action_gap_imputation_scale,
    )
    posterior_imputed_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        imputation_scale=action_gap_imputation_scale,
    )
    posterior_imputed_predictive_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        use_posterior_predictive=True,
        noise_var=1.0,
    )
    adaptive_posterior_imputed_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        imputation_scale=action_gap_imputation_scale,
        adaptive_ema_decay=0.95,
        adaptive_scale_min=0.5,
        adaptive_scale_max=1.25,
        adaptive_power=1.0,
    )
    quantile_posterior_imputed_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        imputation_scale=action_gap_imputation_scale,
        quantile_eps=1e-3,
        center_quantiles=True,
    )
    leaky_mask_posterior_imputed_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        imputation_scale=action_gap_imputation_scale,
        missing_grad_weight=posterior_missing_grad_weight,
    )
    residual_projected_posterior_imputed_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        imputation_scale=action_gap_imputation_scale,
        projection_strength=posterior_projection_strength,
    )
    clipped_posterior_imputed_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        imputation_scale=action_gap_imputation_scale,
        imputation_clip=2.0,
    )
    scenario_posterior_imputed_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        imputation_scale=action_gap_imputation_scale,
        disagreement_samples=16,
        scenario_top_k=3,
    )
    orthogonalized_posterior_imputed_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        imputation_scale=action_gap_imputation_scale,
        offset_shrinkage=1.0,
        min_observed_for_offset=1,
    )
    local_posterior_imputed_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        imputation_scale=action_gap_imputation_scale,
        disagreement_samples=16,
        local_top_k=3,
    )
    target_posterior_imputed_extra = dict(
        imputation_samples=action_gap_imputation_samples,
        imputation_scale=action_gap_imputation_scale,
        target_tau=0.05,
    )
    replay_extra = dict(comparison_samples=comparison_replay_samples)
    spo_kwargs = dict(
        **common,
        mask_grad=mask_spo,
        phased=phased_spo,
        phase_length=phase_length,
        phase_epochs=phase_epochs,
        **({"spo_blend_weight": weber_spo_blend_weight} if weber_blended_spo else {}),
        **(local_posterior_imputed_extra if local_posterior_imputed_spo else {}),
        **(adaptive_posterior_imputed_extra if adaptive_posterior_imputed_spo else {}),
        **(quantile_posterior_imputed_extra if quantile_posterior_imputed_spo else {}),
        **(leaky_mask_posterior_imputed_extra if leaky_mask_posterior_imputed_spo else {}),
        **(residual_projected_posterior_imputed_extra if residual_projected_posterior_imputed_spo else {}),
        **(scenario_posterior_imputed_extra if scenario_posterior_imputed_spo else {}),
        **(clipped_posterior_imputed_extra if clipped_posterior_imputed_spo else {}),
        **(orthogonalized_posterior_imputed_extra if orthogonalized_posterior_imputed_spo else {}),
        **(posterior_imputed_extra if antithetic_posterior_imputed_spo else {}),
        **(target_posterior_imputed_extra if target_posterior_imputed_spo else {}),
        **(posterior_imputed_extra if posterior_imputed_spo else {}),
        **(posterior_imputed_predictive_extra if posterior_imputed_predictive_spo else {}),
        **(action_gap_extra if (action_gap_spo or posterior_imputed_action_gap_spo) else {}),
        **(replay_extra if comparison_balanced_replay_spo else {}),
        **(extra if disagreement_coverage_spo else {}),
    )
    if not (weber_bll or weber_pfy or weber_blended_spo):
        spo_kwargs["replay_batch_size"] = replay_batch_size
    spo = spo_cls(**spo_kwargs)
    mse = mse_cls(
        **common,
        mask_grad=mask_mse,
        phased=phased_mse,
        phase_length=phase_length,
        phase_epochs=phase_epochs,
        replay_batch_size=replay_batch_size,
        **(extra if disagreement_coverage_mse else {}),
    )
    return spo, mse


def _pair_balanced_action(agent, X: np.ndarray) -> np.ndarray:
    """Select one UCB winner from each known pair."""
    agent.t += 1

    if agent.beta_start != agent.beta_end:
        progress = min(agent.t / agent.decay_rounds, 1.0)
        agent.beta = agent.beta_start + (agent.beta_end - agent.beta_start) * progress

    h = agent._features(X)
    ucb_values = np.asarray(
        compute_ucb_values(h, agent.A_inv, agent.b_bayes, agent.beta),
        dtype=float,
    ).reshape(-1)
    selection = np.zeros(agent.n_arms, dtype=float)
    for i, j in PAIR_INDICES:
        selection[i if ucb_values[i] >= ucb_values[j] else j] = 1.0
    return selection


class PairBalancedBayesianSPOAgent(BayesianSPOAgent):
    def act(self, X):
        return _pair_balanced_action(self, X)


class PairBalancedBayesianMSEAgent(BayesianMSEAgent):
    def act(self, X):
        return _pair_balanced_action(self, X)


def _collect_disagreement_candidates(agent, means: np.ndarray, stds: np.ndarray):
    candidate_actions = {}
    candidate_counts = {}

    def _add_candidate(values: np.ndarray):
        action = np.asarray(agent.solver(values), dtype=float).reshape(-1)
        key = tuple((action > 0.5).astype(int).tolist())
        candidate_actions[key] = action
        candidate_counts[key] = candidate_counts.get(key, 0.0) + 1.0

    _add_candidate(means)
    _add_candidate(means + agent.beta * stds)

    for _ in range(agent.disagreement_samples):
        sampled = means + agent.beta * stds * agent.disagreement_rng.standard_normal(agent.n_arms)
        _add_candidate(sampled)

    candidate_keys = list(candidate_actions.keys())
    candidate_matrix = np.vstack([candidate_actions[k] for k in candidate_keys])
    candidate_probs = np.asarray([candidate_counts[k] for k in candidate_keys], dtype=float)
    candidate_probs /= candidate_probs.sum()
    return candidate_matrix, candidate_probs


def _comparison_profile_from_stats(agent, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Per-context action-difference profile over near-boundary alternatives."""
    candidate_matrix, candidate_probs = _collect_disagreement_candidates(agent, means, stds)
    predicted_score = candidate_matrix @ means
    best_idx = int(np.argmax(predicted_score))
    score_gaps = predicted_score[best_idx] - predicted_score
    positive_gaps = score_gaps[score_gaps > 1e-8]
    gap_scale = float(np.median(positive_gaps)) if positive_gaps.size else 1.0
    gap_scale = max(gap_scale, 1e-3)

    boundary_weights = np.exp(-score_gaps / gap_scale) * candidate_probs
    boundary_weights[best_idx] = 0.0
    total = float(boundary_weights.sum())
    if total <= 1e-12:
        diffs = np.abs(candidate_matrix - candidate_matrix[best_idx])
        return diffs.mean(axis=0)
    boundary_weights /= total

    diffs = np.abs(candidate_matrix - candidate_matrix[best_idx])
    return (boundary_weights[:, None] * diffs).sum(axis=0)


def _generic_disagreement_coverage_action(agent, X: np.ndarray) -> np.ndarray:
    """Black-box exploration over contested action comparisons and coverage deficit."""
    agent.t += 1

    if agent.beta_start != agent.beta_end:
        progress = min(agent.t / agent.decay_rounds, 1.0)
        agent.beta = agent.beta_start + (agent.beta_end - agent.beta_start) * progress

    h = agent._features(X)
    means = np.asarray(
        compute_bayesian_predictions(h, agent.A_inv, agent.b_bayes),
        dtype=float,
    ).reshape(-1)
    stds = np.asarray(compute_uncertainties(h, agent.A_inv), dtype=float).reshape(-1)

    candidate_matrix, candidate_probs = _collect_disagreement_candidates(agent, means, stds)

    coverage_deficit = 1.0 / np.sqrt(agent.coverage_counts + 1.0)

    # Score each action by how informative its contested swaps with other plausible
    # actions would be on under-observed components.
    pairwise_swaps = np.abs(candidate_matrix[:, None, :] - candidate_matrix[None, :, :])
    swap_bonus_matrix = (pairwise_swaps * coverage_deficit[None, None, :]).sum(axis=2)
    comparison_bonus = swap_bonus_matrix @ candidate_probs

    predicted_score = candidate_matrix @ means
    total_score = predicted_score + agent.disagreement_lambda * comparison_bonus
    return candidate_matrix[int(np.argmax(total_score))]


def _boundary_disagreement_coverage_action(agent, X: np.ndarray) -> np.ndarray:
    """SPO-aligned exploration that only rewards under-covered near-boundary alternatives."""
    agent.t += 1

    if agent.beta_start != agent.beta_end:
        progress = min(agent.t / agent.decay_rounds, 1.0)
        agent.beta = agent.beta_start + (agent.beta_end - agent.beta_start) * progress

    h = agent._features(X)
    means = np.asarray(
        compute_bayesian_predictions(h, agent.A_inv, agent.b_bayes),
        dtype=float,
    ).reshape(-1)
    stds = np.asarray(compute_uncertainties(h, agent.A_inv), dtype=float).reshape(-1)

    candidate_matrix, candidate_probs = _collect_disagreement_candidates(agent, means, stds)
    predicted_score = candidate_matrix @ means

    best_idx = int(np.argmax(predicted_score))
    score_gaps = predicted_score[best_idx] - predicted_score
    positive_gaps = score_gaps[score_gaps > 1e-8]
    gap_scale = float(np.median(positive_gaps)) if positive_gaps.size else 1.0
    gap_scale = max(gap_scale, 1e-3)

    boundary_weights = np.exp(-score_gaps / gap_scale) * candidate_probs
    boundary_weights[best_idx] = 0.0
    if float(boundary_weights.sum()) <= 1e-12:
        boundary_weights = candidate_probs.copy()
        boundary_weights[best_idx] = 0.0
    total_boundary = float(boundary_weights.sum())
    if total_boundary <= 1e-12:
        return candidate_matrix[best_idx]
    boundary_weights /= total_boundary

    coverage_deficit = 1.0 / np.sqrt(agent.coverage_counts + 1.0)
    pairwise_swaps = np.abs(candidate_matrix[:, None, :] - candidate_matrix[None, :, :])
    swap_bonus_matrix = (pairwise_swaps * coverage_deficit[None, None, :]).sum(axis=2)
    comparison_bonus = swap_bonus_matrix @ boundary_weights

    total_score = predicted_score + agent.disagreement_lambda * comparison_bonus
    return candidate_matrix[int(np.argmax(total_score))]


class _GenericDisagreementCoverageMixin:
    def _init_disagreement_coverage(self, disagreement_samples: int, disagreement_lambda: float, seed: int = 0):
        self.disagreement_samples = int(disagreement_samples)
        self.disagreement_lambda = float(disagreement_lambda)
        self.coverage_counts = np.zeros(self.n_arms, dtype=float)
        self.disagreement_rng = np.random.default_rng(seed)

    def act(self, X):
        return _generic_disagreement_coverage_action(self, X)

    def update(self, X, selection, rewards, lr=None, full_feedback=False, **kwargs):
        observed_mask = np.ones(self.n_arms, dtype=float) if full_feedback else (np.asarray(selection) > 0.5).astype(float)
        self.coverage_counts += observed_mask
        return super().update(X, selection, rewards, lr=lr, full_feedback=full_feedback, **kwargs)


class _BoundaryDisagreementCoverageMixin(_GenericDisagreementCoverageMixin):
    def act(self, X):
        return _boundary_disagreement_coverage_action(self, X)


class DisagreementCoverageBayesianSPOAgent(_BoundaryDisagreementCoverageMixin, BayesianSPOAgent):
    def __init__(self, *args, disagreement_samples: int = 16, disagreement_lambda: float = 1.0, seed: int = 0, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self._init_disagreement_coverage(disagreement_samples, disagreement_lambda, seed=seed)


class DisagreementCoverageBayesianMSEAgent(_GenericDisagreementCoverageMixin, BayesianMSEAgent):
    def __init__(self, *args, disagreement_samples: int = 16, disagreement_lambda: float = 1.0, seed: int = 0, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self._init_disagreement_coverage(disagreement_samples, disagreement_lambda, seed=seed)


# Alias used in the notebook writeup: BayesianSPO_BDC = boundary-disagreement coverage.
BayesianSPO_BDC = DisagreementCoverageBayesianSPOAgent


class _CenteredSPOMixin:
    """Diagnostic variant: remove the shared offset from the SPO feature update."""

    def _compute_spo_grad_h(self, X, h, Y_target, selection_mask, full_feedback):
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

        pred_centered = pred - jnp.mean(pred)
        target_centered = Y_target - jnp.mean(Y_target)
        spo_grad_val = self.loss_grad_fn(pred_centered, target_centered)

        if self.mask_grad and not full_feedback:
            spo_grad_val = jnp.where(selection_mask, spo_grad_val, 0.0)

        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        return jnp.einsum('n,ni->i', spo_grad_val, theta)


class CenteredBayesianSPOAgent(_CenteredSPOMixin, BayesianSPOAgent):
    pass


class CenteredDisagreementCoverageBayesianSPOAgent(
    _CenteredSPOMixin,
    _BoundaryDisagreementCoverageMixin,
    BayesianSPOAgent,
):
    def __init__(self, *args, disagreement_samples: int = 16, disagreement_lambda: float = 1.0, seed: int = 0, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self._init_disagreement_coverage(disagreement_samples, disagreement_lambda, seed=seed)


class ActionGapBayesianSPOAgent(BayesianSPOAgent):
    """SPO variant with an update-side contrastive loss over near-boundary actions."""

    def __init__(
        self,
        *args,
        action_gap_samples: int = 16,
        action_gap_top_k: int = 3,
        action_gap_imputation_samples: int = 1,
        action_gap_imputation_scale: float = 1.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, seed=seed, **kwargs)
        self.disagreement_samples = int(action_gap_samples)
        self.action_gap_top_k = max(int(action_gap_top_k), 1)
        self.action_gap_imputation_samples = max(int(action_gap_imputation_samples), 1)
        self.action_gap_imputation_scale = float(action_gap_imputation_scale)
        self.disagreement_rng = np.random.default_rng(seed)
        self._contrastive_selection = None

    def update(self, X, selection, rewards, lr=None, full_feedback=False, **kwargs):
        self._contrastive_selection = jnp.asarray(selection)
        try:
            return super().update(X, selection, rewards, lr=lr, full_feedback=full_feedback, **kwargs)
        finally:
            self._contrastive_selection = None

    def _selection_and_candidates(self, pred_np: np.ndarray, stds_np: np.ndarray, selection_mask) -> tuple:
        selection = (
            np.asarray(self._contrastive_selection, dtype=float).reshape(-1)
            if self._contrastive_selection is not None
            else np.asarray(selection_mask, dtype=float).reshape(-1)
        )

        candidate_matrix, candidate_probs = _collect_disagreement_candidates(self, pred_np, stds_np)
        candidate_keys = [tuple((a > 0.5).astype(int).tolist()) for a in candidate_matrix]
        selection_key = tuple((selection > 0.5).astype(int).tolist())
        if selection_key not in candidate_keys:
            candidate_matrix = np.vstack([selection, candidate_matrix])
            candidate_probs = np.concatenate([[1.0], candidate_probs])
            candidate_probs /= candidate_probs.sum()
            candidate_keys = [selection_key] + candidate_keys

        sel_idx = candidate_keys.index(selection_key)
        pred_scores = candidate_matrix @ pred_np
        pred_gaps = np.abs(pred_scores - pred_scores[sel_idx])

        near_mask = np.ones(len(candidate_matrix), dtype=bool)
        near_mask[sel_idx] = False
        candidate_indices = np.where(near_mask)[0]
        ordered = candidate_indices[np.argsort(pred_gaps[candidate_indices])]
        top_indices = ordered[: self.action_gap_top_k]
        return selection, candidate_matrix, candidate_probs, sel_idx, pred_scores, pred_gaps, top_indices

    def _target_vectors(self, pred_np: np.ndarray, stds_np: np.ndarray, target_np: np.ndarray, observed_mask_np: np.ndarray):
        return [target_np]

    def _compute_spo_grad_h(self, X, h, Y_target, selection_mask, full_feedback):
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        pred_np = np.asarray(pred, dtype=float).reshape(-1)
        stds_np = np.asarray(compute_uncertainties(h, self.A_inv), dtype=float).reshape(-1)
        target_np = np.asarray(Y_target, dtype=float).reshape(-1)
        observed_mask_np = np.asarray(selection_mask, dtype=bool).reshape(-1)

        (
            selection,
            candidate_matrix,
            candidate_probs,
            sel_idx,
            pred_scores,
            pred_gaps,
            top_indices,
        ) = self._selection_and_candidates(pred_np, stds_np, selection_mask)

        if len(top_indices) == 0:
            return super()._compute_spo_grad_h(X, h, Y_target, selection_mask, full_feedback)

        positive_pred_gaps = pred_gaps[top_indices][pred_gaps[top_indices] > 1e-8]
        gap_scale = float(np.median(positive_pred_gaps)) if positive_pred_gaps.size else 1.0
        gap_scale = max(gap_scale, 1e-3)

        pair_weights = np.exp(-pred_gaps[top_indices] / gap_scale) * candidate_probs[top_indices]
        pair_weights = np.asarray(pair_weights, dtype=float)
        if float(pair_weights.sum()) <= 1e-12:
            pair_weights = np.ones_like(pair_weights) / max(len(pair_weights), 1)
        else:
            pair_weights /= pair_weights.sum()

        grad_pred = np.zeros(self.n_arms, dtype=float)
        target_vectors = self._target_vectors(pred_np, stds_np, target_np, observed_mask_np)
        for target_vec in target_vectors:
            target_scores = candidate_matrix @ target_vec
            sample_grad = np.zeros(self.n_arms, dtype=float)
            for idx, weight in zip(top_indices, pair_weights):
                diff = selection - candidate_matrix[idx]
                target_gap = float(target_scores[sel_idx] - target_scores[idx])
                if abs(target_gap) <= 1e-8:
                    continue
                label = 1.0 if target_gap > 0 else -1.0
                pred_margin = float(pred_scores[sel_idx] - pred_scores[idx])
                logistic = 1.0 / (1.0 + np.exp(label * pred_margin))
                sample_grad += weight * (-label * logistic) * diff
            grad_pred += sample_grad

        grad_pred /= max(len(target_vectors), 1)

        if np.allclose(grad_pred, 0.0):
            return super()._compute_spo_grad_h(X, h, Y_target, selection_mask, full_feedback)

        grad_pred_jnp = jnp.asarray(grad_pred)
        if self.mask_grad and not full_feedback:
            grad_pred_jnp = jnp.where(selection_mask, grad_pred_jnp, 0.0)

        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        return jnp.einsum('n,ni->i', grad_pred_jnp, theta)


class PosteriorImputedActionGapBayesianSPOAgent(ActionGapBayesianSPOAgent):
    """Action-gap SPO with posterior-sampled imputation on unobserved arms."""

    def _target_vectors(self, pred_np: np.ndarray, stds_np: np.ndarray, target_np: np.ndarray, observed_mask_np: np.ndarray):
        targets = [target_np]
        missing_mask = ~observed_mask_np
        if not missing_mask.any():
            return targets

        scale = self.action_gap_imputation_scale * stds_np
        for _ in range(self.action_gap_imputation_samples - 1):
            sampled = target_np.copy()
            sampled[missing_mask] = (
                pred_np[missing_mask]
                + scale[missing_mask] * self.disagreement_rng.standard_normal(missing_mask.sum())
            )
            targets.append(sampled)
        return targets


class ComparisonBalancedReplayBayesianSPOAgent(BayesianSPOAgent):
    """Phased SPO variant with replay weighted toward under-covered action comparisons."""

    def __init__(self, *args, comparison_samples: int = 16, seed: int = 0, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self.comparison_samples = int(comparison_samples)
        self.disagreement_samples = self.comparison_samples
        self.disagreement_rng = np.random.default_rng(seed)
        self.replay_comparison_counts = np.zeros(self.n_arms, dtype=float)

    def _comparison_profiles(self) -> np.ndarray:
        profiles = []
        for X_buf in self._phase_buffer_X:
            h = self._features(X_buf)
            means = np.asarray(
                compute_bayesian_predictions(h, self.A_inv, self.b_bayes),
                dtype=float,
            ).reshape(-1)
            stds = np.asarray(compute_uncertainties(h, self.A_inv), dtype=float).reshape(-1)
            profiles.append(_comparison_profile_from_stats(self, means, stds))
        return np.asarray(profiles, dtype=float)

    def _weighted_replay_order(self, priorities: np.ndarray) -> np.ndarray:
        available = list(range(len(priorities)))
        order = []
        while available:
            local_p = np.asarray([priorities[i] for i in available], dtype=float)
            total = float(local_p.sum())
            if total <= 1e-12:
                chosen = int(self.rng.choice(available))
            else:
                chosen = int(self.rng.choice(available, p=local_p / total))
            order.append(chosen)
            available.remove(chosen)
        return np.asarray(order, dtype=int)

    def _replay_single(self, idx: int) -> None:
        X_buf = self._phase_buffer_X[idx]
        Y_buf = self._phase_buffer_Y[idx]
        prob_buf = self._phase_buffer_prob[idx]
        obs_buf = self._phase_buffer_obs[idx]

        h = self._features(X_buf)
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

        obs_mask = jnp.asarray(obs_buf, dtype=jnp.bool_)
        if prob_buf is not None:
            Y_target = jnp.asarray(prob_buf)
        else:
            Y_target = jnp.where(obs_mask, jnp.asarray(Y_buf), pred)

        grad_h = self._compute_spo_grad_h(X_buf, h, Y_target, obs_mask, False)
        self._backprop_and_step(X_buf, grad_h)

    def _phased_retrain(self):
        n_buffer = len(self._phase_buffer_X)
        if n_buffer == 0:
            return
        batch_size = min(self.replay_batch_size, n_buffer)

        for _epoch in range(self.phase_epochs):
            profiles = self._comparison_profiles()
            deficit = 1.0 / np.sqrt(self.replay_comparison_counts + 1.0)
            priorities = np.maximum(profiles @ deficit, 1e-8)
            order = self._weighted_replay_order(priorities)

            if batch_size <= 1:
                for idx in order:
                    self._replay_single(int(idx))
                    self.replay_comparison_counts += profiles[int(idx)]
            else:
                for start in range(0, n_buffer, batch_size):
                    batch_indices = order[start:start + batch_size]
                    self._phased_retrain_batch(batch_indices)
                    self.replay_comparison_counts += profiles[batch_indices].sum(axis=0)

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            self.n_arms, self.hidden_dim, self.lambda_reg)
        for i in range(len(self._phase_buffer_X)):
            h = self._features(self._phase_buffer_X[i])
            obs_mask = jnp.asarray(self._phase_buffer_obs[i], dtype=jnp.bool_)
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
                self.A, self.b_bayes, self.A_inv, h, rewards, obs_mask)


PAIR_SWAP_PERM = np.array([2, 3, 0, 1], dtype=int)


def swap_pair_context(x: np.ndarray) -> np.ndarray:
    """Swap the two decision coordinates x0 <-> x1, leave nuisance dims unchanged."""
    x = np.asarray(x, dtype=float).copy()
    x[[0, 1]] = x[[1, 0]]
    return x


def swap_pair_items(values: np.ndarray) -> np.ndarray:
    """Swap pair (0,1) with pair (2,3) in item-indexed arrays."""
    values = np.asarray(values)
    return values[PAIR_SWAP_PERM].copy()


def train_with_checkpoints(
    agent,
    episode: dict,
    n_rounds: int,
    checkpoint_interval: int = 50,
    full_feedback: bool = False,
    pair_swap_augment: bool = False,
    eval_contexts: Optional[np.ndarray] = None,
    eval_values: Optional[np.ndarray] = None,
) -> Dict[str, list]:
    """Train an agent on a pregenerated episode, saving W1/b1 checkpoints.

    Returns dict with keys:
    - 'W1_checkpoints': list of (round, W1_array, b1_array) tuples
    - 'cumulative_regret': list of floats
    - 'regrets': list of floats
    """
    rounds_data = episode['rounds_data']
    W1_checkpoints = [(0, np.array(agent.W1), np.array(agent.b1))]
    regrets = []
    cumulative_regret = []
    actions = []
    total_regret = 0.0
    prediction_error_checkpoints = []

    def _prediction_mse() -> float:
        preds = []
        for x in eval_contexts:
            preds.append(np.asarray(agent.predict_mean(x), dtype=float))
        pred_matrix = np.vstack(preds)
        return float(np.mean((pred_matrix - eval_values) ** 2))

    if eval_contexts is not None and eval_values is not None:
        prediction_error_checkpoints.append((0, _prediction_mse()))

    for t in range(n_rounds):
        rd = rounds_data[t]
        X = rd['context']
        true_vals = rd['true_values']
        optimal_val = rd['optimal_value']

        selection = agent.act(X)
        selection = np.asarray(selection).flatten()
        actions.append(selection.copy())
        agent_val = float(np.sum(true_vals * selection))
        instant_regret = optimal_val - agent_val
        total_regret += instant_regret
        regrets.append(instant_regret)
        cumulative_regret.append(total_regret)

        if full_feedback:
            observed = true_vals.copy()
        else:
            observed = np.zeros_like(true_vals)
            mask = selection > 0.5
            observed[mask] = true_vals[mask]

        agent.update(X, selection, observed, full_feedback=full_feedback)

        if pair_swap_augment:
            X_swapped = swap_pair_context(X)
            selection_swapped = swap_pair_items(selection)
            observed_swapped = swap_pair_items(observed)
            agent.update(
                X_swapped,
                selection_swapped,
                observed_swapped,
                full_feedback=full_feedback,
            )

        if (t + 1) % checkpoint_interval == 0:
            W1_checkpoints.append((t + 1, np.array(agent.W1), np.array(agent.b1)))
            if eval_contexts is not None and eval_values is not None:
                prediction_error_checkpoints.append((t + 1, _prediction_mse()))

    return {
        'W1_checkpoints': W1_checkpoints,
        'cumulative_regret': cumulative_regret,
        'regrets': regrets,
        'actions': actions,
        'prediction_error_checkpoints': prediction_error_checkpoints,
    }


def extract_features(agent, X_test: np.ndarray) -> np.ndarray:
    """Extract features h = X @ W1 + b1 for a batch of test contexts.

    Args:
        agent: BayesianSPOAgent or BayesianMSEAgent
        X_test: (n_samples, context_dim) array

    Returns:
        (n_samples, hidden_dim) array of features
    """
    features = []
    for x in X_test:
        h = agent._features(x)
        features.append(np.array(h))
    return np.stack(features)


def generate_test_contexts(env, n_samples: int = 500, seed: int = 99) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test contexts with oracle decisions and true values.

    Returns:
        X_test: (n_samples, context_dim)
        decisions: (n_samples, n_arms) binary optimal selections
        values: (n_samples, n_arms) true cost values
    """
    rng = np.random.default_rng(seed)
    X_test = rng.standard_normal((n_samples, env.context_dim))
    decisions = []
    values = []
    for x in X_test:
        v = env.data._generate_values(x)
        d = env.solve(v)
        decisions.append(d)
        values.append(v)
    return X_test, np.array(decisions), np.array(values)


# ── Task 3: Subspace Alignment Metric ─────────────────────────────────────


def subspace_angle(W_learned: np.ndarray, W_oracle: np.ndarray = None) -> float:
    """Compute the principal angle (degrees) between two 2D subspaces.

    Both W_learned and W_oracle are (d, 2) matrices whose columns span the subspace.
    Returns angle in [0, 90] degrees. 0 = perfectly aligned, 90 = orthogonal.

    If W_oracle is None, uses ORACLE_W1 (projection onto dims 0-1).
    """
    if W_oracle is None:
        W_oracle = ORACLE_W1

    Q1, _ = np.linalg.qr(W_learned)
    Q2, _ = np.linalg.qr(W_oracle)

    M = Q1.T @ Q2
    singular_values = np.linalg.svd(M, compute_uv=False)

    cos_angles = np.clip(singular_values, -1.0, 1.0)
    angles_rad = np.arccos(cos_angles)

    return float(np.degrees(np.max(angles_rad)))


def alignment_over_training(
    checkpoints,
) -> Tuple[List[int], List[float]]:
    """Compute subspace alignment angle at each checkpoint.

    Args:
        checkpoints: list of (round, W1, b1) from train_with_checkpoints

    Returns:
        rounds: list of round numbers
        angles: list of alignment angles in degrees
    """
    rounds = [ckpt[0] for ckpt in checkpoints]
    angles = [subspace_angle(ckpt[1]) for ckpt in checkpoints]
    return rounds, angles


def features_at_checkpoint(W1, b1, X_test: np.ndarray) -> np.ndarray:
    """Compute features h = X @ W1 + b1 using a saved checkpoint."""
    return X_test @ W1 + b1


# ── Trajectory (Sensitivity) Analysis ─────────────────────────────────────


def compute_trajectory_sensitivity(
    agent,
    dim_vary: int,
    n_steps: int = 50,
    n_traces: int = 4,
    sweep_range: Tuple[float, float] = (-2.5, 2.5),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep one context dimension while holding the rest fixed.

    Generates n_traces random base contexts, then varies dim_vary across
    n_steps values. Returns feature trajectories.

    Returns:
        sweep_vals: (n_steps,) the swept values
        trajectories: (n_traces, n_steps, hidden_dim) feature trajectories
    """
    rng = np.random.default_rng(seed)
    sweep_vals = np.linspace(sweep_range[0], sweep_range[1], n_steps)
    trajectories = []

    for _ in range(n_traces):
        base = rng.standard_normal(5)
        trace = []
        for v in sweep_vals:
            x = base.copy()
            x[dim_vary] = v
            h = np.array(agent._features(x))
            trace.append(h)
        trajectories.append(np.stack(trace))

    return sweep_vals, np.array(trajectories)


def trajectory_sensitivity_score(trajectories: np.ndarray) -> float:
    """Mean L2 displacement per step across all traces."""
    # trajectories: (n_traces, n_steps, hidden_dim)
    diffs = np.diff(trajectories, axis=1)  # (n_traces, n_steps-1, hidden_dim)
    step_norms = np.linalg.norm(diffs, axis=2)  # (n_traces, n_steps-1)
    return float(np.mean(step_norms))


def plot_trajectories(
    agent_spo,
    agent_mse,
    seed: int = 0,
) -> 'ggplot':
    """2x2 grid: rows = decision dim (x₀) / nuisance dim (x₂), cols = SPO / MSE.

    Each panel shows feature-space trajectories as lines. If the agent is sensitive
    to that dimension, lines spread across the feature space. If insensitive,
    lines are flat/short.

    Annotates each panel with mean sensitivity (L2 displacement per step).
    """
    from lets_plot import geom_path, geom_text, gggrid

    panels = []
    for dim_idx, dim_name in [(0, 'x₀ (decision)'), (2, 'x₂ (nuisance)')]:
        for agent, agent_name, color in [
            (agent_spo, 'BayesianSPO', COLORS['BayesianSPO']),
            (agent_mse, 'BayesianMSE', COLORS['BayesianMSE']),
        ]:
            sweep, trajs = compute_trajectory_sensitivity(
                agent, dim_vary=dim_idx, n_steps=50, n_traces=6, seed=seed)
            sens = trajectory_sensitivity_score(trajs)

            rows = []
            for t in range(trajs.shape[0]):
                for s in range(trajs.shape[1]):
                    rows.append({
                        'h1': float(trajs[t, s, 0]),
                        'h2': float(trajs[t, s, 1]),
                        'trace': str(t),
                    })
            df = pd.DataFrame(rows)

            # Annotation
            x_ann = float(df['h1'].min() + 0.02 * (df['h1'].max() - df['h1'].min()))
            y_ann = float(df['h2'].max() - 0.02 * (df['h2'].max() - df['h2'].min()))
            df_ann = pd.DataFrame([{'h1': x_ann, 'h2': y_ann,
                                    'text': f'sensitivity = {sens:.3f}'}])

            p = (
                ggplot(df, aes('h1', 'h2'))
                + geom_path(aes(group='trace'), color=color, size=0.8, alpha=0.6)
                + geom_point(size=0.5, color=color, alpha=0.3)
                + geom_text(aes('h1', 'h2', label='text'), data=df_ann,
                            color='black', size=9, hjust=0, fontface='bold')
                + labs(x='Feature dim 1', y='Feature dim 2',
                       title=f'{agent_name} — vary {dim_name}')
                + theme_minimal()
                + ggsize(350, 300)
            )
            panels.append(p)

    return gggrid(panels, ncol=2)


# ── Decoded Latent Analysis ────────────────────────────────────────────────


def decode_latents(features: np.ndarray, X_test: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit linear probes to predict each context dimension from features.

    Returns dict mapping latent name to predicted values.
    """
    from sklearn.linear_model import LinearRegression
    predictions = {}
    latent_names = ['x₀ (decision)', 'x₁ (decision)',
                    'x₂ (nuisance)', 'x₃ (nuisance)', 'x₄ (nuisance)']
    for i, name in enumerate(latent_names):
        reg = LinearRegression().fit(features, X_test[:, i])
        predictions[name] = reg.predict(features)
    return predictions


def plot_decoded_latents(
    features_spo: np.ndarray,
    features_mse: np.ndarray,
    X_test: np.ndarray,
    decisions: np.ndarray,
) -> 'ggplot':
    """Scatter: true latent vs decoded (predicted) latent from features.

    2x2 grid: rows = decision latent x₀ / nuisance latent x₂,
    cols = SPO / MSE. Diagonal = perfect recovery.
    """
    from lets_plot import (
        geom_abline, gggrid, scale_color_manual,
    )

    dec_labels = [decision_label(d) for d in decisions]
    decoded_spo = decode_latents(features_spo, X_test)
    decoded_mse = decode_latents(features_mse, X_test)

    panels = []
    for latent_idx, latent_name, short, dim_type in [
        (0, 'x₀ (decision)', 'x₀', 'DECISION'),
        (2, 'x₂ (nuisance)', 'x₂', 'NUISANCE'),
    ]:
        for agent_name, decoded in [('BayesianSPO', decoded_spo), ('BayesianMSE', decoded_mse)]:
            true_vals = X_test[:, latent_idx]
            pred_vals = decoded[latent_name]
            r2 = float(1 - np.sum((true_vals - pred_vals)**2) / np.sum((true_vals - true_vals.mean())**2))

            rows = [{'true': float(true_vals[i]), 'predicted': float(pred_vals[i]),
                      'decision': dec_labels[i]}
                    for i in range(len(true_vals))]
            df = pd.DataFrame(rows)

            p = (
                ggplot(df, aes('true', 'predicted', color='decision'))
                + geom_point(size=1.5, alpha=0.4)
                + geom_abline(slope=1, intercept=0, linetype='dashed', color='gray', size=0.8)
                + scale_color_manual(values=DECISION_COLORS)
                + labs(
                    x=f'True {short}',
                    y=f'Predicted {short}',
                    title=f'[{dim_type}] {agent_name} → {short} (R²={r2:.2f})',
                )
                + theme_minimal()
                + ggsize(350, 300)
            )
            panels.append(p)

    return gggrid(panels, ncol=2)


# ── Task 4: Plotting Functions (Tiny Problem) ─────────────────────────────


COLORS = {
    'BayesianSPO': '#1f77b4',
    'BayesianMSE': '#d62728',
    'Oracle': '#2ca02c',
}

DECISION_COLORS = {
    '0,2': '#1f77b4',
    '0,3': '#ff7f0e',
    '1,2': '#2ca02c',
    '1,3': '#d62728',
}


def _convex_hull_df(h1, h2, group_label):
    """Compute convex hull vertices for a set of 2D points."""
    from scipy.spatial import ConvexHull
    points = np.column_stack([h1, h2])
    if len(points) < 3:
        return []
    try:
        hull = ConvexHull(points)
        verts = hull.vertices.tolist()
        verts.append(verts[0])  # close the polygon
        return [{'h1': float(points[v, 0]), 'h2': float(points[v, 1]),
                 'decision': group_label}
                for v in verts]
    except Exception:
        return []


def _decision_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    """Linear classification accuracy for predicting decision from features."""
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    scores = cross_val_score(clf, features, labels, cv=5, scoring='accuracy')
    return float(np.mean(scores))


def _cost_correlation(features: np.ndarray, total_cost: np.ndarray) -> float:
    """Max absolute Pearson correlation between any feature dim and total cost."""
    max_corr = 0.0
    for j in range(features.shape[1]):
        r = np.corrcoef(features[:, j], total_cost)[0, 1]
        if np.isfinite(r):
            max_corr = max(max_corr, abs(r))
    return max_corr


def plot_feature_grid(
    features_spo: np.ndarray,
    features_mse: np.ndarray,
    X_test: np.ndarray,
    decisions: np.ndarray,
    values: np.ndarray,
) -> 'ggplot':
    """Main 2x3 figure: Oracle | SPO | MSE, colored by decision (row 1) and cost (row 2).

    Row 1: colored by decision with convex hulls + classification accuracy.
    Row 2: colored by total cost with cost-correlation annotation.
    """
    from lets_plot import (
        scale_color_gradient, gggrid, geom_vline, geom_text, geom_path,
    )

    oracle_h = X_test[:, :2]
    total_cost = values.sum(axis=1)

    dec_labels = np.array([decision_label(d) for d in decisions])

    # --- Build row 1: colored by decision + convex hulls + accuracy ---
    panels_row1 = []
    for label, h in [('Oracle (x₀, x₁)', oracle_h),
                      ('BayesianSPO', features_spo),
                      ('BayesianMSE', features_mse)]:
        acc = _decision_accuracy(h, dec_labels)

        rows = []
        for i in range(len(decisions)):
            dec = decision_label(decisions[i])
            rows.append({'h1': float(h[i, 0]), 'h2': float(h[i, 1]),
                         'decision': dec})
        df = pd.DataFrame(rows)

        # Convex hulls per decision
        hull_rows = []
        for dec in DECISION_COLORS:
            mask = df['decision'] == dec
            if mask.sum() >= 3:
                hull_rows.extend(_convex_hull_df(
                    df.loc[mask, 'h1'].values,
                    df.loc[mask, 'h2'].values,
                    dec))
        df_hull = pd.DataFrame(hull_rows) if hull_rows else None

        # Annotation
        x_ann = float(df['h1'].min() + 0.02 * (df['h1'].max() - df['h1'].min()))
        y_ann = float(df['h2'].max() - 0.02 * (df['h2'].max() - df['h2'].min()))
        df_ann = pd.DataFrame([{'h1': x_ann, 'h2': y_ann,
                                'text': f'accuracy = {acc:.0%}'}])

        p = (
            ggplot(df, aes('h1', 'h2', color='decision'))
            + geom_point(size=2.0, alpha=0.4)
            + scale_color_manual(values=DECISION_COLORS)
            + geom_text(aes('h1', 'h2', label='text'), data=df_ann,
                        color='black', size=9, hjust=0, fontface='bold')
            + labs(x='Feature dim 1', y='Feature dim 2', title=label)
            + theme_minimal()
            + ggsize(350, 320)
        )
        if df_hull is not None and len(df_hull) > 0:
            p = p + geom_path(aes('h1', 'h2', color='decision', group='decision'),
                              data=df_hull, size=1.0, alpha=0.7)
        # Add crosshair on oracle panel
        if label.startswith('Oracle'):
            p = (p
                 + geom_vline(xintercept=0, linetype='dashed', color='gray', size=0.8)
                 + geom_hline(yintercept=0, linetype='dashed', color='gray', size=0.8))
        panels_row1.append(p)

    # --- Build row 2: colored by cost magnitude + correlation ---
    panels_row2 = []
    for label, h in [('Oracle (x₀, x₁)', oracle_h),
                      ('BayesianSPO', features_spo),
                      ('BayesianMSE', features_mse)]:
        corr = _cost_correlation(h, total_cost)

        rows = []
        for i in range(len(decisions)):
            rows.append({'h1': float(h[i, 0]), 'h2': float(h[i, 1]),
                         'cost': float(total_cost[i])})
        df = pd.DataFrame(rows)

        x_ann = float(df['h1'].min() + 0.02 * (df['h1'].max() - df['h1'].min()))
        y_ann = float(df['h2'].max() - 0.02 * (df['h2'].max() - df['h2'].min()))
        df_ann = pd.DataFrame([{'h1': x_ann, 'h2': y_ann,
                                'text': f'|r| = {corr:.2f}'}])

        p = (
            ggplot(df, aes('h1', 'h2', color='cost'))
            + geom_point(size=2.0, alpha=0.6)
            + scale_color_gradient(low='#2166ac', high='#b2182b')
            + geom_text(aes('h1', 'h2', label='text'), data=df_ann,
                        color='black', size=9, hjust=0, fontface='bold')
            + labs(x='Feature dim 1', y='Feature dim 2', title=label)
            + theme_minimal()
            + ggsize(350, 320)
        )
        panels_row2.append(p)

    return gggrid(panels_row1 + panels_row2, ncol=3)


def _rolling_mean(x, window=50):
    """Simple rolling average for smoothing per-round regret."""
    cs = np.cumsum(np.insert(x, 0, 0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = (cs[i + 1] - cs[lo]) / (i - lo + 1)
    return out


def create_feature_evolution_animation(
    checkpoints_spo,
    checkpoints_mse,
    X_test: np.ndarray,
    decisions: np.ndarray,
    regrets_spo: List[float],
    regrets_mse: List[float],
    interval_ms: int = 500,
    smooth_window: int = 50,
):
    """Create animation: feature scatter (top) + per-instance regret (bottom).

    Top row: Oracle | SPO | MSE scatter with convex hulls.
    Bottom row: three aligned regret subplots (Oracle=0, SPO, MSE) revealing
    only past rounds — future regret is hidden.

    Returns an IPython HTML object for inline display.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    from scipy.spatial import ConvexHull

    dec_labels = np.array([decision_label(d) for d in decisions])
    unique_decisions = sorted(set(dec_labels))
    color_map = {d: c for d, c in DECISION_COLORS.items()}

    oracle_h = X_test[:, :2]
    n_frames = min(len(checkpoints_spo), len(checkpoints_mse))
    n_rounds = len(regrets_spo)

    # Pre-compute smoothed per-instance regret
    reg_spo = np.array(regrets_spo)
    reg_mse = np.array(regrets_mse)
    smooth_spo = _rolling_mean(reg_spo, smooth_window)
    smooth_mse = _rolling_mean(reg_mse, smooth_window)
    rounds_all = np.arange(1, n_rounds + 1)

    # Fixed y-limit for regret (shared across all 3 panels)
    reg_ymax = max(smooth_spo.max(), smooth_mse.max()) * 1.1

    # Fixed axis limits from oracle
    pad = 0.3
    ox_min, ox_max = oracle_h[:, 0].min() - pad, oracle_h[:, 0].max() + pad
    oy_min, oy_max = oracle_h[:, 1].min() - pad, oracle_h[:, 1].max() + pad

    # Layout: 2 rows x 3 cols
    fig = plt.figure(figsize=(14, 7.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.4, 1], hspace=0.35, wspace=0.3,
                           top=0.90, bottom=0.08, left=0.06, right=0.96)
    ax_scatter = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_regret = [fig.add_subplot(gs[1, i]) for i in range(3)]

    def draw_frame(frame_idx):
        for ax in ax_scatter + ax_regret:
            ax.clear()

        rnd = checkpoints_spo[frame_idx][0]
        W1_spo, b1_spo = checkpoints_spo[frame_idx][1], checkpoints_spo[frame_idx][2]
        W1_mse, b1_mse = checkpoints_mse[frame_idx][1], checkpoints_mse[frame_idx][2]

        h_spo = features_at_checkpoint(W1_spo, b1_spo, X_test)
        h_mse = features_at_checkpoint(W1_mse, b1_mse, X_test)

        # --- Top row: scatter plots ---
        for ax, h, title in [
            (ax_scatter[0], oracle_h, 'Oracle (x₀, x₁)'),
            (ax_scatter[1], h_spo, 'BayesianSPO'),
            (ax_scatter[2], h_mse, 'BayesianMSE'),
        ]:
            for dec in unique_decisions:
                mask = dec_labels == dec
                c = color_map.get(dec, '#999999')
                pts = h[mask]
                ax.scatter(pts[:, 0], pts[:, 1], c=c, s=12, alpha=0.4, label=dec)

                if pts.shape[0] >= 3:
                    try:
                        hull = ConvexHull(pts)
                        verts = np.append(hull.vertices, hull.vertices[0])
                        ax.plot(pts[verts, 0], pts[verts, 1], c=c, lw=1.2, alpha=0.7)
                    except Exception:
                        pass

            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Feature dim 1', fontsize=9)
            ax.set_ylabel('Feature dim 2', fontsize=9)
            ax.tick_params(labelsize=8)

            if title.startswith('Oracle'):
                ax.set_xlim(ox_min, ox_max)
                ax.set_ylim(oy_min, oy_max)
                ax.axhline(0, ls='--', c='gray', lw=0.8)
                ax.axvline(0, ls='--', c='gray', lw=0.8)

        ax_scatter[2].legend(title='decision', fontsize=7, title_fontsize=8,
                             loc='upper right', markerscale=1.5)

        # --- Bottom row: per-instance regret (only past rounds shown) ---
        end = max(rnd, 1)
        for ax, smooth, color, title in [
            (ax_regret[0], np.zeros(n_rounds), COLORS['Oracle'], 'Oracle regret'),
            (ax_regret[1], smooth_spo, COLORS['BayesianSPO'], 'SPO regret'),
            (ax_regret[2], smooth_mse, COLORS['BayesianMSE'], 'MSE regret'),
        ]:
            ax.plot(rounds_all[:end], smooth[:end], color=color, lw=1.8)
            ax.set_xlim(0, n_rounds)
            ax.set_ylim(-0.02 * reg_ymax, reg_ymax)
            ax.set_xlabel('Round', fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=8)
        ax_regret[0].set_ylabel('Instant regret\n(smoothed)', fontsize=9)

        fig.suptitle(f'Feature Evolution — Round {rnd}', fontsize=13, fontweight='bold')

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=interval_ms, repeat=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_feature_scatter(
    features_spo: np.ndarray,
    features_mse: np.ndarray,
    decisions: np.ndarray,
    title_suffix: str = "",
) -> 'ggplot':
    """2-panel scatter: SPO features (left) vs MSE features (right), colored by decision."""
    rows = []
    for i in range(len(decisions)):
        label = decision_label(decisions[i])
        rows.append({
            'h1': float(features_spo[i, 0]),
            'h2': float(features_spo[i, 1]),
            'decision': label,
            'agent': 'BayesianSPO',
        })
        rows.append({
            'h1': float(features_mse[i, 0]),
            'h2': float(features_mse[i, 1]),
            'decision': label,
            'agent': 'BayesianMSE',
        })
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('h1', 'h2', color='decision'))
        + geom_point(size=2, alpha=0.6)
        + facet_wrap('agent')
        + scale_color_manual(values=DECISION_COLORS)
        + coord_fixed()
        + labs(
            x='Feature dim 1',
            y='Feature dim 2',
            color='Selection',
            title=f'Learned Features by Decision Region{title_suffix}',
        )
        + theme_minimal()
        + ggsize(800, 400)
    )
    return p


def plot_alignment_trajectory(
    rounds_spo: List[int],
    angles_spo: List[float],
    rounds_mse: List[int],
    angles_mse: List[float],
) -> 'ggplot':
    """Line plot of subspace alignment angle over training rounds."""
    rows = []
    for r, a in zip(rounds_spo, angles_spo):
        rows.append({'round': r, 'angle': a, 'agent': 'BayesianSPO'})
    for r, a in zip(rounds_mse, angles_mse):
        rows.append({'round': r, 'angle': a, 'agent': 'BayesianMSE'})
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('round', 'angle', color='agent'))
        + geom_line(size=1.5)
        + geom_point(size=3)
        + scale_color_manual(values={'BayesianSPO': COLORS['BayesianSPO'], 'BayesianMSE': COLORS['BayesianMSE']})
        + labs(
            x='Training Round',
            y='Subspace Angle to Oracle (degrees)',
            color='Agent',
            title='Feature Subspace Alignment During Training',
        )
        + geom_hline(yintercept=0, linetype='dashed', color='gray')
        + theme_minimal()
        + ggsize(600, 400)
    )
    return p


def plot_feature_directions(
    W1_spo: np.ndarray,
    W1_mse: np.ndarray,
    dim_names: Optional[List[str]] = None,
) -> 'ggplot':
    """Stacked horizontal bars: one normalized bar per agent.

    Each segment is the row-norm share for one input dimension:
        ||W1[i, :]|| / sum_j ||W1[j, :]||

    Blue shades = decision dimensions, red shades = nuisance dimensions.
    This isolates allocation across dimensions instead of mixing allocation with
    the overall scale of W1.
    """
    from lets_plot import coord_flip, scale_x_discrete

    if dim_names is None:
        dim_names = ['x₀', 'x₁', 'x₂', 'x₃', 'x₄']

    # Dim type: first 2 = decision, rest = nuisance
    dim_colors = {
        'x₀ (decision)': '#2166ac',
        'x₁ (decision)': '#4393c3',
        'x₂ (nuisance)': '#d6604d',
        'x₃ (nuisance)': '#f4a582',
        'x₄ (nuisance)': '#fddbc7',
    }
    dim_labels = [
        'x₀ (decision)', 'x₁ (decision)',
        'x₂ (nuisance)', 'x₃ (nuisance)', 'x₄ (nuisance)',
    ]

    rows = []
    for agent_name, W1 in [('BayesianSPO', W1_spo), ('BayesianMSE', W1_mse)]:
        row_norms = np.linalg.norm(W1, axis=1)
        total_norm = float(np.sum(row_norms))
        if total_norm <= 1e-12:
            shares = np.zeros_like(row_norms)
        else:
            shares = row_norms / total_norm
        for i, dl in enumerate(dim_labels):
            rows.append({'agent': agent_name, 'dimension': dl, 'weight_share': float(shares[i])})
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('agent', 'weight_share', fill='dimension'))
        + geom_bar(stat='identity', position='stack')
        + coord_flip()
        + scale_fill_manual(values=dim_colors)
        + scale_x_discrete(limits=['BayesianMSE', 'BayesianSPO'])
        + labs(
            x='',
            y='Normalized Feature Weight Share',
            fill='Context Dimension',
            title='Where does each agent allocate feature capacity?',
        )
        + theme_minimal()
        + ggsize(300, 125)
    )
    return p


# ── Regime Experiment Runner & Diagnostics ────────────────────────────────


def run_regime(
    env,
    episode: dict,
    X_test: np.ndarray,
    decisions_test: np.ndarray,
    values_test: np.ndarray,
    n_rounds: int = 5000,
    checkpoint_interval: int = 200,
    full_feedback: bool = True,
    label: str = 'Full',
    **agent_kwargs,
) -> Dict:
    """Run a complete regime experiment and return all results.

    Returns dict with keys: spo_agent, mse_agent, res_spo, res_mse,
    h_spo, h_mse, metrics (dict per agent), label.
    """
    spo, mse = create_agent_pair(env, **agent_kwargs)
    res_spo = train_with_checkpoints(
        spo, episode, n_rounds,
        checkpoint_interval=checkpoint_interval, full_feedback=full_feedback)
    _, mse = create_agent_pair(env, **agent_kwargs)
    res_mse = train_with_checkpoints(
        mse, episode, n_rounds,
        checkpoint_interval=checkpoint_interval, full_feedback=full_feedback)

    h_spo = extract_features(spo, X_test)
    h_mse = extract_features(mse, X_test)

    dec_labels = np.array([decision_label(d) for d in decisions_test])
    total_cost = values_test.sum(axis=1)

    metrics = {}
    for name, h, agent, res in [
        ('SPO', h_spo, spo, res_spo),
        ('MSE', h_mse, mse, res_mse),
    ]:
        metrics[name] = {
            'Decision acc.': f"{_decision_accuracy(h, dec_labels):.0%}",
            'Cost |r|': f"{_cost_correlation(h, total_cost):.2f}",
            'Alignment': f"{subspace_angle(np.array(agent.W1)):.1f}\u00b0",
            'Regret': f"{res['cumulative_regret'][-1]:.0f}",
        }

    return {
        'spo_agent': spo, 'mse_agent': mse,
        'res_spo': res_spo, 'res_mse': res_mse,
        'h_spo': h_spo, 'h_mse': h_mse,
        'metrics': metrics, 'label': label,
    }


def build_regime_comparison_table(regimes: List[Dict]) -> pd.DataFrame:
    """Build a consolidated comparison table from multiple regime results.

    Args:
        regimes: list of dicts from run_regime()

    Returns:
        DataFrame with rows = (regime, agent) and columns = metrics
    """
    rows = {}
    for regime in regimes:
        label = regime['label']
        for agent_name in ['SPO', 'MSE']:
            key = f"{agent_name} ({label})"
            rows[key] = regime['metrics'][agent_name]
    df = pd.DataFrame(rows).T
    df.index.name = 'Agent (regime)'
    return df


def plot_action_timeline(
    res_spo: Dict,
    res_mse: Dict,
    episode: dict,
    n_show: int = 500,
) -> 'ggplot':
    """Heatmap showing action correctness over rounds for both agents.

    Green = correct action, red = wrong action.
    """
    from lets_plot import geom_tile, scale_fill_manual as sfm, gggrid

    rounds_data = episode['rounds_data']
    n = min(n_show, len(res_spo['regrets']))

    panels = []
    for agent_name, res in [('BayesianSPO', res_spo), ('BayesianMSE', res_mse)]:
        rows = []
        for t in range(n):
            correct = float(res['regrets'][t]) < 1e-6
            rows.append({'round': t, 'correct': 'correct' if correct else 'wrong'})
        df = pd.DataFrame(rows)

        p = (
            ggplot(df, aes('round', fill='correct'))
            + geom_bar(stat='count', width=1.0)
            + sfm(values={'correct': '#2ca02c', 'wrong': '#d62728'})
            + labs(x='Round', y='', fill='', title=f'{agent_name} — action correctness')
            + theme_minimal()
            + ggsize(700, 100)
        )
        panels.append(p)
    return gggrid(panels, ncol=1)


def compute_prediction_errors(
    agent,
    episode: dict,
    checkpoints,
    X_test: np.ndarray,
    values_test: np.ndarray,
) -> Tuple[List[int], List[float]]:
    """Compute mean prediction MSE at each checkpoint.

    Uses the agent's current Bayesian predictions to compute MSE
    against true test values at each checkpoint's W1/b1.
    """
    from src.agents.bayesian_utils import compute_bayesian_predictions
    from src.agents.bayesian_utils import init_per_arm_regression, update_bayesian_stats

    rounds = []
    mses = []
    for ckpt in checkpoints:
        rnd, W1, b1 = ckpt[0], ckpt[1], ckpt[2]
        # Compute features at this checkpoint
        H_test = X_test @ W1 + b1
        # We can't reconstruct the Bayesian state at each checkpoint,
        # so use a simple linear regression proxy: fit theta from features
        # This gives an indication of how well the feature space supports prediction
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(
            np.array(H_test), np.array(values_test))
        preds = reg.predict(np.array(H_test))
        mse = float(np.mean((preds - values_test) ** 2))
        rounds.append(rnd)
        mses.append(mse)
    return rounds, mses


def plot_prediction_error_trajectory(
    checkpoints_spo,
    checkpoints_mse,
    X_test: np.ndarray,
    values_test: np.ndarray,
) -> 'ggplot':
    """Plot prediction MSE over training for both agents.

    Shows that MSE features become better predictors while SPO features don't.
    """
    rounds_spo, mses_spo = compute_prediction_errors(
        None, None, checkpoints_spo, X_test, values_test)
    rounds_mse, mses_mse = compute_prediction_errors(
        None, None, checkpoints_mse, X_test, values_test)

    rows = []
    for r, m in zip(rounds_spo, mses_spo):
        rows.append({'round': r, 'mse': m, 'agent': 'BayesianSPO'})
    for r, m in zip(rounds_mse, mses_mse):
        rows.append({'round': r, 'mse': m, 'agent': 'BayesianMSE'})
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('round', 'mse', color='agent'))
        + geom_line(size=1.5)
        + geom_point(size=3)
        + scale_color_manual(values={
            'BayesianSPO': COLORS['BayesianSPO'],
            'BayesianMSE': COLORS['BayesianMSE']})
        + labs(
            x='Training Round',
            y='Prediction MSE (linear probe on features)',
            color='Agent',
            title='Prediction accuracy over training',
        )
        + theme_minimal()
        + ggsize(600, 350)
    )
    return p


# ── Task 5: Realistic Problem (knapsack_32) Utilities ─────────────────────


def compute_effective_dimensionality(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """PCA on features, return explained variance ratios and cumulative ratios."""
    pca = PCA()
    pca.fit(features)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    return explained, cumulative


def project_features_2d(features: np.ndarray) -> Tuple[np.ndarray, 'PCA']:
    """Project features to 2D via PCA."""
    pca = PCA(n_components=2)
    projected = pca.fit_transform(features)
    return projected, pca


def compute_contested_labels(
    values: np.ndarray,
    solver,
    eps: float = 0.5,
) -> np.ndarray:
    """Label items as contested for each test context.

    Item i is contested if solving knapsack(values + eps * e_i) gives a
    different solution than solving knapsack(values).
    """
    n_samples, n_arms = values.shape
    contested = np.zeros((n_samples, n_arms), dtype=bool)

    for i in range(n_samples):
        base_sol = solver(values[i])
        for j in range(n_arms):
            perturbed = values[i].copy()
            perturbed[j] += eps
            new_sol = solver(perturbed)
            if not np.array_equal(base_sol.astype(int), new_sol.astype(int)):
                contested[i, j] = True
    return contested


def contested_probe_accuracy(
    features: np.ndarray,
    contested: np.ndarray,
) -> float:
    """Train linear probe to predict contestedness from features.

    Pools all (context, item) pairs. Returns mean cross-validated accuracy.
    """
    n_samples, n_arms = contested.shape
    X_probe = np.tile(features, (n_arms, 1))
    arm_ids = np.repeat(np.eye(n_arms), n_samples, axis=0)
    X_probe = np.hstack([X_probe, arm_ids])
    y_probe = contested.T.flatten().astype(int)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    scores = cross_val_score(clf, X_probe, y_probe, cv=5, scoring='accuracy')
    return float(np.mean(scores))


def plot_pca_variance(
    cumulative_spo: np.ndarray,
    cumulative_mse: np.ndarray,
) -> 'ggplot':
    """Plot cumulative explained variance for SPO vs MSE features."""
    rows = []
    for i, val in enumerate(cumulative_spo):
        rows.append({'component': i + 1, 'cumulative_variance': float(val), 'agent': 'BayesianSPO'})
    for i, val in enumerate(cumulative_mse):
        rows.append({'component': i + 1, 'cumulative_variance': float(val), 'agent': 'BayesianMSE'})
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('component', 'cumulative_variance', color='agent'))
        + geom_line(size=1.5)
        + geom_point(size=3)
        + geom_hline(yintercept=0.95, linetype='dashed', color='gray')
        + scale_color_manual(values={'BayesianSPO': COLORS['BayesianSPO'], 'BayesianMSE': COLORS['BayesianMSE']})
        + labs(
            x='Principal Component',
            y='Cumulative Explained Variance',
            color='Agent',
            title='Feature Effective Dimensionality (knapsack_32)',
        )
        + theme_minimal()
        + ggsize(600, 400)
    )
    return p


def plot_decision_clusters(
    projected_spo: np.ndarray,
    projected_mse: np.ndarray,
    decision_labels: List[str],
) -> 'ggplot':
    """2-panel PCA scatter colored by decision identity for knapsack_32."""
    rows = []
    counts = Counter(decision_labels)
    top_decisions = {d for d, _ in counts.most_common(6)}

    for i, label in enumerate(decision_labels):
        display_label = label if label in top_decisions else 'other'
        rows.append({
            'pc1': float(projected_spo[i, 0]),
            'pc2': float(projected_spo[i, 1]),
            'decision': display_label,
            'agent': 'BayesianSPO',
        })
        rows.append({
            'pc1': float(projected_mse[i, 0]),
            'pc2': float(projected_mse[i, 1]),
            'decision': display_label,
            'agent': 'BayesianMSE',
        })
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('pc1', 'pc2', color=as_discrete('decision')))
        + geom_point(size=2, alpha=0.5)
        + facet_wrap('agent')
        + labs(
            x='PC 1',
            y='PC 2',
            color='Decision',
            title='Feature Clustering by Decision (knapsack_32)',
        )
        + theme_minimal()
        + ggsize(800, 400)
    )
    return p


# --- Consolidated plotting and analysis helpers ---

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from lets_plot import (
    aes,
    geom_line,
    geom_path,
    geom_point,
    geom_ribbon,
    geom_segment,
    geom_tile,
    gggrid,
    ggplot,
    ggsize,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
    geom_abline,
)
from lets_plot.export import ggsave

from src.agents.base import run_agent_on_episode
from src.agents.contextual import BayesianMSEAgent, BayesianSPOAgent


DEFAULT_LR = 0.001
DEFAULT_BETA = 0.0
DEFAULT_LAMBDA_REG = 1.0
DEFAULT_HIDDEN_DIM = 1
DEFAULT_CONTEXT_DIM = 8
DEFAULT_N_ARMS = 4
DEFAULT_WEIGHTS = np.ones(DEFAULT_N_ARMS, dtype=float)
DEFAULT_CAPACITY = 1.0

DECISION_COLORS = {
    '0,2': '#1f77b4',
    '0,3': '#ff7f0e',
    '1,2': '#2ca02c',
    '1,3': '#d62728',
}


def default_representation_configs() -> List[Tuple[str, Dict]]:
    return [
        (
            'BayesianSPO (partial, masked)',
            dict(agent_name='BayesianSPO', full_feedback=False, mask_grad=True, phased=False),
        ),
        (
            'BayesianMSE (partial)',
            dict(agent_name='BayesianMSE', full_feedback=False, mask_grad=False, phased=False),
        ),
    ]


def top1_solver(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).flatten()
    action = np.zeros_like(values)
    action[int(np.argmax(values))] = 1.0
    return action


def _template_values(z_dec: np.ndarray) -> np.ndarray:
    z_dec = np.asarray(z_dec).reshape(-1, 1)
    centers = np.array([-1.8, -0.6, 0.6, 1.8], dtype=float)[None, :]
    heights = np.array([0.10, 0.40, 0.40, 0.10], dtype=float)[None, :]
    return heights - 0.95 * (z_dec - centers) ** 2


def _offset_and_scale(z_nui: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z_nui = np.asarray(z_nui)
    shared_offset = 2.6 * z_nui[:, 0]
    shared_scale = 1.4 + 0.45 * np.tanh(1.3 * z_nui[:, 1])
    return shared_offset, shared_scale


def _observed_context(z_dec: np.ndarray, z_nui: np.ndarray, noise_scale: float, rng) -> np.ndarray:
    z_dec = np.asarray(z_dec)
    z_nui = np.asarray(z_nui)
    x = np.column_stack(
        [
            z_dec + 0.10 * z_nui[:, 0],
            z_nui[:, 0] + 0.10 * z_dec,
            z_nui[:, 1] - 0.10 * z_dec,
            z_dec * z_nui[:, 0],
            np.sin(z_dec),
            np.sin(z_nui[:, 0]),
            np.cos(z_nui[:, 1]),
            np.tanh(z_dec - 0.5 * z_nui[:, 1]),
        ]
    )
    if noise_scale > 0.0:
        x = x + noise_scale * rng.normal(size=x.shape)
    return x


def make_latent_dataset(
    n_samples: int,
    seed: int,
    noise_scale: float = 0.05,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z_dec = rng.uniform(-2.4, 2.4, size=n_samples)
    z_nui = rng.normal(size=(n_samples, 2))
    template = _template_values(z_dec)
    shared_offset, shared_scale = _offset_and_scale(z_nui)
    values = shared_offset[:, None] + shared_scale[:, None] * template
    context = _observed_context(z_dec, z_nui, noise_scale, rng)
    optimal_arm = np.argmax(values, axis=1)
    sorted_vals = np.sort(values, axis=1)
    decision_margin = sorted_vals[:, -1] - sorted_vals[:, -2]
    return {
        'X': context,
        'Y': values,
        'z_dec': z_dec,
        'z_nui': z_nui,
        'nuisance_score': shared_offset,
        'shared_offset': shared_offset,
        'shared_scale': shared_scale,
        'optimal_arm': optimal_arm,
        'decision_margin': decision_margin,
    }


def make_episode_from_dataset(dataset: Dict[str, np.ndarray]) -> Dict:
    rounds_data = []
    for i in range(len(dataset['X'])):
        true_values = np.asarray(dataset['Y'][i], dtype=float)
        optimal_action = top1_solver(true_values)
        rounds_data.append(
            {
                'context': np.asarray(dataset['X'][i], dtype=float),
                'true_values': true_values,
                'optimal_action': optimal_action,
                'optimal_value': float(np.max(true_values)),
            }
        )
    return {'rounds_data': rounds_data, 'minimize': False}


def _make_agent(
    cfg: Dict,
    hidden_dim: int,
    nonlinear: bool,
    lr: float,
    beta: float,
    lambda_reg: float,
    seed: int,
):
    agent_name = cfg['agent_name']
    common = dict(
        context_dim=DEFAULT_CONTEXT_DIM,
        n_arms=DEFAULT_N_ARMS,
        weights=DEFAULT_WEIGHTS,
        capacity=DEFAULT_CAPACITY,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        minimize=False,
        solver=top1_solver,
        seed=seed,
        nonlinear=nonlinear,
        phased=cfg.get('phased', False),
        phase_length=cfg.get('phase_length', 50),
        phase_epochs=cfg.get('phase_epochs', 5),
    )
    if agent_name == 'BayesianSPO':
        return BayesianSPOAgent(**common, mask_grad=cfg.get('mask_grad', True))
    if agent_name == 'BayesianMSE':
        return BayesianMSEAgent(**common, mask_grad=cfg.get('mask_grad', False))
    raise ValueError(f'Unknown agent: {agent_name}')


def _extract_features(agent, X: np.ndarray) -> np.ndarray:
    features = np.asarray(agent._features(X))
    if features.ndim == 1:
        features = features[:, None]
    return features


def _extract_predictions(agent, X: np.ndarray) -> np.ndarray:
    return np.asarray([np.asarray(agent.predict_mean(x), dtype=float) for x in X])


def _with_intercept(features: np.ndarray) -> np.ndarray:
    return np.column_stack([features, np.ones(len(features))])


def _regression_probe_r2(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
) -> float:
    design = _with_intercept(train_features)
    weights, _, _, _ = np.linalg.lstsq(design, train_targets, rcond=None)
    preds = _with_intercept(test_features) @ weights
    numer = np.mean((preds - test_targets) ** 2)
    denom = np.var(test_targets)
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - numer / denom))


def _fit_linear_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> np.ndarray:
    weights, _, _, _ = np.linalg.lstsq(_with_intercept(train_features), train_targets, rcond=None)
    return weights


def _predict_linear_probe(
    test_features: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return _with_intercept(test_features) @ weights


def _multioutput_r2(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
) -> float:
    values = []
    for i in range(train_targets.shape[1]):
        values.append(
            _regression_probe_r2(train_features, train_targets[:, i], test_features, test_targets[:, i])
        )
    return float(np.mean(values))


def _classification_probe_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
) -> float:
    targets = np.eye(n_classes)[train_labels]
    weights, _, _, _ = np.linalg.lstsq(_with_intercept(train_features), targets, rcond=None)
    logits = _with_intercept(test_features) @ weights
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == test_labels))


def _action_separation_score(features: np.ndarray, labels: np.ndarray) -> float:
    centroids = []
    within = []
    for label in np.unique(labels):
        pts = features[labels == label]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0)
        centroids.append(center)
        within.append(np.mean(np.linalg.norm(pts - center, axis=1)))
    if len(centroids) < 2:
        return 0.0
    centroid_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            centroid_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    return float(np.mean(centroid_dists) / max(np.mean(within), 1e-8))


def _make_probe_summary(
    agent,
    probe_train: Dict[str, np.ndarray],
    probe_test: Dict[str, np.ndarray],
) -> Dict[str, float]:
    train_features = _extract_features(agent, probe_train['X'])
    test_features = _extract_features(agent, probe_test['X'])
    preds = _extract_predictions(agent, probe_test['X'])
    decision_probe_acc = _classification_probe_accuracy(
        train_features,
        probe_train['optimal_arm'],
        test_features,
        probe_test['optimal_arm'],
        DEFAULT_N_ARMS,
    )
    decision_r2 = _regression_probe_r2(
        train_features, probe_train['z_dec'], test_features, probe_test['z_dec']
    )
    nuisance_r2 = _multioutput_r2(
        train_features, probe_train['z_nui'], test_features, probe_test['z_nui']
    )
    nuisance_score_r2 = _regression_probe_r2(
        train_features, probe_train['nuisance_score'], test_features, probe_test['nuisance_score']
    )
    offset_r2 = _regression_probe_r2(
        train_features, probe_train['shared_offset'], test_features, probe_test['shared_offset']
    )
    scale_r2 = _regression_probe_r2(
        train_features, probe_train['shared_scale'], test_features, probe_test['shared_scale']
    )
    margin_r2 = _regression_probe_r2(
        train_features, probe_train['decision_margin'], test_features, probe_test['decision_margin']
    )
    value_probe_r2 = _multioutput_r2(
        train_features, probe_train['Y'], test_features, probe_test['Y']
    )
    value_mse = float(np.mean((preds - probe_test['Y']) ** 2))
    greedy_actions = np.argmax(preds, axis=1)
    decision_acc = float(np.mean(greedy_actions == probe_test['optimal_arm']))
    action_separation = _action_separation_score(test_features, probe_test['optimal_arm'])
    return {
        'probe_action_acc': decision_probe_acc,
        'probe_z_dec_r2': decision_r2,
        'probe_nuisance_r2': nuisance_r2,
        'probe_nuisance_score_r2': nuisance_score_r2,
        'probe_offset_r2': offset_r2,
        'probe_scale_r2': scale_r2,
        'probe_margin_r2': margin_r2,
        'probe_value_r2': value_probe_r2,
        'probe_value_mse': value_mse,
        'probe_greedy_acc': decision_acc,
        'probe_action_separation': action_separation,
    }


def _trajectory_dataset(
    varying: str,
    n_traces: int = 4,
    n_points: int = 61,
) -> Tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(123 if varying == 'decision' else 456)
    rows = []
    contexts = []
    if varying == 'decision':
        dec_values = np.linspace(-2.4, 2.4, n_points)
        nuisances = rng.normal(scale=0.9, size=(n_traces, 3))
        for trace_id in range(n_traces):
            for step, z_dec in enumerate(dec_values):
                z_nui = nuisances[trace_id]
                x = _observed_context(
                    np.array([z_dec]),
                    z_nui[None, :],
                    noise_scale=0.0,
                    rng=np.random.default_rng(trace_id + step),
                )[0]
                contexts.append(x)
                rows.append(
                    {
                        'trace_id': trace_id,
                        'step': step,
                        'varying': 'decision',
                        'z_dec': z_dec,
                        'nuisance_score': _offset_and_scale(z_nui[None, :])[0][0],
                    }
                )
    else:
        nuisance_values = np.linspace(-2.4, 2.4, n_points)
        decision_vals = np.linspace(-2.1, 2.1, n_traces)
        for trace_id, z_dec in enumerate(decision_vals):
            for step, s in enumerate(nuisance_values):
                z_nui = np.array([s, 0.7 * np.sin(s), -0.5 * np.cos(s)], dtype=float)
                x = _observed_context(
                    np.array([z_dec]),
                    z_nui[None, :],
                    noise_scale=0.0,
                    rng=np.random.default_rng(trace_id + step),
                )[0]
                contexts.append(x)
                rows.append(
                    {
                        'trace_id': trace_id,
                        'step': step,
                        'varying': 'nuisance',
                        'z_dec': z_dec,
                        'nuisance_score': _offset_and_scale(z_nui[None, :])[0][0],
                    }
                )
    return np.asarray(contexts, dtype=float), pd.DataFrame(rows)


def _trajectory_summary(agent) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outputs = []
    sensitivity_rows = []
    for varying in ('decision', 'nuisance'):
        contexts, meta = _trajectory_dataset(varying)
        features = _extract_features(agent, contexts)
        meta = meta.copy()
        meta['feature_1'] = features[:, 0]
        meta['feature_2'] = features[:, 1] if features.shape[1] > 1 else 0.0
        diffs = features[1:] - features[:-1]
        same_trace = meta['trace_id'].to_numpy()[1:] == meta['trace_id'].to_numpy()[:-1]
        path_length = float(np.mean(np.linalg.norm(diffs[same_trace], axis=1)))
        sensitivity_rows.append({'varying': varying, 'mean_step_feature_change': path_length})
        outputs.append(meta)
    traces = pd.concat(outputs, ignore_index=True)
    sensitivity = pd.DataFrame(sensitivity_rows)
    decision_change = float(
        sensitivity.loc[sensitivity['varying'] == 'decision', 'mean_step_feature_change'].iloc[0]
    )
    nuisance_change = float(
        sensitivity.loc[sensitivity['varying'] == 'nuisance', 'mean_step_feature_change'].iloc[0]
    )
    sensitivity['decision_to_nuisance_ratio'] = (
        decision_change / max(nuisance_change, 1e-8)
    )
    return traces, sensitivity


def world_overview(
    n_grid: int = 121,
    n_nuisance_samples: int = 256,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    z_grid = np.linspace(-2.4, 2.4, n_grid)
    rng = np.random.default_rng(99)
    nuisance = rng.normal(size=(n_nuisance_samples, 2))
    rows = []
    stability_rows = []
    for z_dec in z_grid:
        z_dec_rep = np.full(n_nuisance_samples, z_dec)
        shared_offset, shared_scale = _offset_and_scale(nuisance)
        values = shared_offset[:, None] + shared_scale[:, None] * _template_values(z_dec_rep)
        optimal = np.argmax(values, axis=1)
        for arm in range(DEFAULT_N_ARMS):
            rows.append(
                {
                    'z_dec': z_dec,
                    'arm': f'Arm {arm + 1}',
                    'mean_value': float(values[:, arm].mean()),
                    'optimal_prob': float(np.mean(optimal == arm)),
                }
            )
        counts = np.bincount(optimal, minlength=DEFAULT_N_ARMS)
        stability_rows.append(
            {
                'z_dec': z_dec,
                'dominant_arm_prob': float(counts.max() / counts.sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(stability_rows)


def run_representation_suite(
    configs: Iterable[Tuple[str, Dict]],
    n_rounds: int = 1500,
    n_seeds: int = 5,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    nonlinear: bool = True,
    lr: float = DEFAULT_LR,
    beta: float = DEFAULT_BETA,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    diagnostics_interval: int = 25,
    probe_train_size: int = 1000,
    probe_test_size: int = 1500,
) -> Dict[str, pd.DataFrame]:
    rounds_records = []
    seed_summary = []
    probe_records = []
    feature_views = []
    trace_records = []
    sensitivity_records = []

    plot_seed = 0

    for seed in range(n_seeds):
        train_data = make_latent_dataset(n_rounds, seed=1000 + seed)
        episode = make_episode_from_dataset(train_data)
        probe_train = make_latent_dataset(probe_train_size, seed=2000 + seed)
        probe_test = make_latent_dataset(probe_test_size, seed=3000 + seed)

        for label, cfg in configs:
            agent = _make_agent(
                cfg,
                hidden_dim=hidden_dim,
                nonlinear=nonlinear,
                lr=lr,
                beta=beta,
                lambda_reg=lambda_reg,
                seed=seed,
            )
            out = run_agent_on_episode(
                agent,
                episode,
                n_rounds=n_rounds,
                full_feedback=cfg.get('full_feedback', False),
                diagnostics=True,
                diagnostics_interval=diagnostics_interval,
            )
            diag_rounds = out['diagnostic_rounds']
            for idx, round_idx in enumerate(diag_rounds):
                rounds_records.append(
                    {
                        'agent': label,
                        'seed': seed,
                        'round': round_idx,
                        'cumulative_regret': float(out['cumulative_regret'][round_idx - 1]),
                        'instant_regret': float(out['regrets'][round_idx - 1]),
                        'decision_correct': float(out['decision_correct'][idx]),
                        'overall_mae': float(out['overall_mae'][idx]),
                        'contested_mae': float(out['contested_mae'][idx]),
                        'rank_corr': float(out['rank_corr'][idx]),
                    }
                )

            probe = _make_probe_summary(agent, probe_train, probe_test)
            probe_records.append({'agent': label, 'seed': seed, **probe})
            seed_summary.append(
                {
                    'agent': label,
                    'seed': seed,
                    'final_regret': float(out['final_regret']),
                    'final_probe_value_mse': probe['probe_value_mse'],
                    'final_probe_greedy_acc': probe['probe_greedy_acc'],
                }
            )

            traces, sensitivity = _trajectory_summary(agent)
            traces['agent'] = label
            traces['seed'] = seed
            trace_records.append(traces)
            sensitivity['agent'] = label
            sensitivity['seed'] = seed
            sensitivity_records.append(sensitivity)

            if seed == plot_seed:
                features = _extract_features(agent, probe_test['X'])
                train_features = _extract_features(agent, probe_train['X'])
                z_dec_probe = _fit_linear_probe(train_features, probe_train['z_dec'])
                offset_probe = _fit_linear_probe(train_features, probe_train['shared_offset'])
                decoded_z_dec = _predict_linear_probe(features, z_dec_probe)
                decoded_offset = _predict_linear_probe(features, offset_probe)
                for i in range(len(features)):
                    feature_views.append(
                        {
                            'agent': label,
                            'seed': seed,
                            'feature_1': float(features[i, 0]),
                            'feature_2': float(features[i, 1] if features.shape[1] > 1 else 0.0),
                            'optimal_arm': f"Arm {int(probe_test['optimal_arm'][i]) + 1}",
                            'z_dec': float(probe_test['z_dec'][i]),
                            'nuisance_score': float(probe_test['nuisance_score'][i]),
                            'shared_offset': float(probe_test['shared_offset'][i]),
                            'decision_margin': float(probe_test['decision_margin'][i]),
                            'decoded_z_dec': float(decoded_z_dec[i]),
                            'decoded_offset': float(decoded_offset[i]),
                        }
                    )

    rounds_df = pd.DataFrame(rounds_records)
    seed_summary_df = pd.DataFrame(seed_summary)
    probe_df = pd.DataFrame(probe_records)
    sensitivity_df = pd.concat(sensitivity_records, ignore_index=True)
    traces_df = pd.concat(trace_records, ignore_index=True)
    feature_df = pd.DataFrame(feature_views)

    train_summary = (
        seed_summary_df.groupby('agent', as_index=False)
        .agg(
            mean_final_regret=('final_regret', 'mean'),
            std_final_regret=('final_regret', 'std'),
            mean_probe_value_mse=('final_probe_value_mse', 'mean'),
            mean_probe_greedy_acc=('final_probe_greedy_acc', 'mean'),
        )
    )

    probe_summary = (
        probe_df.groupby('agent', as_index=False)
        .agg(
            mean_probe_action_acc=('probe_action_acc', 'mean'),
            mean_probe_z_dec_r2=('probe_z_dec_r2', 'mean'),
            mean_probe_nuisance_r2=('probe_nuisance_r2', 'mean'),
            mean_probe_nuisance_score_r2=('probe_nuisance_score_r2', 'mean'),
            mean_probe_offset_r2=('probe_offset_r2', 'mean'),
            mean_probe_scale_r2=('probe_scale_r2', 'mean'),
            mean_probe_margin_r2=('probe_margin_r2', 'mean'),
            mean_probe_value_r2=('probe_value_r2', 'mean'),
            mean_probe_value_mse=('probe_value_mse', 'mean'),
            mean_probe_greedy_acc=('probe_greedy_acc', 'mean'),
            mean_probe_action_separation=('probe_action_separation', 'mean'),
        )
    )

    sensitivity_summary = (
        sensitivity_df.groupby(['agent', 'varying'], as_index=False)
        .agg(
            mean_step_feature_change=('mean_step_feature_change', 'mean'),
            mean_ratio=('decision_to_nuisance_ratio', 'mean'),
        )
    )

    return {
        'rounds': rounds_df,
        'train_summary': train_summary,
        'probe_seed': probe_df,
        'probe_summary': probe_summary,
        'feature_views': feature_df,
        'traces': traces_df,
        'sensitivity_seed': sensitivity_df,
        'sensitivity_summary': sensitivity_summary,
    }


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    series = pd.Series(values)
    return series.rolling(window, min_periods=1, center=True).mean().to_numpy()


def plot_training_regret(rounds: pd.DataFrame, title: str, smooth_window: int = 11):
    if rounds.empty:
        return ggplot() + labs(title=title)
    summary = (
        rounds.groupby(['agent', 'round'], as_index=False)
        .agg(
            mean_regret=('cumulative_regret', 'mean'),
            std_regret=('cumulative_regret', 'std'),
        )
    )
    summary['std_regret'] = summary['std_regret'].fillna(0.0)
    summary['lower'] = summary['mean_regret'] - summary['std_regret']
    summary['upper'] = summary['mean_regret'] + summary['std_regret']
    parts = []
    for agent, df_agent in summary.groupby('agent'):
        df_agent = df_agent.sort_values('round').copy()
        df_agent['mean_regret'] = _smooth(df_agent['mean_regret'], smooth_window)
        df_agent['lower'] = _smooth(df_agent['lower'], smooth_window)
        df_agent['upper'] = _smooth(df_agent['upper'], smooth_window)
        parts.append(df_agent)
    plot_df = pd.concat(parts, ignore_index=True)
    return (
        ggplot(plot_df, aes('round', 'mean_regret', color='agent', fill='agent'))
        + geom_ribbon(aes(ymin='lower', ymax='upper'), alpha=0.16, color='transparent', size=0)
        + geom_line(size=1.1)
        + ggsize(700, 340)
        + labs(title=title, x='Round', y='Cumulative Regret')
        + theme(legend_position='top')
    )


def plot_world_value_curves(curves: pd.DataFrame):
    return (
        ggplot(curves, aes('z_dec', 'mean_value', color='arm'))
        + geom_line(size=1.1)
        + ggsize(650, 320)
        + labs(
            title='True Latent World: Mean Arm Value vs Decision Latent',
            x='Decision latent z_dec',
            y='Mean arm value',
            color='Arm',
        )
        + theme(legend_position='top')
    )


def plot_world_optimal_regions(curves: pd.DataFrame, stability: pd.DataFrame):
    left = (
        ggplot(curves, aes('z_dec', 'optimal_prob', color='arm'))
        + geom_line(size=1.1)
        + ggsize(650, 320)
        + labs(
            title='Optimal-Arm Probability Under Nuisance Variation',
            x='Decision latent z_dec',
            y='P(arm is optimal)',
            color='Arm',
        )
        + theme(legend_position='top')
    )
    right = (
        ggplot(stability, aes('z_dec', 'dominant_arm_prob'))
        + geom_line(size=1.1)
        + ggsize(650, 320)
        + labs(
            title='Decision Stability Under Nuisance Variation',
            x='Decision latent z_dec',
            y='Dominant arm probability',
        )
    )
    return gggrid([left, right], ncol=2)


def plot_feature_scatter(feature_views: pd.DataFrame, color_by: str, title: str):
    plots = []
    for agent, df_agent in feature_views.groupby('agent'):
        plots.append(
            ggplot(df_agent, aes('feature_1', 'feature_2', color=color_by))
            + geom_point(alpha=0.55, size=2.0)
            + ggsize(340, 320)
            + labs(title=agent, x='Feature 1', y='Feature 2', color=color_by.replace('_', ' ').title())
        )
    return gggrid(plots, ncol=len(plots)) + labs(title=title)


def plot_representation_story(feature_views: pd.DataFrame):
    plots = []
    for color_by, subtitle in (
        ('optimal_arm', 'Optimal arm'),
        ('nuisance_score', 'Shared offset'),
    ):
        for agent, df_agent in feature_views.groupby('agent'):
            plots.append(
                ggplot(df_agent, aes('feature_1', 'feature_2', color=color_by))
                + geom_point(alpha=0.60, size=1.8)
                + ggsize(320, 300)
                + labs(
                    title=f'{agent}: {subtitle}',
                    x='Feature 1',
                    y='Feature 2',
                    color=subtitle,
                )
            )
    return gggrid(plots, ncol=2) + labs(
        title='Learned bottleneck: decision regions vs nuisance structure'
    )


def plot_decoded_latent_story(feature_views: pd.DataFrame):
    plots = []
    specs = [
        ('z_dec', 'decoded_z_dec', 'Decision latent', 'Decoded decision latent'),
        ('shared_offset', 'decoded_offset', 'Shared offset', 'Decoded shared offset'),
    ]
    for true_col, pred_col, axis_label, title_suffix in specs:
        for agent, df_agent in feature_views.groupby('agent'):
            low = float(min(df_agent[true_col].min(), df_agent[pred_col].min()))
            high = float(max(df_agent[true_col].max(), df_agent[pred_col].max()))
            line_df = pd.DataFrame({'x': [low, high], 'y': [low, high]})
            plots.append(
                ggplot()
                + geom_line(data=line_df, mapping=aes('x', 'y'), color='#999999', alpha=0.6, size=1.0)
                + geom_point(
                    data=df_agent,
                    mapping=aes(true_col, pred_col, color='optimal_arm'),
                    alpha=0.55,
                    size=1.8,
                )
                + ggsize(320, 300)
                + labs(
                    title=f'{agent}: {title_suffix}',
                    x=f'True {axis_label}',
                    y=f'Decoded {axis_label}',
                    color='Optimal arm',
                )
            )
    return gggrid(plots, ncol=2) + labs(
        title='What the bottleneck linearly encodes'
    )


def plot_feature_trajectories(traces: pd.DataFrame, varying: str, title: str):
    plots = []
    subset = traces[traces['varying'] == varying]
    for agent, df_agent in subset.groupby('agent'):
        df_agent = df_agent.copy()
        df_agent['trace_label'] = df_agent['trace_id'].astype(str)
        plots.append(
            ggplot(df_agent, aes('feature_1', 'feature_2', color='trace_label', group='trace_id'))
            + geom_path(size=1.0, alpha=0.85)
            + geom_point(size=1.6, alpha=0.75)
            + ggsize(340, 320)
            + labs(title=agent, x='Feature 1', y='Feature 2', color='Trace')
        )
    return gggrid(plots, ncol=len(plots)) + labs(title=title)


def _metric_panel_df(
    probe_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    group: str,
) -> pd.DataFrame:
    sensitivity_wide = (
        sensitivity_summary.pivot(index='agent', columns='varying', values='mean_step_feature_change')
        .reset_index()
    )
    ratio_df = (
        sensitivity_summary[sensitivity_summary['varying'] == 'decision'][['agent', 'mean_ratio']]
        .rename(columns={'mean_ratio': 'decision_to_nuisance_ratio'})
    )
    summary = probe_summary.merge(sensitivity_wide, on='agent', how='left').merge(
        ratio_df, on='agent', how='left'
    )

    if group == 'decision':
        metric_map = [
            ('Action probe accuracy', 'mean_probe_action_acc'),
            ('Decision latent R²', 'mean_probe_z_dec_r2'),
            ('Action separation', 'mean_probe_action_separation'),
            ('Decision / nuisance sensitivity', 'decision_to_nuisance_ratio'),
        ]
    else:
        metric_map = [
            ('Full-value probe R²', 'mean_probe_value_r2'),
            ('Offset probe R²', 'mean_probe_offset_r2'),
            ('Value MSE', 'mean_probe_value_mse'),
            ('Nuisance latent R²', 'mean_probe_nuisance_r2'),
        ]

    rows = []
    for metric_label, col in metric_map:
        for _, row in summary.iterrows():
            rows.append(
                {
                    'metric': metric_label,
                    'agent': row['agent'],
                    'value': float(row[col]),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_panels(
    probe_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    group: str,
    title: str,
):
    metric_df = _metric_panel_df(probe_summary, sensitivity_summary, group=group)
    plots = []
    for metric, df_metric in metric_df.groupby('metric'):
        xmin = float(df_metric['value'].min())
        xmax = float(df_metric['value'].max())
        pad = max((xmax - xmin) * 0.15, 0.02)
        segment_df = pd.DataFrame(
            {'x': [xmin], 'xend': [xmax], 'y': [0.0], 'yend': [0.0]}
        )
        plot_df = df_metric.copy()
        plot_df['y'] = 0.0
        plots.append(
            ggplot()
            + geom_segment(
                data=segment_df,
                mapping=aes(x='x', xend='xend', y='y', yend='yend'),
                size=1.2,
                alpha=0.35,
            )
            + geom_point(
                data=plot_df,
                mapping=aes('value', 'y', color='agent'),
                size=4.0,
                alpha=0.9,
            )
            + ggsize(280, 135)
            + labs(title=metric, x='', y='')
            + theme(
                legend_position='none',
                axis_ticks_y='blank',
                axis_text_y='blank',
                axis_title_y='blank',
            )
        )
    return gggrid(plots, ncol=2) + labs(title=title)


def plot_metric_story(
    probe_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    group: str,
    title: str,
):
    metric_df = _metric_panel_df(probe_summary, sensitivity_summary, group=group)
    if metric_df.empty:
        return ggplot() + labs(title=title)
    agents = list(metric_df['agent'].unique())
    if len(agents) != 2:
        return ggplot(metric_df, aes('value', 'metric', color='agent')) + geom_point(size=4) + labs(title=title)

    wide = metric_df.pivot(index='metric', columns='agent', values='value').reset_index()
    wide = wide.rename(columns={agents[0]: 'left_value', agents[1]: 'right_value'})
    long = metric_df.copy()
    return (
        ggplot()
        + geom_segment(
            data=wide,
            mapping=aes(x='left_value', xend='right_value', y='metric', yend='metric'),
            size=1.3,
            alpha=0.35,
        )
        + geom_point(
            data=long,
            mapping=aes('value', 'metric', color='agent'),
            size=4.5,
            alpha=0.9,
        )
        + ggsize(640, 240)
        + labs(title=title, x='', y='', color='Agent')
        + theme(legend_position='top')
    )


def plot_sensitivity_summary(sensitivity_summary: pd.DataFrame):
    df = sensitivity_summary.copy()
    plots = []
    for metric_label, value_col in (
        ('Decision-latent traversal', 'decision'),
        ('Nuisance traversal', 'nuisance'),
    ):
        pass
    pivot = (
        df.pivot(index='agent', columns='varying', values='mean_step_feature_change')
        .reset_index()
    )
    long = pivot.melt(id_vars='agent', value_vars=['decision', 'nuisance'], var_name='varying', value_name='value')
    long['varying'] = long['varying'].map(
        {'decision': 'Decision traversal', 'nuisance': 'Nuisance traversal'}
    )
    left = (
        ggplot(long, aes('varying', 'value', color='agent', group='agent'))
        + geom_line(size=1.0, alpha=0.65)
        + geom_point(size=3.5)
        + ggsize(360, 260)
        + labs(
            title='Mean feature movement by traversal type',
            x='',
            y='Mean step feature change',
            color='Agent',
        )
        + theme(legend_position='top')
    )
    ratio_df = (
        sensitivity_summary[sensitivity_summary['varying'] == 'decision'][['agent', 'mean_ratio']]
        .rename(columns={'mean_ratio': 'ratio'})
    )
    right = (
        ggplot(ratio_df, aes('agent', 'ratio', color='agent'))
        + geom_point(size=4.5)
        + ggsize(260, 260)
        + labs(
            title='Decision / nuisance sensitivity ratio',
            x='',
            y='Ratio',
            color='Agent',
        )
        + theme(legend_position='none')
    )
    return gggrid([left, right], ncol=2)


def print_regret_summary(summary: pd.DataFrame):
    for _, row in summary.sort_values('mean_final_regret').iterrows():
        std = row['std_final_regret']
        std_str = '0.0' if pd.isna(std) else f'{std:.1f}'
        print(f"{row['agent']:<24}: regret = {row['mean_final_regret']:.1f} +/- {std_str}")


LATENT_CONTEXT_DIM = 8
LATENT_N_ARMS = 6
LATENT_K = 2
LATENT_HIDDEN_DIM = 2
LATENT_BETA = 1.0
LATENT_WEIGHTS = np.ones(LATENT_N_ARMS, dtype=float)
LATENT_CAPACITY = float(LATENT_K)
DECISION_CENTERS = np.array([-1.8, 0.0, 1.8], dtype=float)
REGIME_NAMES = np.array(['Regime A', 'Regime B', 'Regime C'])
SET_NAMES = np.array(['{1,2}', '{3,4}', '{5,6}'])
PAIR_REGIME_ADJUST = 0.95 * np.array(
    [
        [1.0, -1.0],
        [-1.0, 1.0],
        [0.0, 0.0],
    ],
    dtype=float,
)
PAIR_BIAS = np.array([0.15, -0.15, 0.15, -0.15, 0.15, -0.15], dtype=float)
PAIR_INDEX = np.array([0, 0, 1, 1, 2, 2], dtype=int)


def latent_representation_configs() -> List[Tuple[str, Dict]]:
    return [
        (
            'BayesianSPO (partial, masked)',
            dict(agent_name='BayesianSPO', full_feedback=False, mask_grad=True, phased=False),
        ),
        (
            'BayesianMSE (partial)',
            dict(agent_name='BayesianMSE', full_feedback=False, mask_grad=False, phased=False),
        ),
    ]


def topk_solver(values: np.ndarray, k: int = LATENT_K) -> np.ndarray:
    values = np.asarray(values).flatten()
    idx = np.argsort(values)[-k:]
    action = np.zeros_like(values)
    action[idx] = 1.0
    return action


def _decision_group_scores(z_dec: np.ndarray) -> np.ndarray:
    z_dec = np.asarray(z_dec).reshape(-1, 1)
    return 4.6 - 1.55 * (z_dec - DECISION_CENTERS[None, :]) ** 2


def _context_from_latents(z_dec: np.ndarray, regime_idx: np.ndarray, noise_scale: float, rng) -> np.ndarray:
    z_dec = np.asarray(z_dec)
    regime_idx = np.asarray(regime_idx)
    regime_embed = np.array(
        [
            [1.0, -0.3, -0.8],
            [-0.6, 1.0, -0.2],
            [-0.4, -0.7, 1.0],
        ],
        dtype=float,
    )[regime_idx]
    x = np.column_stack(
        [
            z_dec + 0.20 * regime_embed[:, 0],
            np.sin(z_dec) + 0.25 * regime_embed[:, 1],
            np.cos(z_dec) + 0.25 * regime_embed[:, 2],
            regime_embed[:, 0] + 0.15 * z_dec,
            regime_embed[:, 1] - 0.10 * z_dec,
            regime_embed[:, 2] + 0.10 * z_dec ** 2,
            z_dec * regime_embed[:, 0],
            z_dec * regime_embed[:, 1],
        ]
    )
    if noise_scale > 0.0:
        x = x + noise_scale * rng.normal(size=x.shape)
    return x


def make_decision_equivalence_dataset(
    n_samples: int,
    seed: int,
    noise_scale: float = 0.04,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    decision_group = rng.integers(0, 3, size=n_samples)
    regime_idx = rng.integers(0, 3, size=n_samples)
    z_dec = rng.normal(loc=DECISION_CENTERS[decision_group], scale=0.22, size=n_samples)
    group_scores = _decision_group_scores(z_dec)
    base_values = group_scores[:, PAIR_INDEX] + PAIR_BIAS[None, :]
    values = base_values.copy()
    pair_cols = np.array([[0, 1], [2, 3], [4, 5]], dtype=int)
    for i in range(n_samples):
        cols = pair_cols[decision_group[i]]
        values[i, cols] += PAIR_REGIME_ADJUST[regime_idx[i]]
    context = _context_from_latents(z_dec, regime_idx, noise_scale, rng)
    selected_set = np.argmax(group_scores, axis=1)
    return {
        'X': context,
        'Y': values,
        'z_dec': z_dec,
        'decision_group': decision_group,
        'regime_idx': regime_idx,
        'selected_set': selected_set,
        'selected_set_label': SET_NAMES[selected_set],
        'regime_label': REGIME_NAMES[regime_idx],
    }


def make_topk_episode(dataset: Dict[str, np.ndarray]) -> Dict:
    rounds_data = []
    for i in range(len(dataset['X'])):
        true_values = np.asarray(dataset['Y'][i], dtype=float)
        optimal_action = topk_solver(true_values)
        rounds_data.append(
            {
                'context': np.asarray(dataset['X'][i], dtype=float),
                'true_values': true_values,
                'optimal_action': optimal_action,
                'optimal_value': float(np.sum(true_values[optimal_action > 0.5])),
            }
        )
    return {'rounds_data': rounds_data, 'minimize': False}


def _make_latent_agent(
    cfg: Dict,
    hidden_dim: int,
    nonlinear: bool,
    lr: float,
    beta: float,
    lambda_reg: float,
    seed: int,
):
    common = dict(
        context_dim=LATENT_CONTEXT_DIM,
        n_arms=LATENT_N_ARMS,
        weights=LATENT_WEIGHTS,
        capacity=LATENT_CAPACITY,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        minimize=False,
        solver=topk_solver,
        seed=seed,
        nonlinear=nonlinear,
        phased=cfg.get('phased', False),
        phase_length=cfg.get('phase_length', 50),
        phase_epochs=cfg.get('phase_epochs', 5),
    )
    if cfg['agent_name'] == 'BayesianSPO':
        return BayesianSPOAgent(**common, mask_grad=cfg.get('mask_grad', True))
    if cfg['agent_name'] == 'BayesianMSE':
        return BayesianMSEAgent(**common, mask_grad=cfg.get('mask_grad', False))
    raise ValueError(f"Unknown agent: {cfg['agent_name']}")


def run_latent_representation_suite(
    configs: Iterable[Tuple[str, Dict]],
    n_rounds: int = 2500,
    n_seeds: int = 3,
    hidden_dim: int = LATENT_HIDDEN_DIM,
    nonlinear: bool = True,
    lr: float = DEFAULT_LR,
    beta: float = LATENT_BETA,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    probe_size: int = 1800,
) -> Dict[str, pd.DataFrame]:
    regret_rows = []
    feature_rows = []

    for seed in range(n_seeds):
        train_data = make_decision_equivalence_dataset(n_rounds, seed=1000 + seed)
        episode = make_topk_episode(train_data)
        probe_data = make_decision_equivalence_dataset(probe_size, seed=2000 + seed, noise_scale=0.0)

        if seed == 0:
            for i in range(len(probe_data['X'])):
                feature_rows.append(
                    {
                        'view': 'original',
                        'seed': seed,
                        'selected_set': probe_data['selected_set_label'][i],
                        'regime': probe_data['regime_label'][i],
                        'regime_order': int(probe_data['regime_idx'][i]),
                        'z_dec': float(probe_data['z_dec'][i]),
                    }
                )

        for label, cfg in configs:
            agent = _make_latent_agent(
                cfg,
                hidden_dim=hidden_dim,
                nonlinear=nonlinear,
                lr=lr,
                beta=beta,
                lambda_reg=lambda_reg,
                seed=seed,
            )
            result = run_agent_on_episode(
                agent,
                episode,
                n_rounds=n_rounds,
                full_feedback=cfg.get('full_feedback', False),
                diagnostics=False,
            )
            regret_rows.append(
                {
                    'agent': label,
                    'seed': seed,
                    'final_regret': float(result['final_regret']),
                }
            )

            if seed == 0:
                pass

            features = _extract_features(agent, probe_data['X'])
            centroid_df = pd.DataFrame(
                {
                    'selected_set': probe_data['selected_set_label'],
                    'regime': probe_data['regime_label'],
                    'regime_order': probe_data['regime_idx'],
                    'feature_1': features[:, 0],
                    'feature_2': features[:, 1] if features.shape[1] > 1 else 0.0,
                }
            )
            centroids = (
                centroid_df.groupby(['selected_set', 'regime', 'regime_order'], as_index=False)
                .agg(
                    feature_1=('feature_1', 'mean'),
                    feature_2=('feature_2', 'mean'),
                )
            )
            for _, row in centroids.iterrows():
                feature_rows.append(
                    {
                        'view': 'learned_centroid',
                        'agent': label,
                        'seed': seed,
                        'selected_set': row['selected_set'],
                        'regime': row['regime'],
                        'regime_order': int(row['regime_order']),
                        'feature_1': float(row['feature_1']),
                        'feature_2': float(row['feature_2']),
                    }
                )

    regret_df = pd.DataFrame(regret_rows)
    regret_summary = (
        regret_df.groupby('agent', as_index=False)
        .agg(
            mean_final_regret=('final_regret', 'mean'),
            std_final_regret=('final_regret', 'std'),
        )
    )
    return {
        'feature_views': pd.DataFrame(feature_rows),
        'regret_seed': regret_df,
        'regret_summary': regret_summary,
    }


def plot_original_latent(feature_views: pd.DataFrame):
    original_df = feature_views[feature_views['view'] == 'original'].copy()
    centroid_df = (
        original_df.groupby(['selected_set', 'regime', 'regime_order'], as_index=False)
        .agg(z_dec=('z_dec', 'mean'))
    )
    return (
        ggplot()
        + geom_point(
            data=original_df,
            mapping=aes('z_dec', 'regime', color='selected_set'),
            alpha=0.28,
            size=1.8,
        )
        + geom_point(
            data=centroid_df,
            mapping=aes('z_dec', 'regime', color='selected_set', shape='regime'),
            alpha=0.95,
            size=4.2,
        )
        + ggsize(620, 300)
        + labs(
            title='Original latent world',
            x='True decision latent z_dec',
            y='Nuisance regime',
            color='Selected set',
            shape='Regime',
        )
        + theme(legend_position='right')
    )


def plot_learned_latent(feature_views: pd.DataFrame, regret_summary: pd.DataFrame | None = None):
    centroid_df = feature_views[feature_views['view'] == 'learned_centroid'].copy()
    centroid_df['path_group'] = centroid_df['selected_set'] + ' / seed ' + centroid_df['seed'].astype(str)
    mean_df = (
        centroid_df.groupby(['agent', 'selected_set', 'regime', 'regime_order'], as_index=False)
        .agg(
            feature_1=('feature_1', 'mean'),
            feature_2=('feature_2', 'mean'),
        )
    )
    mean_df['path_group'] = mean_df['selected_set']
    plots = []
    regret_map = {}
    if regret_summary is not None and not regret_summary.empty:
        for _, row in regret_summary.iterrows():
            std = row['std_final_regret']
            std_str = '0.0' if pd.isna(std) else f'{std:.0f}'
            regret_map[row['agent']] = f"{row['mean_final_regret']:.0f} +/- {std_str}"
    for agent, df_mean in mean_df.groupby('agent'):
        df_mean = df_mean.sort_values(['selected_set', 'regime_order']).copy()
        df_seed = centroid_df[centroid_df['agent'] == agent].sort_values(
            ['selected_set', 'seed', 'regime_order']
        )
        title = agent
        if agent in regret_map:
            title = f'{agent}\nfinal regret = {regret_map[agent]}'
        plots.append(
            ggplot()
            + geom_path(
                data=df_seed,
                mapping=aes('feature_1', 'feature_2', color='selected_set', group='path_group'),
                alpha=0.18,
                size=1.0,
            )
            + geom_path(
                data=df_mean,
                mapping=aes('feature_1', 'feature_2', color='selected_set', group='path_group'),
                alpha=0.95,
                size=1.7,
            )
            + geom_point(
                data=df_seed,
                mapping=aes('feature_1', 'feature_2', color='selected_set', shape='regime'),
                alpha=0.28,
                size=2.2,
            )
            + geom_point(
                data=df_mean,
                mapping=aes('feature_1', 'feature_2', color='selected_set', shape='regime'),
                size=5.2,
                alpha=0.98,
            )
            + ggsize(420, 360)
            + labs(
                title=title,
                x='Feature 1',
                y='Feature 2',
                color='Selected set',
                shape='Regime',
            )
            + theme(legend_position='right')
        )
    return gggrid(plots, ncol=2) + labs(
        title='Learned latent: mean regime trajectories within each selected-set cluster'
    )


# ── Oracle-Aligned Mechanism Experiment ───────────────────────────────────




def oracle_agreement_summary(
    n_verify: int = 200,
    seed: int = 42,
    context_seed: int = 123,
    noise_width: float = 0.0,
) -> pd.DataFrame:
    env = DecisionFocusedEnv(seed=seed, noise_width=noise_width)
    rng = np.random.default_rng(context_seed)
    matches = 0
    for _ in range(n_verify):
        x = rng.standard_normal(env.context_dim)
        v = env.data._generate_values(x)
        oracle_sel = compute_oracle_decision(x)
        solver_sel = env.solve(v)
        if np.array_equal(oracle_sel.astype(int), solver_sel.astype(int)):
            matches += 1
    return pd.DataFrame(
        [
            {
                'noise_width': noise_width,
                'oracle_matches': matches,
                'n_verify': n_verify,
                'agreement_rate': matches / n_verify,
            }
        ]
    )


def plot_decision_dgp_overview(noise_width: float = 0.0):
    from lets_plot import scale_color_manual

    env = DecisionFocusedEnv(seed=42, noise_width=noise_width)
    grid = np.linspace(-2.5, 2.5, 121)
    item_colors = {
        'Item 0': '#1f77b4',
        'Item 1': '#4e79a7',
        'Item 2': '#d62728',
        'Item 3': '#e15759',
    }

    def sweep_df(vary_dim: int, fixed_x: np.ndarray, x_label: str) -> pd.DataFrame:
        rows = []
        for value in grid:
            x = fixed_x.copy()
            x[vary_dim] = value
            vals = env.data._generate_values(x)
            for item_idx, item_value in enumerate(vals):
                rows.append(
                    {
                        'x_value': float(value),
                        'item': f'Item {item_idx}',
                        'value': float(item_value),
                    }
                )
        return pd.DataFrame(rows), x_label

    df_x0, x0_label = sweep_df(0, np.zeros(5), 'Vary x0: pair (0,1) flips')
    df_x1, x1_label = sweep_df(1, np.zeros(5), 'Vary x1: pair (2,3) flips')
    fixed = np.array([0.9, -0.9, 0.0, 0.0, 0.0], dtype=float)
    df_x2, x2_label = sweep_df(
        2,
        fixed,
        'Vary x2 / x3 / x4: all item values shift together',
    )

    plots = []
    for title, df in (
        (x0_label, df_x0),
        (x1_label, df_x1),
        (x2_label, df_x2),
    ):
        plots.append(
            ggplot(df, aes('x_value', 'value', color='item'))
            + geom_line(size=1.3)
            + scale_color_manual(values=item_colors)
            + ggsize(320, 260)
            + labs(
                title=title,
                x='Input value',
                y='Item value',
                color='Item',
            )
            + theme(legend_position='right')
        )

    return gggrid(plots, ncol=3) + labs(
        title='How the synthetic world works'
    )


def plot_decision_pair_schematic():
    from lets_plot import (
        coord_cartesian,
        element_blank,
        element_text,
        geom_rect,
        geom_text,
        scale_fill_manual,
        theme_minimal,
    )

    pair_boxes = pd.DataFrame(
        [
            {'xmin': 0.4, 'xmax': 2.6, 'ymin': 1.20, 'ymax': 2.65},
            {'xmin': 3.4, 'xmax': 5.6, 'ymin': 1.20, 'ymax': 2.65},
        ]
    )
    context_boxes = pd.DataFrame(
        [
            {'xmin': 1.00, 'xmax': 1.45, 'ymin': 3.35, 'ymax': 3.85, 'fill': 'decision_ctx'},
            {'xmin': 1.55, 'xmax': 2.00, 'ymin': 3.35, 'ymax': 3.85, 'fill': 'decision_ctx'},
            {'xmin': 2.35, 'xmax': 2.80, 'ymin': 3.35, 'ymax': 3.85, 'fill': 'nuisance_ctx'},
            {'xmin': 2.90, 'xmax': 3.35, 'ymin': 3.35, 'ymax': 3.85, 'fill': 'nuisance_ctx'},
            {'xmin': 3.45, 'xmax': 3.90, 'ymin': 3.35, 'ymax': 3.85, 'fill': 'nuisance_ctx'},
        ]
    )
    item_boxes = pd.DataFrame(
        [
            {'xmin': 0.75, 'xmax': 1.5, 'ymin': 1.65, 'ymax': 2.25, 'fill': 'item'},
            {'xmin': 1.5, 'xmax': 2.25, 'ymin': 1.65, 'ymax': 2.25, 'fill': 'item'},
            {'xmin': 3.75, 'xmax': 4.5, 'ymin': 1.65, 'ymax': 2.25, 'fill': 'item'},
            {'xmin': 4.5, 'xmax': 5.25, 'ymin': 1.65, 'ymax': 2.25, 'fill': 'item'},
            {'xmin': 0.7, 'xmax': 5.3, 'ymin': -0.96, 'ymax': -0.28, 'fill': 'nuisance'},
        ]
    )
    text = pd.DataFrame(
        [
            {'x': 0.55, 'y': 3.60, 'label': 'Context x =', 'kind': 'context_label'},
            {'x': 1.5, 'y': 2.84, 'label': 'Pair 1', 'kind': 'title'},
            {'x': 4.5, 'y': 2.84, 'label': 'Pair 2', 'kind': 'title'},
            {'x': 1.5, 'y': 0.90, 'label': 'Pick one winner', 'kind': 'body'},
            {'x': 1.5, 'y': 0.66, 'label': 'from pair (0,1)', 'kind': 'body'},
            {'x': 1.5, 'y': 0.40, 'label': 'x0 decides which item wins', 'kind': 'body'},
            {'x': 1.5, 'y': 0.18, 'label': 'inside pair (0,1)', 'kind': 'body'},
            {'x': 4.5, 'y': 0.90, 'label': 'Pick one winner', 'kind': 'body'},
            {'x': 4.5, 'y': 0.66, 'label': 'from pair (2,3)', 'kind': 'body'},
            {'x': 4.5, 'y': 0.40, 'label': 'x1 decides which item wins', 'kind': 'body'},
            {'x': 4.5, 'y': 0.18, 'label': 'inside pair (2,3)', 'kind': 'body'},
            {'x': 3.0, 'y': -0.50, 'label': 'x2, x3, x4 shift overall item values', 'kind': 'body'},
            {'x': 3.0, 'y': -0.72, 'label': 'but do not change the winner inside either pair', 'kind': 'body'},
        ]
    )
    context_labels = pd.DataFrame(
        [
            {'x': 1.225, 'y': 3.60, 'label': 'x0'},
            {'x': 1.775, 'y': 3.60, 'label': 'x1'},
            {'x': 2.575, 'y': 3.60, 'label': 'x2'},
            {'x': 3.125, 'y': 3.60, 'label': 'x3'},
            {'x': 3.675, 'y': 3.60, 'label': 'x4'},
        ]
    )
    item_labels = pd.DataFrame(
        [
            {'x': 1.1, 'y': 1.86, 'label': '0'},
            {'x': 1.9, 'y': 1.86, 'label': '1'},
            {'x': 4.1, 'y': 1.86, 'label': '2'},
            {'x': 4.9, 'y': 1.86, 'label': '3'},
        ]
    )

    return (
        ggplot()
        + geom_rect(
            data=pair_boxes,
            mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax'),
            fill=None,
            color='#6b7280',
            size=1.2,
        )
        + geom_rect(
            data=context_boxes,
            mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'),
            color='#374151',
            size=0.9,
            alpha=0.45,
        )
        + geom_rect(
            data=item_boxes,
            mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'),
            color='#374151',
            size=1.0,
            alpha=0.35,
        )
        + geom_text(
            data=context_labels,
            mapping=aes('x', 'y', label='label'),
            size=10,
            fontface='bold',
        )
        + geom_text(data=item_labels, mapping=aes('x', 'y', label='label'), size=18, fontface='bold')
        + geom_text(
            data=text[text['kind'] == 'title'],
            mapping=aes('x', 'y', label='label'),
            size=16,
            fontface='bold',
        )
        + geom_text(
            data=text[text['kind'] == 'body'],
            mapping=aes('x', 'y', label='label'),
            size=10,
        )
        + geom_text(
            data=text[text['kind'] == 'context_label'],
            mapping=aes('x', 'y', label='label'),
            size=10,
            fontface='bold',
        )
        + scale_fill_manual(
            values={
                'item': '#93c5fd',
                'nuisance': '#fde68a',
                'decision_ctx': '#93c5fd',
                'nuisance_ctx': '#fde68a',
            }
        )
        + coord_cartesian(xlim=(0.0, 6.0), ylim=(-1.08, 4.15))
        + ggsize(920, 520)
        + labs(title='How the decision problem is structured')
        + theme_minimal()
        + theme(
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            axis_title=element_blank(),
            panel_grid=element_blank(),
            legend_position='none',
            plot_title=element_text(size=18, face='bold'),
        )
    )


def _cost_correlation_metric(features: np.ndarray, total_cost: np.ndarray) -> float:
    max_corr = 0.0
    for j in range(features.shape[1]):
        r = np.corrcoef(features[:, j], total_cost)[0, 1]
        if np.isfinite(r):
            max_corr = max(max_corr, abs(r))
    return float(max_corr)


def _predict_mean_matrix(agent, X_test: np.ndarray) -> np.ndarray:
    rows = []
    for x in X_test:
        rows.append(np.asarray(agent.predict_mean(x), dtype=float))
    return np.vstack(rows)


def _safe_decision_probe_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    _, counts = np.unique(labels, return_counts=True)
    min_count = int(counts.min())
    if min_count < 2:
        return float('nan')
    n_splits = min(5, min_count)
    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(clf, features, labels, cv=cv)
    return float(scores.mean())


def _pair_decision_metrics(
    pred_values: np.ndarray,
    true_values: np.ndarray,
    solver,
    contested_quantile: float = 0.25,
) -> Dict[str, float]:
    true_sel = np.vstack([np.asarray(solver(v), dtype=float) for v in true_values])
    pred_sel = np.vstack([np.asarray(solver(v), dtype=float) for v in pred_values])
    greedy_decision_acc = float(np.mean(np.all(pred_sel == true_sel, axis=1)))

    pair_indices = [('pair_01', (0, 1)), ('pair_23', (2, 3))]
    pair_correct = []
    pair_margins = []
    per_pair_correct = {name: [] for name, _ in pair_indices}
    for pred_row, true_row in zip(pred_values, true_values):
        for name, (i, j) in pair_indices:
            pred_winner = i if pred_row[i] >= pred_row[j] else j
            true_winner = i if true_row[i] >= true_row[j] else j
            correct = float(pred_winner == true_winner)
            pair_correct.append(correct)
            pair_margins.append(float(abs(true_row[i] - true_row[j])))
            per_pair_correct[name].append(correct)
    pair_correct = np.asarray(pair_correct, dtype=float)
    pair_margins = np.asarray(pair_margins, dtype=float)
    pair_acc = float(pair_correct.mean())
    threshold = float(np.quantile(pair_margins, contested_quantile))
    contested_mask = pair_margins <= threshold
    contested_pair_acc = float(pair_correct[contested_mask].mean()) if np.any(contested_mask) else float('nan')
    return {
        'greedy_decision_acc': greedy_decision_acc,
        'pair_decision_acc': pair_acc,
        'contested_pair_acc': contested_pair_acc,
        'pair_01_acc': float(np.mean(per_pair_correct['pair_01'])),
        'pair_23_acc': float(np.mean(per_pair_correct['pair_23'])),
    }


def _mechanism_summary_table(
    regime: str,
    res_spo: Dict[str, list],
    res_mse: Dict[str, list],
    spo_agent,
    mse_agent,
    solver,
    h_spo: np.ndarray,
    h_mse: np.ndarray,
    X_test: np.ndarray,
    decisions_test: np.ndarray,
    values_test: np.ndarray,
) -> pd.DataFrame:
    oracle_h = X_test[:, :2]
    decision_labels = np.array([decision_label(d) for d in decisions_test])
    label_to_int = {label: i for i, label in enumerate(np.unique(decision_labels))}
    decision_ids = np.array([label_to_int[label] for label in decision_labels])
    total_cost = values_test.sum(axis=1)
    spo_pred = _predict_mean_matrix(spo_agent, X_test)
    mse_pred = _predict_mean_matrix(mse_agent, X_test)
    _, angles_spo = alignment_over_training(res_spo['W1_checkpoints'])
    _, angles_mse = alignment_over_training(res_mse['W1_checkpoints'])

    rows = [
        {
            'regime': regime,
            'agent': 'Oracle',
            'final_regret': np.nan,
            'decision_probe_acc': _safe_decision_probe_accuracy(oracle_h, decision_ids),
            'greedy_decision_acc': 1.0,
            'pair_decision_acc': 1.0,
            'contested_pair_acc': 1.0,
            'pair_01_acc': 1.0,
            'pair_23_acc': 1.0,
            'cost_correlation': _cost_correlation_metric(oracle_h, total_cost),
            'alignment_angle_deg': 0.0,
            'decision_loading': np.nan,
            'nuisance_loading': np.nan,
        }
    ]
    for label, result, features, checkpoints in (
        ('BayesianSPO', res_spo, h_spo, res_spo['W1_checkpoints']),
        ('BayesianMSE', res_mse, h_mse, res_mse['W1_checkpoints']),
    ):
        W1 = checkpoints[-1][1]
        row_norms = np.linalg.norm(W1, axis=1)
        pred_values = spo_pred if label == 'BayesianSPO' else mse_pred
        pair_metrics = _pair_decision_metrics(pred_values, values_test, solver)
        rows.append(
            {
                'regime': regime,
                'agent': label,
                'final_regret': float(result['cumulative_regret'][-1]),
                'decision_probe_acc': _safe_decision_probe_accuracy(features, decision_ids),
                'greedy_decision_acc': pair_metrics['greedy_decision_acc'],
                'pair_decision_acc': pair_metrics['pair_decision_acc'],
                'contested_pair_acc': pair_metrics['contested_pair_acc'],
                'pair_01_acc': pair_metrics['pair_01_acc'],
                'pair_23_acc': pair_metrics['pair_23_acc'],
                'cost_correlation': _cost_correlation_metric(features, total_cost),
                'alignment_angle_deg': angles_spo[-1] if label == 'BayesianSPO' else angles_mse[-1],
                'decision_loading': float(row_norms[:2].sum()),
                'nuisance_loading': float(row_norms[2:].sum()),
            }
        )
    return pd.DataFrame(rows)


def decode_latents(features: np.ndarray, values: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit linear probes to predict decision margins and shared offset from features."""
    from sklearn.linear_model import LinearRegression

    values = np.asarray(values, dtype=float)
    targets = {
        'margin (0,1)': values[:, 0] - values[:, 1],
        'margin (2,3)': values[:, 2] - values[:, 3],
        'shared offset': values.mean(axis=1),
    }
    predictions = {}
    for name, target in targets.items():
        reg = LinearRegression().fit(features, target)
        predictions[name] = reg.predict(features)
    return predictions


def plot_decoded_latents(
    features_spo: np.ndarray,
    features_mse: np.ndarray,
    values: np.ndarray,
    decisions: np.ndarray,
) -> 'ggplot':
    """Scatter: true task-relevant quantity vs decoded (predicted) quantity from features.

    3x2 grid: rows = pairwise decision margins and shared offset,
    cols = SPO / MSE. Diagonal = perfect recovery.
    """
    from lets_plot import geom_abline, gggrid

    dec_labels = [decision_label(d) for d in decisions]
    decoded_spo = decode_latents(features_spo, values)
    decoded_mse = decode_latents(features_mse, values)

    values = np.asarray(values, dtype=float)
    true_targets = {
        'margin (0,1)': values[:, 0] - values[:, 1],
        'margin (2,3)': values[:, 2] - values[:, 3],
        'shared offset': values.mean(axis=1),
    }

    panels = []
    for target_name, short, dim_type in [
        ('margin (0,1)', 'margin (0,1)', 'DECISION'),
        ('margin (2,3)', 'margin (2,3)', 'DECISION'),
        ('shared offset', 'shared offset', 'NUISANCE'),
    ]:
        for agent_name, decoded in [('BayesianSPO', decoded_spo), ('BayesianMSE', decoded_mse)]:
            true_vals = true_targets[target_name]
            pred_vals = decoded[target_name]
            r2 = float(1 - np.sum((true_vals - pred_vals)**2) / np.sum((true_vals - true_vals.mean())**2))

            rows = [{'true': float(true_vals[i]), 'predicted': float(pred_vals[i]),
                      'decision': dec_labels[i]}
                    for i in range(len(true_vals))]
            df = pd.DataFrame(rows)

            p = (
                ggplot(df, aes('true', 'predicted', color='decision'))
                + geom_point(size=1.5, alpha=0.4)
                + geom_abline(slope=1, intercept=0, linetype='dashed', color='gray', size=0.8)
                + scale_color_manual(values=DECISION_COLORS)
                + labs(
                    x=f'True {short}',
                    y=f'Predicted {short}',
                    title=f'[{dim_type}] {agent_name} → {target_name} (R²={r2:.2f})',
                )
                + theme_minimal()
                + ggsize(460, 420)
            )
            panels.append(p)

    return gggrid(panels, ncol=2)


def plot_feedback_feature_grid(
    features_spo: np.ndarray,
    features_mse: np.ndarray,
    X_test: np.ndarray,
    decisions: np.ndarray,
    values: np.ndarray,
    title_prefix: str,
):
    from lets_plot import (
        geom_hline,
        geom_path,
        geom_text,
        geom_vline,
        gggrid,
        scale_color_gradient,
        scale_color_manual,
        theme_minimal,
    )
    from scipy.spatial import ConvexHull

    oracle_h = X_test[:, :2]
    total_cost = values.sum(axis=1)
    decision_labels = np.array([decision_label(d) for d in decisions])
    label_to_int = {label: i for i, label in enumerate(np.unique(decision_labels))}
    decision_ids = np.array([label_to_int[label] for label in decision_labels])
    decision_colors = {
        '0,2': '#1f77b4',
        '0,3': '#ff7f0e',
        '1,2': '#2ca02c',
        '1,3': '#d62728',
    }

    def convex_hull_df(h1, h2, group_label):
        points = np.column_stack([h1, h2])
        if len(points) < 3:
            return []
        try:
            hull = ConvexHull(points)
        except Exception:
            return []
        verts = hull.vertices.tolist()
        verts.append(verts[0])
        return [
            {
                'h1': float(points[v, 0]),
                'h2': float(points[v, 1]),
                'decision': group_label,
            }
            for v in verts
        ]

    panels_row1 = []
    for label, h in (
        ('Oracle (x0, x1)', oracle_h),
        ('BayesianSPO', features_spo),
        ('BayesianMSE', features_mse),
    ):
        acc = _safe_decision_probe_accuracy(h, decision_ids)
        df = pd.DataFrame(
            {
                'h1': h[:, 0],
                'h2': h[:, 1],
                'decision': decision_labels,
            }
        )
        hull_rows = []
        for decision in decision_colors:
            mask = df['decision'] == decision
            if int(mask.sum()) >= 3:
                hull_rows.extend(
                    convex_hull_df(
                        df.loc[mask, 'h1'].to_numpy(),
                        df.loc[mask, 'h2'].to_numpy(),
                        decision,
                    )
                )
        hull_df = pd.DataFrame(hull_rows)
        x_ann = float(df['h1'].min() + 0.02 * (df['h1'].max() - df['h1'].min()))
        y_ann = float(df['h2'].max() - 0.02 * (df['h2'].max() - df['h2'].min()))
        ann_df = pd.DataFrame([{'h1': x_ann, 'h2': y_ann, 'text': f'accuracy = {acc:.0%}'}])

        plot = (
            ggplot(df, aes('h1', 'h2', color='decision'))
            + geom_point(size=2.0, alpha=0.4)
            + scale_color_manual(values=decision_colors)
            + geom_text(
                data=ann_df,
                mapping=aes('h1', 'h2', label='text'),
                color='black',
                size=9,
                hjust=0,
                fontface='bold',
            )
            + labs(title=label, x='Feature dim 1', y='Feature dim 2')
            + theme_minimal()
            + ggsize(460, 460)
        )
        if not hull_df.empty:
            plot = plot + geom_path(
                data=hull_df,
                mapping=aes('h1', 'h2', color='decision', group='decision'),
                size=1.0,
                alpha=0.7,
            )
        if label.startswith('Oracle'):
            plot = (
                plot
                + geom_vline(xintercept=0, linetype='dashed', color='gray', size=0.8)
                + geom_hline(yintercept=0, linetype='dashed', color='gray', size=0.8)
            )
        panels_row1.append(plot)

    panels_row2 = []
    for label, h in (
        ('Oracle (x0, x1)', oracle_h),
        ('BayesianSPO', features_spo),
        ('BayesianMSE', features_mse),
    ):
        corr = _cost_correlation_metric(h, total_cost)
        df = pd.DataFrame({'h1': h[:, 0], 'h2': h[:, 1], 'cost': total_cost})
        x_ann = float(df['h1'].min() + 0.02 * (df['h1'].max() - df['h1'].min()))
        y_ann = float(df['h2'].max() - 0.02 * (df['h2'].max() - df['h2'].min()))
        ann_df = pd.DataFrame([{'h1': x_ann, 'h2': y_ann, 'text': f'|r| = {corr:.2f}'}])
        panels_row2.append(
            ggplot(df, aes('h1', 'h2', color='cost'))
            + geom_point(size=2.0, alpha=0.6)
            + scale_color_gradient(low='#2166ac', high='#b2182b')
            + geom_text(
                data=ann_df,
                mapping=aes('h1', 'h2', label='text'),
                color='black',
                size=9,
                hjust=0,
                fontface='bold',
            )
            + labs(title=label, x='Feature dim 1', y='Feature dim 2')
            + theme_minimal()
            + ggsize(460, 460)
        )
    return gggrid(panels_row1 + panels_row2, ncol=3) + labs(title=title_prefix)


def plot_action_timeline(
    oracle_actions,
    spo_actions,
    mse_actions,
    title_prefix: str,
):
    from itertools import combinations
    from lets_plot import theme_minimal

    oracle_actions = [np.asarray(a).reshape(-1) for a in oracle_actions]
    n_arms = len(oracle_actions[0])
    action_size = int(round(float(np.sum(oracle_actions[0]))))
    action_space = []
    for combo in combinations(range(n_arms), action_size):
        action = np.zeros(n_arms, dtype=float)
        action[list(combo)] = 1.0
        action_space.append(action)

    def _label(action: np.ndarray) -> str:
        return ','.join(str(i) for i in np.where(np.asarray(action) > 0.5)[0])

    action_labels = [_label(action) for action in action_space]

    def _panel(actions, agent_label):
        rows = []
        for t, (oracle_action, action) in enumerate(zip(oracle_actions, actions), start=1):
            oracle_label = _label(oracle_action)
            action_label = _label(np.asarray(action).reshape(-1))
            for label in action_labels:
                if label == action_label == oracle_label:
                    status = 'Correct action'
                elif label == action_label and action_label != oracle_label:
                    status = 'Chosen wrong'
                elif label == oracle_label and action_label != oracle_label:
                    status = 'Oracle missed'
                else:
                    status = 'Other'
                rows.append({'round': t, 'action': label, 'status': status})
        df = pd.DataFrame(rows, columns=['round', 'action', 'status'])
        return (
            ggplot(df, aes('round', 'action', fill='status'))
            + geom_tile()
            + scale_fill_manual(
                values={
                    'Correct action': '#2ca02c',
                    'Chosen wrong': '#d62728',
                    'Oracle missed': '#ffbf00',
                    'Other': '#f5f5f5',
                }
            )
            + labs(title=agent_label, x='Round', y='Action', fill='')
            + theme_minimal()
            + theme(
                panel_grid_minor='blank',
                panel_grid='blank',
            )
            + ggsize(420, 220)
        )

    plots = [
        _panel(oracle_actions, 'Oracle'),
        _panel(spo_actions, 'BayesianSPO'),
        _panel(mse_actions, 'BayesianMSE'),
    ]
    return gggrid(plots, ncol=3) + labs(title=title_prefix)


def plot_regret_per_round(
    res_spo: Dict[str, list],
    res_mse: Dict[str, list],
    title_prefix: str,
    smooth_window: int = 25,
):
    from lets_plot import scale_color_manual, theme_minimal

    rows = []
    for label, regrets in (('BayesianSPO', res_spo['regrets']), ('BayesianMSE', res_mse['regrets'])):
        series = pd.Series(np.asarray(regrets, dtype=float)).rolling(
            window=smooth_window, min_periods=1
        ).mean()
        for round_idx, value in enumerate(series, start=1):
            rows.append({'round': round_idx, 'regret': float(value), 'agent': label})
    df = pd.DataFrame(rows)
    return (
        ggplot(df, aes('round', 'regret', color='agent'))
        + geom_line(size=1.1)
        + scale_color_manual(values={'BayesianSPO': '#1f77b4', 'BayesianMSE': '#d62728'})
        + labs(title=title_prefix, x='Round', y='Per-round regret')
        + theme_minimal()
        + theme(panel_grid_minor='blank')
        + ggsize(400, 120)
    )


def plot_cumulative_regret(
    res_spo: Dict[str, list],
    res_mse: Dict[str, list],
    title_prefix: str,
):
    from lets_plot import scale_color_manual, theme_minimal

    rows = []
    for label, regrets in (
        ('BayesianSPO', res_spo['cumulative_regret']),
        ('BayesianMSE', res_mse['cumulative_regret']),
    ):
        for round_idx, value in enumerate(np.asarray(regrets, dtype=float), start=1):
            rows.append({'round': round_idx, 'cumulative_regret': float(value), 'agent': label})
    df = pd.DataFrame(rows)
    return (
        ggplot(df, aes('round', 'cumulative_regret', color='agent'))
        + geom_line(size=1.1)
        + scale_color_manual(values={'BayesianSPO': '#1f77b4', 'BayesianMSE': '#d62728'})
        + labs(title=title_prefix, x='Round', y='Cumulative regret')
        + theme_minimal()
        + theme(panel_grid_minor='blank')
        + ggsize(400, 120)
    )


def plot_prediction_error_trajectory(
    res_spo: Dict[str, list],
    res_mse: Dict[str, list],
    title_prefix: str,
):
    from lets_plot import scale_color_manual, theme_minimal

    rows = []
    for label, checkpoints in (
        ('BayesianSPO', res_spo.get('prediction_error_checkpoints', [])),
        ('BayesianMSE', res_mse.get('prediction_error_checkpoints', [])),
    ):
        for round_idx, value in checkpoints:
            rows.append({'round': int(round_idx), 'prediction_error': float(value), 'agent': label})
    df = pd.DataFrame(rows)
    return (
        ggplot(df, aes('round', 'prediction_error', color='agent'))
        + geom_line(size=1.1)
        + geom_point(size=2.0)
        + scale_color_manual(values={'BayesianSPO': '#1f77b4', 'BayesianMSE': '#d62728'})
        + labs(title=title_prefix, x='Round', y='Prediction MSE')
        + theme_minimal()
        + theme(panel_grid_minor='blank')
        + ggsize(300, 120)
    )


def build_regime_dashboard(
    feature_plot,
    decoded_latents_plot,
    regret_plot,
    cumulative_regret_plot,
    prediction_error_plot,
    loading_plot,
    action_plot,
    title_prefix: str,
):
    regret_row = gggrid(
        [regret_plot, cumulative_regret_plot],
        ncol=2,
        widths=[1, 1],
    ) + ggsize(960, 130)
    diagnostics_row = gggrid(
        [prediction_error_plot, loading_plot],
        ncol=2,
        widths=[1, 1],
    ) + ggsize(960, 130)
    return (
        gggrid(
            [
                feature_plot,
                decoded_latents_plot,
                regret_row,
                diagnostics_row,
                action_plot,
            ],
            ncol=1,
            heights=[1.15, 1.1, 0.52, 0.52, 0.95],
            vspace=6,
        )
        + ggsize(1700, 2925)
    )


def _slugify_regime(regime: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', regime.lower()).strip('_')
    return slug or 'regime'


def save_dashboard_plot(dashboard_plot, regime: str) -> str:
    output_dir = Path('results') / 'feature_representation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'{_slugify_regime(regime)}_dashboard.svg'
    return ggsave(dashboard_plot, filename, path=str(output_dir), iframe=False)


def run_feedback_regime_experiment(
    regime: str,
    full_feedback: bool,
    n_rounds: int = 5000,
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
    pair_swap_augment_spo: bool = False,
    pair_swap_augment_mse: bool = False,
    pair_balanced_spo: bool = False,
    pair_balanced_mse: bool = False,
    disagreement_coverage_spo: bool = False,
    disagreement_coverage_mse: bool = False,
    disagreement_samples: int = 16,
    disagreement_lambda: float = 1.0,
) -> Dict[str, object]:
    from src.agents.base import pregenerate_episode

    env = DecisionFocusedEnv(seed=42, noise_width=noise_width)
    episode = pregenerate_episode(env, n_rounds=n_rounds)
    env_test = DecisionFocusedEnv(seed=42, noise_width=noise_width)
    X_test, decisions_test, values_test = generate_test_contexts(env_test, n_samples=n_test, seed=test_seed)
    spo_agent, mse_agent = create_agent_pair(
        env,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        seed=seed,
        mask_spo=mask_spo,
        mask_mse=mask_mse,
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
        pair_swap_augment=pair_swap_augment_spo,
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
        pair_swap_augment=pair_swap_augment_mse,
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
        title_prefix=f'{regime}: decision structure vs nuisance prediction',
    )
    regret_plot = plot_regret_per_round(
        res_spo,
        res_mse,
        title_prefix=f'{regime}: regret per round',
    )
    cumulative_regret_plot = plot_cumulative_regret(
        res_spo,
        res_mse,
        title_prefix=f'{regime}: cumulative regret',
    )
    prediction_error_plot = plot_prediction_error_trajectory(
        res_spo,
        res_mse,
        title_prefix=f'{regime}: prediction error over training',
    )
    rounds_spo, angles_spo = alignment_over_training(res_spo['W1_checkpoints'])
    rounds_mse, angles_mse = alignment_over_training(res_mse['W1_checkpoints'])
    alignment_plot = plot_alignment_trajectory(rounds_spo, angles_spo, rounds_mse, angles_mse)
    loading_plot = plot_feature_directions(
        res_spo['W1_checkpoints'][-1][1],
        res_mse['W1_checkpoints'][-1][1],
    )
    oracle_actions = [rd['optimal_action'] for rd in episode['rounds_data']]
    action_plot = plot_action_timeline(
        oracle_actions,
        res_spo['actions'],
        res_mse['actions'],
        title_prefix=f'{regime}: action-space match vs oracle',
    )
    decoded_latents_plot = plot_decoded_latents(
        h_spo, h_mse, values_test, decisions_test,
    )
    dashboard_plot = build_regime_dashboard(
        feature_plot,
        decoded_latents_plot,
        regret_plot,
        cumulative_regret_plot,
        prediction_error_plot,
        loading_plot,
        action_plot,
        title_prefix=f'{regime}: dashboard',
    )
    dashboard_path = save_dashboard_plot(dashboard_plot, regime)
    return {
        'regime': regime,
        'summary': summary,
        'feature_plot': feature_plot,
        'regret_plot': regret_plot,
        'cumulative_regret_plot': cumulative_regret_plot,
        'prediction_error_plot': prediction_error_plot,
        'alignment_plot': alignment_plot,
        'loading_plot': loading_plot,
        'action_plot': action_plot,
        'decoded_latents_plot': decoded_latents_plot,
        'dashboard_plot': dashboard_plot,
        'dashboard_path': dashboard_path,
        'res_spo': res_spo,
        'res_mse': res_mse,
        'h_spo': h_spo,
        'h_mse': h_mse,
        'X_test': X_test,
        'decisions_test': decisions_test,
        'values_test': values_test,
    }


# --- Consolidated exact-2 helpers ---

from pathlib import Path
import re
from typing import Dict

import numpy as np
import pandas as pd

from src.agents.base import pregenerate_episode

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
    replay_batch_size: int = 1,
    posterior_imputed_spo: bool = False,
    posterior_imputed_predictive_spo: bool = False,
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
    weber_blended_spo: bool = False,
    weber_pfy: bool = False,
    weber_bll: bool = False,
    weber_spo_blend_weight: float = 0.5,
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
        replay_batch_size=replay_batch_size,
        posterior_imputed_spo=posterior_imputed_spo,
        posterior_imputed_predictive_spo=posterior_imputed_predictive_spo,
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
        weber_blended_spo=weber_blended_spo,
        weber_pfy=weber_pfy,
        weber_bll=weber_bll,
        weber_spo_blend_weight=weber_spo_blend_weight,
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
        replay_batch_size=replay_batch_size,
        posterior_imputed_spo=posterior_imputed_spo,
        posterior_imputed_predictive_spo=posterior_imputed_predictive_spo,
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
        weber_blended_spo=weber_blended_spo,
        weber_pfy=weber_pfy,
        weber_bll=weber_bll,
        weber_spo_blend_weight=weber_spo_blend_weight,
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

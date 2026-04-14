from typing import Dict, Iterable, List, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from lets_plot import (
    aes,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    gggrid,
    ggplot,
    ggsize,
    labs,
    theme,
)

from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.agents.bayesian_utils import (
    compute_bayesian_predictions,
    compute_uncertainties,
    init_per_arm_regression,
    update_bayesian_stats,
)
from src.agents.contextual import BayesianMSEAgent, BayesianPFYAgent, BayesianSPOAgent
from src.envs.topk import TopKEnv


DEFAULT_LR = 0.001
DEFAULT_BETA = 1.0
DEFAULT_LAMBDA_REG = 1.0


def default_main_configs(
    include_mse: bool = True,
    include_pfy: bool = True,
) -> List[Tuple[str, Dict]]:
    configs = [
        (
            'BayesSPO (full)',
            dict(agent_name='BayesianSPO', full_feedback=True, mask_grad=True),
        ),
        (
            'BayesSPO (partial, masked)',
            dict(agent_name='BayesianSPO', full_feedback=False, mask_grad=True),
        ),
        (
            'BayesSPO (partial, unmasked)',
            dict(agent_name='BayesianSPO', full_feedback=False, mask_grad=False),
        ),
        (
            'BayesSPO-Phased (partial)',
            dict(
                agent_name='BayesianSPO',
                full_feedback=False,
                mask_grad=True,
                phased=True,
            ),
        ),
    ]
    if include_mse:
        configs.extend(
            [
                (
                    'BayesianMSE (full)',
                    dict(agent_name='BayesianMSE', full_feedback=True, mask_grad=False),
                ),
                (
                    'BayesianMSE (partial)',
                    dict(agent_name='BayesianMSE', full_feedback=False, mask_grad=False),
                ),
                (
                    'BayesianMSE-Phased (partial)',
                    dict(
                        agent_name='BayesianMSE',
                        full_feedback=False,
                        mask_grad=False,
                        phased=True,
                    ),
                ),
            ]
        )
    if include_pfy:
        configs.extend(
            [
                (
                    'BayesianPFY (partial)',
                    dict(agent_name='BayesianPFY', full_feedback=False, mask_grad=True),
                ),
                (
                    'BayesianPFY-Phased (partial)',
                    dict(
                        agent_name='BayesianPFY',
                        full_feedback=False,
                        mask_grad=True,
                        phased=True,
                    ),
                ),
            ]
        )
    return configs


def default_repair_configs() -> List[Tuple[str, Dict]]:
    return [
        (
            'BayesSPO (full)',
            dict(agent_name='BayesianSPO', full_feedback=True, mask_grad=True),
        ),
        (
            'BayesSPO (partial, masked)',
            dict(agent_name='BayesianSPO', full_feedback=False, mask_grad=True),
        ),
        (
            'BayesSPO-Phased (partial)',
            dict(
                agent_name='BayesianSPO',
                full_feedback=False,
                mask_grad=True,
                phased=True,
            ),
        ),
        (
            'BayesSPO (partial, posterior)',
            dict(
                agent_name='PosteriorSampledSPO',
                full_feedback=False,
                mask_grad=True,
                target_samples=16,
            ),
        ),
        (
            'BayesSPO (partial, posterior+boundary)',
            dict(
                agent_name='PosteriorSampledSPO',
                full_feedback=False,
                mask_grad=True,
                target_samples=16,
                boundary_explore=True,
                action_samples=96,
                boundary_bonus=2.0,
            ),
        ),
        (
            'BayesSPO-Phased (posterior)',
            dict(
                agent_name='PosteriorSampledSPO',
                full_feedback=False,
                mask_grad=True,
                phased=True,
                target_samples=16,
            ),
        ),
    ]


def default_structured_repair_configs() -> List[Tuple[str, Dict]]:
    return [
        (
            'BayesSPO (full)',
            dict(agent_name='BayesianSPO', full_feedback=True, mask_grad=True),
        ),
        (
            'BayesSPO-Phased (partial)',
            dict(
                agent_name='BayesianSPO',
                full_feedback=False,
                mask_grad=True,
                phased=True,
            ),
        ),
        (
            'BayesSPO (partial, margin explore)',
            dict(
                agent_name='StructuredRepairSPO',
                full_feedback=False,
                mask_grad=True,
                margin_explore=True,
                phase_length=50,
                phase_epochs=5,
            ),
        ),
        (
            'BayesSPO-Phased (margin only)',
            dict(
                agent_name='StructuredRepairSPO',
                full_feedback=False,
                mask_grad=True,
                phased=True,
                margin_explore=True,
            ),
        ),
        (
            'BayesSPO-Phased (boundary weight)',
            dict(
                agent_name='StructuredRepairSPO',
                full_feedback=False,
                mask_grad=True,
                phased=True,
                boundary_focus=1.0,
                boundary_scale=0.1,
            ),
        ),
        (
            'BayesSPO-Phased (residual shift)',
            dict(
                agent_name='StructuredRepairSPO',
                full_feedback=False,
                mask_grad=True,
                phased=True,
                residual_strength=0.5,
                residual_scale_by_sigma=True,
            ),
        ),
        (
            'BayesSPO-Phased (structured)',
            dict(
                agent_name='StructuredRepairSPO',
                full_feedback=False,
                mask_grad=True,
                phased=True,
                margin_explore=True,
                boundary_focus=1.0,
                boundary_scale=0.1,
                residual_strength=0.5,
                residual_scale_by_sigma=True,
            ),
        ),
    ]


def default_structured_scale_configs() -> List[Tuple[str, Dict]]:
    return [
        (
            'BayesSPO (full)',
            dict(agent_name='BayesianSPO', full_feedback=True, mask_grad=True),
        ),
        (
            'BayesSPO-Phased (partial)',
            dict(
                agent_name='BayesianSPO',
                full_feedback=False,
                mask_grad=True,
                phased=True,
            ),
        ),
        (
            'BayesSPO (partial, margin explore)',
            dict(
                agent_name='StructuredRepairSPO',
                full_feedback=False,
                mask_grad=True,
                margin_explore=True,
                phase_length=50,
                phase_epochs=5,
            ),
        ),
        (
            'BayesSPO-Phased (structured)',
            dict(
                agent_name='StructuredRepairSPO',
                full_feedback=False,
                mask_grad=True,
                phased=True,
                margin_explore=True,
                boundary_focus=1.0,
                boundary_scale=0.1,
                residual_strength=0.5,
                residual_scale_by_sigma=True,
            ),
        ),
    ]


def default_scale_configs() -> List[Tuple[str, Dict]]:
    return default_main_configs(include_mse=True)


class PosteriorSampledSPOAgent(BayesianSPOAgent):

    def __init__(
        self,
        *args,
        target_samples: int = 16,
        boundary_explore: bool = False,
        action_samples: int = 64,
        boundary_bonus: float = 1.0,
        sample_clip=None,
        seed: int = None,
        **kwargs,
    ):
        super().__init__(*args, seed=seed, **kwargs)
        self.target_samples = max(1, int(target_samples))
        self.boundary_explore = boundary_explore
        self.action_samples = max(1, int(action_samples))
        self.boundary_bonus = float(boundary_bonus)
        self.sample_clip = sample_clip
        self.posterior_rng = np.random.default_rng(0 if seed is None else seed + 1729)

    def _sample_costs(self, pred, sigma, n_samples: int) -> np.ndarray:
        samples = np.asarray(pred)[None, :] + np.asarray(sigma)[None, :] * self.posterior_rng.standard_normal((n_samples, self.n_arms))
        if self.sample_clip is not None:
            lo, hi = self.sample_clip
            samples = np.clip(samples, lo, hi)
        return samples

    def _compute_posterior_grad_h(self, h, rewards, obs_mask, full_feedback: bool):
        pred = np.asarray(compute_bayesian_predictions(h, self.A_inv, self.b_bayes))
        if full_feedback:
            target = jnp.asarray(rewards)
            return self._compute_spo_grad_h(None, h, target, jnp.asarray(obs_mask), True)

        sigma = np.asarray(compute_uncertainties(h, self.A_inv))
        y_targets = self._sample_costs(pred, sigma, self.target_samples)
        obs_idx = np.asarray(obs_mask, dtype=bool)
        y_targets[:, obs_idx] = np.asarray(rewards)[obs_idx]
        spo_grads = np.stack([np.asarray(self.loss_grad_fn(pred, y_target)) for y_target in y_targets], axis=0)
        avg_grad = jnp.asarray(spo_grads.mean(axis=0))
        if self.mask_grad:
            avg_grad = jnp.where(jnp.asarray(obs_mask), avg_grad, 0.0)
        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        return jnp.einsum('n,ni->i', avg_grad, theta)

    def act(self, X):
        self.t += 1
        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h = self._features(X)
        pred = np.asarray(compute_bayesian_predictions(h, self.A_inv, self.b_bayes))
        if not self.boundary_explore:
            sigma = np.asarray(compute_uncertainties(h, self.A_inv))
            if self.minimize:
                action_scores = pred - self.beta * sigma
            else:
                action_scores = pred + self.beta * sigma
            return self.solver(action_scores)

        sigma = np.asarray(compute_uncertainties(h, self.A_inv))
        posterior_costs = self._sample_costs(pred, sigma, self.action_samples)
        sampled_actions = np.stack([np.asarray(self.solver(sample)).flatten() for sample in posterior_costs], axis=0)
        inclusion_prob = sampled_actions.mean(axis=0)
        clipped = np.clip(inclusion_prob, 1e-6, 1.0 - 1e-6)
        entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)) / np.log(2.0)
        decision_bonus = self.boundary_bonus * self.beta * entropy * sigma
        if self.minimize:
            action_scores = pred - decision_bonus
        else:
            action_scores = pred + decision_bonus
        return self.solver(action_scores)

    def _phased_retrain(self):
        for _epoch in range(self.phase_epochs):
            for i in range(len(self._phase_buffer_X)):
                X_buf = self._phase_buffer_X[i]
                Y_buf = jnp.asarray(self._phase_buffer_Y[i])
                obs_mask = jnp.asarray(self._phase_buffer_obs[i], dtype=jnp.bool_)
                h = self._features(X_buf)
                grad_h = self._compute_posterior_grad_h(h, Y_buf, obs_mask, bool(jnp.all(obs_mask)))
                self._backprop_and_step(X_buf, grad_h)

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            self.n_arms, self.hidden_dim, self.lambda_reg)
        for i in range(len(self._phase_buffer_X)):
            h = self._features(self._phase_buffer_X[i])
            obs_mask = jnp.asarray(self._phase_buffer_obs[i], dtype=jnp.bool_)
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
                self.A, self.b_bayes, self.A_inv, h, rewards, obs_mask)

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False, **kwargs):
        prob_values = kwargs.get('prob_values')
        if prob_values is not None:
            return super().update(X, selection, rewards, lr=lr, full_feedback=full_feedback, **kwargs)

        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)
        h = self._features(X)
        selection_mask = selection > 0.5
        obs_mask = jnp.ones(self.n_arms, dtype=jnp.bool_) if full_feedback else selection_mask
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, obs_mask
        )

        if self.phased:
            self._append_phase_buffer(X, rewards, selection, prob_values, obs_mask)
            self._rounds_since_retrain += 1
            if self._rounds_since_retrain >= self.phase_length:
                self._phased_retrain()
                self._rounds_since_retrain = 0
            return

        grad_h = self._compute_posterior_grad_h(h, rewards, obs_mask, full_feedback)
        self._backprop_and_step(X, grad_h)


class StructuredRepairSPOAgent(BayesianSPOAgent):

    def __init__(
        self,
        *args,
        margin_explore: bool = False,
        boundary_focus: float = 0.0,
        boundary_scale: float = 0.15,
        residual_strength: float = 0.0,
        residual_scale_by_sigma: bool = True,
        sample_clip=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.margin_explore = margin_explore
        self.boundary_focus = float(boundary_focus)
        self.boundary_scale = max(float(boundary_scale), 1e-6)
        self.residual_strength = float(residual_strength)
        self.residual_scale_by_sigma = residual_scale_by_sigma
        self.sample_clip = sample_clip

    def _boundary_indices(self, pred: np.ndarray):
        k = int(round(float(self.capacity)))
        if self.minimize:
            order = np.argsort(pred)
            inside = order[:k]
            outside = order[k:]
            if len(outside) == 0:
                return inside, outside, None, None
            in_boundary = inside[np.argmax(pred[inside])]
            out_boundary = outside[np.argmin(pred[outside])]
        else:
            order = np.argsort(-pred)
            inside = order[:k]
            outside = order[k:]
            if len(outside) == 0:
                return inside, outside, None, None
            in_boundary = inside[np.argmin(pred[inside])]
            out_boundary = outside[np.argmax(pred[outside])]
        return inside, outside, int(in_boundary), int(out_boundary)

    def _decision_margin(self, pred: np.ndarray) -> float:
        _, outside, in_boundary, out_boundary = self._boundary_indices(pred)
        if len(outside) == 0 or in_boundary is None or out_boundary is None:
            return np.inf
        if self.minimize:
            return float(pred[out_boundary] - pred[in_boundary])
        return float(pred[in_boundary] - pred[out_boundary])

    def _boundary_weight(self, pred: np.ndarray) -> float:
        if self.boundary_focus <= 0.0:
            return 1.0
        margin = abs(self._decision_margin(pred))
        return 1.0 + self.boundary_focus * float(np.exp(-margin / self.boundary_scale))

    def _residual_target(self, h, pred, rewards, obs_mask, full_feedback: bool):
        pred_np = np.asarray(pred, dtype=float)
        rewards_np = np.asarray(rewards, dtype=float)
        obs_idx = np.asarray(obs_mask, dtype=bool)
        if full_feedback or not np.any(obs_idx) or self.residual_strength == 0.0:
            target = pred_np.copy()
            target[obs_idx] = rewards_np[obs_idx]
            if full_feedback:
                target = rewards_np.copy()
            return jnp.asarray(target)

        residual = float(np.mean(rewards_np[obs_idx] - pred_np[obs_idx]))
        target = pred_np.copy()
        if self.residual_scale_by_sigma:
            sigma = np.asarray(compute_uncertainties(h, self.A_inv), dtype=float)
            mean_obs_sigma = max(float(np.mean(sigma[obs_idx])), 1e-6)
            scale = np.clip(sigma / mean_obs_sigma, 0.5, 2.0)
        else:
            scale = 1.0
        target = target + self.residual_strength * residual * scale
        target[obs_idx] = rewards_np[obs_idx]
        if self.sample_clip is not None:
            lo, hi = self.sample_clip
            target = np.clip(target, lo, hi)
        return jnp.asarray(target)

    def act(self, X):
        self.t += 1
        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h = self._features(X)
        pred = np.asarray(compute_bayesian_predictions(h, self.A_inv, self.b_bayes))
        if not self.margin_explore:
            sigma = np.asarray(compute_uncertainties(h, self.A_inv))
            if self.minimize:
                action_scores = pred - self.beta * sigma
            else:
                action_scores = pred + self.beta * sigma
            return self.solver(action_scores)

        sigma = np.asarray(compute_uncertainties(h, self.A_inv))
        inside, outside, in_boundary, out_boundary = self._boundary_indices(pred)
        if len(outside) == 0 or in_boundary is None or out_boundary is None:
            return self.solver(pred)
        adjusted = pred.copy()
        if self.minimize:
            adjusted[inside] = pred[inside] + self.beta * np.sqrt(sigma[inside] ** 2 + sigma[out_boundary] ** 2)
            adjusted[outside] = pred[outside] - self.beta * np.sqrt(sigma[outside] ** 2 + sigma[in_boundary] ** 2)
        else:
            adjusted[inside] = pred[inside] - self.beta * np.sqrt(sigma[inside] ** 2 + sigma[out_boundary] ** 2)
            adjusted[outside] = pred[outside] + self.beta * np.sqrt(sigma[outside] ** 2 + sigma[in_boundary] ** 2)
        return self.solver(adjusted)

    def _apply_structured_update(self, X, h, rewards, obs_mask, full_feedback: bool):
        pred = np.asarray(compute_bayesian_predictions(h, self.A_inv, self.b_bayes))
        y_target = self._residual_target(h, pred, rewards, obs_mask, full_feedback)
        grad_h = self._compute_spo_grad_h(X, h, y_target, jnp.asarray(obs_mask), full_feedback)
        grad_h = grad_h * self._boundary_weight(pred)
        self._backprop_and_step(X, grad_h)

    def _phased_retrain(self):
        for _epoch in range(self.phase_epochs):
            for i in range(len(self._phase_buffer_X)):
                X_buf = self._phase_buffer_X[i]
                Y_buf = jnp.asarray(self._phase_buffer_Y[i])
                obs_mask = jnp.asarray(self._phase_buffer_obs[i], dtype=jnp.bool_)
                h = self._features(X_buf)
                self._apply_structured_update(X_buf, h, Y_buf, obs_mask, bool(jnp.all(obs_mask)))

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            self.n_arms, self.hidden_dim, self.lambda_reg)
        for i in range(len(self._phase_buffer_X)):
            h = self._features(self._phase_buffer_X[i])
            obs_mask = jnp.asarray(self._phase_buffer_obs[i], dtype=jnp.bool_)
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
                self.A, self.b_bayes, self.A_inv, h, rewards, obs_mask)

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False, **kwargs):
        prob_values = kwargs.get('prob_values')
        if prob_values is not None:
            return super().update(X, selection, rewards, lr=lr, full_feedback=full_feedback, **kwargs)

        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)
        h = self._features(X)
        selection_mask = selection > 0.5
        obs_mask = jnp.ones(self.n_arms, dtype=jnp.bool_) if full_feedback else selection_mask
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, obs_mask
        )

        if self.phased:
            self._append_phase_buffer(X, rewards, selection, prob_values, obs_mask)
            self._rounds_since_retrain += 1
            if self._rounds_since_retrain >= self.phase_length:
                self._phased_retrain()
                self._rounds_since_retrain = 0
            return

        self._apply_structured_update(X, h, rewards, obs_mask, full_feedback)


def generate_shared_weights(
    n_arms: int,
    context_dim: int,
    w_spread: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((1, context_dim))
        + rng.standard_normal((n_arms, context_dim)) * w_spread
    )


def build_fixed_topk_env(
    W_true: np.ndarray,
    k: int,
    seed: int,
    cost_fn: str = 'sigmoid',
    noise_width: float = 0.0,
) -> TopKEnv:
    W_true = np.asarray(W_true, dtype=float)
    env = TopKEnv(
        n_arms=W_true.shape[0],
        k=k,
        context_dim=W_true.shape[1],
        cost_fn=cost_fn,
        noise_width=noise_width,
        seed=seed,
    )
    env.W = W_true.copy()
    env.reset()
    return env


def create_agent(
    agent_name: str,
    env: TopKEnv,
    seed: int,
    n_rounds: int,
    hidden_dim: int = 8,
    lr: float = DEFAULT_LR,
    beta: float = DEFAULT_BETA,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    mask_grad: bool = True,
    phased: bool = False,
    phase_length: int = 50,
    phase_epochs: int = 5,
    nonlinear: bool = False,
    target_samples: int = 16,
    boundary_explore: bool = False,
    action_samples: int = 64,
    boundary_bonus: float = 1.0,
    margin_explore: bool = False,
    boundary_focus: float = 0.0,
    boundary_scale: float = 0.15,
    residual_strength: float = 0.0,
    residual_scale_by_sigma: bool = True,
    sample_clip=None,
):
    bayes_beta = -beta if getattr(env, 'minimize', False) else beta
    common = dict(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        hidden_dim=hidden_dim,
        beta=bayes_beta,
        lambda_reg=lambda_reg,
        lr=lr,
        decay_rounds=n_rounds,
        solver=env.solve,
        seed=seed,
        phased=phased,
        phase_length=phase_length,
        phase_epochs=phase_epochs,
        nonlinear=nonlinear,
    )

    if agent_name == 'BayesianSPO':
        return BayesianSPOAgent(
            minimize=getattr(env, 'minimize', False),
            mask_grad=mask_grad,
            **common,
        )
    if agent_name == 'PosteriorSampledSPO':
        return PosteriorSampledSPOAgent(
            minimize=getattr(env, 'minimize', False),
            mask_grad=mask_grad,
            target_samples=target_samples,
            boundary_explore=boundary_explore,
            action_samples=action_samples,
            boundary_bonus=boundary_bonus,
            sample_clip=sample_clip,
            **common,
        )
    if agent_name == 'StructuredRepairSPO':
        return StructuredRepairSPOAgent(
            minimize=getattr(env, 'minimize', False),
            mask_grad=mask_grad,
            margin_explore=margin_explore,
            boundary_focus=boundary_focus,
            boundary_scale=boundary_scale,
            residual_strength=residual_strength,
            residual_scale_by_sigma=residual_scale_by_sigma,
            sample_clip=sample_clip,
            **common,
        )
    if agent_name == 'BayesianMSE':
        return BayesianMSEAgent(
            mask_grad=mask_grad,
            **common,
        )
    if agent_name == 'BayesianPFY':
        return BayesianPFYAgent(
            minimize=getattr(env, 'minimize', False),
            mask_grad=mask_grad,
            **common,
        )
    raise ValueError(f'Unsupported agent for this notebook: {agent_name}')


def _append_round_rows(
    rows: List[Dict],
    agent_label: str,
    seed: int,
    result: Dict,
) -> None:
    n_rounds = len(result['cumulative_regret'])
    metric_keys = [
        'contested_mae',
        'noncontested_mae',
        'rank_corr',
        'mean_sigma',
        'contested_sigma',
        'flip_set_size',
        'decision_correct',
        'contested_sigma_sum',
        'flip_contested_mae',
    ]
    for round_idx in range(n_rounds):
        row = {
            'agent': agent_label,
            'seed': seed,
            'round': round_idx,
            'instant_regret': result['regrets'][round_idx],
            'cumulative_regret': result['cumulative_regret'][round_idx],
        }
        for key in metric_keys:
            values = result.get(key)
            row[key] = np.nan if values is None else values[round_idx]
        rows.append(row)


def summarize_suite(df_rounds: pd.DataFrame) -> pd.DataFrame:
    final = (
        df_rounds.sort_values('round')
        .groupby(['agent', 'seed'], as_index=False)
        .tail(1)[['agent', 'seed', 'cumulative_regret']]
        .rename(columns={'cumulative_regret': 'final_regret'})
    )
    by_seed_metrics = (
        df_rounds.groupby(['agent', 'seed'], as_index=False)
        .agg(
            decision_rate=('decision_correct', 'mean'),
            mean_flip_size=('flip_set_size', 'mean'),
            mean_contested_mae=('contested_mae', 'mean'),
            mean_contested_sigma=('contested_sigma', 'mean'),
            mean_rank_corr=('rank_corr', 'mean'),
        )
    )
    merged = final.merge(by_seed_metrics, on=['agent', 'seed'], how='left')
    summary = (
        merged.groupby('agent', as_index=False)
        .agg(
            regret_mean=('final_regret', 'mean'),
            regret_std=('final_regret', 'std'),
            decision_rate=('decision_rate', 'mean'),
            mean_flip_size=('mean_flip_size', 'mean'),
            mean_contested_mae=('mean_contested_mae', 'mean'),
            mean_contested_sigma=('mean_contested_sigma', 'mean'),
            mean_rank_corr=('mean_rank_corr', 'mean'),
        )
        .sort_values('regret_mean')
        .reset_index(drop=True)
    )
    summary['regret_std'] = summary['regret_std'].fillna(0.0)
    return summary


def print_regret_summary(summary: pd.DataFrame) -> None:
    for row in summary.itertuples(index=False):
        print(f'{row.agent:30s}: regret = {row.regret_mean:.1f} +/- {row.regret_std:.1f}')


def run_suite(
    W_true: np.ndarray,
    k: int,
    configs: Iterable[Tuple[str, Dict]],
    n_rounds: int = 2000,
    n_seeds: int = 5,
    hidden_dim: int = 8,
    cost_fn: str = 'sigmoid',
    lr: float = DEFAULT_LR,
    beta: float = DEFAULT_BETA,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    phase_length: int = 50,
    phase_epochs: int = 5,
    nonlinear: bool = False,
    noise_width: float = 0.0,
    capture_agents: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    round_rows: List[Dict] = []
    artifacts: Dict[str, Dict] = {}
    for seed in range(n_seeds):
        env = build_fixed_topk_env(
            W_true=W_true,
            k=k,
            seed=seed,
            cost_fn=cost_fn,
            noise_width=noise_width,
        )
        episode = pregenerate_episode(env, n_rounds=n_rounds)
        for label, cfg in configs:
            agent = create_agent(
                agent_name=cfg['agent_name'],
                env=env,
                seed=1000 + seed,
                n_rounds=n_rounds,
                hidden_dim=hidden_dim,
                lr=lr,
                beta=beta,
                lambda_reg=lambda_reg,
                mask_grad=cfg.get('mask_grad', True),
                phased=cfg.get('phased', False),
                phase_length=cfg.get('phase_length', phase_length),
                phase_epochs=cfg.get('phase_epochs', phase_epochs),
                nonlinear=cfg.get('nonlinear', nonlinear),
                target_samples=cfg.get('target_samples', 16),
                boundary_explore=cfg.get('boundary_explore', False),
                action_samples=cfg.get('action_samples', 64),
                boundary_bonus=cfg.get('boundary_bonus', 1.0),
                margin_explore=cfg.get('margin_explore', False),
                boundary_focus=cfg.get('boundary_focus', 0.0),
                boundary_scale=cfg.get('boundary_scale', 0.15),
                residual_strength=cfg.get('residual_strength', 0.0),
                residual_scale_by_sigma=cfg.get('residual_scale_by_sigma', True),
                sample_clip=cfg.get('sample_clip', (0.0, 1.0) if cost_fn == 'sigmoid' else None),
            )
            result = run_agent_on_episode(
                agent=agent,
                episode=episode,
                n_rounds=n_rounds,
                lr=lr,
                full_feedback=cfg.get('full_feedback', False),
                diagnostics=True,
            )
            _append_round_rows(round_rows, label, seed, result)
            if capture_agents and seed == 0 and label not in artifacts:
                artifacts[label] = {
                    'agent': agent,
                    'result': result,
                    'config': dict(cfg),
                    'seed': seed,
                }
    df_rounds = pd.DataFrame(round_rows)
    return df_rounds, summarize_suite(df_rounds), artifacts


def run_scale_cases(
    scale_cases: Iterable[Dict],
    configs: Iterable[Tuple[str, Dict]],
    cost_fn: str = 'sigmoid',
    n_seeds: int = 3,
    w_spread: float = 0.4,
    lr: float = DEFAULT_LR,
    beta: float = DEFAULT_BETA,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    phase_length: int = 50,
    phase_epochs: int = 5,
) -> Dict[str, Dict]:
    outputs: Dict[str, Dict] = {}
    for case in scale_cases:
        W_true = generate_shared_weights(
            n_arms=case['n_arms'],
            context_dim=case.get('context_dim', 5),
            w_spread=case.get('w_spread', w_spread),
            seed=case.get('weight_seed', 42),
        )
        df_rounds, summary, _ = run_suite(
            W_true=W_true,
            k=case['k'],
            configs=configs,
            n_rounds=case['n_rounds'],
            n_seeds=case.get('n_seeds', n_seeds),
            hidden_dim=case['hidden_dim'],
            cost_fn=case.get('cost_fn', cost_fn),
            lr=case.get('lr', lr),
            beta=case.get('beta', beta),
            lambda_reg=case.get('lambda_reg', lambda_reg),
            phase_length=case.get('phase_length', phase_length),
            phase_epochs=case.get('phase_epochs', phase_epochs),
            noise_width=case.get('noise_width', 0.0),
            capture_agents=False,
        )
        outputs[case['label']] = {
            'W_true': W_true,
            'rounds': df_rounds,
            'summary': summary,
            'meta': case,
        }
    return outputs


def _true_costs_on_grid(
    W_true: np.ndarray,
    x_grid: np.ndarray,
    cost_fn: str,
) -> np.ndarray:
    if cost_fn == 'linear':
        return W_true @ x_grid.T
    if cost_fn == 'sigmoid':
        logits = W_true @ x_grid.T
        return 1.0 / (1.0 + np.exp(-logits))
    raise ValueError(f'Unsupported cost_fn for plotting: {cost_fn}')


def _mean_std_by_round(df_rounds: pd.DataFrame, metric: str) -> pd.DataFrame:
    agg = (
        df_rounds.groupby(['agent', 'round'])[metric]
        .agg(value_mean='mean', value_std='std')
        .reset_index()
    )
    agg['value_std'] = agg['value_std'].fillna(0.0)
    agg['ymin'] = agg['value_mean'] - agg['value_std']
    agg['ymax'] = agg['value_mean'] + agg['value_std']
    return agg


def _resolve_smooth_window(plot_df: pd.DataFrame, smooth_window: int = None) -> int:
    if smooth_window is not None:
        window = max(1, int(smooth_window))
    else:
        n_rounds = int(plot_df['round'].nunique())
        window = max(7, min(51, 2 * max(1, n_rounds // 40) + 1))
    if window % 2 == 0:
        window += 1
    return window


def _smooth_plot_df(plot_df: pd.DataFrame, smooth_window: int = None) -> pd.DataFrame:
    window = _resolve_smooth_window(plot_df, smooth_window)
    if window <= 1:
        return plot_df

    smoothed_parts = []
    for _, sub in plot_df.groupby('agent', sort=False):
        sub = sub.sort_values('round').copy()
        sub['value_mean'] = (
            sub['value_mean'].rolling(window=window, center=True, min_periods=1).mean()
        )
        sub['value_std'] = (
            sub['value_std'].rolling(window=window, center=True, min_periods=1).mean()
        )
        sub['ymin'] = sub['value_mean'] - sub['value_std']
        sub['ymax'] = sub['value_mean'] + sub['value_std']
        smoothed_parts.append(sub)
    return pd.concat(smoothed_parts, ignore_index=True)


def plot_regret(
    df_rounds: pd.DataFrame,
    title: str,
    smooth_window: int = None,
):
    plot_df = _smooth_plot_df(
        _mean_std_by_round(df_rounds, 'cumulative_regret'),
        smooth_window=smooth_window,
    )
    return (
        ggplot(plot_df, aes('round', 'value_mean', group='agent'))
        + geom_ribbon(aes(ymin='ymin', ymax='ymax', fill='agent'), alpha=0.12, color='transparent', size=0)
        + geom_line(aes(color='agent'), size=1.1)
        + labs(x='Round', y='Cumulative Regret', title=title, color='', fill='')
        + theme(legend_position='right')
        + ggsize(900, 420)
    )


def plot_diagnostics(
    df_rounds: pd.DataFrame,
    title: str,
    metrics: Tuple[str, ...] = (
        'flip_set_size',
        'contested_mae',
        'contested_sigma',
        'decision_correct',
        'rank_corr',
        'flip_contested_mae',
    ),
    smooth_window: int = None,
):
    plots = []
    for metric in metrics:
        plot_df = _smooth_plot_df(
            _mean_std_by_round(df_rounds, metric),
            smooth_window=smooth_window,
        )
        plots.append(
            ggplot(plot_df, aes('round', 'value_mean', group='agent'))
            + geom_ribbon(aes(ymin='ymin', ymax='ymax', fill='agent'), alpha=0.10, color='transparent', size=0)
            + geom_line(aes(color='agent'), size=1.0)
            + labs(
                x='Round',
                y=metric.replace('_', ' ').title(),
                title=metric.replace('_', ' ').title(),
                color='',
                fill='',
            )
            + theme(legend_position='none')
            + ggsize(420, 320)
        )
    return gggrid(plots, ncol=2)


def plot_prediction_profiles(
    artifacts: Dict[str, Dict],
    W_true: np.ndarray,
    cost_fn: str = 'sigmoid',
    x_min: float = -3.0,
    x_max: float = 3.0,
    n_points: int = 250,
    title: str = None,
):
    W_true = np.asarray(W_true, dtype=float)
    if W_true.shape[1] != 1:
        raise ValueError('Prediction profile plot currently supports only 1D context.')
    if not artifacts:
        raise ValueError('No fitted agents supplied for prediction profile plot.')

    x_grid = np.linspace(x_min, x_max, n_points)[:, None]
    true_costs = _true_costs_on_grid(W_true, x_grid, cost_fn)
    plots = []
    for label, artifact in artifacts.items():
        agent = artifact['agent']
        result = artifact['result']
        preds = np.vstack([np.asarray(agent.predict_mean(x)).flatten() for x in x_grid])
        rows = []
        for arm_idx in range(W_true.shape[0]):
            for point_idx, x_val in enumerate(x_grid[:, 0]):
                rows.append(
                    {
                        'x': x_val,
                        'cost': true_costs[arm_idx, point_idx],
                        'arm': f'Arm {arm_idx}',
                        'series': 'True',
                    }
                )
                rows.append(
                    {
                        'x': x_val,
                        'cost': preds[point_idx, arm_idx],
                        'arm': f'Arm {arm_idx}',
                        'series': 'Predicted',
                    }
                )
        plot_df = pd.DataFrame(rows)
        final_regret = result['cumulative_regret'][-1]
        plots.append(
            ggplot(plot_df, aes('x', 'cost', color='arm', linetype='series'))
            + geom_line(size=1.1)
            + labs(
                x='Context x',
                y='Cost',
                title=f'{label}\nfinal regret={final_regret:.1f}',
                color='',
                linetype='',
            )
            + theme(legend_position='none')
            + ggsize(420, 320)
        )
    grid = gggrid(plots, ncol=2)
    return grid if title is None else grid


def _estimate_crossing_from_curves(x_grid: np.ndarray, values_a: np.ndarray, values_b: np.ndarray):
    diff = np.asarray(values_a) - np.asarray(values_b)
    exact = np.where(np.isclose(diff, 0.0))[0]
    if len(exact) > 0:
        return float(x_grid[exact[0]])

    sign = np.sign(diff)
    changes = np.where(sign[:-1] * sign[1:] < 0)[0]
    if len(changes) == 0:
        return np.nan

    idx = int(changes[np.argmin(np.abs(diff[changes]))])
    x0 = float(x_grid[idx])
    x1 = float(x_grid[idx + 1])
    y0 = float(diff[idx])
    y1 = float(diff[idx + 1])
    if np.isclose(y1, y0):
        return 0.5 * (x0 + x1)
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def estimate_crossing_point(
    agent,
    x_min: float = -3.0,
    x_max: float = 3.0,
    n_points: int = 401,
):
    x_grid = np.linspace(x_min, x_max, n_points)
    preds = np.vstack([np.asarray(agent.predict_mean(np.array([x]))).flatten() for x in x_grid])
    if preds.shape[1] != 2:
        raise ValueError('Crossing-point estimate expects a 2-arm agent.')
    return _estimate_crossing_from_curves(x_grid, preds[:, 0], preds[:, 1])


def true_crossing_point(
    W_true: np.ndarray,
    cost_fn: str = 'sigmoid',
    x_min: float = -3.0,
    x_max: float = 3.0,
    n_points: int = 4001,
):
    W_true = np.asarray(W_true, dtype=float)
    if W_true.shape != (2, 1):
        raise ValueError('True crossing-point helper expects W_true with shape (2, 1).')
    x_grid = np.linspace(x_min, x_max, n_points)
    true_vals = _true_costs_on_grid(W_true, x_grid[:, None], cost_fn)
    return _estimate_crossing_from_curves(x_grid, true_vals[0], true_vals[1])


def _evaluate_frozen_policy(
    agent,
    W_true: np.ndarray,
    k: int,
    cost_fn: str = 'sigmoid',
    n_test: int = 2000,
    seed: int = 123,
):
    env = build_fixed_topk_env(W_true=W_true, k=k, seed=seed, cost_fn=cost_fn)
    X_test, Y_test = env.generate_test_set(n_samples=n_test)
    regrets = []
    decision_correct = []
    for X, Y_true in zip(X_test, Y_test):
        pred = np.asarray(agent.predict_mean(X)).flatten()
        action = np.asarray(env.solve(pred)).flatten()
        optimal = np.asarray(env.solve(Y_true)).flatten()
        regret = float(np.sum(Y_true * action) - np.sum(Y_true * optimal))
        regrets.append(regret)
        decision_correct.append(int(np.array_equal((action > 0.5).astype(int), (optimal > 0.5).astype(int))))
    return {
        'frozen_test_regret_mean': float(np.mean(regrets)),
        'frozen_test_regret_std': float(np.std(regrets)),
        'frozen_decision_rate': float(np.mean(decision_correct)),
    }


def run_crossing_trace_suite(
    W_true: np.ndarray,
    k: int,
    configs: Iterable[Tuple[str, Dict]],
    n_rounds: int = 500,
    hidden_dim: int = 8,
    cost_fn: str = 'sigmoid',
    seed: int = 0,
    checkpoint_every: int = 10,
    lr: float = DEFAULT_LR,
    beta: float = DEFAULT_BETA,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    phase_length: int = 50,
    phase_epochs: int = 5,
    x_min: float = -3.0,
    x_max: float = 3.0,
    n_points: int = 401,
):
    env = build_fixed_topk_env(W_true=W_true, k=k, seed=seed, cost_fn=cost_fn)
    episode = pregenerate_episode(env, n_rounds=n_rounds)
    rounds_data = episode['rounds_data']
    trace_rows = []
    frozen_rows = []
    true_crossing = true_crossing_point(
        W_true=W_true,
        cost_fn=cost_fn,
        x_min=x_min,
        x_max=x_max,
        n_points=max(4001, n_points),
    )

    for label, cfg in configs:
        agent = create_agent(
            agent_name=cfg['agent_name'],
            env=env,
            seed=1000 + seed,
            n_rounds=n_rounds,
            hidden_dim=hidden_dim,
            lr=lr,
            beta=beta,
            lambda_reg=lambda_reg,
            mask_grad=cfg.get('mask_grad', True),
            phased=cfg.get('phased', False),
            phase_length=cfg.get('phase_length', phase_length),
            phase_epochs=cfg.get('phase_epochs', phase_epochs),
            target_samples=cfg.get('target_samples', 16),
            boundary_explore=cfg.get('boundary_explore', False),
            action_samples=cfg.get('action_samples', 64),
            boundary_bonus=cfg.get('boundary_bonus', 1.0),
            margin_explore=cfg.get('margin_explore', False),
            boundary_focus=cfg.get('boundary_focus', 0.0),
            boundary_scale=cfg.get('boundary_scale', 0.15),
            residual_strength=cfg.get('residual_strength', 0.0),
            residual_scale_by_sigma=cfg.get('residual_scale_by_sigma', True),
            sample_clip=cfg.get('sample_clip', (0.0, 1.0) if cost_fn == 'sigmoid' else None),
        )
        total_regret = 0.0

        for t, rd in enumerate(rounds_data, start=1):
            X = rd['context']
            true_vals = rd['true_values']
            optimal_val = rd['optimal_value']

            selection = np.asarray(agent.act(X)).flatten()
            agent_val = float(np.sum(true_vals * selection))
            total_regret += agent_val - optimal_val

            if cfg.get('full_feedback', False):
                observed = true_vals.copy()
            else:
                observed = np.zeros_like(true_vals)
                mask = selection > 0.5
                observed[mask] = true_vals[mask]

            agent.update(
                X,
                selection,
                observed,
                lr=lr,
                full_feedback=cfg.get('full_feedback', False),
            )

            if t == 1 or t % checkpoint_every == 0 or t == n_rounds:
                crossing = estimate_crossing_point(
                    agent,
                    x_min=x_min,
                    x_max=x_max,
                    n_points=n_points,
                )
                trace_rows.append(
                    {
                        'agent': label,
                        'round': t,
                        'estimated_crossing': crossing,
                        'cumulative_regret': total_regret,
                    }
                )

        frozen = _evaluate_frozen_policy(
            agent=agent,
            W_true=W_true,
            k=k,
            cost_fn=cost_fn,
        )
        final_crossing = estimate_crossing_point(
            agent,
            x_min=x_min,
            x_max=x_max,
            n_points=n_points,
        )
        frozen_rows.append(
            {
                'agent': label,
                'final_crossing': final_crossing,
                'crossing_error': float(np.abs(final_crossing - true_crossing)) if np.isfinite(final_crossing) else np.nan,
                **frozen,
            }
        )

    return pd.DataFrame(trace_rows), pd.DataFrame(frozen_rows).sort_values('frozen_test_regret_mean').reset_index(drop=True), true_crossing


def plot_crossing_trace(
    trace_df: pd.DataFrame,
    true_crossing: float,
    title: str,
):
    return (
        ggplot(trace_df, aes('round', 'estimated_crossing', color='agent', group='agent'))
        + geom_hline(yintercept=true_crossing, linetype='dashed', alpha=0.6)
        + geom_line(size=1.1)
        + geom_point(size=1.8)
        + labs(
            x='Round',
            y='Estimated crossing x',
            title=title,
            color='',
        )
        + theme(legend_position='right')
        + ggsize(900, 420)
    )

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import jax.numpy as jnp
import numpy as np

from src.agents.bayesian_utils import compute_bayesian_predictions, compute_uncertainties
from src.agents.contextual.bayesian_spo import BayesianSPOAgent
from src.agents.contextual.weber_bll import WeberBLLAgent
from src.agents.feature_extractors import (
    clip_gated_residual_params, gated_residual_backward, gated_residual_forward,
    init_gated_residual_params, validate_residual_activation, validate_residual_weight_init,
)


class GatedResidualBLLAgent(WeberBLLAgent):
    def __init__(
        self,
        context_dim,
        n_arms,
        weights,
        capacity,
        hidden_dim=32,
        seed=None,
        nonlinear=True,
        nonlinear_depth=1,
        residual_gate_scale=1.0,
        residual_activation='relu',
        residual_weight_init='random',
        gradient_mode='current',
        **kwargs
    ):
        if gradient_mode != 'current':
            raise ValueError('The BLL adapter preserves current gradients only')
        depth = int(nonlinear_depth) if nonlinear else 0
        if depth < 0:
            raise ValueError('nonlinear_depth must be non-negative')
        self.residual_gate_scale = float(residual_gate_scale)
        if not np.isfinite(self.residual_gate_scale) or self.residual_gate_scale <= 0:
            raise ValueError('residual_gate_scale must be positive and finite')
        self.residual_activation = validate_residual_activation(residual_activation)
        self.residual_weight_init = validate_residual_weight_init(residual_weight_init)
        kwargs.setdefault('mask_grad', False)
        kwargs.setdefault('posterior_complete_missing', True)
        if kwargs['mask_grad'] or not kwargs['posterior_complete_missing']:
            raise ValueError('Preserve BLL posterior completion with mask_grad=False')
        super().__init__(context_dim, n_arms, weights, capacity, hidden_dim=hidden_dim,
                         seed=seed, nonlinear=False, **kwargs)
        params = init_gated_residual_params(context_dim, hidden_dim, depth,
            seed if seed is not None else 0, self.residual_weight_init)
        params['W1'], params['b1'] = self.W1, self.b1
        self.feature_params = params
        self.nonlinear_depth, self.nonlinear = depth, depth > 0
        self.gradient_mode = gradient_mode
        self._sync_feature_aliases()
        self.opt_state = self.optimizer.init(self.feature_params)

    def _sync_feature_aliases(self):
        self.W1, self.b1 = self.feature_params['W1'], self.feature_params['b1']
        if self.feature_params['blocks']:
            first = self.feature_params['blocks'][0]
            self.W2, self.b2, self.residual_gate = first['weight'], first['bias'], first['gate']

    def _get_params(self):
        if not hasattr(self, 'feature_params'):
            return super()._get_params()
        return self.feature_params

    def _set_params(self, params):
        self.feature_params = clip_gated_residual_params(params)
        self._sync_feature_aliases()

    def _features(self, X):
        return gated_residual_forward(self.feature_params, jnp.asarray(X),
                                      self.residual_gate_scale, self.residual_activation)

    def _backprop_features(self, X, grad_h):
        if not self.nonlinear:
            return {**super()._backprop_features(X, grad_h), 'blocks': ()}
        return gated_residual_backward(self.feature_params, jnp.asarray(X), grad_h,
                                       self.residual_gate_scale, self.residual_activation)


# Replay the historical PI variant (cccca54) to preserve the solver sequence.
class LegacyBayesianSPOPIAgent(BayesianSPOAgent):
    def _compute_spo_grad_h(self, X, h, Y_target, selection_mask, full_feedback):
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        pred_np = np.asarray(pred, dtype=float).reshape(-1)
        stds_np = np.asarray(compute_uncertainties(h, self.A_inv), dtype=float).reshape(-1)
        target_np = np.asarray(Y_target, dtype=float).reshape(-1)
        observed_mask_np = np.asarray(selection_mask, dtype=bool).reshape(-1)

        targets = [target_np]
        missing_mask = ~observed_mask_np
        if missing_mask.any():
            if self.use_posterior_predictive:
                sampling_std = np.sqrt(stds_np ** 2 + self.noise_var)
            else:
                sampling_std = self.imputation_scale * stds_np
            for _ in range(self.imputation_samples - 1):
                sampled = target_np.copy()
                sampled[missing_mask] = (
                    pred_np[missing_mask]
                    + sampling_std[missing_mask]
                    * self.imputation_rng.standard_normal(missing_mask.sum())
                )
                targets.append(sampled)

        theta = jnp.einsum("nij,nj->ni", self.A_inv, self.b_bayes)
        grad_h_total = jnp.zeros_like(h)
        for target_vec in targets:
            spo_grad_val = self.loss_grad_fn(pred, jnp.asarray(target_vec))
            if self.mask_grad and not full_feedback:
                spo_grad_val = jnp.where(selection_mask, spo_grad_val, 0.0)
            grad_h_total = grad_h_total + jnp.einsum("n,ni->i", spo_grad_val, theta)

        return grad_h_total / len(targets)


def make_legacy_pi_agent(env, params, seed, rounds):
    agent_params = {'imputation_samples': 4, 'imputation_scale': 1.0,
                    'use_posterior_predictive': False, 'noise_var': 1.0,
                    'max_grad_norm': 5.0, 'gradient_mode': 'current',
                    **{key: value for key, value in params.items() if key not in ('lr', 'init_lr')}}
    if agent_params['gradient_mode'] != 'current':
        raise ValueError('The historical PI override does not implement posterior gradients')
    return LegacyBayesianSPOPIAgent(
        env.context_dim, env.n_arms, env.weights, env.capacity,
        lr=params.get('init_lr', 0.001), decay_rounds=rounds,
        minimize=getattr(env, 'minimize', False), solver=env.solve, seed=seed,
        mask_grad=True, nonlinear=False, **agent_params,
    )


def write_json(path, value):
    with NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
    Path(handle.name).replace(path)


def load_episode(repo, config, dataset, seed, rounds):
    root = repo / config['episode_sources'][dataset]
    directory = root / 'seeds' / f'seed_{seed}' / 'datasets' / dataset
    metadata = json.loads((directory / 'episode.json').read_text())
    if (metadata['seed'], metadata['dataset']) != (seed, dataset) or rounds > metadata['rounds']:
        raise ValueError('Episode identity mismatch')
    keys = ('context', 'true_values', 'optimal_action', 'optimal_value',
            'mean_values', 'mean_optimal_action', 'mean_optimal_value')
    with np.load(directory / 'episode.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key][:rounds].copy() for key in keys}
    return {'rounds_data': [{key: arrays[key][index] for key in keys} for index in range(rounds)],
            'minimize': metadata['minimize']}


def action_flags(actions, env):
    actions = np.asarray(actions)
    binary_error = np.max(np.abs(actions - np.rint(actions)), axis=1)
    bounds_error = np.maximum(0, np.maximum(-actions.min(axis=1), actions.max(axis=1) - 1))
    integer_tolerance = float(env._solver._model.Params.IntFeasTol) if hasattr(env, 'max_assets') else 1e-5
    feasibility_tolerance = float(env._solver._model.Params.FeasibilityTol) if hasattr(env, 'max_assets') else 1e-6
    invalid = ((binary_error > integer_tolerance + 1e-10) |
               (bounds_error > feasibility_tolerance + 1e-10) | ~np.isfinite(actions).all(axis=1))
    result = {'rounds': len(actions), 'max_binary_error': float(binary_error.max()),
              'max_bounds_error': float(bounds_error.max()), 'integer_tolerance': integer_tolerance,
              'feasibility_tolerance': feasibility_tolerance}
    if hasattr(env, 'max_assets'):
        risk = np.einsum('ti,ij,tj->t', actions, np.asarray(env.data.covariance), actions)
        risk_excess = risk - env._solver.risk_level
        cardinality_error = np.abs(np.rint(actions).sum(axis=1) - env.max_assets)
        bad_risk = risk_excess > feasibility_tolerance + 1e-10
        bad_cardinality = cardinality_error > 0
        invalid |= bad_risk | bad_cardinality
        result.update(risk_violation_rounds=int(bad_risk.sum()), cardinality_violation_rounds=int(bad_cardinality.sum()),
                      max_risk_excess=float(risk_excess.max()),
                      max_raw_cardinality_error=float(np.abs(actions.sum(axis=1) - env.max_assets).max()))
    elif hasattr(env, 'grid_size'):
        n = env.grid_size
        right = actions[:, :n * (n - 1)].reshape(-1, n, n - 1)
        down = actions[:, n * (n - 1):].reshape(-1, n - 1, n)
        balance = np.zeros((len(actions), n, n))
        balance[:, :, :-1] += right
        balance[:, :, 1:] -= right
        balance[:, :-1, :] += down
        balance[:, 1:, :] -= down
        balance[:, 0, 0] -= 1
        balance[:, -1, -1] += 1
        flow_error = np.abs(balance).max(axis=(1, 2))
        bad_flow = flow_error > feasibility_tolerance + 1e-10
        invalid |= bad_flow
        result.update(flow_violation_rounds=int(bad_flow.sum()), max_flow_error=float(flow_error.max()))
    else:
        excess = actions @ np.asarray(env.weights) - env.capacity
        bad_capacity = excess > feasibility_tolerance + 1e-10
        invalid |= bad_capacity
        result.update(capacity_violation_rounds=int(bad_capacity.sum()), max_capacity_excess=float(excess.max()))
    result['invalid_action_rounds'] = int(invalid.sum())
    return result


class SolverMonitor:
    def __init__(self, env):
        self.env, self.original = env, env.solve
        owner = getattr(self.original, '__self__', None)
        solver = env._solver if hasattr(env, 'max_assets') else getattr(owner, 'knapsack_solver', owner)
        self.model = getattr(solver, '_model', None)
        self.reset()
        env.solve = self.solve

    def reset(self):
        self.flags = dict(calls=0, no_incumbent_calls=0, nonoptimal_calls=0, time_limit_calls=0,
                          nonfinite_objective_calls=0, invalid_action_calls=0)

    def solve(self, values):
        nonfinite = not np.isfinite(np.asarray(values)).all()
        action = np.asarray(self.original(values))
        status, solutions = (int(self.model.Status), int(self.model.SolCount)) if self.model is not None else (2, 1)
        self.flags['calls'] += 1
        self.flags['no_incumbent_calls'] += int(solutions == 0)
        self.flags['nonoptimal_calls'] += int(status != 2)
        self.flags['time_limit_calls'] += int(status == 9)
        self.flags['nonfinite_objective_calls'] += int(nonfinite)
        self.flags['invalid_action_calls'] += action_flags(action[None, :], self.env)['invalid_action_rounds']
        return action


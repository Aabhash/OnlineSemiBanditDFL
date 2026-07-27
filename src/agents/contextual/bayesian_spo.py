import jax.numpy as jnp
import numpy as np
import optax
from src.losses import make_spo_grad_fn
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    compute_uncertainties,
    update_bayesian_stats,
)
from src.agents.feature_extractors import (
    clip_gated_residual_params,
    gated_residual_backward,
    gated_residual_forward,
    init_gated_residual_params,
    validate_residual_activation,
    validate_residual_weight_init,
)
from src.agents.agent_base import AgentBase
from src.solver import solve_knapsack


class BayesianSPOAgent(AgentBase):

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        beta_start: float = None,
        beta_end: float = None,
        lambda_reg: float = 1.0,
        lr: float = 0.001,

        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        mask_grad: bool = False,
        nonlinear: bool = False,
        nonlinear_depth: int = 1,
        residual_gate_scale: float = 1.0,
        residual_activation: str = 'relu',
        residual_weight_init: str = 'random',
        minimize: bool = False,
        solver=None,
        seed: int = None,
        imputation_samples: int = 1,
        imputation_scale: float = 1.0,
        use_posterior_predictive: bool = False,
        noise_var: float = 1.0,
        missing_grad_weight: float = 0.0,
        exploration_strategy: str = "ucb",
        ts_scale: float = 1.0,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.mask_grad = mask_grad
        self.nonlinear = nonlinear
        self.nonlinear_depth = int(nonlinear_depth) if nonlinear else 0
        if self.nonlinear_depth < 0:
            raise ValueError('nonlinear_depth must be non-negative')
        if nonlinear and self.nonlinear_depth == 0:
            raise ValueError(
                'nonlinear_depth must be positive when nonlinear=True'
            )
        self.residual_gate_scale = float(residual_gate_scale)
        if self.residual_gate_scale <= 0:
            raise ValueError('residual_gate_scale must be positive')
        self.residual_activation = validate_residual_activation(
            residual_activation
        )
        self.residual_weight_init = validate_residual_weight_init(
            residual_weight_init
        )
        self.minimize = minimize

        if solver is None:
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0
        self.imputation_samples = max(int(imputation_samples), 1)
        self.imputation_scale = float(imputation_scale)
        self.use_posterior_predictive = bool(use_posterior_predictive)
        self.noise_var = float(noise_var)
        self.missing_grad_weight = float(missing_grad_weight)
        seed_value = seed if seed is not None else 0
        self.imputation_rng = np.random.default_rng(seed_value)
        strategy = exploration_strategy.lower()
        strategy_aliases = {
            "ucb": "ucb",
            "greedy": "greedy",
            "ts": "ts",
            "thompson": "ts",
            "thompson_sampling": "ts",
        }
        if strategy not in strategy_aliases:
            raise ValueError(
                f"Unknown exploration_strategy={exploration_strategy!r}; "
                "expected one of 'greedy', 'ucb', or 'ts'."
            )
        self.exploration_strategy = strategy_aliases[strategy]
        self.ts_scale = float(ts_scale)
        self.action_rng = np.random.default_rng(seed_value + 7919)


        self.loss_grad_fn = make_spo_grad_fn(self.solver, minimize)

        if beta_start is not None and beta_end is not None:
            self.beta_start = abs(beta_start)
            self.beta_end = abs(beta_end)
            self.beta = self.beta_start
        else:
            beta = abs(beta)
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        self.feature_params = init_gated_residual_params(
            context_dim,
            hidden_dim,
            self.nonlinear_depth,
            seed_value,
            self.residual_weight_init,
        )
        self._sync_feature_aliases()
        init_params = self.feature_params


        schedule = lambda count: lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule)
        )
        self.opt_state = self.optimizer.init(init_params)

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)


    def _get_params(self):
        return self.feature_params

    def _set_params(self, params):
        self.feature_params = clip_gated_residual_params(params)
        self._sync_feature_aliases()

    def _sync_feature_aliases(self):
        self.W1 = self.feature_params['W1']
        self.b1 = self.feature_params['b1']
        if self.feature_params['blocks']:
            first = self.feature_params['blocks'][0]
            self.W2 = first['weight']
            self.b2 = first['bias']
            self.residual_gate = first['gate']

    def _features(self, X):
        return gated_residual_forward(
            self.feature_params,
            jnp.asarray(X),
            self.residual_gate_scale,
            self.residual_activation,
        )

    def predict_mean(self, X):
        h = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def predict_uncertainty(self, X):
        h = self._features(X)
        return compute_uncertainties(h, self.A_inv)

    def predict_mean_and_uncertainty(self, X):
        h = self._features(X)
        return (compute_bayesian_predictions(h, self.A_inv, self.b_bayes),
                compute_uncertainties(h, self.A_inv))

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h = self._features(X)
        if self.exploration_strategy == "ucb":
            values = compute_ucb_values(
                h, self.A_inv, self.b_bayes, self.beta, minimize=self.minimize
            )
        elif self.exploration_strategy == "greedy":
            values = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        else:
            means = np.asarray(
                compute_bayesian_predictions(h, self.A_inv, self.b_bayes),
                dtype=float,
            )
            stds = np.asarray(compute_uncertainties(h, self.A_inv), dtype=float)
            values = (
                means
                + self.ts_scale
                * stds
                * self.action_rng.standard_normal(self.n_arms)
            )
        return self.solver(np.asarray(values))

    def _compute_spo_grad_h(self, X, h, Y_target, selection_mask, full_feedback):
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        theta = jnp.einsum("nij,nj->ni", self.A_inv, self.b_bayes)

        if self.imputation_samples == 1:
            spo_grad_val = self.loss_grad_fn(pred, Y_target)
            if self.mask_grad and not full_feedback:
                missing_weight = jnp.asarray(
                    self.missing_grad_weight,
                    dtype=spo_grad_val.dtype,
                )
                grad_weight = jnp.where(selection_mask, 1.0, missing_weight)
                spo_grad_val = spo_grad_val * grad_weight
            return jnp.einsum("n,ni->i", spo_grad_val, theta)

        pred_np = np.asarray(pred, dtype=float).reshape(-1)
        stds_np = np.asarray(compute_uncertainties(h, self.A_inv), dtype=float).reshape(-1)
        target_np = np.asarray(Y_target, dtype=float).reshape(-1)
        if full_feedback:
            observed_mask_np = np.ones(self.n_arms, dtype=bool)
        else:
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

        grad_h_total = jnp.zeros_like(h)
        for target_vec in targets:
            spo_grad_val = self.loss_grad_fn(pred, jnp.asarray(target_vec))
            if self.mask_grad and not full_feedback:
                missing_weight = jnp.asarray(self.missing_grad_weight, dtype=spo_grad_val.dtype)
                grad_weight = jnp.where(selection_mask, 1.0, missing_weight)
                spo_grad_val = spo_grad_val * grad_weight
            grad_h_total = grad_h_total + jnp.einsum("n,ni->i", spo_grad_val, theta)

        return grad_h_total / max(len(targets), 1)

    def _backprop_and_step(self, X, grad_h):
        X = jnp.asarray(X)
        if self.nonlinear:
            grads = gated_residual_backward(
                self.feature_params,
                X,
                grad_h,
                self.residual_gate_scale,
                self.residual_activation,
            )
        else:
            grads = {
                'W1': jnp.outer(X, grad_h),
                'b1': grad_h,
                'blocks': (),
            }

        params = self._get_params()
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        self._set_params(optax.apply_updates(params, updates))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False,
               **kwargs):
        prob_values = kwargs.get("prob_values")
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h = self._features(X)

        selection_mask = selection > 0.5
        obs_mask = jnp.ones(self.n_arms, dtype=jnp.bool_) if full_feedback else selection_mask
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, obs_mask
        )

        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

        if prob_values is not None:
            Y_target = jnp.asarray(prob_values)
        elif full_feedback:
            Y_target = rewards
        else:
            Y_target = jnp.where(selection > 0.5, rewards, pred)

        grad_h = self._compute_spo_grad_h(X, h, Y_target, selection_mask, full_feedback)
        self._backprop_and_step(X, grad_h)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    compute_uncertainties,
    linear_forward,
    mlp_forward,
    update_bayesian_stats,
)
from src.agents.agent_base import AgentBase
from src.solver import solve_knapsack


class BayesianMSEAgent(AgentBase):

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
        missing_grad_weight: float = 0.0,
        nonlinear: bool = False,
        minimize: bool | None = None,
        solver=None,
        seed: int = None,
        exploration_strategy: str = "ucb",
        ts_scale: float = 1.0,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        if minimize is None:
            raise ValueError(
                "BayesianMSEAgent requires an explicit minimize=... argument "
                "to avoid ambiguous exploration direction."
            )
        self.mask_grad = mask_grad
        self.missing_grad_weight = float(missing_grad_weight)
        self.nonlinear = nonlinear
        self.minimize = bool(minimize)

        if solver is None:
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0
        seed_value = seed if seed is not None else 0
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

        if beta_start is not None and beta_end is not None:
            self.beta_start = abs(beta_start)
            self.beta_end = abs(beta_end)
            self.beta = self.beta_start
        else:
            beta = abs(beta)
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        key = jax.random.PRNGKey(seed_value)
        if nonlinear:
            k1, k2 = jax.random.split(key)
            self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            self.W2 = jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1
            self.b2 = jnp.zeros(hidden_dim)
            init_params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        else:
            self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            init_params = {'W1': self.W1, 'b1': self.b1}


        schedule = lambda count: lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule)
        )
        self.opt_state = self.optimizer.init(init_params)

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def _get_params(self):
        if self.nonlinear:
            return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        return {'W1': self.W1, 'b1': self.b1}

    def _set_params(self, params):
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']
        if self.nonlinear:
            self.W2 = jnp.clip(params['W2'], -10, 10)
            self.b2 = params['b2']

    def _features(self, X):
        X = jnp.asarray(X)
        if self.nonlinear:
            h, _, _ = mlp_forward(X, self.W1, self.b1, self.W2, self.b2)
            return h
        return linear_forward(X, self.W1, self.b1)

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

    def act(self, X: np.ndarray) -> np.ndarray:
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

    def _compute_mse_grad_h(self, X, h, Y_target, selection_mask, full_feedback):
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        mse_grad = 2 * (pred - Y_target) / self.n_arms
        if self.mask_grad and not full_feedback:
            missing_weight = jnp.asarray(self.missing_grad_weight, dtype=mse_grad.dtype)
            grad_weight = jnp.where(selection_mask, 1.0, missing_weight)
            mse_grad = mse_grad * grad_weight
        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        return jnp.einsum('n,ni->i', mse_grad, theta)

    def _backprop_and_step(self, X, grad_h):
        X = jnp.asarray(X)
        if self.nonlinear:
            z = X @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            grad_a = grad_h @ self.W2.T
            grad_z = grad_a * (z > 0)
            grads = {
                'W1': jnp.outer(X, grad_z), 'b1': grad_z,
                'W2': jnp.outer(a, grad_h), 'b2': grad_h,
            }
        else:
            grads = {'W1': jnp.outer(X, grad_h), 'b1': grad_h}
        params = self._get_params()
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        self._set_params(optax.apply_updates(params, updates))

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False, **kwargs):
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
        if full_feedback:
            Y_target = rewards
        else:
            Y_target = jnp.where(selection > 0.5, rewards, pred)

        grad_h = self._compute_mse_grad_h(X, h, Y_target, selection_mask, full_feedback)
        self._backprop_and_step(X, grad_h)

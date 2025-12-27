import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.losses import spo_plus_grad
from src.solver import solve_knapsack_with_sensitivity
from src.agents.bayesian_utils import (
    init_shared_regression,
    compute_shared_ucb_values,
    compute_shared_predictions,
    compute_shared_uncertainties,
    update_shared_bayesian_stats,
)


class PerArmFullSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta_start: float = 2.0,
        beta_end: float = 0.1,
        decay_rounds: int = 300,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.hidden_dim = hidden_dim

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = beta_start
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        key = jax.random.PRNGKey(seed if seed else 0)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)

        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=schedule)
            )
        else:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr)
            )
        self.opt_state = self.optimizer.init({'W1': self.W1, 'b1': self.b1})

        self.A, self.b_bayes, self.A_inv = init_shared_regression(hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        Z1 = X @ self.W1 + self.b1
        H = jnp.maximum(0, Z1)
        return H, Z1

    def predict_mean(self, X):
        H, _ = self._features(X)
        return compute_shared_predictions(H, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1
        progress = min(self.t / self.decay_rounds, 1.0)
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        H, _ = self._features(X)
        ucb_values = compute_shared_ucb_values(H, self.A_inv, self.b_bayes, self.beta)
        return self.solver(np.asarray(ucb_values))

    def update(self, X, selection, rewards, lr=None, full_feedback=False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        H, Z1 = self._features(X)

        mask = jnp.ones(self.n_arms, dtype=bool) if full_feedback else (selection > 0.5)
        self.A, self.b_bayes, self.A_inv = update_shared_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, H, rewards, mask
        )

        pred = compute_shared_predictions(H, self.A_inv, self.b_bayes)

        if full_feedback:
            Y_target = rewards
        else:
            Y_target = jnp.where(selection > 0.5, rewards, pred)

        spo_grad_val = spo_plus_grad(pred, Y_target, self.weights, self.capacity)

        theta = self.A_inv @ self.b_bayes
        grad_H = jnp.outer(spo_grad_val, theta)
        grad_Z1 = grad_H * (Z1 > 0)

        grad_W1 = X.T @ grad_Z1
        grad_b1 = jnp.sum(grad_Z1, axis=0)

        grads = {'W1': grad_W1, 'b1': grad_b1}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']


class PerArmDecisionUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta_pred_start: float = 2.0,
        beta_pred_end: float = 0.1,
        beta_decision: float = 1.0,
        decay_rounds: int = 300,
        alpha: float = 0.0,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.hidden_dim = hidden_dim

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver

        self.beta_pred_start = beta_pred_start
        self.beta_pred_end = beta_pred_end
        self.beta_pred = beta_pred_start
        self.beta_decision = beta_decision
        self.decay_rounds = decay_rounds
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.t = 0

        key = jax.random.PRNGKey(seed if seed else 0)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)

        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=schedule)
            )
        else:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr)
            )
        self.opt_state = self.optimizer.init({'W1': self.W1, 'b1': self.b1})

        self.A, self.b_bayes, self.A_inv = init_shared_regression(hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        Z1 = X @ self.W1 + self.b1
        H = jnp.maximum(0, Z1)
        return H, Z1

    def predict_mean(self, X):
        H, _ = self._features(X)
        return compute_shared_predictions(H, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1
        progress = min(self.t / self.decay_rounds, 1.0)
        self.beta_pred = self.beta_pred_start + (self.beta_pred_end - self.beta_pred_start) * progress

        H, _ = self._features(X)

        pred_values = compute_shared_predictions(H, self.A_inv, self.b_bayes)
        pred_uncertainty = compute_shared_uncertainties(H, self.A_inv)

        sensitivity_info = solve_knapsack_with_sensitivity(
            np.asarray(pred_values), self.weights_np, self.capacity
        )
        decision_sensitivity = jnp.asarray(sensitivity_info["decision_relevance"])

        ucb_values = pred_values + self.beta_pred * pred_uncertainty + self.beta_decision * decision_sensitivity

        return self.solver(np.asarray(ucb_values))

    def update(self, X, selection, rewards, lr=None, full_feedback=False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        H, Z1 = self._features(X)

        mask = jnp.ones(self.n_arms, dtype=bool) if full_feedback else (selection > 0.5)
        self.A, self.b_bayes, self.A_inv = update_shared_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, H, rewards, mask
        )

        if full_feedback:
            Y_target = rewards
        else:
            pred_values = compute_shared_predictions(H, self.A_inv, self.b_bayes)
            pred_uncertainty = compute_shared_uncertainties(H, self.A_inv)

            sensitivity_info = solve_knapsack_with_sensitivity(
                np.asarray(pred_values), self.weights_np, self.capacity
            )
            decision_sensitivity = jnp.asarray(sensitivity_info["decision_relevance"])

            ucb_values = pred_values + self.beta_pred * pred_uncertainty + self.beta_decision * decision_sensitivity
            Y_target = jnp.where(selection > 0.5, rewards, ucb_values)

        Y_pred = compute_shared_predictions(H, self.A_inv, self.b_bayes)
        mse_grad = 2 * (Y_pred - Y_target) / self.n_arms
        spo_grad_val = spo_plus_grad(Y_pred, Y_target, self.weights, self.capacity)
        hybrid_grad = self.alpha * mse_grad + (1 - self.alpha) * spo_grad_val

        theta = self.A_inv @ self.b_bayes
        grad_H = jnp.outer(hybrid_grad, theta)
        grad_Z1 = grad_H * (Z1 > 0)

        grad_W1 = X.T @ grad_Z1
        grad_b1 = jnp.sum(grad_Z1, axis=0)

        grads = {'W1': grad_W1, 'b1': grad_b1}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']

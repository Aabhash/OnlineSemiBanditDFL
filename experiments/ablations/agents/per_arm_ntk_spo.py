import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.losses import spo_plus_grad
from src.solver import solve_knapsack_with_sensitivity
from src.agents.bayesian_utils import (
    init_shared_regression,
    compute_shared_predictions,
    update_shared_bayesian_stats,
)


class PerArmNTKSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        gamma: float = 1.0,
        gamma_start: float = None,
        gamma_end: float = None,
        lambda_reg: float = 1.0,
        lambda_ntk: float = 1.0,
        lr: float = 0.01,
        decay_rounds: int = 300,
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

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver

        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.lambda_ntk = lambda_ntk
        self.decay_rounds = decay_rounds
        self.t = 0

        if gamma_start is not None and gamma_end is not None:
            self.gamma_start = gamma_start
            self.gamma_end = gamma_end
            self.gamma = gamma_start
        else:
            self.gamma_start = gamma
            self.gamma_end = gamma
            self.gamma = gamma

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

        self.total_params = context_dim * hidden_dim + hidden_dim
        self.Z_diag = lambda_ntk * jnp.ones(self.total_params)

    def _features(self, X):
        X = jnp.asarray(X)
        Z1 = X @ self.W1 + self.b1
        H = jnp.maximum(0, Z1)
        return H, Z1

    def _compute_gradient_features(self, X, Z1=None):
        X = jnp.asarray(X)
        if Z1 is None:
            _, Z1 = self._features(X)

        theta = self.A_inv @ self.b_bayes
        relu_mask = (Z1 > 0).astype(jnp.float32)
        grad_Z1 = relu_mask * theta

        g_W1 = jnp.einsum('nd,nh->ndh', X, grad_Z1).reshape(self.n_arms, -1)
        g_b1 = grad_Z1

        return jnp.concatenate([g_W1, g_b1], axis=1)

    def _compute_ntk_uncertainties(self, g_features):
        g_scaled = g_features / jnp.sqrt(self.total_params)
        uncertainties = jnp.sqrt(jnp.sum(g_scaled ** 2 / self.Z_diag, axis=1))
        return uncertainties

    def _update_ntk_matrix(self, g_features, mask):
        g_scaled = g_features / jnp.sqrt(self.total_params)
        masked_g2 = jnp.where(mask[:, None], g_scaled ** 2, 0.0)
        self.Z_diag = self.Z_diag + jnp.sum(masked_g2, axis=0)

    def predict_mean(self, X):
        H, _ = self._features(X)
        return compute_shared_predictions(H, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1

        if self.gamma_start != self.gamma_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * progress

        X = jnp.asarray(X)
        H, Z1 = self._features(X)

        predictions = compute_shared_predictions(H, self.A_inv, self.b_bayes)

        g_features = self._compute_gradient_features(X, Z1)
        uncertainties = self._compute_ntk_uncertainties(g_features)

        ucb_values = predictions + self.gamma * uncertainties

        return self.solver(np.asarray(ucb_values))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        H, Z1 = self._features(X)

        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_shared_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, H, rewards, selection_mask
        )

        g_features = self._compute_gradient_features(X, Z1)
        self._update_ntk_matrix(g_features, selection_mask)

        pred = compute_shared_predictions(H, self.A_inv, self.b_bayes)

        if full_feedback:
            Y_target = rewards
        else:
            uncertainties = self._compute_ntk_uncertainties(g_features)
            ucb_values = pred + self.gamma * uncertainties
            Y_target = jnp.where(selection > 0.5, rewards, ucb_values)

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


class PerArmNTKDecisionUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        gamma: float = 1.0,
        gamma_start: float = None,
        gamma_end: float = None,
        beta_decision: float = 1.0,
        lambda_reg: float = 1.0,
        lambda_ntk: float = 1.0,
        lr: float = 0.01,
        decay_rounds: int = 300,
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
        self.beta_decision = beta_decision

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver

        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.lambda_ntk = lambda_ntk
        self.decay_rounds = decay_rounds
        self.t = 0

        if gamma_start is not None and gamma_end is not None:
            self.gamma_start = gamma_start
            self.gamma_end = gamma_end
            self.gamma = gamma_start
        else:
            self.gamma_start = gamma
            self.gamma_end = gamma
            self.gamma = gamma

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

        self.total_params = context_dim * hidden_dim + hidden_dim
        self.Z_diag = lambda_ntk * jnp.ones(self.total_params)

    def _features(self, X):
        X = jnp.asarray(X)
        Z1 = X @ self.W1 + self.b1
        H = jnp.maximum(0, Z1)
        return H, Z1

    def _compute_gradient_features(self, X, Z1=None):
        X = jnp.asarray(X)
        if Z1 is None:
            _, Z1 = self._features(X)

        theta = self.A_inv @ self.b_bayes
        relu_mask = (Z1 > 0).astype(jnp.float32)
        grad_Z1 = relu_mask * theta

        g_W1 = jnp.einsum('nd,nh->ndh', X, grad_Z1).reshape(self.n_arms, -1)
        g_b1 = grad_Z1

        return jnp.concatenate([g_W1, g_b1], axis=1)

    def _compute_ntk_uncertainties(self, g_features):
        g_scaled = g_features / jnp.sqrt(self.total_params)
        uncertainties = jnp.sqrt(jnp.sum(g_scaled ** 2 / self.Z_diag, axis=1))
        return uncertainties

    def _update_ntk_matrix(self, g_features, mask):
        g_scaled = g_features / jnp.sqrt(self.total_params)
        masked_g2 = jnp.where(mask[:, None], g_scaled ** 2, 0.0)
        self.Z_diag = self.Z_diag + jnp.sum(masked_g2, axis=0)

    def predict_mean(self, X):
        H, _ = self._features(X)
        return compute_shared_predictions(H, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1

        if self.gamma_start != self.gamma_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * progress

        X = jnp.asarray(X)
        H, Z1 = self._features(X)

        predictions = compute_shared_predictions(H, self.A_inv, self.b_bayes)

        g_features = self._compute_gradient_features(X, Z1)
        ntk_uncertainties = self._compute_ntk_uncertainties(g_features)

        sensitivity_info = solve_knapsack_with_sensitivity(
            np.asarray(predictions), self.weights_np, self.capacity
        )
        decision_sensitivity = jnp.asarray(sensitivity_info["decision_relevance"])

        ucb_values = (predictions
                      + self.gamma * ntk_uncertainties
                      + self.beta_decision * decision_sensitivity)

        return self.solver(np.asarray(ucb_values))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        H, Z1 = self._features(X)

        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_shared_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, H, rewards, selection_mask
        )

        g_features = self._compute_gradient_features(X, Z1)
        self._update_ntk_matrix(g_features, selection_mask)

        pred = compute_shared_predictions(H, self.A_inv, self.b_bayes)

        if full_feedback:
            Y_target = rewards
        else:
            ntk_uncertainties = self._compute_ntk_uncertainties(g_features)
            sensitivity_info = solve_knapsack_with_sensitivity(
                np.asarray(pred), self.weights_np, self.capacity
            )
            decision_sensitivity = jnp.asarray(sensitivity_info["decision_relevance"])
            ucb_values = pred + self.gamma * ntk_uncertainties + self.beta_decision * decision_sensitivity
            Y_target = jnp.where(selection > 0.5, rewards, ucb_values)

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

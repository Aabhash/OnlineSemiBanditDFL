import jax.numpy as jnp
import numpy as np
from src.agents.utils.bayesian import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    update_bayesian_stats,
)


class InverseRegretAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        bound_weight: float = 1.0,
        mse_weight: float = 1.0,
        lambda_reg: float = 1.0,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.hidden_dim = hidden_dim

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver
        self.beta = beta
        self.bound_weight = bound_weight
        self.mse_weight = mse_weight
        self.lambda_reg = lambda_reg
        self.rng = np.random.default_rng(seed)
        self.t = 0

        self.W1 = jnp.asarray(np.random.default_rng(seed).standard_normal((context_dim, hidden_dim)) * 0.1)
        self.b1 = jnp.zeros(hidden_dim)

        self.A, self.b, self.A_inv = init_per_arm_regression(
            n_arms, hidden_dim, lambda_reg
        )

        self.regret_bound_history = []
        self.gradient_norm_history = []

    def _get_features(self, X: np.ndarray) -> jnp.ndarray:
        X = jnp.asarray(X)
        z1 = X @ self.W1 + self.b1
        h = jnp.maximum(0, z1)
        return h

    def _compute_ucb_values(self, features: jnp.ndarray) -> jnp.ndarray:
        return compute_ucb_values(features, self.A_inv, self.b, self.beta)

    def _compute_means(self, features: jnp.ndarray) -> jnp.ndarray:
        return compute_bayesian_predictions(features, self.A_inv, self.b)

    def _get_thetas(self) -> jnp.ndarray:
        return jnp.einsum('nij,nj->ni', self.A_inv, self.b)

    def predict_mean(self, X: np.ndarray) -> jnp.ndarray:
        features = self._get_features(X)
        return self._compute_means(features)

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1
        features = self._get_features(X)
        ucb_values = self._compute_ucb_values(features)
        return self.solver(np.asarray(ucb_values))

    def _compute_regret_bound(self, selection: np.ndarray, rewards: np.ndarray,
                               ucb_values: jnp.ndarray) -> float:
        selected = np.where(np.asarray(selection) > 0.5)[0]
        unselected = np.where(np.asarray(selection) < 0.5)[0]
        ucb_np = np.asarray(ucb_values)

        if len(selected) == 0:
            return 0.0

        efficiency = rewards[selected] / (self.weights[selected] + 1e-8)
        threshold = efficiency.min()

        bound = 0.0
        for j in unselected:
            ucb_efficiency = ucb_np[j] / (self.weights[j] + 1e-8)
            excess = max(0, ucb_efficiency - threshold)
            bound += excess * self.weights[j]

        return bound

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = 0.01, full_feedback: bool = False):
        features = self._get_features(X)
        rewards_jax = jnp.asarray(rewards)
        selected = np.where(np.asarray(selection) > 0.5)[0]
        unselected = np.where(np.asarray(selection) < 0.5)[0]

        means = np.asarray(self._compute_means(features))
        ucb_values = self._compute_ucb_values(features)
        thetas = np.asarray(self._get_thetas())

        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = jnp.asarray(selection) > 0.5

        self.A, self.b, self.A_inv = update_bayesian_stats(
            self.A, self.b, self.A_inv, features, rewards_jax, mask
        )

        bound = self._compute_regret_bound(selection, rewards, ucb_values)
        self.regret_bound_history.append(bound)

        if full_feedback:
            grad_h = np.zeros(self.hidden_dim)
            for i in range(self.n_arms):
                error = means[i] - rewards[i]
                grad_h += 2 * error * thetas[i] / self.n_arms
        else:
            grad_h = np.zeros(self.hidden_dim)

            if len(selected) > 0:
                efficiency = rewards[selected] / (self.weights[selected] + 1e-8)
                threshold = efficiency.min()
            else:
                threshold = 0.0

            n_selected = max(len(selected), 1)
            for i in selected:
                error = means[i] - rewards[i]
                grad_h += self.mse_weight * 2 * error * thetas[i] / n_selected

            ucb_np = np.asarray(ucb_values)
            n_unselected = max(len(unselected), 1)
            for j in unselected:
                ucb_efficiency = ucb_np[j] / (self.weights[j] + 1e-8)
                excess = ucb_efficiency - threshold

                if excess > 0:
                    grad_h += self.bound_weight * thetas[j] * excess / n_unselected

        grad_h = np.clip(grad_h, -5, 5)
        self.gradient_norm_history.append(np.linalg.norm(grad_h))

        X_jax = jnp.asarray(X)
        z1 = X_jax @ self.W1 + self.b1
        grad_z1 = jnp.asarray(grad_h) * (z1 > 0)

        grad_W1 = jnp.outer(X_jax, grad_z1)
        grad_b1 = grad_z1

        grad_W1 = jnp.clip(grad_W1, -5, 5)
        grad_b1 = jnp.clip(grad_b1, -5, 5)

        self.W1 = self.W1 - lr * grad_W1
        self.b1 = self.b1 - lr * grad_b1

        self.W1 = jnp.clip(self.W1, -10, 10)

    def get_diagnostics(self) -> dict:
        if not self.regret_bound_history:
            return {}

        return {
            'mean_regret_bound': np.mean(self.regret_bound_history),
            'final_regret_bound': self.regret_bound_history[-1],
            'regret_bound_trend': (
                self.regret_bound_history[-1] - self.regret_bound_history[0]
                if len(self.regret_bound_history) > 1 else 0
            ),
            'mean_gradient_norm': np.mean(self.gradient_norm_history) if self.gradient_norm_history else 0,
        }

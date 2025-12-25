import jax.numpy as jnp
import numpy as np

from src.models import MLPModel
from src.solver import solve_knapsack, perturbed_soft_knapsack
from src.losses import spo_plus_grad
from src.agents.utils.bayesian import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    update_bayesian_stats,
)


class ConservativeUnifiedAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        diversity_threshold: float = 0.5,
        temperature: float = 0.1,
        n_perturbations: int = 10,
        lambda_reg: float = 1.0,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.diversity_threshold = diversity_threshold
        self.temperature = temperature
        self.n_perturbations = n_perturbations
        self.lambda_reg = lambda_reg
        self.rng = np.random.default_rng(seed)
        self.t = 0

        self.model = MLPModel(context_dim, n_arms, hidden_dim)
        local_rng = np.random.default_rng(seed)
        self.model.W1 = jnp.asarray(local_rng.standard_normal((context_dim, hidden_dim)) * 0.1)
        self.model.W2 = jnp.asarray(local_rng.standard_normal((hidden_dim, n_arms)) * 0.1)

        self.A, self.b, self.A_inv = init_per_arm_regression(
            n_arms, hidden_dim, lambda_reg
        )

        self.mode_history = []

    def _forward(self, X: np.ndarray) -> tuple:
        X = jnp.asarray(X)
        z1 = X @ self.model.W1 + self.model.b1
        h = jnp.maximum(0, z1)
        out = h @ self.model.W2 + self.model.b2
        return out, h

    def _compute_ucb_values(self, features: jnp.ndarray) -> jnp.ndarray:
        return compute_ucb_values(features, self.A_inv, self.b, self.beta)

    def predict_mean(self, X: np.ndarray) -> jnp.ndarray:
        _, features = self._forward(X)
        return compute_bayesian_predictions(features, self.A_inv, self.b)

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1
        _, features = self._forward(X)
        ucb_values = compute_ucb_values(features, self.A_inv, self.b, self.beta)
        return solve_knapsack(np.asarray(ucb_values), self.weights, self.capacity)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = 0.01, full_feedback: bool = False):
        pred, features = self._forward(X)
        rewards_jax = jnp.asarray(rewards)

        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = jnp.asarray(selection) > 0.5

        self.A, self.b, self.A_inv = update_bayesian_stats(
            self.A, self.b, self.A_inv, features, rewards_jax, mask
        )

        if full_feedback:
            X_jax = jnp.asarray(X)
            z1 = X_jax @ self.model.W1 + self.model.b1
            h = jnp.maximum(0, z1)

            Y_pred = compute_bayesian_predictions(h, self.A_inv, self.b)
            x_opt = solve_knapsack(np.asarray(rewards), self.weights, self.capacity)
            x_spo = solve_knapsack(np.asarray(2 * Y_pred - rewards), self.weights, self.capacity)
            grad = jnp.asarray(2 * (x_spo - x_opt))

            grad = jnp.clip(grad, -5, 5)

            theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b)
            grad_h = jnp.einsum('n,ni->i', grad, theta)

            grad_z1 = grad_h * (z1 > 0)

            grads = {
                'W1': jnp.outer(X_jax, grad_z1),
                'b1': grad_z1,
                'W2': jnp.zeros_like(self.model.W2),
                'b2': jnp.zeros_like(self.model.b2),
            }
            self.model.update(grads, lr)
        else:
            Y_pred = self.model.forward_with_cache(X)
            pred_std = float(jnp.std(Y_pred))

            if pred_std > self.diversity_threshold:
                self.mode_history.append('pkl')
                grad = self._conservative_pkl_gradient(Y_pred, selection, rewards)
            else:
                self.mode_history.append('spo')
                grad = self._conservative_spo_gradient(Y_pred, selection, rewards, features)

            selected = np.where(np.asarray(selection) > 0.5)[0]
            bayesian_pred = compute_bayesian_predictions(features, self.A_inv, self.b)
            alignment_grad = np.asarray(Y_pred) - np.asarray(bayesian_pred)

            alignment_weight = 0.1
            combined_grad = np.zeros(self.n_arms)
            combined_grad[selected] = grad[selected] + alignment_weight * alignment_grad[selected]

            combined_grad = jnp.clip(jnp.asarray(combined_grad), -5, 5)
            grads = self.model.backward(combined_grad)
            self.model.update(grads, lr)

    def _conservative_pkl_gradient(self, Y_pred, selection, rewards):
        selected = np.where(np.asarray(selection) > 0.5)[0]

        soft_sel, samples = perturbed_soft_knapsack(
            np.asarray(Y_pred), self.weights, self.capacity,
            temperature=self.temperature,
            n_samples=self.n_perturbations,
            rng=self.rng
        )

        observed_reward = np.sum(selection * rewards)
        soft_reward_on_observed = np.sum(soft_sel[selected] * rewards[selected])
        reward_gap = soft_reward_on_observed - observed_reward

        grad = np.zeros(self.n_arms)
        for sample in samples:
            sample_diff = sample - selection
            for i in selected:
                grad[i] += sample_diff[i] * reward_gap / self.temperature
        grad /= len(samples)

        return grad

    def _conservative_spo_gradient(self, Y_pred, selection, rewards, features):
        selected = np.where(np.asarray(selection) > 0.5)[0]

        ucb_values = np.asarray(self._compute_ucb_values(features))
        Y_target = ucb_values.copy()
        Y_target[selected] = rewards[selected]

        full_grad = np.asarray(spo_plus_grad(np.asarray(Y_pred), Y_target, self.weights, self.capacity))

        grad = np.zeros(self.n_arms)
        grad[selected] = full_grad[selected]

        return grad

    def get_diagnostics(self) -> dict:
        if not self.mode_history:
            return {}
        pkl_count = sum(1 for m in self.mode_history if m == 'pkl')
        return {
            'pkl_fraction': pkl_count / len(self.mode_history),
            'spo_fraction': 1 - pkl_count / len(self.mode_history),
        }

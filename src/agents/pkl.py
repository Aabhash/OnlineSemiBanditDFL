import jax.numpy as jnp
import numpy as np
from src.models import MLPModel
from src.solver import perturbed_soft_knapsack
from src.agents.utils.bayesian import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    update_bayesian_stats,
)


class SimplePKLAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        temperature: float = 0.5,
        n_perturbations: int = 10,
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

    def _forward(self, X: np.ndarray) -> tuple:
        X = jnp.asarray(X)
        z1 = X @ self.model.W1 + self.model.b1
        h = jnp.maximum(0, z1)
        out = h @ self.model.W2 + self.model.b2
        return out, h

    def predict_mean(self, X: np.ndarray) -> jnp.ndarray:
        _, features = self._forward(X)
        return compute_bayesian_predictions(features, self.A_inv, self.b)

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1
        _, features = self._forward(X)
        ucb_values = compute_ucb_values(features, self.A_inv, self.b, self.beta)
        return self.solver(np.asarray(ucb_values))

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

        selected = np.where(np.asarray(selection) > 0.5)[0]

        if full_feedback:
            X_jax = jnp.asarray(X)
            z1 = X_jax @ self.model.W1 + self.model.b1
            h = jnp.maximum(0, z1)

            Y_pred = compute_bayesian_predictions(h, self.A_inv, self.b)
            x_opt = self.solver(np.asarray(rewards))
            x_spo = self.solver(np.asarray(2 * Y_pred - rewards))
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

            bayesian_pred = compute_bayesian_predictions(features, self.A_inv, self.b)
            alignment_grad = np.asarray(Y_pred) - np.asarray(bayesian_pred)

            alignment_weight = 0.1
            combined_grad = np.zeros(self.n_arms)
            combined_grad[selected] = grad[selected] + alignment_weight * alignment_grad[selected]

            combined_grad = jnp.clip(jnp.asarray(combined_grad), -5, 5)

            grads = self.model.backward(combined_grad)
            self.model.update(grads, lr)

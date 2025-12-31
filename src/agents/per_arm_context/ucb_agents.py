import jax
import jax.numpy as jnp
import numpy as np
from src.agents.bayesian_utils import (
    init_shared_regression,
    compute_shared_ucb_values,
    update_shared_bayesian_stats,
)


class PerArmLinUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        minimize: bool = False,
        solver=None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.minimize = minimize

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

        self.A, self.b, self.A_inv = init_shared_regression(context_dim, lambda_reg)

    def act(self, X: np.ndarray) -> np.ndarray:
        H = jnp.asarray(X)
        ucb_values = compute_shared_ucb_values(H, self.A_inv, self.b, self.alpha, self.minimize)
        return self.solver(np.asarray(ucb_values))

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        H = jnp.asarray(X)
        rewards_jax = jnp.asarray(rewards)

        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = jnp.asarray(selection) > 0.5

        self.A, self.b, self.A_inv = update_shared_bayesian_stats(
            self.A, self.b, self.A_inv, H, rewards_jax, mask
        )


class PerArmUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        beta: float = 1.0,
        lambda_reg: float = 1.0,
        minimize: bool = False,
        solver=None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.minimize = minimize

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

        self.A, self.b, self.A_inv = init_shared_regression(context_dim, lambda_reg)

    def act(self, X: np.ndarray) -> np.ndarray:
        H = jnp.asarray(X)
        ucb_values = compute_shared_ucb_values(H, self.A_inv, self.b, self.beta, self.minimize)
        return self.solver(np.asarray(ucb_values))

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        H = jnp.asarray(X)
        rewards_jax = jnp.asarray(rewards)

        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = jnp.asarray(selection) > 0.5

        self.A, self.b, self.A_inv = update_shared_bayesian_stats(
            self.A, self.b, self.A_inv, H, rewards_jax, mask
        )


class PerArmNeuralLinUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 32,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        minimize: bool = False,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.minimize = minimize

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

        key = jax.random.PRNGKey(seed if seed else 0)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) / np.sqrt(context_dim)
        self.b1 = jnp.zeros(hidden_dim)
        self.A, self.b, self.A_inv = init_shared_regression(hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        Z1 = X @ self.W1 + self.b1
        H = jnp.maximum(0, Z1)
        return H

    def act(self, X: np.ndarray) -> np.ndarray:
        H = self._features(X)
        ucb_values = compute_shared_ucb_values(H, self.A_inv, self.b, self.alpha, self.minimize)
        return self.solver(np.asarray(ucb_values))

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        H = self._features(X)
        rewards_jax = jnp.asarray(rewards)

        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = jnp.asarray(selection) > 0.5

        self.A, self.b, self.A_inv = update_shared_bayesian_stats(
            self.A, self.b, self.A_inv, H, rewards_jax, mask
        )



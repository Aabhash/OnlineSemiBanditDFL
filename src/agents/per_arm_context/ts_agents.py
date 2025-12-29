import jax.numpy as jnp
import numpy as np
from src.agents.bayesian_utils import (
    init_shared_regression,
    update_shared_bayesian_stats,
)


class PerArmThompsonSamplingAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        lambda_reg: float = 1.0,
        sigma: float = 1.0,
        minimize: bool = False,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.lambda_reg = lambda_reg
        self.sigma = sigma
        self.minimize = minimize

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

        self.rng = np.random.default_rng(seed if seed else 0)
        self.A, self.b, self.A_inv = init_shared_regression(context_dim, lambda_reg)

    def act(self, X: np.ndarray) -> np.ndarray:
        H = jnp.asarray(X)
        theta_mean = np.asarray(self.A_inv @ self.b)
        A_inv_np = np.asarray(self.A_inv)

        try:
            L = np.linalg.cholesky(A_inv_np)
            z = self.rng.standard_normal(self.context_dim)
            theta_sample = theta_mean + self.sigma * L @ z
        except np.linalg.LinAlgError:
            A_inv_diag = np.diag(A_inv_np)
            theta_sample = theta_mean + self.sigma * self.rng.standard_normal(self.context_dim) * np.sqrt(np.maximum(A_inv_diag, 0))

        values = np.asarray(H) @ theta_sample
        return self.solver(values)

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

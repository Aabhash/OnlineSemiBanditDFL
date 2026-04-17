import jax
import jax.numpy as jnp
import numpy as np
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_bayesian_predictions,
    compute_uncertainties,
    update_bayesian_stats,
)
from src.agents.agent_base import AgentBase
from src.solver import solve_knapsack


class ThompsonSamplingAgent(AgentBase):

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        prior_var: float = 1.0,
        noise_var: float = 0.1,
        minimize: bool = False,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.minimize = minimize
        self.rng_key = jax.random.PRNGKey(seed if seed is not None else 0)
        lambda_reg = 1.0 / prior_var
        self.A, self.b, self.A_inv = init_per_arm_regression(
            n_arms, context_dim, lambda_reg
        )

        if solver is None:
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

    def act(self, X: np.ndarray) -> np.ndarray:
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        features = jnp.asarray(X)
        means = compute_bayesian_predictions(features, self.A_inv, self.b)
        stds = compute_uncertainties(features, self.A_inv)
        noise = jax.random.normal(sample_key, (self.n_arms,))
        y_sample = means + stds * noise
        return self.solver(np.asarray(y_sample))

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        features = jnp.asarray(X)
        return np.asarray(compute_bayesian_predictions(features, self.A_inv, self.b))

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        features = jnp.asarray(X)
        return np.asarray(compute_uncertainties(features, self.A_inv))

    def predict_mean_and_uncertainty(self, X):
        features = jnp.asarray(X)
        return (
            np.asarray(compute_bayesian_predictions(features, self.A_inv, self.b)),
            np.asarray(compute_uncertainties(features, self.A_inv)),
        )

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False, **kwargs):
        scale = jnp.sqrt(self.noise_var)
        features = jnp.asarray(X) / scale
        rewards_scaled = jnp.asarray(rewards) / scale

        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = jnp.asarray(selection) > 0.5

        self.A, self.b, self.A_inv = update_bayesian_stats(
            self.A, self.b, self.A_inv, features, rewards_scaled, mask
        )

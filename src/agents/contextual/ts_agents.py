import jax
import numpy as np
from src.models import BayesianLinearModel
from src.agents.agent_base import AgentBase
from src.solver import solve_knapsack


class ThompsonSamplingAgent(AgentBase):

    def __init__(
        self,
        model: BayesianLinearModel,
        weights: np.ndarray,
        capacity: float,
        minimize: bool = False,
        solver=None,
        seed: int = None,
    ):
        self.model = model
        self.weights = weights
        self.capacity = capacity
        self.n_arms = len(weights)
        self.minimize = minimize
        self.rng_key = jax.random.PRNGKey(seed if seed is not None else 0)

        if solver is None:
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

    def act(self, X: np.ndarray) -> np.ndarray:
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        Y_sample = self.model.predict_sample(X, key=sample_key)
        return self.solver(Y_sample)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_mean(X)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        return self.model.get_uncertainty(X)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False, **kwargs):
        self.model.update_batch(X, selection, rewards, full_feedback=full_feedback)

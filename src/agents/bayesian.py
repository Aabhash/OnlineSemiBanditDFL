import numpy as np
from src.models import BayesianLinearModel
from src.solver import solve_knapsack


class ThompsonSamplingAgent:

    def __init__(
        self,
        model: BayesianLinearModel,
        weights: np.ndarray,
        capacity: float,
    ):
        self.model = model
        self.weights = weights
        self.capacity = capacity
        self.n_arms = len(weights)

    def act(self, X: np.ndarray) -> np.ndarray:
        Y_sample = self.model.predict_sample(X)
        return solve_knapsack(Y_sample, self.weights, self.capacity)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        self.model.update_batch(X, selection, rewards, full_feedback=full_feedback)


class UCBAgent:

    def __init__(
        self,
        model: BayesianLinearModel,
        weights: np.ndarray,
        capacity: float,
        beta: float = 1.0,
    ):
        self.model = model
        self.weights = weights
        self.capacity = capacity
        self.beta = beta
        self.n_arms = len(weights)

    def act(self, X: np.ndarray) -> np.ndarray:
        Y_ucb = self.model.predict_ucb(X, self.beta)
        return solve_knapsack(Y_ucb, self.weights, self.capacity)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        self.model.update_batch(X, selection, rewards, full_feedback=full_feedback)

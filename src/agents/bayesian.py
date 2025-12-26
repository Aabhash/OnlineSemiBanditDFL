import numpy as np
from src.models import BayesianLinearModel


class ThompsonSamplingAgent:

    def __init__(
        self,
        model: BayesianLinearModel,
        weights: np.ndarray,
        capacity: float,
        minimize: bool = False,
        solver=None,
    ):
        self.model = model
        self.weights = weights
        self.capacity = capacity
        self.n_arms = len(weights)
        self.minimize = minimize

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

    def act(self, X: np.ndarray) -> np.ndarray:
        Y_sample = self.model.predict_sample(X)
        # For TS, we sample from posterior - no need to negate for minimization
        # since the solver handles finding the optimal (min or max)
        return self.solver(Y_sample)

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
        minimize: bool = False,
        solver=None,
    ):
        self.model = model
        self.weights = weights
        self.capacity = capacity
        self.beta = beta
        self.n_arms = len(weights)
        self.minimize = minimize

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

    def act(self, X: np.ndarray) -> np.ndarray:
        Y_ucb = self.model.predict_ucb(X, self.beta, self.minimize)
        return self.solver(Y_ucb)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        self.model.update_batch(X, selection, rewards, full_feedback=full_feedback)

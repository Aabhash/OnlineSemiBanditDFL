import numpy as np
from src.solver import solve_knapsack
from src.agents.utils.bayesian import sherman_morrison_update, init_per_arm_regression


class LinUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        self.A, self.b, self.A_inv = init_per_arm_regression(
            n_arms, context_dim, lambda_reg
        )

    def act(self, X: np.ndarray) -> np.ndarray:
        ucb_values = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            theta_i = self.A_inv[i] @ self.b[i]
            mean = theta_i @ X
            std = np.sqrt(X @ self.A_inv[i] @ X)
            ucb_values[i] = mean + self.alpha * std

        return solve_knapsack(ucb_values, self.weights, self.capacity)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        for i in range(self.n_arms):
            if full_feedback or selection[i] > 0.5:
                self.A[i] += np.outer(X, X)
                self.b[i] += X * rewards[i]
                self.A_inv[i] = sherman_morrison_update(self.A_inv[i], X)


class NeuralUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 64,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        lr: float = 0.001,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.rng = np.random.default_rng(seed)

        self.W1 = self.rng.standard_normal((hidden_dim, context_dim)) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.standard_normal((n_arms, hidden_dim)) * 0.1
        self.b2 = np.zeros(n_arms)

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            n_arms, hidden_dim, lambda_reg
        )

        self.data_buffer = []
        self.update_freq = 10
        self.t = 0

    def _forward(self, X: np.ndarray) -> tuple:
        h = np.maximum(0, self.W1 @ X + self.b1)
        out = self.W2 @ h + self.b2
        return out, h

    def act(self, X: np.ndarray) -> np.ndarray:
        _, features = self._forward(X)

        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            theta_i = self.A_inv[i] @ self.b_bayes[i]
            mean = theta_i @ features
            std = np.sqrt(features @ self.A_inv[i] @ features)
            ucb_values[i] = mean + self.alpha * std

        return solve_knapsack(ucb_values, self.weights, self.capacity)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        self.t += 1

        pred, features = self._forward(X)

        for i in range(self.n_arms):
            if full_feedback or selection[i] > 0.5:
                self.A[i] += np.outer(features, features)
                self.b_bayes[i] += features * rewards[i]
                self.A_inv[i] = sherman_morrison_update(self.A_inv[i], features)

        self.data_buffer.append((X, selection, rewards, full_feedback))

        if self.t % self.update_freq == 0 and len(self.data_buffer) >= self.update_freq:
            self._update_network()

    def _update_network(self):
        for X, selection, rewards, full_feedback in self.data_buffer[-self.update_freq:]:
            pred, h = self._forward(X)

            for i in range(self.n_arms):
                if full_feedback or selection[i] > 0.5:
                    error = np.clip(pred[i] - rewards[i], -1, 1)

                    self.W2[i] -= self.lr * error * h
                    self.b2[i] -= self.lr * error

                    d_h = error * self.W2[i] * (h > 0)
                    d_h = np.clip(d_h, -1, 1)
                    self.W1 -= self.lr * np.outer(d_h, X)
                    self.b1 -= self.lr * d_h

        self.W1 = np.clip(self.W1, -5, 5)
        self.W2 = np.clip(self.W2, -5, 5)
        self.b1 = np.clip(self.b1, -5, 5)
        self.b2 = np.clip(self.b2, -5, 5)

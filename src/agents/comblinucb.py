import numpy as np
from src.solver import solve_top_k, solve_knapsack


class CombLinUCBAgent:
    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray = None,
        capacity: float = None,
        alpha: float = 5.0,
        lambda_reg: float = 1.0,
        constraint_type: str = 'top_k',
        K: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = np.ones(n_arms) if weights is None else np.asarray(weights)
        self.capacity = K if capacity is None else capacity
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.constraint_type = constraint_type
        self.K = K if K is not None else n_arms // 5

        self.A = [lambda_reg * np.eye(context_dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(context_dim) / lambda_reg for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

        self.t = 0

    def act(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        self.t += 1

        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            x_i = X[i]
            theta_i = self.A_inv[i] @ self.b[i]
            pred = x_i @ theta_i
            uncertainty = np.sqrt(x_i @ self.A_inv[i] @ x_i)
            ucb_values[i] = pred + self.alpha * uncertainty

        if self.constraint_type == 'top_k':
            return solve_top_k(ucb_values, self.K)
        else:
            return solve_knapsack(ucb_values, self.weights, self.capacity)

    def update(
        self,
        X: np.ndarray,
        selection: np.ndarray,
        rewards: np.ndarray,
        lr: float = None,
        full_feedback: bool = False,
    ):
        X = np.asarray(X)
        rewards = np.asarray(rewards)
        selection = np.asarray(selection)

        if full_feedback:
            mask = np.ones(self.n_arms, dtype=bool)
        else:
            mask = selection > 0.5

        for i in range(self.n_arms):
            if mask[i]:
                x_i = X[i]  # (context_dim,)
                r_i = rewards[i]

                self.A[i] = self.A[i] + np.outer(x_i, x_i)

                self.b[i] = self.b[i] + r_i * x_i

                Ax = self.A_inv[i] @ x_i
                denom = 1.0 + x_i @ Ax
                self.A_inv[i] = self.A_inv[i] - np.outer(Ax, Ax) / denom


class CombLinTSAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray = None,
        capacity: float = None,
        nu: float = 1.0,
        lambda_reg: float = 1.0,
        constraint_type: str = 'top_k',
        K: int = None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = np.ones(n_arms) if weights is None else np.asarray(weights)
        self.capacity = K if capacity is None else capacity
        self.nu = nu
        self.lambda_reg = lambda_reg
        self.constraint_type = constraint_type
        self.K = K if K is not None else n_arms // 5

        self.A = lambda_reg * np.eye(context_dim)
        self.A_inv = np.eye(context_dim) / lambda_reg
        self.b = np.zeros(context_dim)

        self.rng = np.random.default_rng(seed)
        self.t = 0

    def act(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        self.t += 1

        theta_hat = self.A_inv @ self.b

        try:
            L = np.linalg.cholesky(self.nu**2 * self.A_inv)
            z = self.rng.standard_normal(self.context_dim)
            theta_sample = theta_hat + L @ z
        except np.linalg.LinAlgError:
            theta_sample = theta_hat

        sampled_values = X @ theta_sample

        if self.constraint_type == 'top_k':
            return solve_top_k(sampled_values, self.K)
        else:
            return solve_knapsack(sampled_values, self.weights, self.capacity)

    def update(
        self,
        X: np.ndarray,
        selection: np.ndarray,
        rewards: np.ndarray,
        lr: float = None,
        full_feedback: bool = False,
    ):
        X = np.asarray(X)
        rewards = np.asarray(rewards)
        selection = np.asarray(selection)

        if full_feedback:
            mask = np.ones(self.n_arms, dtype=bool)
        else:
            mask = selection > 0.5

        for i in range(self.n_arms):
            if mask[i]:
                x_i = X[i]
                r_i = rewards[i]

                self.A = self.A + np.outer(x_i, x_i)
                self.b = self.b + r_i * x_i

                Ax = self.A_inv @ x_i
                denom = 1.0 + x_i @ Ax
                self.A_inv = self.A_inv - np.outer(Ax, Ax) / denom

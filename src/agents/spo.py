import numpy as np
from src.models import MLPModel
from src.solver import solve_knapsack
from src.losses import spo_plus_grad
from src.agents.utils.bayesian import (
    sherman_morrison_update,
    init_per_arm_regression,
)


class FullSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 32,
        beta_start: float = 2.0,
        beta_end: float = 0.1,
        decay_rounds: int = 300,
        lambda_reg: float = 1.0,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.decay_rounds = decay_rounds
        self.beta = beta_start
        self.lambda_reg = lambda_reg
        self.rng = np.random.default_rng(seed)
        self.t = 0

        self.W1 = self.rng.standard_normal((context_dim, hidden_dim)) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.standard_normal((hidden_dim, n_arms)) * 0.1
        self.b2 = np.zeros(n_arms)

        self.A, _, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def _forward(self, X: np.ndarray) -> tuple:
        z1 = X @ self.W1 + self.b1
        h = np.maximum(0, z1)
        out = h @ self.W2 + self.b2
        return out, h

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred, _ = self._forward(X)
        return pred

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1

        progress = min(self.t / self.decay_rounds, 1.0)
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        pred, h = self._forward(X)

        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            std = np.sqrt(h @ self.A_inv[i] @ h)
            ucb_values[i] = pred[i] + self.beta * std

        return solve_knapsack(ucb_values, self.weights, self.capacity)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float, full_feedback: bool = False):
        pred, h = self._forward(X)

        for i in range(self.n_arms):
            if full_feedback or selection[i] > 0.5:
                self.A[i] += np.outer(h, h)
                self.A_inv[i] = sherman_morrison_update(self.A_inv[i], h)

        if full_feedback:
            Y_target = rewards
        else:
            Y_target = pred.copy()
            selected_idx = selection > 0.5
            Y_target[selected_idx] = rewards[selected_idx]

        spo_grad = spo_plus_grad(pred, Y_target, self.weights, self.capacity)

        d_W2 = np.outer(h, spo_grad)
        d_b2 = spo_grad
        d_h = self.W2 @ spo_grad
        d_h = d_h * (h > 0)
        d_W1 = np.outer(X, d_h)
        d_b1 = d_h

        d_W1 = np.clip(d_W1, -5, 5)
        d_W2 = np.clip(d_W2, -5, 5)
        d_b1 = np.clip(d_b1, -5, 5)
        d_b2 = np.clip(d_b2, -5, 5)

        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2

        self.W1 = np.clip(self.W1, -10, 10)
        self.W2 = np.clip(self.W2, -10, 10)


class OptimisticSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 32,
        beta_start: float = 2.0,
        beta_end: float = 0.1,
        decay_rounds: int = 300,
        alpha: float = 0.0,
        lambda_reg: float = 1.0,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.decay_rounds = decay_rounds
        self.beta = beta_start
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.rng = np.random.default_rng(seed)
        self.t = 0

        self.model = MLPModel(context_dim, n_arms, hidden_dim)
        local_rng = np.random.default_rng(seed)
        self.model.W1 = local_rng.standard_normal((context_dim, hidden_dim)) * 0.1
        self.model.W2 = local_rng.standard_normal((hidden_dim, n_arms)) * 0.1

        self.A, self.b, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def _forward(self, X: np.ndarray) -> tuple:
        z1 = X @ self.model.W1 + self.model.b1
        h = np.maximum(0, z1)
        out = h @ self.model.W2 + self.model.b2
        return out, h

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        _, features = self._forward(X)
        means = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            theta_i = self.A_inv[i] @ self.b[i]
            means[i] = theta_i @ features
        return means

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1

        progress = min(self.t / self.decay_rounds, 1.0)
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        _, features = self._forward(X)

        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            theta_i = self.A_inv[i] @ self.b[i]
            mean = theta_i @ features
            std = np.sqrt(features @ self.A_inv[i] @ features)
            ucb_values[i] = mean + self.beta * std

        return solve_knapsack(ucb_values, self.weights, self.capacity)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float, full_feedback: bool = False):
        pred, features = self._forward(X)

        for i in range(self.n_arms):
            if full_feedback or selection[i] > 0.5:
                self.A[i] += np.outer(features, features)
                self.b[i] += features * rewards[i]
                self.A_inv[i] = sherman_morrison_update(self.A_inv[i], features)

        if full_feedback:
            Y_target = rewards
        else:
            ucb_values = np.zeros(self.n_arms)
            for i in range(self.n_arms):
                theta_i = self.A_inv[i] @ self.b[i]
                mean = theta_i @ features
                std = np.sqrt(features @ self.A_inv[i] @ features)
                ucb_values[i] = mean + self.beta * std

            Y_target = ucb_values.copy()
            selected_idx = selection > 0.5
            Y_target[selected_idx] = rewards[selected_idx]

        Y_pred = self.model.forward_with_cache(X)
        mse_grad = 2 * (Y_pred - Y_target) / self.n_arms
        spo_grad = spo_plus_grad(Y_pred, Y_target, self.weights, self.capacity)
        hybrid_grad = self.alpha * mse_grad + (1 - self.alpha) * spo_grad

        grads = self.model.backward(hybrid_grad)
        self.model.update(grads, lr)


class EndToEndSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 32,
        beta_start: float = 2.0,
        beta_end: float = 0.1,
        decay_rounds: int = 300,
        alpha: float = 0.0,
        lambda_reg: float = 1.0,
        theta_lr_scale: float = 1.0,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.decay_rounds = decay_rounds
        self.beta = beta_start
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.theta_lr_scale = theta_lr_scale
        self.rng = np.random.default_rng(seed)
        self.t = 0

        self.model = MLPModel(context_dim, n_arms, hidden_dim)
        local_rng = np.random.default_rng(seed)
        self.model.W1 = local_rng.standard_normal((context_dim, hidden_dim)) * 0.1
        self.model.W2 = local_rng.standard_normal((hidden_dim, n_arms)) * 0.1

        self.theta = np.zeros((n_arms, hidden_dim))

        self.A, _, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def _forward(self, X: np.ndarray) -> tuple:
        z1 = X @ self.model.W1 + self.model.b1
        h = np.maximum(0, z1)
        out = h @ self.model.W2 + self.model.b2
        return out, h

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        _, features = self._forward(X)
        means = np.array([self.theta[i] @ features for i in range(self.n_arms)])
        return means

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1

        progress = min(self.t / self.decay_rounds, 1.0)
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        _, features = self._forward(X)

        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            mean = self.theta[i] @ features
            std = np.sqrt(features @ self.A_inv[i] @ features)
            ucb_values[i] = mean + self.beta * std

        return solve_knapsack(ucb_values, self.weights, self.capacity)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float, full_feedback: bool = False):
        _, features = self._forward(X)

        for i in range(self.n_arms):
            if full_feedback or selection[i] > 0.5:
                self.A[i] += np.outer(features, features)
                self.A_inv[i] = sherman_morrison_update(self.A_inv[i], features)

        if full_feedback:
            Y_target = rewards
        else:
            ucb_values = np.zeros(self.n_arms)
            for i in range(self.n_arms):
                mean = self.theta[i] @ features
                std = np.sqrt(features @ self.A_inv[i] @ features)
                ucb_values[i] = mean + self.beta * std

            Y_target = ucb_values.copy()
            selected_idx = selection > 0.5
            Y_target[selected_idx] = rewards[selected_idx]

        Y_pred = np.array([self.theta[i] @ features for i in range(self.n_arms)])

        spo_grad = spo_plus_grad(Y_pred, Y_target, self.weights, self.capacity)
        mse_grad = 2 * (Y_pred - Y_target) / self.n_arms
        pred_grad = self.alpha * mse_grad + (1 - self.alpha) * spo_grad

        theta_lr = lr * self.theta_lr_scale
        for i in range(self.n_arms):
            grad_theta_i = pred_grad[i] * features
            grad_theta_i = np.clip(grad_theta_i, -1, 1)
            self.theta[i] -= theta_lr * grad_theta_i

        Y_mlp = self.model.forward_with_cache(X)
        mlp_spo_grad = spo_plus_grad(Y_mlp, Y_target, self.weights, self.capacity)
        mlp_mse_grad = 2 * (Y_mlp - Y_target) / self.n_arms
        mlp_grad = self.alpha * mlp_mse_grad + (1 - self.alpha) * mlp_spo_grad

        grads = self.model.backward(mlp_grad)
        self.model.update(grads, lr)

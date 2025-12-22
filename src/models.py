import numpy as np
from typing import Tuple, Optional


class LinearModel:

    def __init__(self, context_dim: int, n_arms: int):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.theta = np.random.randn(n_arms, context_dim) * 0.1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.theta @ X

    def gradient_mse(self, X: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        Y_pred = self.predict(X)
        error = Y_pred - Y_true
        grad = 2 * np.outer(error, X) / self.n_arms
        return grad

    def gradient_spo(self, X: np.ndarray, spo_grad: np.ndarray) -> np.ndarray:
        grad = np.outer(spo_grad, X)
        return grad

    def update(self, grad: np.ndarray, lr: float):
        grad = np.clip(grad, -1, 1)
        self.theta -= lr * grad

    def copy(self) -> "LinearModel":
        new = LinearModel(self.context_dim, self.n_arms)
        new.theta = self.theta.copy()
        return new


class MLPModel:

    def __init__(self, context_dim: int, n_arms: int, hidden_dim: int = 64):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.hidden_dim = hidden_dim

        self.W1 = np.random.randn(context_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_arms) * 0.1
        self.b2 = np.zeros(n_arms)
        self._cache = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        out = h1 @ self.W2 + self.b2
        return out

    def forward_with_cache(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        out = h1 @ self.W2 + self.b2
        self._cache = {'X': X, 'z1': z1, 'h1': h1}
        return out

    def backward(self, grad_out: np.ndarray) -> dict:
        X = self._cache['X']
        z1 = self._cache['z1']
        h1 = self._cache['h1']

        grad_W2 = np.outer(h1, grad_out)
        grad_b2 = grad_out
        grad_h1 = self.W2 @ grad_out
        grad_z1 = grad_h1 * (z1 > 0)
        grad_W1 = np.outer(X, grad_z1)
        grad_b1 = grad_z1

        return {
            'W1': grad_W1, 'b1': grad_b1,
            'W2': grad_W2, 'b2': grad_b2,
        }

    def gradient_mse(self, X: np.ndarray, Y_true: np.ndarray) -> dict:
        Y_pred = self.forward_with_cache(X)
        grad_out = 2 * (Y_pred - Y_true) / self.n_arms
        return self.backward(grad_out)

    def gradient_spo(self, X: np.ndarray, spo_grad: np.ndarray) -> dict:
        _ = self.forward_with_cache(X)
        return self.backward(spo_grad)

    def update(self, grads: dict, lr: float):
        for key in ['W1', 'b1', 'W2', 'b2']:
            grad = np.clip(grads[key], -1, 1)
            setattr(self, key, getattr(self, key) - lr * grad)

    def copy(self) -> "MLPModel":
        new = MLPModel(self.context_dim, self.n_arms, self.hidden_dim)
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2.copy()
        return new


class BayesianLinearModel:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        prior_var: float = 1.0,
        noise_var: float = 0.1,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.prior_var = prior_var
        self.noise_var = noise_var

        self.mu = [np.zeros(context_dim) for _ in range(n_arms)]
        self.Sigma = [np.eye(context_dim) * prior_var for _ in range(n_arms)]
        self.Lambda = [np.eye(context_dim) / prior_var for _ in range(n_arms)]
        self.eta = [np.zeros(context_dim) for _ in range(n_arms)]

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return np.array([X @ self.mu[a] for a in range(self.n_arms)])

    def predict_sample(self, X: np.ndarray) -> np.ndarray:
        samples = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta_sample = np.random.multivariate_normal(self.mu[a], self.Sigma[a])
            samples[a] = X @ theta_sample
        return samples

    def predict_ucb(self, X: np.ndarray, beta: float = 1.0) -> np.ndarray:
        means = np.zeros(self.n_arms)
        stds = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            means[a] = X @ self.mu[a]
            stds[a] = np.sqrt(X @ self.Sigma[a] @ X)
        return means + beta * stds

    def update(self, X: np.ndarray, arm: int, reward: float):
        self.Lambda[arm] += np.outer(X, X) / self.noise_var
        self.eta[arm] += X * reward / self.noise_var
        self.Sigma[arm] = np.linalg.inv(self.Lambda[arm])
        self.mu[arm] = self.Sigma[arm] @ self.eta[arm]

    def update_batch(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray, full_feedback: bool = False):
        for arm in range(self.n_arms):
            if full_feedback or selection[arm] > 0.5:
                self.update(X, arm, rewards[arm])

    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        return np.array([np.sqrt(X @ self.Sigma[a] @ X) for a in range(self.n_arms)])


def create_model(
    model_type: str,
    context_dim: int,
    n_arms: int,
    **kwargs
):
    if model_type == "linear":
        return LinearModel(context_dim, n_arms)
    elif model_type == "mlp":
        hidden_dim = kwargs.get('hidden_dim', 32)
        return MLPModel(context_dim, n_arms, hidden_dim)
    elif model_type == "bayesian":
        prior_var = kwargs.get('prior_var', 1.0)
        noise_var = kwargs.get('noise_var', 0.1)
        return BayesianLinearModel(context_dim, n_arms, prior_var, noise_var)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

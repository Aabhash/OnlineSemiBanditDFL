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


class PerArmNeuralUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 32,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        minimize: bool = False,
        solver=None,
        seed: int = None,
        train_interval: int = 10,
        train_epochs: int = 50,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.minimize = minimize
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.train_interval = train_interval
        self.train_epochs = train_epochs
        self.t = 0

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver

        key = jax.random.PRNGKey(seed if seed else 0)
        k1, k2 = jax.random.split(key)

        self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) / np.sqrt(hidden_dim)
        self.b1 = jnp.zeros(hidden_dim)
        self.W2 = jax.random.normal(k2, (hidden_dim, 1)) / np.sqrt(hidden_dim)
        self.b2 = jnp.zeros(1)

        self.W1_init = self.W1.copy()
        self.b1_init = self.b1.copy()
        self.W2_init = self.W2.copy()
        self.b2_init = self.b2.copy()

        self.total_params = (context_dim * hidden_dim + hidden_dim +
                            hidden_dim * 1 + 1)
        self.U_diag = lambda_reg * jnp.ones(self.total_params)

        self.history_X = []
        self.history_Y = []
        self.history_mask = []
        self.rng = np.random.default_rng(seed if seed else 0)

        self.output_scale = 1.0
        self.output_bias = 0.0
        self.target_mean = 0.0
        self.target_std = 1.0
        self.target_count = 0

    def _forward(self, X):
        X = jnp.asarray(X)
        Z1 = X @ self.W1 + self.b1
        H = jnp.maximum(0, Z1)
        out = (H @ self.W2 + self.b2).squeeze(-1)
        return out

    def predict(self, X):
        raw = self._forward(X)
        return raw * self.output_scale + self.output_bias

    def _compute_gradient_features(self, X):
        X = jnp.asarray(X)

        def forward_flat(flat_params, x_single):
            w1_size = self.context_dim * self.hidden_dim
            b1_size = self.hidden_dim
            w2_size = self.hidden_dim * 1

            W1 = flat_params[:w1_size].reshape(self.context_dim, self.hidden_dim)
            b1 = flat_params[w1_size:w1_size + b1_size]
            W2 = flat_params[w1_size + b1_size:w1_size + b1_size + w2_size].reshape(self.hidden_dim, 1)
            b2 = flat_params[w1_size + b1_size + w2_size:]

            z1 = x_single @ W1 + b1
            h = jnp.maximum(0, z1)
            out = (h @ W2 + b2).squeeze()
            return out

        flat_params = jnp.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel()
        ])

        jacobians = jax.vmap(lambda x: jax.grad(lambda p: forward_flat(p, x))(flat_params))(X)
        return jacobians

    def _compute_uncertainties(self, g_features):
        g_scaled = g_features / jnp.sqrt(self.total_params)
        sigma2 = g_scaled ** 2 / self.U_diag
        return self.alpha * jnp.sqrt(jnp.sum(sigma2, axis=1))

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1
        X = jnp.asarray(X)

        predictions = self.predict(X)
        g_features = self._compute_gradient_features(X)
        uncertainties = self._compute_uncertainties(g_features)
        scaled_uncertainties = uncertainties * self.output_scale

        self._cached_g_features = g_features

        if self.minimize:
            lcb_values = predictions - scaled_uncertainties
            return self.solver(np.asarray(lcb_values))
        else:
            ucb_values = predictions + scaled_uncertainties
            return self.solver(np.asarray(ucb_values))

    def _update_information_matrix(self, g_features, mask):
        g_scaled = g_features / jnp.sqrt(self.total_params)
        masked_g2 = jnp.where(mask[:, None], g_scaled ** 2, 0.0)
        self.U_diag = self.U_diag + jnp.sum(masked_g2, axis=0)

    def _train_batch(self):
        if len(self.history_X) == 0:
            return

        all_X = jnp.asarray(np.array(self.history_X))
        all_Y = jnp.asarray(np.array(self.history_Y))
        all_mask = jnp.asarray(np.array(self.history_mask))
        n_samples = len(all_X)

        batch_size = min(100, n_samples)

        for epoch in range(self.train_epochs):
            if n_samples <= batch_size:
                batch_idx = np.arange(n_samples)
            else:
                batch_idx = self.rng.choice(n_samples, size=batch_size, replace=False)

            X_batch = all_X[batch_idx]
            Y_batch = all_Y[batch_idx]
            mask_batch = all_mask[batch_idx]

            def loss_fn(params):
                W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

                z1 = jnp.einsum('bac,ch->bah', X_batch, W1) + b1
                h = jnp.maximum(0, z1)
                pred = (jnp.einsum('bah,ho->bao', h, W2) + b2).squeeze(-1)

                diff = (pred - Y_batch) * mask_batch
                n_obs = jnp.maximum(jnp.sum(mask_batch), 1.0)
                mse_loss = 0.5 * jnp.sum(diff ** 2) / n_obs

                reg_loss = 0.0
                reg_loss += jnp.mean((W1 - self.W1_init) ** 2)
                reg_loss += jnp.mean((b1 - self.b1_init) ** 2)
                reg_loss += jnp.mean((W2 - self.W2_init) ** 2)
                reg_loss += jnp.mean((b2 - self.b2_init) ** 2)
                reg_loss = 0.5 * self.lambda_reg * reg_loss

                return mse_loss + reg_loss

            params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
            grads = jax.grad(loss_fn)(params)

            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads.values()))
            clip_coef = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-6))

            self.W1 = self.W1 - self.lr * grads['W1'] * clip_coef
            self.b1 = self.b1 - self.lr * grads['b1'] * clip_coef
            self.W2 = self.W2 - self.lr * grads['W2'] * clip_coef
            self.b2 = self.b2 - self.lr * grads['b2'] * clip_coef

    def _update_output_scaling(self, rewards, mask):
        observed = np.asarray(rewards)[np.asarray(mask)]
        if len(observed) == 0:
            return

        for y in observed:
            self.target_count += 1
            delta = y - self.target_mean
            self.target_mean += delta / self.target_count
            delta2 = y - self.target_mean
            self.target_std += delta * delta2

        if self.target_count > 1:
            variance = self.target_std / (self.target_count - 1)
            std = np.sqrt(max(variance, 1e-6))
            self.output_scale = std
            self.output_bias = self.target_mean

    def _normalize_targets(self, targets):
        if self.target_count > 1 and self.output_scale > 1e-6:
            return (np.asarray(targets) - self.target_mean) / self.output_scale
        return np.asarray(targets)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False):
        X_np = np.asarray(X)
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        obs_mask_scaling = jnp.ones(self.n_arms, dtype=bool) if full_feedback else (selection > 0.5)
        self._update_output_scaling(rewards, obs_mask_scaling)

        selection_mask = selection > 0.5

        if hasattr(self, '_cached_g_features'):
            g_features = self._cached_g_features
            del self._cached_g_features
        else:
            g_features = self._compute_gradient_features(X)

        self._update_information_matrix(g_features, selection_mask)

        raw_predictions = np.asarray(self._forward(X))
        rewards_np = np.asarray(rewards)
        selection_np = np.asarray(selection) > 0.5

        if full_feedback:
            Y_target = np.where(
                selection_np,
                self._normalize_targets(rewards_np),
                raw_predictions
            )
        else:
            Y_target = np.where(
                selection_np,
                self._normalize_targets(rewards_np),
                raw_predictions
            )
        obs_mask = selection_np.astype(float)

        self.history_X.append(X_np)
        self.history_Y.append(Y_target)
        self.history_mask.append(obs_mask)

        if self.t % self.train_interval == 0:
            self._train_batch()

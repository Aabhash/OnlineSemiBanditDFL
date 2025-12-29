import jax
import jax.numpy as jnp
import numpy as np
from src.models import BayesianLinearModel
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_ucb_values,
    update_bayesian_stats,
)


class LinUCBAgent:

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

        self.A, self.b, self.A_inv = init_per_arm_regression(
            n_arms, context_dim, lambda_reg
        )

    def act(self, X: np.ndarray) -> np.ndarray:
        features = jnp.asarray(X)
        ucb_values = compute_ucb_values(features, self.A_inv, self.b, self.alpha, self.minimize)
        return self.solver(np.asarray(ucb_values))

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        features = jnp.asarray(X)
        rewards_jax = jnp.asarray(rewards)

        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = jnp.asarray(selection) > 0.5

        self.A, self.b, self.A_inv = update_bayesian_stats(
            self.A, self.b, self.A_inv, features, rewards_jax, mask
        )


class NeuralUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 100,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        minimize: bool = False,
        solver=None,
        seed: int = None,
        train_interval: int = 10,
        train_epochs: int = 100,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.minimize = minimize

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver
        self.nu = alpha
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.train_interval = train_interval
        self.train_epochs = train_epochs
        self.t = 0

        key = jax.random.PRNGKey(seed if seed else 0)
        k1, k2 = jax.random.split(key)
        self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) / np.sqrt(hidden_dim)
        self.b1 = jnp.zeros(hidden_dim)
        self.W2 = jax.random.normal(k2, (hidden_dim, n_arms)) / np.sqrt(hidden_dim)
        self.b2 = jnp.zeros(n_arms)

        self.W1_init = self.W1.copy()
        self.b1_init = self.b1.copy()
        self.W2_init = self.W2.copy()
        self.b2_init = self.b2.copy()

        self.total_params = (context_dim * hidden_dim + hidden_dim +
                            hidden_dim * n_arms + n_arms)

        self.U_diag = lambda_reg * jnp.ones(self.total_params)

        self.history_X = []
        self.history_Y = []
        self.history_mask = []
        self.rng = np.random.default_rng(seed if seed else 0)

        self._cached_g_features = None
        self._cached_X = None

        self._build_jit_functions()

    def _build_jit_functions(self):
        context_dim = self.context_dim
        hidden_dim = self.hidden_dim
        n_arms = self.n_arms

        w1_size = context_dim * hidden_dim
        b1_size = hidden_dim
        w2_size = hidden_dim * n_arms

        @jax.jit
        def forward(W1, b1, W2, b2, X):
            z1 = X @ W1 + b1
            h = jnp.maximum(0, z1)
            return h @ W2 + b2

        @jax.jit
        def compute_jacobian(flat_params, X):
            def forward_flat(fp):
                W1 = fp[:w1_size].reshape(context_dim, hidden_dim)
                b1 = fp[w1_size:w1_size + b1_size]
                W2 = fp[w1_size + b1_size:w1_size + b1_size + w2_size].reshape(
                    hidden_dim, n_arms)
                b2 = fp[w1_size + b1_size + w2_size:]
                z1 = X @ W1 + b1
                h = jnp.maximum(0, z1)
                return h @ W2 + b2
            return jax.jacobian(forward_flat)(flat_params)

        @jax.jit
        def compute_uncertainties(g_features, U_diag, lambda_reg, nu, m):
            g_scaled = g_features / jnp.sqrt(m)
            sigma2 = g_scaled ** 2 / U_diag
            return nu * jnp.sqrt(jnp.sum(sigma2, axis=1))

        @jax.jit
        def update_U_diag(U_diag, g_features, mask, m):
            g_scaled = g_features / jnp.sqrt(m)
            masked_g2 = jnp.where(mask[:, None], g_scaled ** 2, 0.0)
            return U_diag + jnp.sum(masked_g2, axis=0)

        self._jit_forward = forward
        self._jit_jacobian = compute_jacobian
        self._jit_uncertainties = compute_uncertainties
        self._jit_update_U = update_U_diag

    def _get_params(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

    def _flatten_params(self, params):
        return jnp.concatenate([
            params['W1'].ravel(),
            params['b1'].ravel(),
            params['W2'].ravel(),
            params['b2'].ravel()
        ])

    def _forward(self, X):
        return self._jit_forward(self.W1, self.b1, self.W2, self.b2, X)

    def _compute_gradient_features(self, X):
        flat_params = self._flatten_params(self._get_params())
        return self._jit_jacobian(flat_params, X)

    def _compute_uncertainties(self, g_features):
        return self._jit_uncertainties(g_features, self.U_diag, self.lambda_reg, self.nu, self.total_params)

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1
        X = jnp.asarray(X)

        predictions = self._forward(X)
        g_features = self._compute_gradient_features(X)
        uncertainties = self._compute_uncertainties(g_features)

        self._cached_g_features = g_features
        self._cached_X = X

        if self.minimize:
            lcb_values = predictions - uncertainties
            return self.solver(np.asarray(lcb_values))
        else:
            ucb_values = predictions + uncertainties
            return self.solver(np.asarray(ucb_values))

    def _update_information_matrix(self, g_features, mask):
        self.U_diag = self._jit_update_U(self.U_diag, g_features, mask, self.total_params)

    def _train_batch(self, batch_size=100):
        if len(self.history_X) == 0:
            return

        all_X = jnp.asarray(self.history_X)
        all_Y = jnp.asarray(self.history_Y)
        all_mask = jnp.asarray(self.history_mask)
        n_samples = len(all_X)

        params_init = {
            'W1': self.W1_init, 'b1': self.b1_init,
            'W2': self.W2_init, 'b2': self.b2_init
        }

        for epoch in range(self.train_epochs):
            if n_samples <= batch_size:
                batch_idx = np.arange(n_samples)
            else:
                batch_idx = self.rng.choice(n_samples, size=batch_size, replace=False)

            X_batch = all_X[batch_idx]
            Y_batch = all_Y[batch_idx]
            mask_batch = all_mask[batch_idx]

            def loss_fn(params):
                z1 = X_batch @ params['W1'] + params['b1']
                h = jnp.maximum(0, z1)
                pred = h @ params['W2'] + params['b2']

                diff = (pred - Y_batch) * mask_batch
                n_obs = jnp.maximum(jnp.sum(mask_batch), 1.0)
                mse_loss = 0.5 * jnp.sum(diff ** 2) / n_obs

                reg_loss = 0.0
                for key in params:
                    reg_loss = reg_loss + jnp.mean((params[key] - params_init[key]) ** 2)
                reg_loss = 0.5 * self.lambda_reg * reg_loss

                return mse_loss + reg_loss

            params = self._get_params()
            grads = jax.grad(loss_fn)(params)

            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads.values()))
            clip_coef = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-6))

            self.W1 = self.W1 - self.lr * grads['W1'] * clip_coef
            self.b1 = self.b1 - self.lr * grads['b1'] * clip_coef
            self.W2 = self.W2 - self.lr * grads['W2'] * clip_coef
            self.b2 = self.b2 - self.lr * grads['b2'] * clip_coef

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        X_jax = jnp.asarray(X)

        if self._cached_g_features is not None and self._cached_X is not None:
            if jnp.allclose(X_jax, self._cached_X):
                g_features = self._cached_g_features
            else:
                g_features = self._compute_gradient_features(X_jax)
        else:
            g_features = self._compute_gradient_features(X_jax)

        self._cached_g_features = None
        self._cached_X = None

        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = jnp.asarray(selection) > 0.5

        self._update_information_matrix(g_features, mask)

        predictions = np.asarray(self._forward(X_jax))
        Y_target = np.where(mask, rewards, predictions)
        obs_mask = np.asarray(mask).astype(float)

        self.history_X.append(np.asarray(X))
        self.history_Y.append(Y_target)
        self.history_mask.append(obs_mask)

        if self.t % self.train_interval == 0:
            self._train_batch()


class UCBBayesianAgent:

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


UCBAgent = UCBBayesianAgent

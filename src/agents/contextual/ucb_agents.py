import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    compute_uncertainties as compute_bayesian_uncertainties,
    update_bayesian_stats,
)
from src.agents.agent_base import AgentBase
from src.solver import solve_knapsack


@jax.jit
def _neuralucb_forward_jit(W1, b1, W2, b2, X):
    z1 = X @ W1 + b1
    h = jnp.maximum(0, z1)
    return h @ W2 + b2


@partial(jax.jit, static_argnames=('context_dim', 'hidden_dim', 'n_arms'))
def _neuralucb_compute_jacobian_jit(flat_params, X, *, context_dim, hidden_dim, n_arms):
    w1_size = context_dim * hidden_dim
    b1_size = hidden_dim
    w2_size = hidden_dim * n_arms

    def forward_flat(fp):
        W1 = fp[:w1_size].reshape(context_dim, hidden_dim)
        b1 = fp[w1_size:w1_size + b1_size]
        W2 = fp[w1_size + b1_size:w1_size + b1_size + w2_size].reshape(
            hidden_dim, n_arms
        )
        b2 = fp[w1_size + b1_size + w2_size:]
        z1 = X @ W1 + b1
        h = jnp.maximum(0, z1)
        return h @ W2 + b2

    return jax.jacobian(forward_flat)(flat_params)


@jax.jit
def _neuralucb_compute_uncertainties_jit(g_features, U_diag, nu, m):
    g_scaled = g_features / jnp.sqrt(m)
    sigma2 = g_scaled ** 2 / U_diag
    return nu * jnp.sqrt(jnp.sum(sigma2, axis=1))


@jax.jit
def _neuralucb_update_U_diag_jit(U_diag, g_features, mask, m):
    g_scaled = g_features / jnp.sqrt(m)
    masked_g2 = jnp.where(mask[:, None], g_scaled ** 2, 0.0)
    return U_diag + jnp.sum(masked_g2, axis=0)


@jax.jit
def _neuralucb_compute_grads_jit(params, params_init, X_batch, Y_batch, mask_batch, lambda_reg):
    def loss_fn(p):
        pred = _neuralucb_forward_jit(p['W1'], p['b1'], p['W2'], p['b2'], X_batch)
        diff = (pred - Y_batch) * mask_batch
        n_obs = jnp.maximum(jnp.sum(mask_batch), 1.0)
        mse_loss = 0.5 * jnp.sum(diff ** 2) / n_obs
        reg_loss = sum(jnp.mean((p[k] - params_init[k]) ** 2) for k in p)
        return mse_loss + 0.5 * lambda_reg * reg_loss

    return jax.grad(loss_fn)(params)


class LinUCBAgent(AgentBase):

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

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        features = jnp.asarray(X)
        return np.asarray(compute_bayesian_predictions(features, self.A_inv, self.b))

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        features = jnp.asarray(X)
        return np.asarray(compute_bayesian_uncertainties(features, self.A_inv))

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False, **kwargs):
        features = jnp.asarray(X)
        rewards_jax = jnp.asarray(rewards)

        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = jnp.asarray(selection) > 0.5

        self.A, self.b, self.A_inv = update_bayesian_stats(
            self.A, self.b, self.A_inv, features, rewards_jax, mask
        )


class NeuralUCBAgent(AgentBase):

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 100,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        lr: float = 0.001,
        minimize: bool = False,
        solver=None,
        seed: int = None,

        phased: bool = False,
        phase_length: int = 50,
        phase_epochs: int = 5,
        max_history_size: int = 512,
        batch_size: int = 32,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.minimize = minimize

        if solver is None:
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver
        self.nu = alpha
        self.lambda_reg = lambda_reg
        self.init_lr = lr
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
        self.nn_width = hidden_dim

        self.U_diag = lambda_reg * jnp.ones(self.total_params)

        self.max_history_size = max(int(max_history_size), 1)
        self.batch_size = max(int(batch_size), 1)
        self.history_X = []
        self.history_Y = []
        self.history_mask = []
        self.rng = np.random.default_rng(seed if seed else 0)

        self._cached_g_features = None
        self._cached_X = None


        schedule = lambda count: self.init_lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(5.0),
            optax.adam(schedule),
        )
        init_params = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
        }
        self.opt_state = self.optimizer.init(init_params)


        self.phased = phased
        self.phase_length = phase_length
        self.phase_epochs = phase_epochs
        self._rounds_since_retrain = 0

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
        return _neuralucb_forward_jit(self.W1, self.b1, self.W2, self.b2, X)

    def _compute_gradient_features(self, X):
        flat_params = self._flatten_params(self._get_params())
        return _neuralucb_compute_jacobian_jit(
            flat_params, X,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            n_arms=self.n_arms,
        )

    def _compute_uncertainties(self, g_features):
        return _neuralucb_compute_uncertainties_jit(
            g_features, self.U_diag, self.nu, self.nn_width
        )

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

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        X_jax = jnp.asarray(X)
        return np.asarray(self._forward(X_jax))

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        X_jax = jnp.asarray(X)

        if self._cached_g_features is not None and self._cached_X is not None:
            if jnp.allclose(X_jax, self._cached_X):
                g_features = self._cached_g_features
            else:
                g_features = self._compute_gradient_features(X_jax)
        else:
            g_features = self._compute_gradient_features(X_jax)

        return np.asarray(self._compute_uncertainties(g_features))

    def predict_mean_and_uncertainty(self, X):
        X_jax = jnp.asarray(X)
        pred = self._forward(X_jax)
        g_features = self._compute_gradient_features(X_jax)
        sigmas = self._compute_uncertainties(g_features)
        return np.asarray(pred), np.asarray(sigmas)

    def _update_information_matrix(self, g_features, mask):
        self.U_diag = _neuralucb_update_U_diag_jit(
            self.U_diag, g_features, mask, self.nn_width
        )

    def _train_step(self, batch_size=32):
        n_samples = len(self.history_X)
        if n_samples == 0:
            return

        if n_samples <= batch_size:
            batch_idx = np.arange(n_samples)
        else:
            batch_idx = self.rng.choice(n_samples, size=batch_size, replace=False)

        X_batch = jnp.stack([self.history_X[i] for i in batch_idx])
        Y_batch = jnp.stack([self.history_Y[i] for i in batch_idx])
        mask_batch = jnp.stack([self.history_mask[i] for i in batch_idx])

        params_init = {
            'W1': self.W1_init, 'b1': self.b1_init,
            'W2': self.W2_init, 'b2': self.b2_init
        }

        params = self._get_params()
        grads = _neuralucb_compute_grads_jit(
            params, params_init, X_batch, Y_batch, mask_batch, self.lambda_reg
        )
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        self.W1 = new_params['W1']
        self.b1 = new_params['b1']
        self.W2 = new_params['W2']
        self.b2 = new_params['b2']

    def _append_history(self, X, Y_target, mask):
        if len(self.history_X) >= self.max_history_size:
            self.history_X.pop(0)
            self.history_Y.pop(0)
            self.history_mask.pop(0)
        self.history_X.append(np.asarray(X))
        self.history_Y.append(np.asarray(Y_target))
        self.history_mask.append(np.asarray(mask))

    def _phased_retrain(self):
        n_samples = len(self.history_X)
        if n_samples == 0:
            return
        for _ in range(self.phase_epochs):
            # Keep the original replay volume (one sampled update per buffered example
            # per epoch), but use the JITted minibatch train step for speed.
            for _ in range(n_samples):
                self._train_step(batch_size=min(self.batch_size, n_samples))

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False, **kwargs):
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

        predictions = self._forward(X_jax)
        Y_target = jnp.where(mask, jnp.asarray(rewards), predictions)
        obs_mask = mask.astype(jnp.float32)

        self._append_history(X_jax, Y_target, obs_mask)

        if self.phased:

            self._rounds_since_retrain += 1
            if self._rounds_since_retrain >= self.phase_length:
                self._phased_retrain()
                self._rounds_since_retrain = 0
        else:
            self._train_step()


class UCBBayesianAgent(AgentBase):

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
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

        self.A, self.b, self.A_inv = init_per_arm_regression(
            n_arms, context_dim, lambda_reg
        )

    def act(self, X: np.ndarray) -> np.ndarray:
        features = jnp.asarray(X)
        ucb_values = compute_ucb_values(features, self.A_inv, self.b, self.beta, self.minimize)
        return self.solver(np.asarray(ucb_values))

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        features = jnp.asarray(X)
        return np.asarray(compute_bayesian_predictions(features, self.A_inv, self.b))

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        features = jnp.asarray(X)
        return np.asarray(compute_bayesian_uncertainties(features, self.A_inv))

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False, **kwargs):
        features = jnp.asarray(X)
        rewards_jax = jnp.asarray(rewards)
        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = jnp.asarray(selection) > 0.5
        self.A, self.b, self.A_inv = update_bayesian_stats(
            self.A, self.b, self.A_inv, features, rewards_jax, mask
        )


UCBAgent = UCBBayesianAgent

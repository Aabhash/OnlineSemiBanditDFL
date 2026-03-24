import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.models import BayesianLinearModel
from src.losses import make_spo_grad_fn
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    compute_uncertainties as compute_bayesian_uncertainties,
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


class LinearSPOAgent:
    """Pure linear model with SPO+ training and non-Bayesian exploration.

    Prediction: pred_i = theta_i^T x  (per-arm linear weights)
    Training: SPO+ gradient descent on theta

    Exploration (two modes, can be combined):
      - Prediction-variance UCB (alpha > 0): online running std of recent
        predictions as uncertainty bonus/penalty.
      - Epsilon-greedy (epsilon > 0): with probability epsilon, add random
        noise to predictions before solving.
    """

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        alpha: float = 0.0,
        alpha_start: float = None,
        alpha_end: float = None,
        epsilon: float = 0.0,
        epsilon_start: float = None,
        epsilon_end: float = None,
        decay_rounds: int = 300,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        noise_scale: float = 1.0,
        var_window: int = 50,
        mask_grad: bool = False,
        minimize: bool = False,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.lambda_reg = lambda_reg
        self.minimize = minimize
        self.mask_grad = mask_grad
        self.decay_rounds = decay_rounds
        self.lr = lr
        self.noise_scale = noise_scale
        self.var_window = var_window
        self.t = 0

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

        self.loss_grad_fn = make_spo_grad_fn(self.solver, minimize)

        # Alpha schedule (prediction-variance UCB)
        if alpha_start is not None and alpha_end is not None:
            self.alpha_start = alpha_start
            self.alpha_end = alpha_end
            self.alpha = alpha_start
        else:
            self.alpha_start = alpha
            self.alpha_end = alpha
            self.alpha = alpha

        # Epsilon schedule (epsilon-greedy)
        if epsilon_start is not None and epsilon_end is not None:
            self.epsilon_start = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon = epsilon_start
        else:
            self.epsilon_start = epsilon
            self.epsilon_end = epsilon
            self.epsilon = epsilon

        # Per-arm linear weights: theta_i in R^context_dim
        key = jax.random.PRNGKey(seed if seed else 0)
        self.theta = jax.random.normal(key, (n_arms, context_dim)) * 0.01
        self.rng = np.random.default_rng(seed if seed else 0)

        # Online prediction variance: ring buffer of recent predictions
        self._pred_buffer = np.zeros((var_window, n_arms))
        self._buf_idx = 0
        self._buf_count = 0

    def _update_schedules(self):
        """Decay alpha and epsilon linearly over decay_rounds."""
        if self.decay_rounds > 0 and self.t <= self.decay_rounds:
            frac = min(self.t / self.decay_rounds, 1.0)
            self.alpha = self.alpha_start + frac * (self.alpha_end - self.alpha_start)
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _forward(self, X):
        """Predict: pred_i = theta_i^T x."""
        return self.theta @ jnp.asarray(X)

    def _record_prediction(self, pred_np):
        """Store prediction in ring buffer."""
        self._pred_buffer[self._buf_idx] = pred_np
        self._buf_idx = (self._buf_idx + 1) % self.var_window
        self._buf_count = min(self._buf_count + 1, self.var_window)

    def _compute_uncertainties(self):
        """Per-arm std from recent prediction history."""
        if self._buf_count < 2:
            return np.ones(self.n_arms)
        buf = self._pred_buffer[:self._buf_count]
        return np.std(buf, axis=0)

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1
        self._update_schedules()

        predictions = self._forward(X)
        pred_np = np.asarray(predictions)

        values = pred_np.copy()

        # Prediction-variance UCB bonus
        if self.alpha != 0.0:
            stds = self._compute_uncertainties()
            if self.minimize:
                values = values - self.alpha * stds
            else:
                values = values + self.alpha * stds

        # Epsilon-greedy: with prob epsilon, add random noise
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            noise = self.rng.normal(0, self.noise_scale, size=self.n_arms)
            values = values + noise

        return self.solver(values)

    def _train_step(self, X, rewards, mask):
        """Single SPO+ gradient step on the current observation."""
        X_jax = jnp.asarray(X)
        pred = self._forward(X_jax)
        pred_np = np.asarray(pred)

        # Build full target: use predictions for unobserved arms
        Y_target = np.where(np.asarray(mask), np.asarray(rewards), pred_np)

        # Compute SPO+ gradient w.r.t. predictions
        spo_grad = np.asarray(self.loss_grad_fn(pred_np, Y_target))
        if self.mask_grad:
            spo_grad = spo_grad * np.asarray(mask)

        # Gradient of pred_i = theta_i^T x w.r.t. theta_i is just x
        spo_grad_jax = jnp.asarray(spo_grad)
        d_theta = spo_grad_jax[:, None] * X_jax[None, :]  # (n_arms, context_dim)

        # Add regularization: lambda * theta
        d_theta = d_theta + self.lambda_reg * self.theta

        self.theta = self.theta - self.lr * d_theta

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        if full_feedback:
            mask = np.ones(self.n_arms, dtype=bool)
        else:
            mask = np.asarray(selection) > 0.5

        # Record current prediction before training step (for variance tracking)
        pred_np = np.asarray(self._forward(X))
        self._record_prediction(pred_np)

        # SPO+ gradient step
        self._train_step(X, rewards, mask)


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
        lr: float = 0.001,
        minimize: bool = False,
        solver=None,
        seed: int = None,
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

        self.U_diag = lambda_reg * jnp.ones(self.total_params)

        self.history_X = []
        self.history_Y = []
        self.history_mask = []
        self.rng = np.random.default_rng(seed if seed else 0)

        self._cached_g_features = None
        self._cached_X = None

        # Inverse-sqrt LR schedule: lr / sqrt(t+1)
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

    def _train_step(self, batch_size=32):
        """Single gradient step on a mini-batch from history.

        Analogous to Bayesian agents' sufficient statistics update which
        implicitly uses all history — one gradient step per round, but on
        a batch sampled from accumulated data.
        """
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

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        self.W1 = new_params['W1']
        self.b1 = new_params['b1']
        self.W2 = new_params['W2']
        self.b2 = new_params['b2']

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

        predictions = self._forward(X_jax)
        Y_target = jnp.where(mask, jnp.asarray(rewards), predictions)
        obs_mask = mask.astype(jnp.float32)

        self.history_X.append(X_jax)
        self.history_Y.append(Y_target)
        self.history_mask.append(obs_mask)

        self._train_step()


class NeuralSPOAgent:
    """NeuralUCB-style agent with SPO+ training instead of MSE.

    Architecture and NTK uncertainty are identical to NeuralUCBAgent.
    Only the training objective differs: SPO+ gradient backpropagated
    through the network instead of MSE loss.
    """

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 100,
        alpha: float = 1.0,
        alpha_start: float = None,
        alpha_end: float = None,
        decay_rounds: int = 300,
        lambda_reg: float = 1.0,
        lr: float = 0.001,
        minimize: bool = False,
        mask_grad: bool = False,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.minimize = minimize
        self.mask_grad = mask_grad
        self.decay_rounds = decay_rounds

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

        self.loss_grad_fn = make_spo_grad_fn(self.solver, minimize)

        # Exploration schedule
        if alpha_start is not None and alpha_end is not None:
            self.alpha_start = alpha_start
            self.alpha_end = alpha_end
            self.nu = alpha_start
        else:
            self.alpha_start = alpha
            self.alpha_end = alpha
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

        self.U_diag = lambda_reg * jnp.ones(self.total_params)

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

    def _update_alpha(self):
        """Decay exploration parameter linearly over decay_rounds."""
        if self.decay_rounds > 0 and self.t <= self.decay_rounds:
            frac = min(self.t / self.decay_rounds, 1.0)
            self.nu = self.alpha_start + frac * (self.alpha_end - self.alpha_start)

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1
        self._update_alpha()
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

    def _train_step(self, batch_size=32):
        """Single SPO+ gradient step on a mini-batch from history."""
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

        # Compute SPO+ gradient for each sample, then backprop through network
        params = self._get_params()

        def loss_fn(params):
            z1 = X_batch @ params['W1'] + params['b1']
            h = jnp.maximum(0, z1)
            pred_batch = h @ params['W2'] + params['b2']  # (B, n_arms)
            return pred_batch  # return predictions for custom grad

        # Forward pass
        z1 = X_batch @ params['W1'] + params['b1']
        h = jnp.maximum(0, z1)
        pred_batch = h @ params['W2'] + params['b2']  # (B, n_arms)

        # Compute SPO+ grad for each sample (not JIT-able due to solver calls)
        spo_grads = []
        for i in range(len(batch_idx)):
            pred_i = np.asarray(pred_batch[i])
            y_i = np.asarray(Y_batch[i])
            g = self.loss_grad_fn(pred_i, y_i)
            if self.mask_grad:
                g = g * np.asarray(mask_batch[i])
            spo_grads.append(g)
        # (B, n_arms) — the "loss gradient" w.r.t. network output
        dl_dpred = jnp.stack(spo_grads) / len(batch_idx)

        # Backprop SPO+ gradient through network using VJP
        def fwd(params):
            z1 = X_batch @ params['W1'] + params['b1']
            h = jnp.maximum(0, z1)
            return h @ params['W2'] + params['b2']

        _, vjp_fn = jax.vjp(fwd, params)
        (grads,) = vjp_fn(dl_dpred)

        # Add regularization gradient
        for key in grads:
            grads[key] = grads[key] + self.lambda_reg * (params[key] - params_init[key])

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        self.W1 = new_params['W1']
        self.b1 = new_params['b1']
        self.W2 = new_params['W2']
        self.b2 = new_params['b2']

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

        predictions = self._forward(X_jax)
        Y_target = jnp.where(mask, jnp.asarray(rewards), predictions)
        obs_mask = mask.astype(jnp.float32)

        self.history_X.append(X_jax)
        self.history_Y.append(Y_target)
        self.history_mask.append(obs_mask)

        self._train_step()


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

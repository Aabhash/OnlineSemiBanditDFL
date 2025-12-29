import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.solver import solve_top_k

DEFAULT_TRAIN_INTERVAL = 1
DEFAULT_TRAIN_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_GRAD_CLIP = 5.0


def _flatten_params(params):
    flat = []
    for key in ['W1', 'b1', 'W2', 'b2']:
        flat.append(params[key].ravel())
    return jnp.concatenate(flat)


@jax.jit
def _forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h = jnp.maximum(0, z1)
    out = h @ W2 + b2
    return out.squeeze()


def _compute_all_gradients(params, X, n_arms, total_params):
    def forward_flat(flat_params):
        w1_size = params['W1'].size
        b1_size = params['b1'].size
        w2_size = params['W2'].size

        idx = 0
        W1 = flat_params[idx:idx + w1_size].reshape(params['W1'].shape)
        idx += w1_size
        b1 = flat_params[idx:idx + b1_size].reshape(params['b1'].shape)
        idx += b1_size
        W2 = flat_params[idx:idx + w2_size].reshape(params['W2'].shape)
        idx += w2_size
        b2 = flat_params[idx:].reshape(params['b2'].shape)

        z1 = X @ W1 + b1
        h = jnp.maximum(0, z1)
        out = h @ W2 + b2
        return out.squeeze()

    flat_params = _flatten_params(params)
    jacobian = jax.jacobian(forward_flat)(flat_params)
    return jacobian


class CNUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 64,
        lambda_reg: float = 1.0,
        gamma: float = 1.0,
        lr: float = 0.01,
        constraint_type: str = 'knapsack',
        K: int = None,
        solver=None,
        seed: int = None,
        train_interval: int = None,
        train_epochs: int = None,
        batch_size: int = None,
        per_arm_context: bool = True,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.hidden_dim = hidden_dim

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.lr = lr
        self.constraint_type = constraint_type
        self.K = K if K is not None else n_arms // 5
        self.t = 0
        self.per_arm_context = per_arm_context

        self.train_interval = train_interval if train_interval is not None else DEFAULT_TRAIN_INTERVAL
        self.train_epochs = train_epochs if train_epochs is not None else DEFAULT_TRAIN_EPOCHS
        self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE

        key = jax.random.PRNGKey(seed if seed else 0)
        k1, k2 = jax.random.split(key)

        self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) / np.sqrt(hidden_dim)
        self.b1 = jnp.zeros(hidden_dim)

        if per_arm_context:
            self.W2 = jax.random.normal(k2, (hidden_dim, 1)) / np.sqrt(hidden_dim)
            self.b2 = jnp.zeros(1)
            self.output_dim = 1
        else:
            self.W2 = jax.random.normal(k2, (hidden_dim, n_arms)) / np.sqrt(hidden_dim)
            self.b2 = jnp.zeros(n_arms)
            self.output_dim = n_arms

        self.W1_init = self.W1.copy()
        self.b1_init = self.b1.copy()
        self.W2_init = self.W2.copy()
        self.b2_init = self.b2.copy()

        self.param_shapes = {
            'W1': (context_dim, hidden_dim),
            'b1': (hidden_dim,),
            'W2': (hidden_dim, self.output_dim),
            'b2': (self.output_dim,),
        }
        self.total_params = sum(int(np.prod(s)) for s in self.param_shapes.values())

        self.Z_diag = lambda_reg * jnp.ones(self.total_params)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(DEFAULT_GRAD_CLIP),
            optax.adam(learning_rate=lr)
        )
        self.opt_state = self.optimizer.init({
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2
        })

        self.history_X = []
        self.history_Y = []
        self.history_mask = []

        self.rng = np.random.default_rng(seed if seed else 0)

        self.output_scale = 1.0
        self.output_bias = 0.0
        self.target_mean = 0.0
        self.target_std = 1.0
        self.target_count = 0

    def _get_params(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

    def _compute_gradient_features(self, X):
        params = self._get_params()
        return _compute_all_gradients(params, X, self.n_arms, self.total_params)

    def _compute_uncertainties(self, g_features):
        g_scaled = g_features / np.sqrt(self.total_params)
        uncertainties = jnp.sqrt(jnp.sum(g_scaled ** 2 / self.Z_diag, axis=1))
        return uncertainties

    def _raw_predict(self, X):
        X = jnp.asarray(X)
        return _forward(X, self.W1, self.b1, self.W2, self.b2)

    def predict(self, X):
        raw = self._raw_predict(X)
        return raw * self.output_scale + self.output_bias

    def act(self, X):
        self.t += 1
        X = jnp.asarray(X)

        predictions = self.predict(X)

        g_features = self._compute_gradient_features(X)
        uncertainties = self._compute_uncertainties(g_features)
        scaled_uncertainties = uncertainties * self.output_scale

        ucb_values = predictions + self.gamma * scaled_uncertainties

        if self.constraint_type == 'top_k':
            return solve_top_k(np.asarray(ucb_values), self.K)
        else:
            return self.solver(np.asarray(ucb_values))

    def _update_gram_matrix(self, g_features, mask):
        g_scaled = g_features / np.sqrt(self.total_params)
        for i in range(self.n_arms):
            if mask[i]:
                g = g_scaled[i]
                self.Z_diag = self.Z_diag + g ** 2

    def _get_initial_params(self):
        return {
            'W1': self.W1_init, 'b1': self.b1_init,
            'W2': self.W2_init, 'b2': self.b2_init
        }

    def _train_batch(self):
        if len(self.history_X) == 0:
            return

        all_X = np.array(self.history_X)
        all_Y = np.array(self.history_Y)
        all_mask = np.array(self.history_mask)
        n_samples = len(all_X)

        params_init = self._get_initial_params()

        for epoch in range(self.train_epochs):
            if n_samples <= self.batch_size:
                batch_idx = np.arange(n_samples)
            else:
                batch_idx = self.rng.choice(n_samples, size=self.batch_size, replace=False)

            X_batch = jnp.asarray(all_X[batch_idx])
            Y_batch = jnp.asarray(all_Y[batch_idx])
            mask_batch = jnp.asarray(all_mask[batch_idx])

            def loss_fn(params):
                z1 = X_batch @ params['W1'] + params['b1']
                h = jnp.maximum(0, z1)
                pred = h @ params['W2'] + params['b2']
                pred = pred.squeeze()

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

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X_np = np.asarray(X)
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        obs_mask = jnp.ones(self.n_arms, dtype=bool) if full_feedback else (selection > 0.5)
        self._update_output_scaling(rewards, obs_mask)

        selection_mask = selection > 0.5
        g_features = self._compute_gradient_features(X)
        self._update_gram_matrix(g_features, selection_mask)

        raw_predictions = np.asarray(self._raw_predict(X))

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


class CNTSAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 64,
        lambda_reg: float = 1.0,
        nu: float = 1.0,
        M: int = 10,
        lr: float = 0.01,
        constraint_type: str = 'knapsack',
        K: int = None,
        solver=None,
        seed: int = None,
        train_interval: int = None,
        train_epochs: int = None,
        batch_size: int = None,
        per_arm_context: bool = True,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.hidden_dim = hidden_dim

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.lambda_reg = lambda_reg
        self.nu = nu
        self.M = M
        self.lr = lr
        self.constraint_type = constraint_type
        self.K = K if K is not None else n_arms // 5
        self.t = 0
        self.per_arm_context = per_arm_context

        self.train_interval = train_interval if train_interval is not None else DEFAULT_TRAIN_INTERVAL
        self.train_epochs = train_epochs if train_epochs is not None else DEFAULT_TRAIN_EPOCHS
        self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE

        self.rng = np.random.default_rng(seed if seed else 0)

        key = jax.random.PRNGKey(seed if seed else 0)
        k1, k2 = jax.random.split(key)

        self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) / np.sqrt(hidden_dim)
        self.b1 = jnp.zeros(hidden_dim)

        if per_arm_context:
            self.W2 = jax.random.normal(k2, (hidden_dim, 1)) / np.sqrt(hidden_dim)
            self.b2 = jnp.zeros(1)
            self.output_dim = 1
        else:
            self.W2 = jax.random.normal(k2, (hidden_dim, n_arms)) / np.sqrt(hidden_dim)
            self.b2 = jnp.zeros(n_arms)
            self.output_dim = n_arms

        self.W1_init = self.W1.copy()
        self.b1_init = self.b1.copy()
        self.W2_init = self.W2.copy()
        self.b2_init = self.b2.copy()

        self.param_shapes = {
            'W1': (context_dim, hidden_dim),
            'b1': (hidden_dim,),
            'W2': (hidden_dim, self.output_dim),
            'b2': (self.output_dim,),
        }
        self.total_params = sum(int(np.prod(s)) for s in self.param_shapes.values())

        self.Z_diag = lambda_reg * jnp.ones(self.total_params)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(DEFAULT_GRAD_CLIP),
            optax.adam(learning_rate=lr)
        )
        self.opt_state = self.optimizer.init({
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2
        })

        self.history_X = []
        self.history_Y = []
        self.history_mask = []

        self.output_scale = 1.0
        self.output_bias = 0.0
        self.target_mean = 0.0
        self.target_std = 1.0
        self.target_count = 0

    def _get_params(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

    def _compute_gradient_features(self, X):
        params = self._get_params()
        return _compute_all_gradients(params, X, self.n_arms, self.total_params)

    def _compute_uncertainties(self, g_features):
        g_scaled = g_features / np.sqrt(self.total_params)
        uncertainties = jnp.sqrt(jnp.sum(g_scaled ** 2 / self.Z_diag, axis=1))
        return uncertainties

    def _raw_predict(self, X):
        X = jnp.asarray(X)
        return _forward(X, self.W1, self.b1, self.W2, self.b2)

    def predict(self, X):
        raw = self._raw_predict(X)
        return raw * self.output_scale + self.output_bias

    def act(self, X):
        self.t += 1
        X = jnp.asarray(X)

        predictions = np.asarray(self.predict(X))
        g_features = self._compute_gradient_features(X)
        uncertainties = np.asarray(self._compute_uncertainties(g_features))
        scaled_uncertainties = uncertainties * self.output_scale

        optimistic_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            samples = self.rng.normal(
                loc=predictions[i],
                scale=self.nu * scaled_uncertainties[i],
                size=self.M
            )
            optimistic_values[i] = np.max(samples)

        if self.constraint_type == 'top_k':
            return solve_top_k(optimistic_values, self.K)
        else:
            return self.solver(optimistic_values)

    def _update_gram_matrix(self, g_features, mask):
        g_scaled = g_features / np.sqrt(self.total_params)
        for i in range(self.n_arms):
            if mask[i]:
                g = g_scaled[i]
                self.Z_diag = self.Z_diag + g ** 2

    def _get_initial_params(self):
        return {
            'W1': self.W1_init, 'b1': self.b1_init,
            'W2': self.W2_init, 'b2': self.b2_init
        }

    def _train_batch(self):
        if len(self.history_X) == 0:
            return

        all_X = np.array(self.history_X)
        all_Y = np.array(self.history_Y)
        all_mask = np.array(self.history_mask)
        n_samples = len(all_X)

        params_init = self._get_initial_params()

        for epoch in range(self.train_epochs):
            if n_samples <= self.batch_size:
                batch_idx = np.arange(n_samples)
            else:
                batch_idx = self.rng.choice(n_samples, size=self.batch_size, replace=False)

            X_batch = jnp.asarray(all_X[batch_idx])
            Y_batch = jnp.asarray(all_Y[batch_idx])
            mask_batch = jnp.asarray(all_mask[batch_idx])

            def loss_fn(params):
                z1 = X_batch @ params['W1'] + params['b1']
                h = jnp.maximum(0, z1)
                pred = h @ params['W2'] + params['b2']
                pred = pred.squeeze()

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

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X_np = np.asarray(X)
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        obs_mask = jnp.ones(self.n_arms, dtype=bool) if full_feedback else (selection > 0.5)
        self._update_output_scaling(rewards, obs_mask)

        selection_mask = selection > 0.5
        g_features = self._compute_gradient_features(X)
        self._update_gram_matrix(g_features, selection_mask)

        raw_predictions = np.asarray(self._raw_predict(X))

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

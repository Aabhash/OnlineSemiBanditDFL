import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from src.solver import solve_knapsack, solve_top_k

TRAIN_INTERVAL = 10
TRAIN_EPOCHS = 100
BATCH_SIZE = 100
GRAD_CLIP = 1.0


def _flatten_params(params):
    flat = []
    for key in ['W1', 'b1', 'W2', 'b2']:
        flat.append(params[key].ravel())
    return jnp.concatenate(flat)


def _unflatten_params(flat, shapes):
    params = {}
    idx = 0
    for key, shape in shapes.items():
        size = int(np.prod(shape))
        params[key] = flat[idx:idx + size].reshape(shape)
        idx += size
    return params


@jax.jit
def _forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h = jnp.maximum(0, z1)
    out = h @ W2 + b2
    return out.squeeze()


def _forward_single_arm(params, X, arm_idx, n_arms, hidden_dim):
    z1 = X @ params['W1'] + params['b1']
    h = jnp.maximum(0, z1)
    out = h @ params['W2'] + params['b2']
    return out[arm_idx]


def _forward_params(params, X):
    z1 = X @ params['W1'] + params['b1']
    h = jnp.maximum(0, z1)
    out = h @ params['W2'] + params['b2']
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
        hidden_dim: int = 100,
        lambda_reg: float = 1.0,
        gamma: float = 1.0,
        lr: float = 0.01,
        constraint_type: str = 'knapsack',
        K: int = None,
        seed: int = None,
        train_interval: int = TRAIN_INTERVAL,
        train_epochs: int = TRAIN_EPOCHS,
        batch_size: int = BATCH_SIZE,
        per_arm_context: bool = True,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.lr = lr
        self.constraint_type = constraint_type
        self.K = K if K is not None else n_arms // 5
        self.t = 0
        self.per_arm_context = per_arm_context

        self.train_interval = train_interval
        self.train_epochs = train_epochs
        self.batch_size = batch_size

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
            optax.clip_by_global_norm(GRAD_CLIP),
            optax.sgd(learning_rate=lr)
        )
        self.opt_state = self.optimizer.init({
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2
        })

        self._grad_fn = jax.grad(_forward_single_arm, argnums=0)

        self.history_X = []
        self.history_Y = []
        self.history_mask = []

        self.rng = np.random.default_rng(seed if seed else 0)

    def _get_params(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

    def _compute_gradient_features(self, X):
        params = self._get_params()
        return _compute_all_gradients(params, X, self.n_arms, self.total_params)

    def _compute_uncertainties(self, g_features):
        g_scaled = g_features / np.sqrt(self.hidden_dim)
        uncertainties = jnp.sqrt(jnp.sum(g_scaled ** 2 / self.Z_diag, axis=1))

        return uncertainties

    def predict(self, X):
        X = jnp.asarray(X)
        return _forward(X, self.W1, self.b1, self.W2, self.b2)

    def act(self, X):
        self.t += 1
        X = jnp.asarray(X)

        predictions = self.predict(X)

        g_features = self._compute_gradient_features(X)
        uncertainties = self._compute_uncertainties(g_features)

        ucb_values = predictions + self.gamma * uncertainties

        if self.constraint_type == 'top_k':
            return solve_top_k(np.asarray(ucb_values), self.K)
        else:
            return solve_knapsack(np.asarray(ucb_values), self.weights_np, self.capacity)

    def _update_gram_matrix(self, g_features, mask):
        g_scaled = g_features / np.sqrt(self.hidden_dim)
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

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X_np = np.asarray(X)
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        g_features = self._compute_gradient_features(X)

        mask = jnp.ones(self.n_arms, dtype=bool) if full_feedback else (selection > 0.5)
        self._update_gram_matrix(g_features, mask)

        predictions = self.predict(X)
        if full_feedback:
            Y_target = np.asarray(rewards)
            obs_mask = np.ones(self.n_arms)
        else:
            Y_target = np.asarray(jnp.where(selection > 0.5, rewards, predictions))
            obs_mask = np.asarray(selection > 0.5).astype(float)

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
        hidden_dim: int = 100,
        lambda_reg: float = 1.0,
        nu: float = 1.0,
        M: int = 10,
        lr: float = 0.01,
        constraint_type: str = 'knapsack',
        K: int = None,
        seed: int = None,
        train_interval: int = TRAIN_INTERVAL,
        train_epochs: int = TRAIN_EPOCHS,
        batch_size: int = BATCH_SIZE,
        per_arm_context: bool = True,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.nu = nu
        self.M = M
        self.lr = lr
        self.constraint_type = constraint_type
        self.K = K if K is not None else n_arms // 5
        self.t = 0
        self.per_arm_context = per_arm_context

        self.train_interval = train_interval
        self.train_epochs = train_epochs
        self.batch_size = batch_size

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
            optax.clip_by_global_norm(GRAD_CLIP),
            optax.sgd(learning_rate=lr)
        )
        self.opt_state = self.optimizer.init({
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2
        })

        self._grad_fn = jax.grad(_forward_single_arm, argnums=0)

        self.history_X = []
        self.history_Y = []
        self.history_mask = []

    def _get_params(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

    def _compute_gradient_features(self, X):
        params = self._get_params()
        return _compute_all_gradients(params, X, self.n_arms, self.total_params)

    def _compute_uncertainties(self, g_features):
        g_scaled = g_features / np.sqrt(self.hidden_dim)
        uncertainties = jnp.sqrt(jnp.sum(g_scaled ** 2 / self.Z_diag, axis=1))
        return uncertainties

    def predict(self, X):
        X = jnp.asarray(X)
        return _forward(X, self.W1, self.b1, self.W2, self.b2)

    def act(self, X):
        self.t += 1
        X = jnp.asarray(X)

        predictions = np.asarray(self.predict(X))
        g_features = self._compute_gradient_features(X)
        uncertainties = np.asarray(self._compute_uncertainties(g_features))

        optimistic_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            samples = self.rng.normal(
                loc=predictions[i],
                scale=self.nu * uncertainties[i],
                size=self.M
            )
            optimistic_values[i] = np.max(samples)

        if self.constraint_type == 'top_k':
            return solve_top_k(optimistic_values, self.K)
        else:
            return solve_knapsack(optimistic_values, self.weights_np, self.capacity)

    def _update_gram_matrix(self, g_features, mask):
        g_scaled = g_features / np.sqrt(self.hidden_dim)
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

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X_np = np.asarray(X)
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        g_features = self._compute_gradient_features(X)
        mask = jnp.ones(self.n_arms, dtype=bool) if full_feedback else (selection > 0.5)
        self._update_gram_matrix(g_features, mask)

        predictions = self.predict(X)
        if full_feedback:
            Y_target = np.asarray(rewards)
            obs_mask = np.ones(self.n_arms)
        else:
            Y_target = np.asarray(jnp.where(selection > 0.5, rewards, predictions))
            obs_mask = np.asarray(selection > 0.5).astype(float)

        self.history_X.append(X_np)
        self.history_Y.append(Y_target)
        self.history_mask.append(obs_mask)

        if self.t % self.train_interval == 0:
            self._train_batch()

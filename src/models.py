import jax
import jax.numpy as jnp
import optax
from typing import Optional, Dict, Any


class LinearModel:

    def __init__(self, context_dim: int, n_arms: int, seed: int = 0):
        self.context_dim = context_dim
        self.n_arms = n_arms
        key = jax.random.PRNGKey(seed)
        self.theta = jax.random.normal(key, (n_arms, context_dim)) * 0.1

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.theta @ X

    def gradient_mse(self, X: jnp.ndarray, Y_true: jnp.ndarray) -> jnp.ndarray:
        Y_pred = self.predict(X)
        error = Y_pred - Y_true
        grad = 2 * jnp.outer(error, X) / self.n_arms
        return grad

    def gradient_spo(self, X: jnp.ndarray, grad_out: jnp.ndarray) -> jnp.ndarray:
        return jnp.outer(jnp.asarray(grad_out), jnp.asarray(X))

    def update(self, grad: jnp.ndarray, lr: float):
        grad = jnp.clip(grad, -1, 1)
        self.theta = self.theta - lr * grad


class MLPModel:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        hidden_dim: int = 64,
        lr: float = 0.01,
        decay_steps: Optional[int] = None,
        min_lr_ratio: float = 0.1,
        max_grad_norm: float = 5.0,
        seed: int = 0,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.hidden_dim = hidden_dim

        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        self.params = {
            'W1': jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1,
            'b1': jnp.zeros(hidden_dim),
            'W2': jax.random.normal(k2, (hidden_dim, n_arms)) * 0.1,
            'b2': jnp.zeros(n_arms),
        }
        self._cache = {}

        if decay_steps is not None and decay_steps > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_steps, alpha=min_lr_ratio)
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=schedule)
            )
        else:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr)
            )

        self.opt_state = self.optimizer.init(self.params)

    @property
    def W1(self):
        return self.params['W1']

    @W1.setter
    def W1(self, value):
        self.params['W1'] = value

    @property
    def b1(self):
        return self.params['b1']

    @b1.setter
    def b1(self, value):
        self.params['b1'] = value

    @property
    def W2(self):
        return self.params['W2']

    @W2.setter
    def W2(self, value):
        self.params['W2'] = value

    @property
    def b2(self):
        return self.params['b2']

    @b2.setter
    def b2(self, value):
        self.params['b2'] = value

    def predict(self, X) -> jnp.ndarray:
        X = jnp.asarray(X)
        z1 = X @ self.params['W1'] + self.params['b1']
        h1 = jnp.maximum(0, z1)
        out = h1 @ self.params['W2'] + self.params['b2']
        return out

    def forward_with_cache(self, X) -> jnp.ndarray:
        X = jnp.asarray(X)
        z1 = X @ self.params['W1'] + self.params['b1']
        h1 = jnp.maximum(0, z1)
        out = h1 @ self.params['W2'] + self.params['b2']
        self._cache = {'X': X, 'z1': z1, 'h1': h1}
        return out

    def backward(self, grad_out) -> Dict[str, jnp.ndarray]:
        grad_out = jnp.asarray(grad_out)
        X = self._cache['X']
        z1 = self._cache['z1']
        h1 = self._cache['h1']

        grad_W2 = jnp.outer(h1, grad_out)
        grad_b2 = grad_out
        grad_h1 = self.params['W2'] @ grad_out
        grad_z1 = grad_h1 * (z1 > 0)
        grad_W1 = jnp.outer(X, grad_z1)
        grad_b1 = grad_z1

        return {
            'W1': grad_W1, 'b1': grad_b1,
            'W2': grad_W2, 'b2': grad_b2,
        }

    def gradient_mse(self, X, Y_true) -> Dict[str, jnp.ndarray]:
        Y_pred = self.forward_with_cache(X)
        grad_out = 2 * (Y_pred - jnp.asarray(Y_true)) / self.n_arms
        return self.backward(grad_out)

    def gradient_spo(self, X, grad_out) -> Dict[str, jnp.ndarray]:
        self.forward_with_cache(X)
        return self.backward(grad_out)

    def update(self, grads: Dict[str, jnp.ndarray], lr: float = None):
        grads = {k: jnp.asarray(v) for k, v in grads.items()}

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)

        self.params['W1'] = jnp.clip(self.params['W1'], -10, 10)
        self.params['W2'] = jnp.clip(self.params['W2'], -10, 10)

    def copy(self) -> "MLPModel":
        new = MLPModel(self.context_dim, self.n_arms, self.hidden_dim)
        new.params = {k: v.copy() for k, v in self.params.items()}
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

        eye = jnp.eye(context_dim)
        self.mu = jnp.zeros((n_arms, context_dim))
        self.Sigma = jnp.broadcast_to(eye * prior_var, (n_arms, context_dim, context_dim)).copy()
        self.Lambda = jnp.broadcast_to(eye / prior_var, (n_arms, context_dim, context_dim)).copy()
        self.eta = jnp.zeros((n_arms, context_dim))

    def predict_mean(self, X) -> jnp.ndarray:
        X = jnp.asarray(X)
        return self.mu @ X

    def predict_sample(self, X, key=None) -> jnp.ndarray:
        X = jnp.asarray(X)
        if key is None:
            key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, (self.n_arms, self.context_dim))
        L = jnp.linalg.cholesky(self.Sigma)
        theta_samples = self.mu + jnp.einsum('nij,nj->ni', L, noise)
        return theta_samples @ X

    def predict_ucb(self, X, beta: float = 1.0, minimize: bool = False) -> jnp.ndarray:
        X = jnp.asarray(X)
        means = self.mu @ X
        Sx = jnp.einsum('nij,j->ni', self.Sigma, X)
        variances = jnp.einsum('i,ni->n', X, Sx)
        stds = jnp.sqrt(jnp.maximum(variances, 0.0))
        if minimize:
            return means - beta * stds
        return means + beta * stds

    def update(self, X, arm: int, reward: float):
        X = jnp.asarray(X)
        outer = jnp.outer(X, X) / self.noise_var
        self.Lambda = self.Lambda.at[arm].set(self.Lambda[arm] + outer)
        self.eta = self.eta.at[arm].set(self.eta[arm] + X * reward / self.noise_var)
        self.Sigma = self.Sigma.at[arm].set(jnp.linalg.inv(self.Lambda[arm]))
        self.mu = self.mu.at[arm].set(self.Sigma[arm] @ self.eta[arm])

    def update_batch(self, X, selection, rewards, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        if full_feedback:
            mask = jnp.ones(self.n_arms, dtype=bool)
        else:
            mask = selection > 0.5

        outer = jnp.outer(X, X) / self.noise_var
        mask_3d = mask[:, None, None]
        mask_2d = mask[:, None]

        self.Lambda = jnp.where(mask_3d, self.Lambda + outer, self.Lambda)
        self.eta = jnp.where(mask_2d, self.eta + X * rewards[:, None] / self.noise_var, self.eta)

        for arm in range(self.n_arms):
            if mask[arm]:
                self.Sigma = self.Sigma.at[arm].set(jnp.linalg.inv(self.Lambda[arm]))
                self.mu = self.mu.at[arm].set(self.Sigma[arm] @ self.eta[arm])

    def get_uncertainty(self, X) -> jnp.ndarray:
        X = jnp.asarray(X)
        Sx = jnp.einsum('nij,j->ni', self.Sigma, X)
        variances = jnp.einsum('i,ni->n', X, Sx)
        return jnp.sqrt(jnp.maximum(variances, 0.0))


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
        lr = kwargs.get('lr', 0.01)
        decay_steps = kwargs.get('decay_steps', None)
        return MLPModel(context_dim, n_arms, hidden_dim, lr=lr, decay_steps=decay_steps)
    elif model_type == "bayesian":
        prior_var = kwargs.get('prior_var', 1.0)
        noise_var = kwargs.get('noise_var', 0.1)
        return BayesianLinearModel(context_dim, n_arms, prior_var, noise_var)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

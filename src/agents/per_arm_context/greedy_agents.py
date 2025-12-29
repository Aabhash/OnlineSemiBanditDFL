import jax
import jax.numpy as jnp
import numpy as np
import optax


class PerArmEpsilonGreedyAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        epsilon: float = 0.1,
        decay_epsilon: bool = True,
        decay_rounds: int = 300,
        lr: float = 0.01,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        solver=None,
        seed: int = None,
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

        self.epsilon = epsilon
        self.decay_epsilon = decay_epsilon
        self.decay_rounds = decay_rounds
        self.t = 0

        self.rng = np.random.default_rng(seed if seed else 0)

        key = jax.random.PRNGKey(seed if seed else 0)
        k1, k2 = jax.random.split(key)
        self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)
        self.W2 = jax.random.normal(k2, (hidden_dim, 1)) * 0.1
        self.b2 = jnp.zeros(1)

        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=schedule)
            )
        else:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr)
            )
        self.opt_state = self.optimizer.init({
            'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2
        })

    def _forward(self, X):
        X = jnp.asarray(X)
        Z1 = X @ self.W1 + self.b1
        H = jnp.maximum(0, Z1)
        out = (H @ self.W2 + self.b2).squeeze(-1)
        return out, Z1, H

    def _random_knapsack(self):
        order = self.rng.permutation(self.n_arms)
        selection = np.zeros(self.n_arms)
        remaining = self.capacity

        for i in order:
            if self.weights_np[i] <= remaining:
                selection[i] = 1
                remaining -= self.weights_np[i]

        return selection

    def act(self, X):
        self.t += 1

        if self.decay_epsilon:
            current_eps = self.epsilon * max(0, 1 - self.t / self.decay_rounds)
        else:
            current_eps = self.epsilon

        if self.rng.random() < current_eps:
            return self._random_knapsack()
        else:
            pred, _, _ = self._forward(X)
            return self.solver(np.asarray(pred))

    def update(self, X, selection, rewards, lr=None, full_feedback=False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        pred, Z1, H = self._forward(X)

        if full_feedback:
            mask = jnp.ones(self.n_arms)
            Y_target = rewards
        else:
            mask = (selection > 0.5).astype(float)
            Y_target = jnp.where(selection > 0.5, rewards, pred)

        error = (pred - Y_target) * mask
        n_obs = jnp.maximum(jnp.sum(mask), 1.0)
        grad_out = 2 * error / n_obs

        grad_W2 = H.T @ grad_out[:, None]
        grad_b2 = jnp.sum(grad_out, keepdims=True)

        grad_H = jnp.outer(grad_out, self.W2.squeeze())
        grad_Z1 = grad_H * (Z1 > 0)

        grad_W1 = X.T @ grad_Z1
        grad_b1 = jnp.sum(grad_Z1, axis=0)

        grads = {'W1': grad_W1, 'b1': grad_b1, 'W2': grad_W2, 'b2': grad_b2}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state,
            {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        )
        params = optax.apply_updates(
            {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}, updates
        )
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']
        self.W2 = jnp.clip(params['W2'], -10, 10)
        self.b2 = params['b2']

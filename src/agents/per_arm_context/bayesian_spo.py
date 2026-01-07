import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.losses import spo_plus_grad
from src.agents.bayesian_utils import (
    init_shared_regression,
    compute_shared_ucb_values,
    compute_shared_predictions,
    compute_shared_uncertainties,
    update_shared_bayesian_stats,
)


class PerArmBayesianSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        beta_start: float = None,
        beta_end: float = None,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        decay_rounds: int = 300,
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

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver

        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        key = jax.random.PRNGKey(seed if seed else 0)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)

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
        self.opt_state = self.optimizer.init({'W1': self.W1, 'b1': self.b1})

        self.A, self.b_bayes, self.A_inv = init_shared_regression(hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        Z1 = X @ self.W1 + self.b1
        H = jnp.maximum(0, Z1)
        return H, Z1

    def predict_mean(self, X):
        H, _ = self._features(X)
        return compute_shared_predictions(H, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        H, _ = self._features(X)
        ucb_values = compute_shared_ucb_values(H, self.A_inv, self.b_bayes, self.beta)
        return self.solver(np.asarray(ucb_values))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        H, Z1 = self._features(X)

        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_shared_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, H, rewards, selection_mask
        )

        pred = compute_shared_predictions(H, self.A_inv, self.b_bayes)

        if full_feedback:
            Y_target = rewards
        else:
            ucb_values = compute_shared_ucb_values(H, self.A_inv, self.b_bayes, self.beta)
            Y_target = jnp.where(selection > 0.5, rewards, ucb_values)

        spo_grad_val = spo_plus_grad(pred, Y_target, self.weights, self.capacity)

        theta = self.A_inv @ self.b_bayes
        grad_H = jnp.outer(spo_grad_val, theta)
        grad_Z1 = grad_H * (Z1 > 0)
        grad_W1 = X.T @ grad_Z1
        grad_b1 = jnp.sum(grad_Z1, axis=0)

        grads = {'W1': grad_W1, 'b1': grad_b1}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Union
from src.models import LinearModel, MLPModel
from src.losses import spo_plus_grad
from src.agents.utils.bayesian import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    update_bayesian_stats,
)


class RandomAgent:

    def __init__(
        self,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        seed: int = None,
    ):
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

        self.reward_sum = jnp.zeros(n_arms)
        self.reward_count = jnp.zeros(n_arms)

    def act(self, X: np.ndarray) -> np.ndarray:
        order = self.rng.permutation(self.n_arms)
        selection = np.zeros(self.n_arms)
        remaining = self.capacity

        for i in order:
            if self.weights[i] <= remaining:
                selection[i] = 1
                remaining -= self.weights[i]

        return selection

    def predict(self, X: np.ndarray) -> jnp.ndarray:
        pred = jnp.zeros(self.n_arms)
        mask = self.reward_count > 0
        pred = jnp.where(mask, self.reward_sum / jnp.maximum(self.reward_count, 1), pred)
        return pred

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        rewards = jnp.asarray(rewards)
        selection = jnp.asarray(selection)

        if full_feedback:
            self.reward_sum = self.reward_sum + rewards
            self.reward_count = self.reward_count + 1
        else:
            selected = selection > 0.5
            self.reward_sum = jnp.where(selected, self.reward_sum + rewards, self.reward_sum)
            self.reward_count = jnp.where(selected, self.reward_count + 1, self.reward_count)


class GreedyAgent:

    def __init__(
        self,
        model: Union[LinearModel, MLPModel],
        weights: np.ndarray,
        capacity: float,
        loss_type: str = "mse",
        solver=None,
    ):
        self.model = model
        self.weights = weights
        self.capacity = capacity
        self.loss_type = loss_type
        self.n_arms = len(weights)

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

    def act(self, X: np.ndarray) -> np.ndarray:
        Y_pred = self.model.predict(X)
        return self.solver(np.asarray(Y_pred))

    def update(self, X: np.ndarray, Y_true: np.ndarray, lr: float):
        if self.loss_type == "mse":
            grad = self.model.gradient_mse(X, Y_true)
        elif self.loss_type == "spo":
            Y_pred = self.model.predict(X)
            spo_grad = spo_plus_grad(Y_pred, Y_true, self.weights, self.capacity)
            grad = self.model.gradient_spo(X, spo_grad)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        self.model.update(grad, lr)


class EpsilonGreedyAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
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
        self.W2 = jax.random.normal(k2, (hidden_dim, n_arms)) * 0.1
        self.b2 = jnp.zeros(n_arms)

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
        z1 = X @ self.W1 + self.b1
        h = jnp.maximum(0, z1)
        return h @ self.W2 + self.b2, z1, h

    def _random_knapsack(self):
        order = self.rng.permutation(self.n_arms)
        selection = np.zeros(self.n_arms)
        remaining = self.capacity

        for i in order:
            if self.weights_np[i] <= remaining:
                selection[i] = 1
                remaining -= self.weights_np[i]

        return selection

    def act(self, X: np.ndarray) -> np.ndarray:
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

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        pred, z1, h = self._forward(X)

        if full_feedback:
            mask = jnp.ones(self.n_arms)
            Y_target = rewards
        else:
            mask = (selection > 0.5).astype(float)
            Y_target = jnp.where(selection > 0.5, rewards, pred)

        error = (pred - Y_target) * mask
        n_obs = jnp.maximum(jnp.sum(mask), 1.0)
        grad_out = 2 * error / n_obs

        grad_W2 = jnp.outer(h, grad_out)
        grad_b2 = grad_out
        grad_h = grad_out @ self.W2.T
        grad_z1 = grad_h * (z1 > 0)
        grad_W1 = jnp.outer(X, grad_z1)
        grad_b1 = grad_z1

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


class MSENeuralAgent:
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
        self.hidden_dim = hidden_dim

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
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

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        z1 = X @ self.W1 + self.b1
        return jnp.maximum(0, z1), z1

    def predict_mean(self, X):
        h, _ = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h, _ = self._features(X)
        ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
        return self.solver(np.asarray(ucb_values))

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h, z1 = self._features(X)

        mask = jnp.ones(self.n_arms, dtype=bool) if full_feedback else (selection > 0.5)
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, mask
        )

        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

        if full_feedback:
            Y_target = rewards
        else:
            ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
            Y_target = jnp.where(selection > 0.5, rewards, ucb_values)

        mse_grad = 2 * (pred - Y_target) / self.n_arms

        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        grad_h = jnp.einsum('n,ni->i', mse_grad, theta)

        grad_z1 = grad_h * (z1 > 0)

        grads = {'W1': jnp.outer(X, grad_z1), 'b1': grad_z1}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']

import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.losses import make_spo_grad_fn
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    update_bayesian_stats,
)


class BayesianSPOAgent:

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
        lr: float = 0.001,

        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        mask_grad: bool = False,
        nonlinear: bool = False,
        minimize: bool = False,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.mask_grad = mask_grad
        self.nonlinear = nonlinear
        self.minimize = minimize

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        # SPO+ gradient using the correct solver and minimize direction
        self.loss_grad_fn = make_spo_grad_fn(self.solver, minimize)

        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        key = jax.random.PRNGKey(seed if seed else 0)
        if nonlinear:
            k1, k2 = jax.random.split(key)
            self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            self.W2 = jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1
            self.b2 = jnp.zeros(hidden_dim)
            init_params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        else:
            self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            init_params = {'W1': self.W1, 'b1': self.b1}

        # Inverse-sqrt LR schedule: lr / sqrt(t+1)
        schedule = lambda count: lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule)
        )
        self.opt_state = self.optimizer.init(init_params)

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        if self.nonlinear:
            z = X @ self.W1 + self.b1
            a = jnp.maximum(0, z)  # ReLU
            h = a @ self.W2 + self.b2
            return h
        else:
            h = X @ self.W1 + self.b1
            return h

    def predict_mean(self, X):
        h = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h = self._features(X)
        ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
        return self.solver(np.asarray(ucb_values))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False,
               prob_values=None):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h = self._features(X)

        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, selection_mask
        )

        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

        # Use smooth probabilities for SPO gradient when available
        if prob_values is not None:
            Y_target = jnp.asarray(prob_values)
        elif full_feedback:
            Y_target = rewards
        else:
            Y_target = jnp.where(selection > 0.5, rewards, pred)

        spo_grad_val = self.loss_grad_fn(pred, Y_target)

        # Zero gradient on unobserved arms when mask_grad is enabled
        if self.mask_grad and not full_feedback:
            spo_grad_val = jnp.where(selection > 0.5, spo_grad_val, 0.0)

        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        grad_h = jnp.einsum('n,ni->i', spo_grad_val, theta)

        if self.nonlinear:
            # Backprop: grad_h -> W2 -> ReLU -> W1
            z = X @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            grad_a = grad_h @ self.W2.T  # through W2
            grad_z = grad_a * (z > 0)     # through ReLU
            grads = {
                'W1': jnp.outer(X, grad_z), 'b1': grad_z,
                'W2': jnp.outer(a, grad_h), 'b2': grad_h,
            }
            params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        else:
            grads = {'W1': jnp.outer(X, grad_h), 'b1': grad_h}
            params = {'W1': self.W1, 'b1': self.b1}

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        self.W1 = jnp.clip(new_params['W1'], -10, 10)
        self.b1 = new_params['b1']
        if self.nonlinear:
            self.W2 = jnp.clip(new_params['W2'], -10, 10)
            self.b2 = new_params['b2']

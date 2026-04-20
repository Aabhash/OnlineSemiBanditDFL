import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.losses import make_spo_grad_fn
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    compute_uncertainties,
    linear_forward,
    mlp_forward,
    update_bayesian_stats,
)
from src.agents.agent_base import AgentBase
from src.solver import solve_knapsack


class BayesianSPOAgent(AgentBase):

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

        phased: bool = False,
        phase_length: int = 50,
        phase_epochs: int = 5,
        max_buffer_size: int = None,
        replay_batch_size: int = 1,
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
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0


        self.loss_grad_fn = make_spo_grad_fn(self.solver, minimize)

        if beta_start is not None and beta_end is not None:
            self.beta_start = abs(beta_start)
            self.beta_end = abs(beta_end)
            self.beta = self.beta_start
        else:
            beta = abs(beta)
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


        schedule = lambda count: lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule)
        )
        self.opt_state = self.optimizer.init(init_params)

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)


        self.phased = phased
        self.phase_length = phase_length
        self.phase_epochs = phase_epochs
        self.replay_batch_size = max(int(replay_batch_size), 1)
        self.rng = np.random.default_rng(seed if seed else 0)
        if phased:
            self._init_phase_buffer(max_buffer_size)
            self._rounds_since_retrain = 0

    def _get_params(self):
        if self.nonlinear:
            return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        return {'W1': self.W1, 'b1': self.b1}

    def _set_params(self, params):
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']
        if self.nonlinear:
            self.W2 = jnp.clip(params['W2'], -10, 10)
            self.b2 = params['b2']

    def _features(self, X):
        X = jnp.asarray(X)
        if self.nonlinear:
            h, _, _ = mlp_forward(X, self.W1, self.b1, self.W2, self.b2)
            return h
        return linear_forward(X, self.W1, self.b1)

    def predict_mean(self, X):
        h = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def predict_uncertainty(self, X):
        h = self._features(X)
        return compute_uncertainties(h, self.A_inv)

    def predict_mean_and_uncertainty(self, X):
        h = self._features(X)
        return (compute_bayesian_predictions(h, self.A_inv, self.b_bayes),
                compute_uncertainties(h, self.A_inv))

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h = self._features(X)
        ucb_values = compute_ucb_values(
            h, self.A_inv, self.b_bayes, self.beta, minimize=self.minimize
        )
        return self.solver(np.asarray(ucb_values))

    def _compute_spo_grad_h(self, X, h, Y_target, selection_mask, full_feedback):
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

        spo_grad_val = self.loss_grad_fn(pred, Y_target)

        if self.mask_grad and not full_feedback:
            spo_grad_val = jnp.where(selection_mask, spo_grad_val, 0.0)

        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        return jnp.einsum('n,ni->i', spo_grad_val, theta)

    def _backprop_and_step(self, X, grad_h):
        X = jnp.asarray(X)
        if self.nonlinear:
            z = X @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            grad_a = grad_h @ self.W2.T
            grad_z = grad_a * (z > 0)
            grads = {
                'W1': jnp.outer(X, grad_z), 'b1': grad_z,
                'W2': jnp.outer(a, grad_h), 'b2': grad_h,
            }
        else:
            grads = {'W1': jnp.outer(X, grad_h), 'b1': grad_h}

        params = self._get_params()
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        self._set_params(optax.apply_updates(params, updates))

    def _backprop_and_step_batch(self, X_batch, grad_h_batch):
        X_batch = jnp.asarray(X_batch)
        grad_h_batch = jnp.asarray(grad_h_batch)
        batch_size = max(int(X_batch.shape[0]), 1)
        if self.nonlinear:
            z = X_batch @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            grad_a = grad_h_batch @ self.W2.T
            grad_z = grad_a * (z > 0)
            grads = {
                'W1': (X_batch.T @ grad_z) / batch_size,
                'b1': jnp.mean(grad_z, axis=0),
                'W2': (a.T @ grad_h_batch) / batch_size,
                'b2': jnp.mean(grad_h_batch, axis=0),
            }
        else:
            grads = {
                'W1': (X_batch.T @ grad_h_batch) / batch_size,
                'b1': jnp.mean(grad_h_batch, axis=0),
            }

        params = self._get_params()
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        self._set_params(optax.apply_updates(params, updates))

    def _phased_retrain_batch(self, batch_indices):
        X_batch = jnp.stack([self._phase_buffer_X[i] for i in batch_indices])
        h_batch = self._features(X_batch)
        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        pred_batch = h_batch @ theta.T

        y_targets = []
        obs_masks = []
        grad_h_batch = []

        for pos, idx in enumerate(batch_indices):
            obs_mask = jnp.asarray(self._phase_buffer_obs[idx], dtype=jnp.bool_)
            prob_buf = self._phase_buffer_prob[idx]
            if prob_buf is not None:
                y_target = jnp.asarray(prob_buf)
            else:
                y_target = jnp.where(obs_mask, jnp.asarray(self._phase_buffer_Y[idx]), pred_batch[pos])

            spo_grad_val = self.loss_grad_fn(pred_batch[pos], y_target)
            if self.mask_grad:
                spo_grad_val = jnp.where(obs_mask, spo_grad_val, 0.0)

            y_targets.append(y_target)
            obs_masks.append(obs_mask)
            grad_h_batch.append(jnp.einsum('n,ni->i', spo_grad_val, theta))

        self._backprop_and_step_batch(X_batch, jnp.stack(grad_h_batch))

    def _phased_retrain(self):
        n_buffer = len(self._phase_buffer_X)
        if n_buffer == 0:
            return
        batch_size = min(self.replay_batch_size, n_buffer)

        for _epoch in range(self.phase_epochs):
            if batch_size <= 1:
                for i in range(n_buffer):
                    X_buf = self._phase_buffer_X[i]
                    Y_buf = self._phase_buffer_Y[i]
                    prob_buf = self._phase_buffer_prob[i]
                    obs_buf = self._phase_buffer_obs[i]

                    h = self._features(X_buf)
                    pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

                    obs_mask = jnp.asarray(obs_buf, dtype=jnp.bool_)
                    if prob_buf is not None:
                        Y_target = jnp.asarray(prob_buf)
                    else:
                        Y_target = jnp.where(obs_mask, jnp.asarray(Y_buf), pred)

                    grad_h = self._compute_spo_grad_h(X_buf, h, Y_target, obs_mask, False)
                    self._backprop_and_step(X_buf, grad_h)
            else:
                indices = self.rng.permutation(n_buffer)
                for start in range(0, n_buffer, batch_size):
                    batch_indices = indices[start:start + batch_size]
                    self._phased_retrain_batch(batch_indices)


        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            self.n_arms, self.hidden_dim, self.lambda_reg)
        for i in range(len(self._phase_buffer_X)):
            h = self._features(self._phase_buffer_X[i])
            obs_mask = jnp.asarray(self._phase_buffer_obs[i], dtype=jnp.bool_)
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
                self.A, self.b_bayes, self.A_inv, h, rewards, obs_mask)

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False,
               **kwargs):
        prob_values = kwargs.get("prob_values")
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h = self._features(X)

        selection_mask = selection > 0.5
        obs_mask = jnp.ones(self.n_arms, dtype=jnp.bool_) if full_feedback else selection_mask
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, obs_mask
        )

        if self.phased:

            self._append_phase_buffer(X, rewards, selection, prob_values, obs_mask)

            self._rounds_since_retrain += 1
            if self._rounds_since_retrain >= self.phase_length:
                self._phased_retrain()
                self._rounds_since_retrain = 0
        else:

            pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

            if prob_values is not None:
                Y_target = jnp.asarray(prob_values)
            elif full_feedback:
                Y_target = rewards
            else:
                Y_target = jnp.where(selection > 0.5, rewards, pred)

            grad_h = self._compute_spo_grad_h(X, h, Y_target, selection_mask, full_feedback)
            self._backprop_and_step(X, grad_h)

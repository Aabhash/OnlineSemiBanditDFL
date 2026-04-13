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
            z = X @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            h = a @ self.W2 + self.b2
            return h
        else:
            h = X @ self.W1 + self.b1
            return h

    def predict_mean(self, X):
        h = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def predict_uncertainty(self, X):
        h = self._features(X)
        return compute_uncertainties(h, self.A_inv)

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h = self._features(X)
        ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
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

    def _phased_retrain(self):
        for _epoch in range(self.phase_epochs):
            for i in range(len(self._phase_buffer_X)):
                X_buf = self._phase_buffer_X[i]
                Y_buf = self._phase_buffer_Y[i]
                sel_buf = self._phase_buffer_sel[i]
                prob_buf = self._phase_buffer_prob[i]

                h = self._features(X_buf)
                pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

                sel_mask = jnp.asarray(sel_buf) > 0.5
                if prob_buf is not None:
                    Y_target = jnp.asarray(prob_buf)
                else:
                    Y_target = jnp.where(sel_mask, jnp.asarray(Y_buf), pred)

                grad_h = self._compute_spo_grad_h(X_buf, h, Y_target, sel_mask, False)
                self._backprop_and_step(X_buf, grad_h)


        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            self.n_arms, self.hidden_dim, self.lambda_reg)
        for i in range(len(self._phase_buffer_X)):
            h = self._features(self._phase_buffer_X[i])
            sel_mask = jnp.asarray(self._phase_buffer_sel[i]) > 0.5
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
                self.A, self.b_bayes, self.A_inv, h, rewards, sel_mask)

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

            self._append_phase_buffer(X, rewards, selection, prob_values)

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

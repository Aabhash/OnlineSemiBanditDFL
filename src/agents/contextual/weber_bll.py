import jax
import jax.numpy as jnp
import numpy as np
import optax
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


class WeberBLLAgent(AgentBase):

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
        mask_grad: bool = True,
        nonlinear: bool = False,
        minimize: bool = False,
        solver=None,
        seed: int = None,

        noise_var: float = 1.0,
        fisher_beta: float = 0.1,
        phased: bool = True,
        phase_length: int = 50,
        phase_epochs: int = 5,
        fisher_eps: float = 1e-8,
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
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        if solver is None:
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver


        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta


        self.noise_var = noise_var
        self.fisher_beta = fisher_beta
        self.phased = phased
        self.phase_length = phase_length
        self.phase_epochs = phase_epochs
        self.fisher_eps = fisher_eps


        if phased:
            self._init_phase_buffer(max_buffer_size)
            self._rounds_since_retrain = 0
        self._fisher_diag = None
        self._param_anchor = None


        key = jax.random.PRNGKey(seed if seed is not None else 0)
        key, init_key = jax.random.split(key)
        self.rng_key = key


        if nonlinear:
            k1, k2 = jax.random.split(init_key)
            self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            self.W2 = jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1
            self.b2 = jnp.zeros(hidden_dim)
        else:
            self.W1 = jax.random.normal(init_key, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)


        schedule = lambda count: lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule),
        )
        self.opt_state = self.optimizer.init(self._get_params())


        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)


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
        ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
        return self.solver(np.asarray(ucb_values))


    def _marginalized_grad_h(self, h, rewards, selection_mask, full_feedback):

        mu = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)


        pred = mu @ h


        residuals = pred - rewards


        if full_feedback:
            obs_mask = jnp.ones(self.n_arms, dtype=jnp.bool_)
        else:
            obs_mask = selection_mask

        if self.mask_grad:
            residuals = jnp.where(obs_mask, residuals, 0.0)


        mse_term = jnp.einsum('n,ni->i', residuals, mu)


        sigma_h = jnp.einsum('nij,j->ni', self.A_inv, h)
        if self.mask_grad:
            sigma_h = jnp.where(obs_mask[:, None], sigma_h, 0.0)
        uncertainty_term = jnp.sum(sigma_h, axis=0)

        return (1.0 / self.noise_var) * (mse_term + uncertainty_term)


    def _backprop_features(self, X, grad_h):
        X = jnp.asarray(X)
        if self.nonlinear:
            z = X @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            grad_a = grad_h @ self.W2.T
            grad_z = grad_a * (z > 0)
            return {
                'W1': jnp.outer(X, grad_z), 'b1': grad_z,
                'W2': jnp.outer(a, grad_h), 'b2': grad_h,
            }
        return {'W1': jnp.outer(X, grad_h), 'b1': grad_h}


    def _compute_fisher_diagonal(self):
        N = len(self._phase_buffer_X)
        if N == 0:
            return {k: jnp.zeros_like(v) for k, v in self._get_params().items()}

        fisher = {k: jnp.zeros_like(v) for k, v in self._get_params().items()}
        for i in range(N):
            h = self._features(self._phase_buffer_X[i])
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            obs_mask = jnp.asarray(self._phase_buffer_obs[i], dtype=jnp.bool_)
            grad_h = self._marginalized_grad_h(h, rewards, obs_mask, full_feedback=False)
            grads = self._backprop_features(self._phase_buffer_X[i], grad_h)
            for k in fisher:
                fisher[k] = fisher[k] + grads[k] ** 2

        return {k: v / N + self.fisher_eps for k, v in fisher.items()}

    def _add_fisher_reg(self, grads):
        if self.fisher_beta <= 0.0 or self._fisher_diag is None or self._param_anchor is None:
            return grads
        grads = dict(grads)
        params = self._get_params()
        for k in grads:
            grads[k] = grads[k] + self.fisher_beta * self._fisher_diag[k] * (params[k] - self._param_anchor[k])
        return grads


    def _apply_gradient_step(self, grads):
        grads = self._add_fisher_reg(grads)
        params = self._get_params()
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        self._set_params(new_params)


    def _reset_bayesian_stats(self):
        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            self.n_arms, self.hidden_dim, self.lambda_reg)

    def _replay_buffer_bayesian(self):
        self._reset_bayesian_stats()
        for i in range(len(self._phase_buffer_X)):
            h = self._features(self._phase_buffer_X[i])
            obs_mask = jnp.asarray(self._phase_buffer_obs[i], dtype=jnp.bool_)
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
                self.A, self.b_bayes, self.A_inv, h, rewards, obs_mask)

    def _phased_retrain(self):

        self._fisher_diag = self._compute_fisher_diagonal()


        self._param_anchor = {k: v.copy() for k, v in self._get_params().items()}


        for _epoch in range(self.phase_epochs):
            for i in range(len(self._phase_buffer_X)):
                X_buf = self._phase_buffer_X[i]
                Y_buf = self._phase_buffer_Y[i]
                prob_buf = self._phase_buffer_prob[i]
                obs_buf = self._phase_buffer_obs[i]

                h = self._features(X_buf)


                rewards = jnp.asarray(prob_buf) if prob_buf is not None else jnp.asarray(Y_buf)
                obs_mask = jnp.asarray(obs_buf, dtype=jnp.bool_)

                grad_h = self._marginalized_grad_h(h, rewards, obs_mask, full_feedback=False)
                grads = self._backprop_features(X_buf, grad_h)
                self._apply_gradient_step(grads)


        self._replay_buffer_bayesian()


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
            self.A, self.b_bayes, self.A_inv, h, rewards, obs_mask)

        if self.phased:

            self._append_phase_buffer(X, rewards, selection, prob_values, obs_mask)

            self._rounds_since_retrain += 1
            if self._rounds_since_retrain >= self.phase_length:
                self._phased_retrain()
                self._rounds_since_retrain = 0
        else:

            r = jnp.asarray(prob_values) if prob_values is not None else rewards
            grad_h = self._marginalized_grad_h(h, r, selection_mask, full_feedback)
            grads = self._backprop_features(X, grad_h)
            self._apply_gradient_step(grads)

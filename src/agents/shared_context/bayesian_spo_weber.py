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


class BayesianSPO_Weber_Agent:
    """BayesianSPO with Weber posterior-marginalized feature gradient.

    Instead of using only the posterior mean theta_hat for the chain-rule
    multiplier, we draw MC samples from the posterior N(theta_hat, A_inv)
    and average the resulting SPO+ feature gradients. This makes the
    feature gradient uncertainty-aware: uncertain arms contribute more
    diverse gradient signals, preventing premature feature collapse.
    """

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
        # Weber-specific parameters
        n_mc_samples: int = 5,
        fisher_beta: float = 0.0,
        cholesky_jitter: float = 1e-6,
        # Phased mode parameters
        phased: bool = False,
        phase_length: int = 50,
        phase_epochs: int = 5,
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

        # SPO+ gradient
        self.loss_grad_fn = make_spo_grad_fn(self.solver, minimize)

        # Exploration schedule
        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        # Weber-specific
        self.n_mc_samples = n_mc_samples
        self.fisher_beta = fisher_beta
        self.cholesky_jitter = cholesky_jitter

        # Phased mode
        self.phased = phased
        self.phase_length = phase_length
        self.phase_epochs = phase_epochs
        if phased:
            self._phase_buffer_X = []
            self._phase_buffer_Y = []
            self._phase_buffer_sel = []
            self._phase_buffer_prob = []

        # RNG
        key = jax.random.PRNGKey(seed if seed is not None else 0)
        key, init_key = jax.random.split(key)
        self.rng_key = key

        # Initialize feature extractor
        if nonlinear:
            k1, k2 = jax.random.split(init_key)
            self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            self.W2 = jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1
            self.b2 = jnp.zeros(hidden_dim)
            init_params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        else:
            self.W1 = jax.random.normal(init_key, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            init_params = {'W1': self.W1, 'b1': self.b1}

        # Anchor weights for Fisher regularization
        self.W1_anchor = self.W1.copy()
        if nonlinear:
            self.W2_anchor = self.W2.copy()

        # Optimizer with inverse-sqrt LR schedule
        schedule = lambda count: lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule)
        )
        self.opt_state = self.optimizer.init(init_params)

        # Bayesian stats
        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

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

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h = self._features(X)
        ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
        return self.solver(np.asarray(ucb_values))

    def _mc_marginalized_gradient(self, h, Y_target, selection_mask, full_feedback):
        """Compute MC-marginalized SPO+ feature gradient (Weber et al.)."""
        theta_hat = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)

        # Cholesky of A_inv (posterior covariance) with jitter for stability
        jitter = self.cholesky_jitter * jnp.eye(self.hidden_dim)
        A_inv_jittered = self.A_inv + jitter[None, :, :]
        # Use eigendecomposition-based sqrt for robustness
        L = jnp.linalg.cholesky(A_inv_jittered)

        grad_h_accum = jnp.zeros(self.hidden_dim)

        for s in range(self.n_mc_samples):
            self.rng_key, subkey = jax.random.split(self.rng_key)
            z = jax.random.normal(subkey, (self.n_arms, self.hidden_dim))
            # theta_s ~ N(theta_hat, A_inv)
            theta_s = theta_hat + jnp.einsum('nij,nj->ni', L, z)

            # Predictions at sampled weights
            pred_s = theta_s @ h

            # SPO+ gradient at sampled predictions
            spo_grad_s = self.loss_grad_fn(pred_s, Y_target)

            # Mask gradient on unobserved arms
            if self.mask_grad and not full_feedback:
                spo_grad_s = jnp.where(selection_mask, spo_grad_s, 0.0)

            # Chain rule: d_loss/d_h = sum_n spo_grad[n] * theta_s[n]
            grad_h_accum = grad_h_accum + jnp.einsum('n,ni->i', spo_grad_s, theta_s)

        return grad_h_accum / self.n_mc_samples

    def _backprop_features(self, X, grad_h):
        """Backprop grad_h through the feature extractor to get parameter gradients."""
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
        return grads

    def _add_fisher_reg(self, grads):
        """Add Fisher L2 regularization toward anchor weights."""
        if self.fisher_beta <= 0.0:
            return grads
        grads = dict(grads)
        grads['W1'] = grads['W1'] + self.fisher_beta * (self.W1 - self.W1_anchor)
        if self.nonlinear and 'W2' in grads:
            grads['W2'] = grads['W2'] + self.fisher_beta * (self.W2 - self.W2_anchor)
        return grads

    def _apply_gradient_step(self, grads):
        """Apply one optimizer step with the given gradients."""
        grads = self._add_fisher_reg(grads)

        if self.nonlinear:
            params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        else:
            params = {'W1': self.W1, 'b1': self.b1}

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        self.W1 = jnp.clip(new_params['W1'], -10, 10)
        self.b1 = new_params['b1']
        if self.nonlinear:
            self.W2 = jnp.clip(new_params['W2'], -10, 10)
            self.b2 = new_params['b2']

    def _reset_bayesian_stats(self):
        """Reset Bayesian stats (used after phased retraining)."""
        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            self.n_arms, self.hidden_dim, self.lambda_reg)

    def _replay_buffer_bayesian(self):
        """Replay buffered data through updated features to rebuild Bayesian stats."""
        self._reset_bayesian_stats()
        for i in range(len(self._phase_buffer_X)):
            h = self._features(self._phase_buffer_X[i])
            sel_mask = jnp.asarray(self._phase_buffer_sel[i]) > 0.5
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
                self.A, self.b_bayes, self.A_inv, h, rewards, sel_mask
            )

    def _phased_retrain(self, lr):
        """Run phased retraining: multiple gradient steps over the buffer."""
        # Snapshot anchors
        self.W1_anchor = self.W1.copy()
        if self.nonlinear:
            self.W2_anchor = self.W2.copy()

        for epoch in range(self.phase_epochs):
            for i in range(len(self._phase_buffer_X)):
                X_buf = self._phase_buffer_X[i]
                Y_buf = self._phase_buffer_Y[i]
                sel_buf = self._phase_buffer_sel[i]
                prob_buf = self._phase_buffer_prob[i]

                h = self._features(X_buf)
                pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

                if prob_buf is not None:
                    Y_target = jnp.asarray(prob_buf)
                else:
                    sel_mask = jnp.asarray(sel_buf) > 0.5
                    Y_target = jnp.where(sel_mask, jnp.asarray(Y_buf), pred)

                sel_mask = jnp.asarray(sel_buf) > 0.5
                grad_h = self._mc_marginalized_gradient(h, Y_target, sel_mask, False)
                grads = self._backprop_features(X_buf, grad_h)
                self._apply_gradient_step(grads)

        # Replay buffer with new features to rebuild Bayesian stats
        self._replay_buffer_bayesian()

        # Clear buffer
        self._phase_buffer_X.clear()
        self._phase_buffer_Y.clear()
        self._phase_buffer_sel.clear()
        self._phase_buffer_prob.clear()

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

        if prob_values is not None:
            Y_target = jnp.asarray(prob_values)
        elif full_feedback:
            Y_target = rewards
        else:
            Y_target = jnp.where(selection > 0.5, rewards, pred)

        if self.phased:
            # Buffer data for phased retraining
            self._phase_buffer_X.append(X)
            self._phase_buffer_Y.append(rewards)
            self._phase_buffer_sel.append(selection)
            self._phase_buffer_prob.append(prob_values)

            # Retrain every phase_length rounds
            if len(self._phase_buffer_X) >= self.phase_length:
                self._phased_retrain(lr)
        else:
            # Per-round: one MC-marginalized gradient step
            grad_h = self._mc_marginalized_gradient(h, Y_target, selection_mask, full_feedback)
            grads = self._backprop_features(X, grad_h)
            self._apply_gradient_step(grads)

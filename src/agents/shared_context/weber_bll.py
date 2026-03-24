import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    update_bayesian_stats,
)


class WeberBLLAgent:
    """Faithful implementation of Weber et al. 'Optimizing over a Bayesian Last Layer' (2018).

    Trains a feature extractor by maximizing the Gaussian log-likelihood
    marginalized over the Bayesian last-layer posterior.  For the Gaussian case
    this has a closed-form gradient (no MC sampling needed):

        grad_h = -(1/sigma^2) * sum_n [(mu_n^T h - y_n)*mu_n + Sigma_n @ h]

    The second term (Sigma_n @ h) is the *marginalization correction* that
    pushes features toward directions where the posterior is uncertain.

    Training is strictly phased (Weber Section 4):
      - Online phase: fixed features, update Bayesian posterior, UCB exploration
      - Offline phase: retrain features with marginalized LL + diagonal Fisher
        regularization, then replay buffer to rebuild posteriors
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
        mask_grad: bool = True,
        nonlinear: bool = False,
        minimize: bool = False,
        solver=None,
        seed: int = None,
        # Weber-specific
        noise_var: float = 1.0,
        fisher_beta: float = 0.1,
        phased: bool = True,
        phase_length: int = 50,
        phase_epochs: int = 5,
        fisher_eps: float = 1e-8,
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
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver

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
        self.noise_var = noise_var
        self.fisher_beta = fisher_beta
        self.phased = phased
        self.phase_length = phase_length
        self.phase_epochs = phase_epochs
        self.fisher_eps = fisher_eps

        # Phased training state
        self._phase_buffer_X = []
        self._phase_buffer_Y = []
        self._phase_buffer_sel = []
        self._phase_buffer_prob = []
        self._rounds_since_retrain = 0
        self._fisher_diag = None
        self._param_anchor = None

        # RNG
        key = jax.random.PRNGKey(seed if seed is not None else 0)
        key, init_key = jax.random.split(key)
        self.rng_key = key

        # Feature extractor
        if nonlinear:
            k1, k2 = jax.random.split(init_key)
            self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            self.W2 = jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1
            self.b2 = jnp.zeros(hidden_dim)
        else:
            self.W1 = jax.random.normal(init_key, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)

        # Optimizer with inverse-sqrt LR schedule
        schedule = lambda count: lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule),
        )
        self.opt_state = self.optimizer.init(self._get_params())

        # Per-arm Bayesian regression
        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            return a @ self.W2 + self.b2
        return X @ self.W1 + self.b1

    # ------------------------------------------------------------------
    # Prediction / action
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Closed-form marginalized gradient  (Weber core contribution)
    # ------------------------------------------------------------------

    def _marginalized_grad_h(self, h, rewards, selection_mask, full_feedback):
        """Closed-form gradient of the Gaussian marginalized log-likelihood.

        For observed arm n:
          grad_n = (1/sigma^2) * [(mu_n^T h - y_n) * mu_n  +  Sigma_n @ h]
                    ~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~    ~~~~~~~~~~~~
                    scale           MSE term                   uncertainty correction

        Returns the gradient in the *descent* direction (minimising neg-LL).
        """
        # Posterior mean: mu_n = A_inv_n @ b_n
        mu = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)  # (n_arms, hidden_dim)

        # Predictions: mu_n^T h
        pred = mu @ h  # (n_arms,)

        # Residuals
        residuals = pred - rewards  # (n_arms,)

        # Observation mask
        if full_feedback:
            obs_mask = jnp.ones(self.n_arms, dtype=jnp.bool_)
        else:
            obs_mask = selection_mask

        if self.mask_grad:
            residuals = jnp.where(obs_mask, residuals, 0.0)

        # Term 1: MSE gradient  sum_n (pred_n - y_n) * mu_n
        mse_term = jnp.einsum('n,ni->i', residuals, mu)  # (hidden_dim,)

        # Term 2: Uncertainty correction  sum_n Sigma_n @ h  (only observed)
        sigma_h = jnp.einsum('nij,j->ni', self.A_inv, h)  # (n_arms, hidden_dim)
        if self.mask_grad:
            sigma_h = jnp.where(obs_mask[:, None], sigma_h, 0.0)
        uncertainty_term = jnp.sum(sigma_h, axis=0)  # (hidden_dim,)

        return (1.0 / self.noise_var) * (mse_term + uncertainty_term)

    # ------------------------------------------------------------------
    # Backprop through feature extractor
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Diagonal Fisher regularization  (Weber Eq. 7, Section 7.2)
    # ------------------------------------------------------------------

    def _compute_fisher_diagonal(self):
        """Diagonal Fisher: F_diag[i] = (1/N) sum_t (dL/dtheta_i)^2."""
        N = len(self._phase_buffer_X)
        if N == 0:
            return {k: jnp.zeros_like(v) for k, v in self._get_params().items()}

        fisher = {k: jnp.zeros_like(v) for k, v in self._get_params().items()}
        for i in range(N):
            h = self._features(self._phase_buffer_X[i])
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            sel_mask = jnp.asarray(self._phase_buffer_sel[i]) > 0.5
            grad_h = self._marginalized_grad_h(h, rewards, sel_mask, full_feedback=False)
            grads = self._backprop_features(self._phase_buffer_X[i], grad_h)
            for k in fisher:
                fisher[k] = fisher[k] + grads[k] ** 2

        return {k: v / N + self.fisher_eps for k, v in fisher.items()}

    def _add_fisher_reg(self, grads):
        """Fisher-weighted regularization: fisher_beta * F_diag * (theta - anchor)."""
        if self.fisher_beta <= 0.0 or self._fisher_diag is None or self._param_anchor is None:
            return grads
        grads = dict(grads)
        params = self._get_params()
        for k in grads:
            grads[k] = grads[k] + self.fisher_beta * self._fisher_diag[k] * (params[k] - self._param_anchor[k])
        return grads

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _apply_gradient_step(self, grads):
        grads = self._add_fisher_reg(grads)
        params = self._get_params()
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        self._set_params(new_params)

    # ------------------------------------------------------------------
    # Phased retraining  (Weber Section 4)
    # ------------------------------------------------------------------

    def _reset_bayesian_stats(self):
        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            self.n_arms, self.hidden_dim, self.lambda_reg)

    def _replay_buffer_bayesian(self):
        """Replay full buffer through updated features to rebuild posteriors."""
        self._reset_bayesian_stats()
        for i in range(len(self._phase_buffer_X)):
            h = self._features(self._phase_buffer_X[i])
            sel_mask = jnp.asarray(self._phase_buffer_sel[i]) > 0.5
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
                self.A, self.b_bayes, self.A_inv, h, rewards, sel_mask)

    def _phased_retrain(self):
        """Offline retraining: Fisher computation -> gradient epochs -> replay."""
        # 1. Diagonal Fisher on current (pre-retrain) params
        self._fisher_diag = self._compute_fisher_diagonal()

        # 2. Snapshot anchor weights
        self._param_anchor = {k: v.copy() for k, v in self._get_params().items()}

        # 3. Multi-epoch gradient descent over the buffer
        for _epoch in range(self.phase_epochs):
            for i in range(len(self._phase_buffer_X)):
                X_buf = self._phase_buffer_X[i]
                Y_buf = self._phase_buffer_Y[i]
                sel_buf = self._phase_buffer_sel[i]
                prob_buf = self._phase_buffer_prob[i]

                h = self._features(X_buf)

                # Use prob_values as targets when available (e.g. Criteo)
                rewards = jnp.asarray(prob_buf) if prob_buf is not None else jnp.asarray(Y_buf)
                sel_mask = jnp.asarray(sel_buf) > 0.5

                grad_h = self._marginalized_grad_h(h, rewards, sel_mask, full_feedback=False)
                grads = self._backprop_features(X_buf, grad_h)
                self._apply_gradient_step(grads)

        # 4. Replay full buffer to rebuild posteriors with new features
        self._replay_buffer_bayesian()

    # ------------------------------------------------------------------
    # Main update  (online phase + phase boundary trigger)
    # ------------------------------------------------------------------

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False,
               prob_values=None):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h = self._features(X)

        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, selection_mask)

        if self.phased:
            # Online phase: posterior update only, features frozen until phase boundary
            self._phase_buffer_X.append(X)
            self._phase_buffer_Y.append(rewards)
            self._phase_buffer_sel.append(selection)
            self._phase_buffer_prob.append(prob_values)

            self._rounds_since_retrain += 1
            if self._rounds_since_retrain >= self.phase_length:
                self._phased_retrain()
                self._rounds_since_retrain = 0
        else:
            # Per-round: update features immediately with marginalized gradient
            r = jnp.asarray(prob_values) if prob_values is not None else rewards
            grad_h = self._marginalized_grad_h(h, r, selection_mask, full_feedback)
            grads = self._backprop_features(X, grad_h)
            self._apply_gradient_step(grads)

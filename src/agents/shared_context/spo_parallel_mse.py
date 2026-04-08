"""SPO+ agent with a parallel MSE imputer for unobserved arms.

Decouples the decision-focused model from the imputation model:
- Model A (SPO+): Bayesian shared-feature SPO+ agent.  Handles decisions
  (action selection via Bayesian UCB) and SPO+ feature learning.
- Model B (MSE imputer): Per-arm OLS on raw (augmented) context.  No
  shared features, no Bayesian head, no solver involvement — just
  closed-form ridge regression via Sherman-Morrison per arm.

During Model A's SPO+ gradient computation, unobserved arms are imputed
with Model B's predictions instead of Model A's own predictions.  This
breaks the feedback loop where bad SPO+ predictions corrupt the solver
input, which then corrupts future SPO+ gradients.

Because Model B is trained only with MSE on observed arms and never calls
the solver, it is immune to the solver contamination problem, and its
predictions converge toward the best linear-in-raw-x approximation of the
true costs.  Model A is then free to specialize its shared features for
decision-focused learning.
"""

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


class SPO_ParallelMSE_Agent:
    """BayesianSPOAgent with a parallel per-arm OLS imputer for unobserved arms.

    The SPO+ model (features + Bayesian head) is unchanged from
    ``BayesianSPOAgent``.  The only addition is a per-arm closed-form ridge
    regression (OLS) on raw augmented context ``[x, 1]``, used solely to
    impute unobserved arms in the SPO+ gradient computation.

    The MSE imputer has no shared features, no hidden layer, no Bayesian
    head — it is the simplest possible per-arm linear predictor.  This keeps
    its training dynamics completely independent of the SPO+ model's feature
    collapse issues.
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
        # Phased training (Weber-style)
        phased: bool = False,
        phase_length: int = 50,
        phase_epochs: int = 5,
        # Parallel MSE imputer
        mse_lambda_reg: float = 1.0,
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

        def _lr_schedule(count):
            return lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(_lr_schedule),
        )
        self.opt_state = self.optimizer.init(init_params)

        # SPO+ Bayesian head (shared-feature side)
        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

        # Parallel MSE imputer: per-arm OLS on augmented raw context [x, 1]
        self._aug_dim = context_dim + 1
        self.M_mse_inv = np.stack([np.eye(self._aug_dim) / mse_lambda_reg
                                   for _ in range(n_arms)], axis=0)
        self.b_mse_raw = np.zeros((n_arms, self._aug_dim), dtype=np.float64)

        # Phased training state
        self.phased = phased
        self.phase_length = phase_length
        self.phase_epochs = phase_epochs
        if phased:
            self._phase_buffer_X = []
            self._phase_buffer_Y = []
            self._phase_buffer_sel = []
            self._phase_buffer_prob = []
            self._rounds_since_retrain = 0

    # ------------------------------------------------------------------
    # Feature extractor + Bayesian head helpers
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

    def predict_mean(self, X):
        h = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    # ------------------------------------------------------------------
    # Parallel MSE imputer: per-arm OLS on augmented raw context
    # ------------------------------------------------------------------

    def _aug(self, x):
        x_np = np.asarray(x, dtype=np.float64).flatten()
        return np.concatenate([x_np, [1.0]])

    def _predict_mse(self, x):
        """Predict costs for all arms using the parallel MSE imputer."""
        x_aug = self._aug(x)
        means = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            w_i = self.M_mse_inv[i] @ self.b_mse_raw[i]
            means[i] = w_i @ x_aug
        return means

    def _update_mse_imputer(self, x, true_costs, obs_mask):
        """Closed-form ridge regression update via Sherman-Morrison, per arm."""
        x_aug = self._aug(x)
        true_np = np.asarray(true_costs, dtype=np.float64)
        obs_np = np.asarray(obs_mask)
        for i in range(self.n_arms):
            if obs_np[i]:
                Mx = self.M_mse_inv[i] @ x_aug
                self.M_mse_inv[i] -= np.outer(Mx, Mx) / (1.0 + x_aug @ Mx)
                self.b_mse_raw[i] += x_aug * true_np[i]

    # ------------------------------------------------------------------
    # Action selection (unchanged from BayesianSPOAgent)
    # ------------------------------------------------------------------

    def act(self, X):
        self.t += 1
        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h = self._features(X)
        ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
        return self.solver(np.asarray(ucb_values))

    # ------------------------------------------------------------------
    # SPO+ gradient with parallel-MSE imputation of unobserved arms
    # ------------------------------------------------------------------

    def _compute_spo_grad_h(self, X, h, rewards, selection_mask, full_feedback):
        """SPO+ feature gradient.  Unobserved arms are imputed with the
        parallel MSE model's predictions (not Model A's own predictions),
        decontaminating the solver input."""
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

        if full_feedback:
            Y_target = rewards
        else:
            mse_pred = jnp.asarray(self._predict_mse(X))
            Y_target = jnp.where(selection_mask, rewards, mse_pred)

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

    # ------------------------------------------------------------------
    # Phased retraining
    # ------------------------------------------------------------------

    def _phased_retrain(self):
        for _ in range(self.phase_epochs):
            for i in range(len(self._phase_buffer_X)):
                X_buf = self._phase_buffer_X[i]
                Y_buf = self._phase_buffer_Y[i]
                sel_buf = self._phase_buffer_sel[i]

                h = self._features(X_buf)
                sel_mask = jnp.asarray(sel_buf) > 0.5
                grad_h = self._compute_spo_grad_h(
                    X_buf, h, jnp.asarray(Y_buf), sel_mask, False)
                self._backprop_and_step(X_buf, grad_h)

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            self.n_arms, self.hidden_dim, self.lambda_reg)
        for i in range(len(self._phase_buffer_X)):
            h = self._features(self._phase_buffer_X[i])
            sel_mask = jnp.asarray(self._phase_buffer_sel[i]) > 0.5
            rewards = jnp.asarray(self._phase_buffer_Y[i])
            self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
                self.A, self.b_bayes, self.A_inv, h, rewards, sel_mask)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, X, selection, rewards, lr=None, full_feedback: bool = False,
               prob_values=None):
        X_j = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h = self._features(X_j)
        selection_mask = selection > 0.5

        # Update SPO+ Bayesian head (same as BayesianSPOAgent)
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, selection_mask
        )

        # Update parallel MSE imputer (closed-form, never uses solver)
        self._update_mse_imputer(X, rewards, selection_mask)

        if self.phased:
            self._phase_buffer_X.append(X_j)
            self._phase_buffer_Y.append(rewards)
            self._phase_buffer_sel.append(selection)
            self._phase_buffer_prob.append(prob_values)

            self._rounds_since_retrain += 1
            if self._rounds_since_retrain >= self.phase_length:
                self._phased_retrain()
                self._rounds_since_retrain = 0
        else:
            grad_h = self._compute_spo_grad_h(
                X_j, h, rewards, selection_mask, full_feedback)
            self._backprop_and_step(X_j, grad_h)

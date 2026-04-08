"""SPO+ agent with a parallel MSE imputer for unobserved arms.

Architecture (no Bayesian last layer anywhere on the SPO+ side):
  - Feature extractor: linear ``h = W1 x + b1`` (optional MLP via ``nonlinear=True``).
  - Prediction head: point-estimate ``W_pred ∈ R^{n_arms × hidden_dim}``.
    Predictions are ``pred_i = W_pred[i] @ h`` — plain network output.
    Both ``W_pred`` and features are trained via SPO+ gradient (chain rule).
  - Uncertainty (LinUCB/NeuralUCB style): per-arm feature design matrices
    ``U_i = Σ_t I[arm i observed] h_t h_t^T + λI`` with their inverses
    maintained by Sherman-Morrison.  Bonus ``σ_i(x) = sqrt(h(x)^T U_i^{-1} h(x))``.
  - Parallel MSE imputer: per-arm closed-form OLS on augmented raw context
    ``[x, 1]``.  Updated via Sherman-Morrison.  No features, no Bayesian head,
    no solver involvement.  Its predictions are fed into the SPO+ gradient
    computation as the imputed values for unobserved arms.

This decouples three things that were entangled in ``BayesianSPOAgent``:
  1. Predictions (from ``W_pred`` trained on SPO+ gradient — decision-focused)
  2. Uncertainty (from feature design matrices — LinUCB style)
  3. Imputation (from a separate per-arm OLS trained only with MSE —
     immune to solver contamination)
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.losses import make_spo_grad_fn


class SPO_ParallelMSE_Agent:
    """Decoupled-imputation SPO+ agent.

    Predictions come from a point-estimate head (no Bayesian posterior),
    exploration uses LinUCB-style feature uncertainty, and unobserved arms
    are imputed using a separate per-arm OLS model on raw context.
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
        else:
            self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)

        # Point-estimate prediction head (no Bayesian posterior)
        self.W_pred = jnp.zeros((n_arms, hidden_dim))

        init_params = self._get_params()

        def _lr_schedule(count):
            return lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(_lr_schedule),
        )
        self.opt_state = self.optimizer.init(init_params)

        # LinUCB-style per-arm design matrices on features (for uncertainty only)
        eye = np.eye(hidden_dim)
        self._U_inv = np.stack([eye / lambda_reg for _ in range(n_arms)], axis=0)

        # Parallel MSE imputer: per-arm OLS on augmented raw context [x, 1]
        self._aug_dim = context_dim + 1
        aug_eye = np.eye(self._aug_dim)
        self.M_mse_inv = np.stack([aug_eye / mse_lambda_reg for _ in range(n_arms)], axis=0)
        self.b_mse_raw = np.zeros((n_arms, self._aug_dim), dtype=np.float64)

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _get_params(self):
        if self.nonlinear:
            return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2,
                    'W_pred': self.W_pred}
        return {'W1': self.W1, 'b1': self.b1, 'W_pred': self.W_pred}

    def _set_params(self, params):
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']
        if self.nonlinear:
            self.W2 = jnp.clip(params['W2'], -10, 10)
            self.b2 = params['b2']
        self.W_pred = params['W_pred']

    # ------------------------------------------------------------------
    # Network forward (features + prediction head)
    # ------------------------------------------------------------------

    def _features(self, X):
        X = jnp.asarray(X)
        if self.nonlinear:
            z = X @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            return a @ self.W2 + self.b2
        return X @ self.W1 + self.b1

    def _predict(self, h):
        return self.W_pred @ h

    def predict_mean(self, X):
        return self._predict(self._features(X))

    # ------------------------------------------------------------------
    # LinUCB-style uncertainty on features
    # ------------------------------------------------------------------

    def _compute_uncertainty(self, h):
        h_np = np.asarray(h, dtype=np.float64)
        stds = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            var = h_np @ self._U_inv[i] @ h_np
            stds[i] = np.sqrt(max(var, 0.0))
        return stds

    def _update_uncertainty(self, h, obs_mask):
        h_np = np.asarray(h, dtype=np.float64)
        obs_np = np.asarray(obs_mask)
        for i in range(self.n_arms):
            if obs_np[i]:
                Uh = self._U_inv[i] @ h_np
                self._U_inv[i] -= np.outer(Uh, Uh) / (1.0 + h_np @ Uh)

    # ------------------------------------------------------------------
    # Parallel MSE imputer (per-arm OLS on raw augmented context)
    # ------------------------------------------------------------------

    def _aug(self, x):
        x_np = np.asarray(x, dtype=np.float64).flatten()
        return np.concatenate([x_np, [1.0]])

    def _predict_mse(self, x):
        x_aug = self._aug(x)
        means = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            w_i = self.M_mse_inv[i] @ self.b_mse_raw[i]
            means[i] = w_i @ x_aug
        return means

    def _update_mse_imputer(self, x, true_costs, obs_mask):
        x_aug = self._aug(x)
        true_np = np.asarray(true_costs, dtype=np.float64)
        obs_np = np.asarray(obs_mask)
        for i in range(self.n_arms):
            if obs_np[i]:
                Mx = self.M_mse_inv[i] @ x_aug
                self.M_mse_inv[i] -= np.outer(Mx, Mx) / (1.0 + x_aug @ Mx)
                self.b_mse_raw[i] += x_aug * true_np[i]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, X):
        self.t += 1
        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h = self._features(X)
        means = np.asarray(self._predict(h))
        stds = self._compute_uncertainty(h)
        if self.minimize:
            values = means - self.beta * stds
        else:
            values = means + self.beta * stds
        return self.solver(values)

    # ------------------------------------------------------------------
    # SPO+ training (chain rule through network, no Bayesian posterior)
    # ------------------------------------------------------------------

    def update(self, X, selection, rewards, lr=None, full_feedback: bool = False,
               prob_values=None):
        X_j = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h = self._features(X_j)
        pred = self._predict(h)
        selection_mask = selection > 0.5

        # Track LinUCB design matrices on features (observed arms only)
        if full_feedback:
            self._update_uncertainty(h, np.ones(self.n_arms, dtype=bool))
        else:
            self._update_uncertainty(h, selection_mask)

        # Update the parallel MSE imputer (closed-form, never uses solver)
        self._update_mse_imputer(X, rewards, selection_mask)

        # Imputation target for SPO+ gradient: parallel MSE model on unobserved
        if prob_values is not None:
            Y_target = jnp.asarray(prob_values)
        elif full_feedback:
            Y_target = rewards
        else:
            mse_pred = jnp.asarray(self._predict_mse(X))
            Y_target = jnp.where(selection_mask, rewards, mse_pred)

        # SPO+ per-arm loss gradient
        spo_grad_val = self.loss_grad_fn(pred, Y_target)
        if self.mask_grad and not full_feedback:
            spo_grad_val = jnp.where(selection_mask, spo_grad_val, 0.0)

        # Chain rule through prediction head:
        #   grad_h = Σ_n spo_grad[n] · W_pred[n]
        #   grad_W_pred[n] = spo_grad[n] · h
        grad_h = jnp.einsum('n,ni->i', spo_grad_val, self.W_pred)
        grad_W_pred = jnp.outer(spo_grad_val, h)

        # Backprop grad_h through the feature extractor
        X_aug = X_j
        if self.nonlinear:
            z = X_aug @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            grad_a = grad_h @ self.W2.T
            grad_z = grad_a * (z > 0)
            grads = {
                'W1': jnp.outer(X_aug, grad_z), 'b1': grad_z,
                'W2': jnp.outer(a, grad_h), 'b2': grad_h,
                'W_pred': grad_W_pred,
            }
        else:
            grads = {
                'W1': jnp.outer(X_aug, grad_h), 'b1': grad_h,
                'W_pred': grad_W_pred,
            }

        params = self._get_params()
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        self._set_params(optax.apply_updates(params, updates))

"""SPO+ agent with point-estimate head + LinUCB uncertainty + own-prediction imputation.

This is the "bridge" variant between SPO_EmpUCB_Agent and SPO_ParallelMSE_Agent,
isolating the effect of the uncertainty source.

Architecture:
  - Feature extractor: linear h = W1 x + b1 (optional MLP via nonlinear=True).
  - Prediction head: point-estimate W_pred (no Bayesian posterior).
    pred_i = W_pred[i] @ h
    Both W_pred and features are trained via SPO+ gradient (chain rule).
  - Uncertainty: LinUCB-style per-arm feature design matrices
    U_i = Σ_t I[arm i observed] h_t h_t^T + λI,
    sigma_i(x) = sqrt(h(x)^T U_i^{-1} h(x)).
  - Imputation: own predictions (no parallel MSE) — unobserved arms are
    imputed with W_pred @ h, same as BayesianSPOAgent.

Compared to:
  - SPO_EmpUCB_Agent: same predictions + imputation, different uncertainty
    (UCB1 count-based vs LinUCB feature-based).
  - SPO_ParallelMSE_Agent: same predictions + uncertainty, different
    imputation (own predictions vs parallel per-arm raw-x OLS).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.losses import make_spo_grad_fn


class SPO_LinUCB_Agent:
    """SPO+ with point-estimate head, LinUCB feature uncertainty, own-prediction imputation."""

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

        # Point-estimate prediction head
        self.W_pred = jnp.zeros((n_arms, hidden_dim))

        init_params = self._get_params()

        def _lr_schedule(count):
            return lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(_lr_schedule),
        )
        self.opt_state = self.optimizer.init(init_params)

        # LinUCB-style per-arm design matrices on features
        eye = np.eye(hidden_dim)
        self._U_inv = np.stack([eye / lambda_reg for _ in range(n_arms)], axis=0)

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
    # Network forward + uncertainty
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
    # SPO+ training
    # ------------------------------------------------------------------

    def update(self, X, selection, rewards, lr=None, full_feedback: bool = False,
               prob_values=None):
        X_j = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h = self._features(X_j)
        pred = self._predict(h)
        selection_mask = selection > 0.5

        # Update per-arm feature design matrices (for LinUCB uncertainty)
        if full_feedback:
            self._update_uncertainty(h, np.ones(self.n_arms, dtype=bool))
        else:
            self._update_uncertainty(h, selection_mask)

        # Imputation: own predictions (same as BayesianSPOAgent)
        if prob_values is not None:
            Y_target = jnp.asarray(prob_values)
        elif full_feedback:
            Y_target = rewards
        else:
            Y_target = jnp.where(selection_mask, rewards, pred)

        # SPO+ per-arm loss gradient
        spo_grad_val = self.loss_grad_fn(pred, Y_target)
        if self.mask_grad and not full_feedback:
            spo_grad_val = jnp.where(selection_mask, spo_grad_val, 0.0)

        # Chain rule
        grad_h = jnp.einsum('n,ni->i', spo_grad_val, self.W_pred)
        grad_W_pred = jnp.outer(spo_grad_val, h)

        # Backprop grad_h through features
        if self.nonlinear:
            z = X_j @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            grad_a = grad_h @ self.W2.T
            grad_z = grad_a * (z > 0)
            grads = {
                'W1': jnp.outer(X_j, grad_z), 'b1': grad_z,
                'W2': jnp.outer(a, grad_h), 'b2': grad_h,
                'W_pred': grad_W_pred,
            }
        else:
            grads = {
                'W1': jnp.outer(X_j, grad_h), 'b1': grad_h,
                'W_pred': grad_W_pred,
            }

        params = self._get_params()
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        self._set_params(optax.apply_updates(params, updates))

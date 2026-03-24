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

    Supports two marginalization modes (``marginalization`` parameter):

    - ``'mc'`` (default): Full MC marginalization — sample theta from the
      posterior and average the SPO+ gradient across samples.
    - ``'hybrid'``: Closed-form hybrid — uses the SPO+ gradient at the
      posterior mean for the signal term, plus Weber's closed-form
      uncertainty correction ``sum_n Sigma_n @ h``.  No MC sampling,
      so it is faster and deterministic.
    - ``'selective'``: SPO+ on observed arms (where we have ground-truth
      feedback) + Weber uncertainty correction on unobserved arms only.
      Avoids using SPO+ on imputed targets; the uncertainty term drives
      feature learning toward reducing posterior uncertainty on unseen arms.
    - ``'interpolated'``: Uses Weber's full marginalized gradient structure
      (signal + Sigma_n h) but replaces the residual with an interpolation:
      ``alpha * spo_grad + (1-alpha) * mse_grad`` where alpha = spo_weight.
      MSE provides continuous base signal; SPO+ steers toward decisions.
    - ``'perturbed_fy'``: Perturbed Fenchel-Young loss (Berthet et al. 2020).
      Smooths the SPO+ gradient by averaging solver(2*pred - y + eps) over
      N Gaussian perturbations.  Gives continuous, decision-focused gradients
      that combine naturally with Weber's uncertainty correction.
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
        uncertainty_weight: float = 1.0,
        spo_weight: float = 0.5,
        n_perturbations: int = 10,
        perturbation_sigma: float = 0.5,
        marginalization: str = 'mc',
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
        self.uncertainty_weight = uncertainty_weight
        self.spo_weight = spo_weight
        self.n_perturbations = n_perturbations
        self.perturbation_sigma = perturbation_sigma
        self.marginalization = marginalization

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

    def _hybrid_weber_gradient(self, h, Y_target, selection_mask, full_feedback):
        """Closed-form hybrid: SPO+ signal at posterior mean + Weber Σh correction.

        Term 1 (signal):      sum_n  spo_grad(mu^T h, y)_n  *  mu_n
        Term 2 (uncertainty):  sum_n  Sigma_n @ h

        Term 1 is identical to standard BayesianSPO's feature gradient.
        Term 2 is Weber's closed-form uncertainty correction that pushes
        features toward directions where the posterior is uncertain.
        """
        mu = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)  # (n_arms, hidden_dim)
        pred = mu @ h  # (n_arms,)

        # SPO+ gradient at posterior mean predictions
        spo_grad = self.loss_grad_fn(pred, Y_target)
        if self.mask_grad and not full_feedback:
            spo_grad = jnp.where(selection_mask, spo_grad, 0.0)

        # Term 1: SPO+ signal through posterior mean
        signal_term = jnp.einsum('n,ni->i', spo_grad, mu)

        # Term 2: Weber uncertainty correction  sum_n Sigma_n @ h
        sigma_h = jnp.einsum('nij,j->ni', self.A_inv, h)  # (n_arms, hidden_dim)
        if self.mask_grad and not full_feedback:
            sigma_h = jnp.where(selection_mask[:, None], sigma_h, 0.0)
        uncertainty_term = jnp.sum(sigma_h, axis=0)

        return signal_term + self.uncertainty_weight * uncertainty_term

    def _selective_weber_gradient(self, h, Y_target, selection_mask, full_feedback):
        """SPO+ on observed arms, Weber uncertainty on unobserved.

        Observed arms have ground-truth feedback, so the SPO+ gradient is
        meaningful.  Unobserved arms only have imputed (model-mean) targets,
        so instead of a noisy SPO+ signal we use Weber's closed-form
        uncertainty correction Sigma_n @ h, which directly pushes features
        toward directions that reduce posterior uncertainty on those arms.
        """
        mu = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        pred = mu @ h

        # SPO+ gradient — only on observed arms
        spo_grad = self.loss_grad_fn(pred, Y_target)
        if not full_feedback:
            spo_grad = jnp.where(selection_mask, spo_grad, 0.0)
        signal_term = jnp.einsum('n,ni->i', spo_grad, mu)

        # Weber uncertainty correction — only on unobserved arms
        sigma_h = jnp.einsum('nij,j->ni', self.A_inv, h)
        if full_feedback:
            # No unobserved arms
            uncertainty_term = jnp.zeros(self.hidden_dim)
        else:
            sigma_h_unobs = jnp.where(selection_mask[:, None], 0.0, sigma_h)
            uncertainty_term = jnp.sum(sigma_h_unobs, axis=0)

        return signal_term + self.uncertainty_weight * uncertainty_term

    def _interpolated_weber_gradient(self, h, Y_target, selection_mask, full_feedback):
        """Interpolated SPO-MSE with Weber's marginalized gradient structure.

        Uses Weber's closed-form marginalized gradient (MSE term + Sigma_n h)
        as the backbone, but replaces the pure MSE residual with a blend:

            residual_n = alpha * spo_grad_n + (1 - alpha) * mse_grad_n

        where alpha = self.spo_weight.  This gives:
        - Continuous, always-nonzero gradients (from MSE) that work with phasing
        - Decision focus (from SPO+) that steers learning toward flip-set arms
        - Weber's uncertainty correction (Sigma_n h) that prevents feature collapse
        """
        mu = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        pred = mu @ h

        # MSE gradient: 2 * (pred - y) / n_arms  (always nonzero, continuous)
        mse_grad = 2.0 * (pred - Y_target) / self.n_arms

        # SPO+ gradient: piecewise constant, decision-focused
        spo_grad = self.loss_grad_fn(pred, Y_target)

        # Interpolate
        alpha = self.spo_weight
        blended_grad = alpha * spo_grad + (1.0 - alpha) * mse_grad

        # Mask unobserved if requested
        if self.mask_grad and not full_feedback:
            blended_grad = jnp.where(selection_mask, blended_grad, 0.0)

        # Term 1: blended signal through posterior mean (Weber Eq. 5, modified)
        signal_term = jnp.einsum('n,ni->i', blended_grad, mu)

        # Term 2: Weber uncertainty correction  sum_n Sigma_n @ h
        sigma_h = jnp.einsum('nij,j->ni', self.A_inv, h)
        if self.mask_grad and not full_feedback:
            sigma_h = jnp.where(selection_mask[:, None], sigma_h, 0.0)
        uncertainty_term = jnp.sum(sigma_h, axis=0)

        return signal_term + self.uncertainty_weight * uncertainty_term

    def _perturbed_fy_gradient(self, h, Y_target, selection_mask, full_feedback):
        """Perturbed Fenchel-Young loss gradient + Weber uncertainty correction.

        Replaces SPO+'s piecewise-constant gradient with a smooth approximation:
        instead of solving w*(2*pred - y) once, we solve w*(2*pred - y + eps)
        for N perturbations eps ~ N(0, sigma^2 I) and average.  This gives a
        continuous gradient that is still decision-focused (uses the solver).

        Reference: Berthet et al. (2020), "Learning with Differentiable
        Perturbed Optimizers".
        """
        mu = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        pred = mu @ h

        spo_input = np.asarray(2.0 * pred - Y_target, dtype=np.float64)
        y_true_np = np.asarray(Y_target, dtype=np.float64)
        w_true = np.asarray(self.solver(y_true_np), dtype=np.float64)

        # Average solver output over perturbations
        avg_sol = np.zeros(self.n_arms, dtype=np.float64)
        for _ in range(self.n_perturbations):
            self.rng_key, subkey = jax.random.split(self.rng_key)
            eps = self.perturbation_sigma * np.asarray(
                jax.random.normal(subkey, (self.n_arms,)))
            avg_sol += np.asarray(self.solver(spo_input + eps), dtype=np.float64)
        avg_sol /= self.n_perturbations

        # Smooth per-arm gradient (same structure as SPO+ but continuous)
        if self.minimize:
            fy_grad = jnp.asarray(2.0 * (w_true - avg_sol) / self.n_arms)
        else:
            fy_grad = jnp.asarray(2.0 * (avg_sol - w_true) / self.n_arms)

        if self.mask_grad and not full_feedback:
            fy_grad = jnp.where(selection_mask, fy_grad, 0.0)

        # Term 1: smooth decision-focused signal through posterior mean
        signal_term = jnp.einsum('n,ni->i', fy_grad, mu)

        # Term 2: Weber uncertainty correction
        sigma_h = jnp.einsum('nij,j->ni', self.A_inv, h)
        if self.mask_grad and not full_feedback:
            sigma_h = jnp.where(selection_mask[:, None], sigma_h, 0.0)
        uncertainty_term = jnp.sum(sigma_h, axis=0)

        return signal_term + self.uncertainty_weight * uncertainty_term

    def _compute_weber_gradient(self, h, Y_target, selection_mask, full_feedback):
        """Dispatch to the selected marginalization mode."""
        if self.marginalization == 'hybrid':
            return self._hybrid_weber_gradient(h, Y_target, selection_mask, full_feedback)
        elif self.marginalization == 'selective':
            return self._selective_weber_gradient(h, Y_target, selection_mask, full_feedback)
        elif self.marginalization == 'interpolated':
            return self._interpolated_weber_gradient(h, Y_target, selection_mask, full_feedback)
        elif self.marginalization == 'perturbed_fy':
            return self._perturbed_fy_gradient(h, Y_target, selection_mask, full_feedback)
        return self._mc_marginalized_gradient(h, Y_target, selection_mask, full_feedback)

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
                grad_h = self._compute_weber_gradient(h, Y_target, sel_mask, False)
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
            # Per-round: one Weber gradient step (MC or hybrid)
            grad_h = self._compute_weber_gradient(h, Y_target, selection_mask, full_feedback)
            grads = self._backprop_features(X, grad_h)
            self._apply_gradient_step(grads)

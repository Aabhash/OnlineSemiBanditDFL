import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.losses import spo_plus_grad
from src.agents.bayesian_utils import (
    init_shared_regression,
    compute_shared_ucb_values,
    compute_shared_predictions,
    update_shared_bayesian_stats,
)


class PerArmBayesianIERMAgent:

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
        alpha: float = 0.0,
        lambda_reg: float = 1.0,
        score_type: str = 'dm',
        cov_decay: float = 0.99,
        lr: float = 0.01,
        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver

        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.score_type = score_type.lower()
        self.cov_decay = cov_decay
        self.decay_rounds = decay_rounds
        self.t = 0

        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        key = jax.random.PRNGKey(seed if seed else 0)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)

        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=schedule)
            )
        else:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr)
            )
        self.opt_state = self.optimizer.init({'W1': self.W1, 'b1': self.b1})

        self.A, self.b_bayes, self.A_inv = init_shared_regression(hidden_dim, lambda_reg)
        self.selection_cov = lambda_reg * np.eye(n_arms)

    def _features(self, X):
        X = jnp.asarray(X)
        Z1 = X @ self.W1 + self.b1
        H = jnp.maximum(0, Z1)
        return H, Z1

    def predict_mean(self, X):
        H, _ = self._features(X)
        return compute_shared_predictions(H, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        H, _ = self._features(X)
        ucb_values = compute_shared_ucb_values(H, self.A_inv, self.b_bayes, self.beta)
        return self.solver(np.asarray(ucb_values))

    def _compute_score_dm(self, predictions, selection, rewards):
        return jnp.where(selection > 0.5, rewards, predictions)

    def _compute_score_isw(self, predictions, selection, rewards):
        selection_np = np.asarray(selection)
        rewards_np = np.asarray(rewards)
        predictions_np = np.asarray(predictions)
        selected_mask = selection_np > 0.5

        cov_reg = self.selection_cov + self.lambda_reg * np.eye(self.n_arms)
        try:
            cov_inv = np.linalg.pinv(cov_reg)
        except:
            cov_inv = np.eye(self.n_arms) / self.lambda_reg

        observed_cost = np.sum(selection_np * rewards_np)
        isw_estimate = cov_inv @ selection_np * observed_cost

        score = np.where(selected_mask, rewards_np, isw_estimate)
        return jnp.asarray(score)

    def _compute_score_dr(self, predictions, selection, rewards):
        selection_np = np.asarray(selection)
        rewards_np = np.asarray(rewards)
        predictions_np = np.asarray(predictions)
        selected_mask = selection_np > 0.5

        cov_reg = self.selection_cov + self.lambda_reg * np.eye(self.n_arms)
        try:
            cov_inv = np.linalg.pinv(cov_reg)
        except:
            cov_inv = np.eye(self.n_arms) / self.lambda_reg

        observed_cost = np.sum(selection_np * rewards_np)
        predicted_cost = np.sum(selection_np * predictions_np)
        cost_residual = observed_cost - predicted_cost

        correction = cov_inv @ selection_np * cost_residual
        dr_estimate = predictions_np + correction

        score = np.where(selected_mask, rewards_np, dr_estimate)
        return jnp.asarray(score)

    def _compute_score(self, predictions, selection, rewards):
        if self.score_type == 'dm':
            return self._compute_score_dm(predictions, selection, rewards)
        elif self.score_type == 'isw':
            return self._compute_score_isw(predictions, selection, rewards)
        elif self.score_type == 'dr':
            return self._compute_score_dr(predictions, selection, rewards)
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

    def _update_selection_covariance(self, selection):
        selection_np = np.asarray(selection)
        outer = np.outer(selection_np, selection_np)
        self.selection_cov = self.cov_decay * self.selection_cov + (1 - self.cov_decay) * outer

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        H, Z1 = self._features(X)

        bayes_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_shared_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, H, rewards, bayes_mask
        )

        pred = compute_shared_predictions(H, self.A_inv, self.b_bayes)

        if full_feedback:
            Y_target = rewards
        else:
            self._update_selection_covariance(selection)
            Y_target = self._compute_score(pred, selection, rewards)

        spo_grad_val = spo_plus_grad(pred, Y_target, self.weights, self.capacity)

        if self.alpha > 0:
            mse_grad = 2 * (pred - Y_target) / self.n_arms
            hybrid_grad = self.alpha * mse_grad + (1 - self.alpha) * spo_grad_val
        else:
            hybrid_grad = spo_grad_val

        theta = self.A_inv @ self.b_bayes
        grad_H = jnp.outer(hybrid_grad, theta)
        grad_Z1 = grad_H * (Z1 > 0)
        grad_W1 = X.T @ grad_Z1
        grad_b1 = jnp.sum(grad_Z1, axis=0)

        grads = {'W1': grad_W1, 'b1': grad_b1}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']


class PerArmBayesianIERMDMAgent(PerArmBayesianIERMAgent):
    def __init__(self, *args, **kwargs):
        kwargs['score_type'] = 'dm'
        super().__init__(*args, **kwargs)


class PerArmBayesianIERMDRAgent(PerArmBayesianIERMAgent):
    def __init__(self, *args, **kwargs):
        kwargs['score_type'] = 'dr'
        super().__init__(*args, **kwargs)

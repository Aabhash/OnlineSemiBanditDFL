import jax.numpy as jnp
import numpy as np
from src.models import MLPModel
from src.solver import solve_knapsack_with_sensitivity
from src.losses import spo_plus_grad
from src.agents.bayesian_utils import (
    mlp_forward,
    init_per_arm_regression,
    compute_bayesian_predictions,
    compute_uncertainties,
    update_bayesian_stats,
)


class DecisionUCBAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta_pred_start: float = 2.0,
        beta_pred_end: float = 0.1,
        beta_decision: float = 1.0,
        decay_rounds: int = 300,
        alpha: float = 0.0,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.hidden_dim = hidden_dim

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver

        self.beta_pred_start = beta_pred_start
        self.beta_pred_end = beta_pred_end
        self.beta_pred = beta_pred_start
        self.beta_decision = beta_decision
        self.decay_rounds = decay_rounds
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.t = 0

        self.model = MLPModel(
            context_dim, n_arms, hidden_dim,
            lr=lr, decay_steps=decay_rounds, seed=seed if seed else 0
        )

        self.A, self.b, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def predict_mean(self, X):
        X = jnp.asarray(X)
        _, h, _ = mlp_forward(X, self.model.W1, self.model.b1, self.model.W2, self.model.b2)
        return compute_bayesian_predictions(h, self.A_inv, self.b)

    def act(self, X):
        self.t += 1
        progress = min(self.t / self.decay_rounds, 1.0)
        self.beta_pred = self.beta_pred_start + (self.beta_pred_end - self.beta_pred_start) * progress

        X = jnp.asarray(X)
        _, h, _ = mlp_forward(X, self.model.W1, self.model.b1, self.model.W2, self.model.b2)

        pred_values = compute_bayesian_predictions(h, self.A_inv, self.b)
        pred_uncertainty = compute_uncertainties(h, self.A_inv)

        sensitivity_info = solve_knapsack_with_sensitivity(
            np.asarray(pred_values), self.weights_np, self.capacity
        )
        decision_sensitivity = jnp.asarray(sensitivity_info["decision_relevance"])

        ucb_values = pred_values + self.beta_pred * pred_uncertainty + self.beta_decision * decision_sensitivity

        return self.solver(np.asarray(ucb_values))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        _, h, z1 = mlp_forward(X, self.model.W1, self.model.b1, self.model.W2, self.model.b2)

        selection_mask = selection > 0.5
        self.A, self.b, self.A_inv = update_bayesian_stats(
            self.A, self.b, self.A_inv, h, rewards, selection_mask
        )

        if full_feedback:
            Y_target = rewards
        else:
            pred_values = compute_bayesian_predictions(h, self.A_inv, self.b)
            pred_uncertainty = compute_uncertainties(h, self.A_inv)

            sensitivity_info = solve_knapsack_with_sensitivity(
                np.asarray(pred_values), self.weights_np, self.capacity
            )
            decision_sensitivity = jnp.asarray(sensitivity_info["decision_relevance"])

            ucb_values = pred_values + self.beta_pred * pred_uncertainty + self.beta_decision * decision_sensitivity
            Y_target = jnp.where(selection > 0.5, rewards, ucb_values)

        Y_pred = compute_bayesian_predictions(h, self.A_inv, self.b)
        mse_grad = 2 * (Y_pred - Y_target) / self.n_arms
        spo_grad_val = spo_plus_grad(Y_pred, Y_target, self.weights, self.capacity)
        hybrid_grad = self.alpha * mse_grad + (1 - self.alpha) * spo_grad_val

        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b)
        grad_h = jnp.einsum('n,ni->i', hybrid_grad, theta)

        grad_z1 = grad_h * (z1 > 0)

        grads = {
            'W1': jnp.outer(X, grad_z1),
            'b1': grad_z1,
            'W2': jnp.zeros_like(self.model.W2),
            'b2': jnp.zeros_like(self.model.b2),
        }
        self.model.update(grads, lr)

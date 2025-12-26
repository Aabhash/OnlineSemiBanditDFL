import jax.numpy as jnp
import numpy as np
from src.models import MLPModel
from src.losses import spo_plus_grad
from src.agents.utils.mlp import mlp_forward, mlp_backward
from src.agents.utils.bayesian import (
    init_per_arm_regression,
    compute_uncertainties,
    update_bayesian_stats,
)


class FullSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta_start: float = 2.0,
        beta_end: float = 0.1,
        decay_rounds: int = 300,
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
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.decay_rounds = decay_rounds
        self.beta = beta_start
        self.lambda_reg = lambda_reg
        self.t = 0

        self.model = MLPModel(
            context_dim, n_arms, hidden_dim,
            lr=lr, decay_steps=decay_rounds, seed=seed if seed else 0
        )

        self.A, self.b, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def predict(self, X):
        X = jnp.asarray(X)
        out, _, _ = mlp_forward(X, self.model.W1, self.model.b1, self.model.W2, self.model.b2)
        return out

    def act(self, X):
        self.t += 1
        progress = min(self.t / self.decay_rounds, 1.0)
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        X = jnp.asarray(X)
        pred, h, _ = mlp_forward(X, self.model.W1, self.model.b1, self.model.W2, self.model.b2)

        stds = compute_uncertainties(h, self.A_inv)
        ucb_values = pred + self.beta * stds

        return self.solver(np.asarray(ucb_values))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        pred, h, z1 = mlp_forward(X, self.model.W1, self.model.b1, self.model.W2, self.model.b2)

        mask = jnp.ones(self.n_arms, dtype=bool) if full_feedback else (selection > 0.5)
        self.A, self.b, self.A_inv = update_bayesian_stats(
            self.A, self.b, self.A_inv, h, rewards, mask
        )

        if full_feedback:
            Y_target = rewards
        else:
            Y_target = jnp.where(selection > 0.5, rewards, pred)

        spo_grad_val = spo_plus_grad(pred, Y_target, self.weights, self.capacity)
        grads = mlp_backward(spo_grad_val, h, z1, X, self.model.W2)
        self.model.update(grads, lr)

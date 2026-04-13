import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Union
from src.models import LinearModel, MLPModel
from src.losses import spo_plus_grad
from src.agents.agent_base import AgentBase
from src.solver import solve_knapsack


class GreedyAgent(AgentBase):

    def __init__(
        self,
        model: Union[LinearModel, MLPModel],
        weights: np.ndarray,
        capacity: float,
        loss_type: str = "mse",
        solver=None,
    ):
        self.model = model
        self.weights = weights
        self.capacity = capacity
        self.loss_type = loss_type
        self.n_arms = len(weights)

        if solver is None:
            self.solver = lambda v: solve_knapsack(np.asarray(v), weights, capacity)
        else:
            self.solver = solver

    def act(self, X: np.ndarray) -> np.ndarray:
        Y_pred = self.model.predict(X)
        return self.solver(np.asarray(Y_pred))

    def predict_mean(self, X: np.ndarray):
        return self.model.predict(X)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False, **kwargs):

        if full_feedback:
            Y_target = np.asarray(rewards)
        else:
            Y_pred = np.asarray(self.model.predict(X))
            Y_target = Y_pred.copy()
            mask = np.asarray(selection) > 0.5
            Y_target[mask] = np.asarray(rewards)[mask]

        if self.loss_type == "mse":
            grad = self.model.gradient_mse(X, Y_target)
        elif self.loss_type == "spo":
            Y_pred = self.model.predict(X)
            spo_grad = spo_plus_grad(Y_pred, Y_target, self.weights, self.capacity)
            grad = self.model.gradient_spo(X, spo_grad)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        self.model.update(grad, lr if lr is not None else 0.01)


class EpsilonGreedyAgent(AgentBase):

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        hidden_dim: int = 32,
        epsilon: float = 0.1,
        decay_epsilon: bool = True,
        decay_rounds: int = 300,
        lr: float = 0.001,

        max_grad_norm: float = 5.0,
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
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.epsilon = epsilon
        self.decay_epsilon = decay_epsilon
        self.decay_rounds = decay_rounds
        self.t = 0

        self.rng = np.random.default_rng(seed if seed else 0)

        key = jax.random.PRNGKey(seed if seed else 0)
        k1, k2 = jax.random.split(key)
        self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)
        self.W2 = jax.random.normal(k2, (hidden_dim, n_arms)) * 0.1
        self.b2 = jnp.zeros(n_arms)


        schedule = lambda count: lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule)
        )
        self.opt_state = self.optimizer.init({
            'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2
        })

    def _forward(self, X):
        X = jnp.asarray(X)
        z1 = X @ self.W1 + self.b1
        h = jnp.maximum(0, z1)
        return h @ self.W2 + self.b2, z1, h

    def _random_knapsack(self):
        order = self.rng.permutation(self.n_arms)
        selection = np.zeros(self.n_arms)
        remaining = self.capacity

        for i in order:
            if self.weights_np[i] <= remaining:
                selection[i] = 1
                remaining -= self.weights_np[i]

        return selection

    def act(self, X: np.ndarray) -> np.ndarray:
        self.t += 1

        if self.decay_epsilon:
            current_eps = self.epsilon * max(0, 1 - self.t / self.decay_rounds)
        else:
            current_eps = self.epsilon

        if self.rng.random() < current_eps:
            return self._random_knapsack()
        else:
            pred, _, _ = self._forward(X)
            return self.solver(np.asarray(pred))

    def predict_mean(self, X: np.ndarray):
        pred, _, _ = self._forward(X)
        return pred

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False, **kwargs):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        pred, z1, h = self._forward(X)

        if full_feedback:
            mask = jnp.ones(self.n_arms)
            Y_target = rewards
        else:
            mask = (selection > 0.5).astype(float)
            Y_target = jnp.where(selection > 0.5, rewards, pred)

        error = (pred - Y_target) * mask
        n_obs = jnp.maximum(jnp.sum(mask), 1.0)
        grad_out = 2 * error / n_obs

        grad_W2 = jnp.outer(h, grad_out)
        grad_b2 = grad_out
        grad_h = grad_out @ self.W2.T
        grad_z1 = grad_h * (z1 > 0)
        grad_W1 = jnp.outer(X, grad_z1)
        grad_b1 = grad_z1

        grads = {'W1': grad_W1, 'b1': grad_b1, 'W2': grad_W2, 'b2': grad_b2}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state,
            {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        )
        params = optax.apply_updates(
            {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}, updates
        )
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']
        self.W2 = jnp.clip(params['W2'], -10, 10)
        self.b2 = params['b2']

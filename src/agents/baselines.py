import jax.numpy as jnp
import numpy as np
from typing import Union
from src.models import LinearModel, MLPModel
from src.solver import solve_knapsack
from src.losses import spo_plus_grad


class RandomAgent:

    def __init__(
        self,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        seed: int = None,
    ):
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

        self.reward_sum = jnp.zeros(n_arms)
        self.reward_count = jnp.zeros(n_arms)

    def act(self, X: np.ndarray) -> np.ndarray:
        order = self.rng.permutation(self.n_arms)
        selection = np.zeros(self.n_arms)
        remaining = self.capacity

        for i in order:
            if self.weights[i] <= remaining:
                selection[i] = 1
                remaining -= self.weights[i]

        return selection

    def predict(self, X: np.ndarray) -> jnp.ndarray:
        pred = jnp.zeros(self.n_arms)
        mask = self.reward_count > 0
        pred = jnp.where(mask, self.reward_sum / jnp.maximum(self.reward_count, 1), pred)
        return pred

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        rewards = jnp.asarray(rewards)
        selection = jnp.asarray(selection)

        if full_feedback:
            self.reward_sum = self.reward_sum + rewards
            self.reward_count = self.reward_count + 1
        else:
            selected = selection > 0.5
            self.reward_sum = jnp.where(selected, self.reward_sum + rewards, self.reward_sum)
            self.reward_count = jnp.where(selected, self.reward_count + 1, self.reward_count)


class GreedyAgent:

    def __init__(
        self,
        model: Union[LinearModel, MLPModel],
        weights: np.ndarray,
        capacity: float,
        loss_type: str = "mse",
    ):
        self.model = model
        self.weights = weights
        self.capacity = capacity
        self.loss_type = loss_type
        self.n_arms = len(weights)

    def act(self, X: np.ndarray) -> np.ndarray:
        Y_pred = self.model.predict(X)
        return solve_knapsack(np.asarray(Y_pred), self.weights, self.capacity)

    def update(self, X: np.ndarray, Y_true: np.ndarray, lr: float):
        if self.loss_type == "mse":
            grad = self.model.gradient_mse(X, Y_true)
        elif self.loss_type == "spo":
            Y_pred = self.model.predict(X)
            spo_grad = spo_plus_grad(Y_pred, Y_true, self.weights, self.capacity)
            grad = self.model.gradient_spo(X, spo_grad)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        self.model.update(grad, lr)

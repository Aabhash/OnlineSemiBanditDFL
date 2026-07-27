import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional
from src.agents.agent_base import AgentBase


class RandomAgent(AgentBase):

    def __init__(
        self,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        seed: int = None,
        random_action_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.n_arms = n_arms
        self.weights = weights
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.random_action_fn = random_action_fn

        self.reward_sum = jnp.zeros(n_arms)
        self.reward_count = jnp.zeros(n_arms)

    def act(self, X: np.ndarray) -> np.ndarray:
        if self.random_action_fn is not None:
            priorities = self.rng.standard_normal(self.n_arms)
            return np.asarray(self.random_action_fn(priorities), dtype=float)

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

    def predict_mean(self, X: np.ndarray) -> jnp.ndarray:
        return self.predict(X)

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               lr: float = None, full_feedback: bool = False, **kwargs):
        rewards = jnp.asarray(rewards)
        selection = jnp.asarray(selection)

        if full_feedback:
            self.reward_sum = self.reward_sum + rewards
            self.reward_count = self.reward_count + 1
        else:
            selected = selection > 0.5
            self.reward_sum = jnp.where(selected, self.reward_sum + rewards, self.reward_sum)
            self.reward_count = jnp.where(selected, self.reward_count + 1, self.reward_count)

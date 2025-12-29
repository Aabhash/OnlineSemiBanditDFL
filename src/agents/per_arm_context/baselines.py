import numpy as np


class PerArmRandomAgent:

    def __init__(
        self,
        n_arms: int,
        weights: np.ndarray,
        capacity: float,
        seed: int = None,
    ):
        self.n_arms = n_arms
        self.weights = np.asarray(weights)
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

    def act(self, X: np.ndarray) -> np.ndarray:
        order = self.rng.permutation(self.n_arms)
        selection = np.zeros(self.n_arms)
        remaining = self.capacity

        for i in order:
            if self.weights[i] <= remaining:
                selection[i] = 1
                remaining -= self.weights[i]

        return selection

    def update(self, X: np.ndarray, selection: np.ndarray, rewards: np.ndarray,
               full_feedback: bool = False):
        pass

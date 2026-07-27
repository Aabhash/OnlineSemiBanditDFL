import numpy as np
from typing import Tuple


class KnapsackData:
    def __init__(
        self,
        n_items: int = 32,
        n_features: int = None,
        context_dim: int = None,
        capacity: float = 20.0,
        deg: int = 4,
        noise_width: float = 0.5,
        seed: int = 42,
        unit_weights: bool = False,
        **kwargs,
    ):
        self.n_items = n_items
        if context_dim is not None:
            self.n_features = context_dim
        elif n_features is not None:
            self.n_features = n_features
        else:
            self.n_features = 5
        self.context_dim = self.n_features
        self.capacity = capacity
        self.deg = deg
        self.noise_width = noise_width
        self.rng = np.random.default_rng(seed)


        self.B = self.rng.integers(0, 2, size=(n_items, self.n_features)).astype(float)
        if unit_weights:
            self.weights = np.ones(n_items)
        else:
            self.weights = self.rng.uniform(1.0, 5.0, n_items)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        context = self.rng.standard_normal(self.n_features)
        values = self._generate_values(context)
        return context, values

    def conditional_mean(self, context: np.ndarray) -> np.ndarray:
        linear = self.B @ context / np.sqrt(self.n_features)
        mean_values = (linear + 3) ** self.deg + 1
        return mean_values * 5 / (3.5 ** self.deg)

    def step(self):
        pass

    def _generate_values(self, x: np.ndarray) -> np.ndarray:
        mean_values = self.conditional_mean(x)

        noise = self.rng.uniform(
            1 - self.noise_width,
            1 + self.noise_width,
            self.n_items
        )
        values = mean_values * noise

        return values

    def get_config(self) -> dict:
        return {
            'n_items': self.n_items,
            'n_features': self.n_features,
            'context_dim': self.context_dim,
            'capacity': self.capacity,
            'deg': self.deg,
            'noise_width': self.noise_width,
            'weights': self.weights.copy(),
        }

    def reset(self):
        pass

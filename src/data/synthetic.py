import numpy as np
from typing import Tuple


class SyntheticKnapsackData:

    def __init__(
        self,
        n_items: int = 32,
        n_features: int = None,
        context_dim: int = None,
        capacity: float = 20.0,
        deg: int = 4,
        noise_width: float = 0.5,
        seed: int = 42,
    ):
        self.n_items = n_items
        # Support both n_features and context_dim (context_dim takes precedence)
        if context_dim is not None:
            self.n_features = context_dim
        elif n_features is not None:
            self.n_features = n_features
        else:
            self.n_features = 5  # default
        self.context_dim = self.n_features  # alias for consistency
        self.capacity = capacity
        self.deg = deg
        self.noise_width = noise_width
        self.rng = np.random.default_rng(seed)

        self.B = self.rng.integers(0, 2, size=(n_items, self.n_features)).astype(float)
        self.weights = self.rng.uniform(1.0, 5.0, n_items)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        context = self.rng.standard_normal(self.n_features)
        values = self._generate_values(context)
        return context, values

    def _generate_values(self, x: np.ndarray) -> np.ndarray:
        linear = self.B @ x / np.sqrt(self.n_features)
        poly = (linear + 3) ** self.deg + 1
        poly = poly * 5 / (3.5 ** self.deg)

        noise = self.rng.uniform(
            1 - self.noise_width,
            1 + self.noise_width,
            self.n_items
        )
        values = poly * noise

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

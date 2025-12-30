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
        # Non-stationarity parameters
        nonstationarity: str = 'none',  # 'none', 'drift', 'shift', 'drift_and_shift'
        drift_rate: float = 0.01,  # Std of Gaussian noise added to B each round
        shift_interval: int = 500,  # Rounds between major shifts
        shift_fraction: float = 0.2,  # Fraction of B entries to flip on shift
    ):
        self.n_items = n_items
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

        self.nonstationarity = nonstationarity
        self.drift_rate = drift_rate
        self.shift_interval = shift_interval
        self.shift_fraction = shift_fraction
        self.round = 0

        # Initialize B matrix (binary feature mask for each item)
        self.B = self.rng.integers(0, 2, size=(n_items, self.n_features)).astype(float)
        self.B_original = self.B.copy()  # Keep original for reference
        self.weights = self.rng.uniform(1.0, 5.0, n_items)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a context and compute item values."""
        context = self.rng.standard_normal(self.n_features)
        values = self._generate_values(context)
        return context, values

    def step(self):
        self.round += 1

        if self.nonstationarity == 'none':
            return

        elif self.nonstationarity == 'drift':
            self._apply_drift()

        elif self.nonstationarity == 'shift':
            if self.round % self.shift_interval == 0:
                self._apply_shift()

        elif self.nonstationarity == 'drift_and_shift':
            self._apply_drift()
            if self.round % self.shift_interval == 0:
                self._apply_shift()

    def _apply_drift(self):
        noise = self.rng.normal(0, self.drift_rate, self.B.shape)
        self.B = self.B + noise
        self.B = np.clip(self.B, -0.2, 1.2)

    def _apply_shift(self):
        n_entries = self.B.size
        n_flip = int(n_entries * self.shift_fraction)

        flat_indices = self.rng.choice(n_entries, size=n_flip, replace=False)

        B_flat = self.B.flatten()
        for idx in flat_indices:
            if B_flat[idx] > 0.5:
                B_flat[idx] = self.rng.uniform(0, 0.3)
            else:
                B_flat[idx] = self.rng.uniform(0.7, 1.0)
        self.B = B_flat.reshape(self.B.shape)

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
            'nonstationarity': self.nonstationarity,
            'drift_rate': self.drift_rate,
            'shift_interval': self.shift_interval,
            'shift_fraction': self.shift_fraction,
            'weights': self.weights.copy(),
        }

    def reset(self):
        self.B = self.B_original.copy()
        self.round = 0

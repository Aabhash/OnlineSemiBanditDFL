import numpy as np
from typing import Tuple


class SyntheticGridData:

    def __init__(
        self,
        grid_size: int = 5,
        context_dim: int = 10,
        deg: int = 4,
        noise_width: float = 0.5,
        seed: int = 42,
        # Non-stationarity parameters
        nonstationarity: str = 'none',
        drift_rate: float = 0.01,
        shift_interval: int = 500,
        shift_fraction: float = 0.2,
    ):
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.context_dim = context_dim
        self.deg = deg
        self.noise_width = noise_width
        self.rng = np.random.default_rng(seed)

        # Non-stationarity state
        self.nonstationarity = nonstationarity
        self.drift_rate = drift_rate
        self.shift_interval = shift_interval
        self.shift_fraction = shift_fraction
        self.round = 0

        # Binary feature mask (same as SyntheticKnapsackData)
        self.B = self.rng.integers(0, 2, size=(self.n_cells, context_dim)).astype(float)
        self.B_original = self.B.copy()

    def _compute_costs(self, context: np.ndarray) -> np.ndarray:
        linear = self.B @ context / np.sqrt(self.context_dim)
        poly = (linear + 3) ** self.deg + 1
        poly = poly * 5 / (3.5 ** self.deg)

        noise = self.rng.uniform(
            1 - self.noise_width,
            1 + self.noise_width,
            self.n_cells,
        )
        costs = poly * noise

        return costs.reshape(self.grid_size, self.grid_size)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        context = self.rng.standard_normal(self.context_dim)
        costs = self._compute_costs(context)
        return context, costs

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

    def reset(self):
        self.B = self.B_original.copy()
        self.round = 0

    def get_config(self) -> dict:
        return {
            'grid_size': self.grid_size,
            'n_cells': self.n_cells,
            'context_dim': self.context_dim,
            'deg': self.deg,
            'noise_width': self.noise_width,
            'nonstationarity': self.nonstationarity,
            'drift_rate': self.drift_rate,
            'shift_interval': self.shift_interval,
            'shift_fraction': self.shift_fraction,
        }

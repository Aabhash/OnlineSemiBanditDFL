import numpy as np
from typing import Tuple
import os


class SyntheticGridData:
    def __init__(
        self,
        grid_size: int = 5,
        context_dim: int = 10,
        cost_type: str = 'linear',
        noise_std: float = 0.0,
        seed: int = 42,
        # Non-stationarity parameters
        nonstationarity: str = 'none',  # 'none', 'drift', 'shift', 'drift_and_shift'
        drift_rate: float = 0.01,  # Std of Gaussian noise added to coefficients each round
        shift_interval: int = 500,  # Rounds between major shifts
        shift_fraction: float = 0.2,  # Fraction of coefficient entries to flip on shift
    ):
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.context_dim = context_dim
        self.cost_type = cost_type
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # Non-stationarity state
        self.nonstationarity = nonstationarity
        self.drift_rate = drift_rate
        self.shift_interval = shift_interval
        self.shift_fraction = shift_fraction
        self.round = 0

        self.coefficients = np.zeros((self.n_cells, context_dim))
        for i in range(self.n_cells):
            a = self.rng.standard_normal(context_dim)
            self.coefficients[i] = a / (np.linalg.norm(a) + 1e-8)

        self.coefficients_original = self.coefficients.copy()  # Keep original for reset
        self.biases = self.rng.uniform(0.5, 1.5, self.n_cells)
        self.biases_original = self.biases.copy()

    def _compute_costs(self, context: np.ndarray) -> np.ndarray:
        projections = self.coefficients @ context

        if self.cost_type == 'linear':
            costs = np.abs(projections) + self.biases
        elif self.cost_type == 'quadratic':
            costs = projections ** 2 + self.biases
        elif self.cost_type == 'cosine':
            costs = (1 + np.cos(np.pi * projections)) / 2 + self.biases
        else:
            raise ValueError(f"Unknown cost_type: {self.cost_type}")

        return costs.reshape(self.grid_size, self.grid_size)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        context = self.rng.standard_normal(self.context_dim)
        context = context / (np.linalg.norm(context) + 1e-8)
        u = self.rng.uniform(0, 1)
        context = context * (u ** (1.0 / self.context_dim))

        costs = self._compute_costs(context)

        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, costs.shape)
            costs = np.maximum(costs + noise, 0.01)  # Ensure positive costs

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
        noise = self.rng.normal(0, self.drift_rate, self.coefficients.shape)
        self.coefficients = self.coefficients + noise
        norms = np.linalg.norm(self.coefficients, axis=1, keepdims=True) + 1e-8
        self.coefficients = self.coefficients / norms

    def _apply_shift(self):
        n_entries = self.coefficients.size
        n_flip = int(n_entries * self.shift_fraction)

        flat_indices = self.rng.choice(n_entries, size=n_flip, replace=False)

        coef_flat = self.coefficients.flatten()
        for idx in flat_indices:
            # Flip sign and add some noise
            coef_flat[idx] = -coef_flat[idx] + self.rng.normal(0, 0.2)
        self.coefficients = coef_flat.reshape(self.coefficients.shape)

        # Re-normalize
        norms = np.linalg.norm(self.coefficients, axis=1, keepdims=True) + 1e-8
        self.coefficients = self.coefficients / norms

    def reset(self):
        """Reset to original coefficients."""
        self.coefficients = self.coefficients_original.copy()
        self.biases = self.biases_original.copy()
        self.round = 0

    def get_config(self) -> dict:
        return {
            'grid_size': self.grid_size,
            'n_cells': self.n_cells,
            'context_dim': self.context_dim,
            'cost_type': self.cost_type,
            'noise_std': self.noise_std,
            'nonstationarity': self.nonstationarity,
            'drift_rate': self.drift_rate,
            'shift_interval': self.shift_interval,
            'shift_fraction': self.shift_fraction,
        }

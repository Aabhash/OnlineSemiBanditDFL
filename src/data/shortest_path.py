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
    ):
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.context_dim = context_dim
        self.cost_type = cost_type
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.coefficients = np.zeros((self.n_cells, context_dim))
        for i in range(self.n_cells):
            a = self.rng.standard_normal(context_dim)
            self.coefficients[i] = a / (np.linalg.norm(a) + 1e-8)

        self.biases = self.rng.uniform(0.5, 1.5, self.n_cells)

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

    def get_config(self) -> dict:
        return {
            'grid_size': self.grid_size,
            'n_cells': self.n_cells,
            'context_dim': self.context_dim,
            'cost_type': self.cost_type,
            'noise_std': self.noise_std,
        }

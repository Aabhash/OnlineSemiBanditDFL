import numpy as np
from typing import Tuple


class SyntheticGridData:

    def __init__(
        self,
        grid_size: int = 5,
        context_dim: int = 10,
        deg: int = 6,
        noise_width: float = 0.5,
        seed: int = 42,
        **kwargs,
    ):
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.context_dim = context_dim
        self.deg = deg
        self.noise_width = noise_width
        self.rng = np.random.default_rng(seed)


        self.B = self.rng.integers(0, 2, size=(self.n_cells, context_dim)).astype(float)

    def _compute_costs(self, context: np.ndarray) -> np.ndarray:
        linear = self.B @ context / np.sqrt(self.context_dim)
        poly = (linear + 3) ** self.deg + 1
        poly = poly / (3.5 ** self.deg)

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
        pass

    def reset(self):
        pass

    def get_config(self) -> dict:
        return {
            'grid_size': self.grid_size,
            'n_cells': self.n_cells,
            'context_dim': self.context_dim,
            'deg': self.deg,
            'noise_width': self.noise_width,
        }

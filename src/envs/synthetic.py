import numpy as np
from typing import Tuple

from src.envs.base import ConstrainedBanditEnv
from src.data.synthetic import SyntheticKnapsackData


class SyntheticKnapsackEnv(ConstrainedBanditEnv):

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
        if context_dim is not None:
            effective_dim = context_dim
        elif n_features is not None:
            effective_dim = n_features
        else:
            effective_dim = 5  # default

        self.data = SyntheticKnapsackData(
            n_items=n_items,
            context_dim=effective_dim,
            capacity=capacity,
            deg=deg,
            noise_width=noise_width,
            seed=seed,
        )

        super().__init__(
            n_arms=n_items,
            context_dim=effective_dim,
            weights=self.data.weights,
            capacity=capacity,
            seed=seed,
        )

        self.current_context, self.current_values = self._generate_sample()

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data.sample()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self.data.get_config())
        return config

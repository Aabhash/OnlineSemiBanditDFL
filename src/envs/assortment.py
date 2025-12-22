import numpy as np
from typing import Tuple

from src.envs.base import ConstrainedBanditEnv
from src.data.assortment import InstacartData


class AssortmentEnv(ConstrainedBanditEnv):

    def __init__(
        self,
        n_products: int = 50,
        n_slots: int = 5,
        context_dim: int = 10,
        data_dir: str = None,
        seed: int = 42,
    ):
        self.n_slots = n_slots

        self.data = InstacartData(
            n_products=n_products,
            context_dim=context_dim,
            data_dir=data_dir,
            seed=seed,
        )

        weights = np.ones(n_products)
        capacity = float(n_slots)

        super().__init__(
            n_arms=n_products,
            context_dim=context_dim,
            weights=weights,
            capacity=capacity,
            seed=seed,
        )

        self.current_context, self.current_values = self._generate_sample()

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data.sample()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self.data.get_config())
        config['n_slots'] = self.n_slots
        return config

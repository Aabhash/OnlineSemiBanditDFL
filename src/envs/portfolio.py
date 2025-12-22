import numpy as np
from typing import Tuple

from src.envs.base import ConstrainedBanditEnv
from src.data.portfolio import FamaFrenchData


class PortfolioEnv(ConstrainedBanditEnv):

    def __init__(
        self,
        n_portfolios: int = 50,
        n_select: int = 5,
        lookback: int = 12,
        start_date: str = "2000-01",
        end_date: str = "2023-12",
        data_dir: str = None,
        seed: int = 42,
    ):
        self.n_select = n_select

        self.data = FamaFrenchData(
            n_portfolios=n_portfolios,
            lookback=lookback,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            seed=seed,
        )

        weights = np.ones(n_portfolios)
        capacity = float(n_select)

        super().__init__(
            n_arms=n_portfolios,
            context_dim=self.data.context_dim,
            weights=weights,
            capacity=capacity,
            seed=seed,
        )

        self.current_context, self.current_values = self._generate_sample()

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data.sample()

    def reset(self) -> np.ndarray:
        self.data.reset()
        return super().reset()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self.data.get_config())
        config['n_select'] = self.n_select
        return config

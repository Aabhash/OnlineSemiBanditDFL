import numpy as np
from typing import Tuple, Dict, Any, Optional

from src.envs.base import ConstrainedBanditEnv
from src.data.criteo import CriteoData


class CriteoEnv(ConstrainedBanditEnv):
    def __init__(
        self,
        n_campaigns: int = 100,
        n_select: int = 10,
        context_dim: int = 50,
        subsample_rate: float = 0.1,
        data_dir: Optional[str] = None,
        seed: int = 42,
        n_estimators: int = 200,
        max_depth: int = 6,
    ):
        self.n_select = n_select
        self.subsample_rate = subsample_rate

        self.data = CriteoData(
            n_campaigns=n_campaigns,
            context_dim=context_dim,
            subsample_rate=subsample_rate,
            data_dir=data_dir,
            seed=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

        weights = self.data.campaign_costs.copy()
        capacity = float(n_select) * np.mean(weights)

        super().__init__(
            n_arms=n_campaigns,
            context_dim=context_dim,
            weights=weights,
            capacity=capacity,
            seed=seed,
        )

        self.current_context, self.current_values = self._generate_sample()

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data.sample()

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'n_campaigns': self.n_arms,
            'n_select': self.n_select,
            'subsample_rate': self.subsample_rate,
            'weight_range': (float(self.weights.min()), float(self.weights.max())),
            'weight_std': float(np.std(self.weights)),
        })
        config.update(self.data.get_config())
        return config

import numpy as np
from typing import Tuple, Dict, Any

from src.envs.base import ConstrainedBanditEnv
from src.data.knapsack import KnapsackData


class KnapsackEnv(ConstrainedBanditEnv):

    def __init__(
        self,
        n_items: int = 32,
        n_features: int = None,
        context_dim: int = None,
        capacity: float = 20.0,
        deg: int = 4,
        noise_width: float = 0.5,
        seed: int = 42,
        unit_weights: bool = False,
        **kwargs,  # absorb legacy nonstationarity params
    ):
        if context_dim is not None:
            effective_dim = context_dim
        elif n_features is not None:
            effective_dim = n_features
        else:
            effective_dim = 5  # default

        self.data = KnapsackData(
            n_items=n_items,
            context_dim=effective_dim,
            capacity=capacity,
            deg=deg,
            noise_width=noise_width,
            seed=seed,
            unit_weights=unit_weights,
        )

        super().__init__(
            n_arms=n_items,
            context_dim=effective_dim,
            weights=self.data.weights,
            capacity=capacity,
            seed=seed,
        )

        self.current_context, self.current_values = self._generate_sample()

    def step(
        self,
        action: np.ndarray,
        full_feedback: bool = False,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        observed_values, reward, info = super().step(action, full_feedback)
        self.data.step()
        return observed_values, reward, info

    def _advance(self):
        super()._advance()
        self.data.step()

    def reset(self) -> np.ndarray:
        self.data.reset()
        return super().reset()

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data.sample()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self.data.get_config())
        return config

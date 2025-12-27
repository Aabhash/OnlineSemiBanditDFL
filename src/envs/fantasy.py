import numpy as np
from typing import Tuple, Dict, Any, Optional

from src.envs.base import ConstrainedBanditEnv
from src.data.fantasy import FantasyData


class FantasyEnv(ConstrainedBanditEnv):

    def __init__(
        self,
        n_players: int = 100,
        salary_cap: float = 50000,
        context_dim: int = 20,
        season: int = 2020,
        data_dir: Optional[str] = None,
        use_position_constraint: bool = False,
        seed: int = 42,
    ):
        self.salary_cap = salary_cap
        self.season = season
        self.use_position_constraint = use_position_constraint

        self.data = FantasyData(
            n_players=n_players,
            context_dim=context_dim,
            season=season,
            data_dir=data_dir,
            seed=seed,
            verbose=False,
        )

        weights = self.data.base_salaries.copy()
        self.player_positions = self.data.player_positions.copy()

        super().__init__(
            n_arms=n_players,
            context_dim=context_dim,
            weights=weights,
            capacity=salary_cap,
            seed=seed,
        )

        self.current_context, self.current_values = self._generate_sample()

    @property
    def per_arm_context(self) -> bool:
        return True

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data.sample_per_arm()

    def _solve_optimal(self, values: np.ndarray) -> np.ndarray:
        if self.use_position_constraint:
            from src.solver import solve_fantasy_lineup
            return solve_fantasy_lineup(
                values,
                self.weights,
                self.player_positions,
                self.capacity
            )
        else:
            from src.solver import solve_knapsack
            return solve_knapsack(values, self.weights, self.capacity)

    def solve(self, values: np.ndarray) -> np.ndarray:
        return self._solve_optimal(values)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'n_players': self.n_arms,
            'salary_cap': self.salary_cap,
            'season': self.season,
            'use_position_constraint': self.use_position_constraint,
            'weight_range': (float(self.weights.min()), float(self.weights.max())),
            'weight_std': float(np.std(self.weights)),
        })
        config.update(self.data.get_config())
        return config

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
        validation_weeks: int = 0,  # For HP tuning
        holdout_weeks: int = 0,     # For final test
        seed: int = 42,
    ):
        self.salary_cap = salary_cap
        self.season = season
        self.use_position_constraint = use_position_constraint
        self.validation_weeks = validation_weeks
        self.holdout_weeks = holdout_weeks

        self.data = FantasyData(
            n_players=n_players,
            context_dim=context_dim,
            season=season,
            data_dir=data_dir,
            validation_weeks=validation_weeks,
            holdout_weeks=holdout_weeks,
            seed=seed,
            verbose=False,
        )

        weights = self.data.base_salaries.copy()
        self.player_positions = self.data.player_positions.copy()

        super().__init__(
            n_arms=self.data.n_players,  # Use actual count from data loader
            context_dim=context_dim,
            weights=weights,
            capacity=salary_cap,
            seed=seed,
        )

        self.current_context, self.current_values = self._generate_sample()

    @property
    def per_arm_context(self) -> bool:
        return True

    def reset(self):
        self.data.reset_sequential()
        return super().reset()

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data.sample_per_arm(sequential=True)

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

    def get_holdout_set(self):
        return self.data.get_holdout_set()

    def get_validation_set(self):
        return self.data.get_validation_set()

    def _evaluate_on_weeks(self, weeks_data, agent) -> Dict[str, float]:
        if not weeks_data:
            return {}

        total_reward = 0.0
        total_optimal = 0.0

        for contexts, values in weeks_data:
            selection = agent.act(contexts)
            reward = np.sum(np.asarray(selection) * values)
            optimal_selection = self.solve(values)
            optimal_reward = np.sum(optimal_selection * values)

            total_reward += reward
            total_optimal += optimal_reward

        n_weeks = len(weeks_data)
        total_regret = total_optimal - total_reward

        return {
            'regret': total_regret,
            'total_reward': total_reward,
            'total_optimal': total_optimal,
            'avg_reward': total_reward / n_weeks,
            'avg_optimal': total_optimal / n_weeks,
            'avg_regret': total_regret / n_weeks,
            'pct_optimal': 100 * total_reward / total_optimal if total_optimal > 0 else 0,
            'n_weeks': n_weeks,
        }

    def evaluate_on_holdout(self, agent) -> Dict[str, float]:
        result = self._evaluate_on_weeks(self.get_holdout_set(), agent)
        if result:
            result['n_holdout_weeks'] = result.pop('n_weeks')
        return result

    def evaluate_on_validation(self, agent) -> Dict[str, float]:
        """Evaluate agent on validation weeks for HP selection."""
        result = self._evaluate_on_weeks(self.get_validation_set(), agent)
        if result:
            result['n_validation_weeks'] = result.pop('n_weeks')
        return result

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'n_players': self.n_arms,
            'salary_cap': self.salary_cap,
            'season': self.season,
            'use_position_constraint': self.use_position_constraint,
            'validation_weeks': self.validation_weeks,
            'holdout_weeks': self.holdout_weeks,
            'n_train_weeks': self.data.n_weeks,
            'n_validation_weeks': self.data.n_validation_weeks,
            'n_holdout_weeks': self.data.n_holdout_weeks,
            'weight_range': (float(self.weights.min()), float(self.weights.max())),
            'weight_std': float(np.std(self.weights)),
        })
        config.update(self.data.get_config())
        return config

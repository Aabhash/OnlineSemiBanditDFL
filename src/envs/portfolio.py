import numpy as np
from typing import Tuple, Dict, Any

from src.envs.base import ConstrainedBanditEnv
from src.data.portfolio import PortfolioData
from src.solver import PortfolioSolver


class PortfolioEnv(ConstrainedBanditEnv):

    def __init__(
        self,
        num_assets: int = 32,
        max_assets: int = 5,
        n_features: int = None,
        context_dim: int = None,
        deg: int = 4,
        noise_width: float = 0.5,
        signal_scale: float = 0.05,
        baseline_return: float = 0.1,
        gamma: float = 1.0,
        solver_time_limit: float | None = 0.1,
        seed: int = 42,
        **kwargs,
    ):
        if context_dim is not None:
            effective_dim = context_dim
        elif n_features is not None:
            effective_dim = n_features
        else:
            effective_dim = 5

        self.gamma = gamma
        self.max_assets = int(max_assets)
        self.solver_time_limit = None if solver_time_limit is None else float(solver_time_limit)
        self.data = PortfolioData(
            num_assets=num_assets,
            context_dim=effective_dim,
            deg=deg,
            noise_width=noise_width,
            signal_scale=signal_scale,
            baseline_return=baseline_return,
            seed=seed,
        )
        self._solver = PortfolioSolver(
            covariance=self.data.covariance,
            gamma=self.gamma,
            max_assets=self.max_assets,
            factor_loadings=self.data.L,
            idiosyncratic_var=self.data.idiosyncratic_var,
            time_limit=self.solver_time_limit,
        )

        super().__init__(
            n_arms=num_assets,
            context_dim=effective_dim,
            weights=np.ones(num_assets, dtype=np.float64),
            capacity=float(self.max_assets),
            seed=seed,
        )

        self.current_context, self.current_values = self._generate_sample()

    def _solve_optimal(self, values: np.ndarray) -> np.ndarray:
        return self._solver.solve(values)

    def solve(self, values: np.ndarray) -> np.ndarray:
        return self._solve_optimal(values)

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
        config.update(
            {
                'gamma': self.gamma,
                'max_assets': self.max_assets,
                'solver_time_limit': self.solver_time_limit,
            }
        )
        return config

import numpy as np
from typing import Tuple, Dict, Any, Optional

from src.envs.base import ConstrainedBanditEnv
from src.data.movielens import MovieLensData


class MovieLensEnv(ConstrainedBanditEnv):
    """MovieLens recommendation environment.

    Creates a semi-bandit setting for movie recommendation where:
    - Context: User embedding capturing preferences
    - Arms: Movies (top n by popularity)
    - Constraint: Recommend exactly k movies (knapsack with uniform weights)
    - Reward: Sum of ratings for selected movies

    Args:
        n_movies: Number of movies to use
        n_slots: Number of recommendation slots (k)
        context_dim: User embedding dimension
        data_dir: Path to MovieLens data
        version: Dataset version ('1m' or '100k')
        noise_std: Noise added to ratings
        seed: Random seed
    """

    def __init__(
        self,
        n_movies: int = 50,
        n_slots: int = 5,
        context_dim: int = 20,
        data_dir: Optional[str] = None,
        version: str = '1m',
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        self.n_slots = n_slots
        self.version = version

        self.data = MovieLensData(
            n_movies=n_movies,
            context_dim=context_dim,
            data_dir=data_dir,
            version=version,
            noise_std=noise_std,
            seed=seed,
        )

        weights = np.ones(n_movies)
        capacity = float(n_slots)

        super().__init__(
            n_arms=n_movies,
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
            'n_movies': self.n_arms,
            'n_slots': self.n_slots,
            'version': self.version,
            'rating_range': (1, 5),
        })
        config.update(self.data.get_config())
        return config

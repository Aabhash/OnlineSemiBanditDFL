from src.envs.base import ConstrainedBanditEnv
from src.envs.knapsack import KnapsackEnv
from src.envs.portfolio import PortfolioEnv
from src.envs.shortest_path import ShortestPathEdgeEnv
from src.envs.energy_scheduling import EnergySchedulingEnv

__all__ = [
    'ConstrainedBanditEnv',
    'KnapsackEnv',
    'PortfolioEnv',
    'ShortestPathEdgeEnv',
    'EnergySchedulingEnv',
]


def create_env(env_type: str, seed: int = 42, **kwargs):
    if env_type.startswith('knapsack'):
        return KnapsackEnv(seed=seed, **kwargs)
    elif env_type.startswith('portfolio'):
        return PortfolioEnv(seed=seed, **kwargs)
    elif env_type.startswith('shortest_path'):
        kwargs_copy = kwargs.copy()
        parts = env_type.split('_')
        grid_size = kwargs_copy.pop('grid_size', None)
        if grid_size is None:
            for part in parts[2:]:
                if part.isdigit():
                    grid_size = int(part)
                    break
            if grid_size is None:
                grid_size = 8
        kwargs_copy.pop('data_type', None)
        kwargs_copy.pop('minimize', None)
        return ShortestPathEdgeEnv(
            grid_size=grid_size,
            seed=seed,
            **kwargs_copy
        )
    elif env_type.startswith('energy_scheduling'):
        return EnergySchedulingEnv(seed=seed, **kwargs)
    else:
        raise ValueError(
            f"Unknown environment type: {env_type}. "
            "Available: knapsack, portfolio, shortest_path_*, "
            "energy_scheduling"
        )

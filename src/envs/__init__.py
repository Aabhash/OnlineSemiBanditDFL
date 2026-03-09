from src.envs.base import ConstrainedBanditEnv
from src.envs.knapsack import KnapsackEnv
from src.envs.shortest_path import ShortestPathEdgeEnv
from src.envs.criteo import CriteoEnv

__all__ = [
    'ConstrainedBanditEnv',
    'KnapsackEnv',
    'ShortestPathEdgeEnv',
    'CriteoEnv',
]


def create_env(env_type: str, seed: int = 42, **kwargs):
    if env_type.startswith('knapsack'):
        return KnapsackEnv(seed=seed, **kwargs)
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
                grid_size = 8  # default
        kwargs_copy.pop('data_type', None)
        # Remove minimize as it's a property, not a parameter
        kwargs_copy.pop('minimize', None)
        return ShortestPathEdgeEnv(
            grid_size=grid_size,
            seed=seed,
            **kwargs_copy
        )
    elif env_type.startswith('criteo'):
        return CriteoEnv(seed=seed, **kwargs)
    else:
        raise ValueError(
            f"Unknown environment type: {env_type}. "
            f"Available: knapsack, knapsack_unit_weight, "
            f"shortest_path_*, criteo"
        )

from src.envs.base import ConstrainedBanditEnv
from src.envs.synthetic import SyntheticKnapsackEnv
from src.envs.portfolio import PortfolioEnv  # Kept for backwards compatibility
from src.envs.assortment import AssortmentEnv
from src.data.cn_synthetic import CNSyntheticEnv
from src.envs.shortest_path import ShortestPathEnv
from src.envs.movielens import MovieLensEnv
from src.envs.recsim import RecSimEnv, StationaryRecSimEnv, MultiUserRecSimEnv

__all__ = [
    'ConstrainedBanditEnv',
    'SyntheticKnapsackEnv',
    'AssortmentEnv',
    'CNSyntheticEnv',
    'ShortestPathEnv',
    'MovieLensEnv',
    'RecSimEnv',
    'StationaryRecSimEnv',
    'MultiUserRecSimEnv',
    # PortfolioEnv removed from exports - predictions too good, decisions don't vary
]


def create_env(env_type: str, seed: int = 42, **kwargs):
    if env_type == 'synthetic':
        return SyntheticKnapsackEnv(seed=seed, **kwargs)
    elif env_type == 'portfolio':
        return PortfolioEnv(seed=seed, **kwargs)
    elif env_type == 'assortment':
        return AssortmentEnv(seed=seed, **kwargs)
    elif env_type.startswith('cn_'):
        # CN synthetic benchmarks (Hwang et al., ICML 2023)
        score_type = env_type[3:]  # Remove 'cn_' prefix
        return CNSyntheticEnv(seed=seed, score_type=score_type, **kwargs)
    elif kwargs.get('type') == 'cn_synthetic':
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('type', None)
        return CNSyntheticEnv(seed=seed, **kwargs_copy)
    elif env_type.startswith('shortest_path') or env_type.startswith('warcraft'):
        kwargs_copy = kwargs.copy()
        parts = env_type.split('_')
        grid_size = kwargs_copy.pop('grid_size', int(parts[2]) if len(parts) > 2 else 5)
        data_type = kwargs_copy.pop('data_type', 'synthetic')
        return ShortestPathEnv(
            grid_size=grid_size,
            data_type=data_type,
            seed=seed,
            **kwargs_copy
        )
    elif env_type.startswith('movielens'):
        return MovieLensEnv(seed=seed, **kwargs)
    elif env_type.startswith('recsim'):
        if 'stationary' in env_type:
            return StationaryRecSimEnv(seed=seed, **kwargs)
        elif 'multi' in env_type:
            return MultiUserRecSimEnv(seed=seed, **kwargs)
        else:
            return RecSimEnv(seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

from src.envs.base import ConstrainedBanditEnv
from src.envs.synthetic import SyntheticKnapsackEnv
from src.data.cn_synthetic import CNSyntheticEnv
from src.envs.shortest_path import ShortestPathEnv
from src.envs.criteo import CriteoEnv
from src.envs.fantasy import FantasyEnv
from src.envs.fpl import FPLEnv

__all__ = [
    'ConstrainedBanditEnv',
    'SyntheticKnapsackEnv',
    'CNSyntheticEnv',
    'ShortestPathEnv',
    'CriteoEnv',
    'FantasyEnv',
    'FPLEnv',
]


def create_env(env_type: str, seed: int = 42, **kwargs):
    if env_type == 'synthetic':
        return SyntheticKnapsackEnv(seed=seed, **kwargs)
    elif env_type.startswith('cn_'):
        score_type = env_type[3:]
        return CNSyntheticEnv(seed=seed, score_type=score_type, **kwargs)
    elif kwargs.get('type') == 'cn_synthetic':
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('type', None)
        return CNSyntheticEnv(seed=seed, **kwargs_copy)
    elif env_type.startswith('shortest_path'):
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
    elif env_type.startswith('criteo'):
        return CriteoEnv(seed=seed, **kwargs)
    elif env_type.startswith('fantasy'):
        return FantasyEnv(seed=seed, **kwargs)
    elif env_type.startswith('fpl'):
        return FPLEnv(seed=seed, **kwargs)
    else:
        raise ValueError(
            f"Unknown environment type: {env_type}. "
            f"Available: synthetic, cn_linear, cn_quadratic, cn_cosine, "
            f"shortest_path_5, shortest_path_8, criteo_ads, fantasy, fpl"
        )

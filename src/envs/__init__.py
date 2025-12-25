from src.envs.base import ConstrainedBanditEnv
from src.envs.synthetic import SyntheticKnapsackEnv
from src.envs.portfolio import PortfolioEnv
from src.envs.assortment import AssortmentEnv
from src.data.cn_synthetic import CNSyntheticEnv

__all__ = [
    'ConstrainedBanditEnv',
    'SyntheticKnapsackEnv',
    'PortfolioEnv',
    'AssortmentEnv',
    'CNSyntheticEnv',
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
        # Extract score type from env_type (e.g., 'cn_linear' -> 'linear')
        score_type = env_type[3:]  # Remove 'cn_' prefix
        return CNSyntheticEnv(seed=seed, score_type=score_type, **kwargs)
    elif kwargs.get('type') == 'cn_synthetic':
        # Handle type specified in kwargs
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('type', None)
        return CNSyntheticEnv(seed=seed, **kwargs_copy)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

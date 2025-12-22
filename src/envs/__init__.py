from src.envs.base import ConstrainedBanditEnv
from src.envs.synthetic import SyntheticKnapsackEnv
from src.envs.portfolio import PortfolioEnv
from src.envs.assortment import AssortmentEnv

__all__ = [
    'ConstrainedBanditEnv',
    'SyntheticKnapsackEnv',
    'PortfolioEnv',
    'AssortmentEnv',
]


def create_env(env_type: str, seed: int = 42, **kwargs):
    if env_type == 'synthetic':
        return SyntheticKnapsackEnv(seed=seed, **kwargs)
    elif env_type == 'portfolio':
        return PortfolioEnv(seed=seed, **kwargs)
    elif env_type == 'assortment':
        return AssortmentEnv(seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

import numpy as np
from typing import Tuple, Dict, Any, Optional

from src.envs.base import ConstrainedBanditEnv
from src.data.recsim_data import RecSimData, MultiUserRecSimData


class RecSimEnv(ConstrainedBanditEnv):
    """RecSim-based recommendation environment.

    Creates a semi-bandit setting for slate recommendation where:
    - Context: User interest embedding (evolving over time)
    - Arms: Documents with topic distributions
    - Constraint: Recommend exactly k documents (slate size)
    - Reward: Sum of engagement for selected documents

    Key feature: User interests evolve, creating a non-stationary
    bandit setting where exploration has long-term value.

    Args:
        n_docs: Number of documents in corpus
        slate_size: Number of documents to recommend (k)
        n_topics: Number of latent topics
        context_dim: User embedding dimension
        interest_decay: How quickly interests evolve (0.98 = slow drift)
        stationary: If True, user interests don't evolve (ablation)
        multi_user: If True, sample different users (alternative setting)
        seed: Random seed
    """

    def __init__(
        self,
        n_docs: int = 50,
        slate_size: int = 5,
        n_topics: int = 10,
        context_dim: int = 20,
        interest_decay: float = 0.98,
        noise_std: float = 0.1,
        stationary: bool = False,
        multi_user: bool = False,
        seed: int = 42,
    ):
        self.slate_size = slate_size
        self.n_topics = n_topics
        self.interest_decay = interest_decay
        self.stationary = stationary
        self.multi_user = multi_user

        if multi_user:
            self.data = MultiUserRecSimData(
                n_users=100,
                n_docs=n_docs,
                n_topics=n_topics,
                context_dim=context_dim,
                noise_std=noise_std,
                seed=seed,
            )
        else:
            self.data = RecSimData(
                n_docs=n_docs,
                n_topics=n_topics,
                context_dim=context_dim,
                interest_decay=1.0 if stationary else interest_decay,
                noise_std=noise_std,
                seed=seed,
            )

        weights = np.ones(n_docs)
        capacity = float(slate_size)

        super().__init__(
            n_arms=n_docs,
            context_dim=context_dim,
            weights=weights,
            capacity=capacity,
            seed=seed,
        )

        self.current_context, self.current_values = self._generate_sample()

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data.sample()

    def reset(self) -> np.ndarray:
        self.t = 0
        if hasattr(self.data, 'reset'):
            self.data.reset()
        self.current_context, self.current_values = self._generate_sample()
        return self.current_context

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'n_docs': self.n_arms,
            'slate_size': self.slate_size,
            'n_topics': self.n_topics,
            'interest_decay': self.interest_decay,
            'stationary': self.stationary,
            'multi_user': self.multi_user,
        })
        config.update(self.data.get_config())
        return config


class StationaryRecSimEnv(RecSimEnv):

    def __init__(self, **kwargs):
        kwargs['stationary'] = True
        super().__init__(**kwargs)


class MultiUserRecSimEnv(RecSimEnv):

    def __init__(self, **kwargs):
        kwargs['multi_user'] = True
        super().__init__(**kwargs)

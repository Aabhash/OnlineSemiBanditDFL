import numpy as np
from typing import Tuple, Dict, Optional


class RecSimData:
    """Simplified RecSim-style recommendation simulator.

    Models a single user whose interests evolve over time, interacting
    with a fixed corpus of documents. This creates a non-stationary
    bandit setting where exploration has long-term value.

    User Model:
    - User has latent interest vector (n_topics,)
    - Interests evolve slowly via random walk
    - Satisfaction level affects engagement (optional)

    Document Model:
    - Each document has topic distribution + quality score
    - Topics are sampled from Dirichlet
    - Quality captures intrinsic item value

    Engagement Model:
    - engagement = interest_match * quality + noise
    - interest_match = user_interests @ doc_topics

    Args:
        n_docs: Number of documents in corpus
        n_topics: Number of latent topics
        context_dim: User state dimension for agent
        interest_decay: How quickly interests evolve (1.0 = static)
        quality_range: (min, max) document quality
        noise_std: Noise in engagement observation
        seed: Random seed
    """

    def __init__(
        self,
        n_docs: int = 50,
        n_topics: int = 10,
        context_dim: int = 20,
        interest_decay: float = 0.98,
        quality_range: Tuple[float, float] = (0.5, 1.5),
        noise_std: float = 0.1,
        seed: int = 42,
    ):
        self.n_docs = n_docs
        self.n_topics = n_topics
        self.context_dim = context_dim
        self.interest_decay = interest_decay
        self.quality_range = quality_range
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self._init_documents()

        self._init_user()

        self._init_projection()

    def _init_documents(self):
        self.doc_topics = self.rng.dirichlet(
            np.ones(self.n_topics) * 0.5,
            size=self.n_docs
        )  # (n_docs, n_topics)

        q_min, q_max = self.quality_range
        self.doc_quality = self.rng.uniform(q_min, q_max, self.n_docs)

        n_popular = max(1, self.n_docs // 10)
        popular_idx = self.rng.choice(self.n_docs, size=n_popular, replace=False)
        self.doc_quality[popular_idx] *= 1.5

    def _init_user(self):
        self.user_interests = self.rng.dirichlet(np.ones(self.n_topics))

        self.satisfaction = 0.5  # [0, 1]

        self.t = 0

    def _init_projection(self):
        self.proj_matrix = self.rng.standard_normal(
            (self.n_topics, self.context_dim)
        ) * 0.5
        if self.context_dim <= self.n_topics:
            u, _, _ = np.linalg.svd(self.proj_matrix, full_matrices=False)
            self.proj_matrix = u[:, :self.context_dim]

    def reset(self):
        self._init_user()
        return self.sample()[0]

    def _evolve_interests(self):
        noise = self.rng.dirichlet(np.ones(self.n_topics) * 10)

        self.user_interests = (
            self.interest_decay * self.user_interests +
            (1 - self.interest_decay) * noise
        )

        self.user_interests = self.user_interests / self.user_interests.sum()

    def _compute_context(self) -> np.ndarray:
        context = self.user_interests @ self.proj_matrix

        context += self.rng.standard_normal(self.context_dim) * 0.05

        context = context / (np.linalg.norm(context) + 1e-8)

        return context

    def _compute_engagement(self) -> np.ndarray:
        interest_match = self.doc_topics @ self.user_interests  # (n_docs,)
        engagement = interest_match * self.doc_quality
        engagement += self.rng.standard_normal(self.n_docs) * self.noise_std
        engagement = np.clip(engagement, 0.0, 2.0)

        return engagement

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        self._evolve_interests()
        context = self._compute_context()
        values = self._compute_engagement()
        self.t += 1
        return context, values

    def get_config(self) -> Dict:
        return {
            'n_docs': self.n_docs,
            'n_topics': self.n_topics,
            'context_dim': self.context_dim,
            'interest_decay': self.interest_decay,
            'quality_range': self.quality_range,
        }


class StationaryRecSimData(RecSimData):
    def _evolve_interests(self):
        pass


class MultiUserRecSimData:
    def __init__(
        self,
        n_users: int = 100,
        n_docs: int = 50,
        n_topics: int = 10,
        context_dim: int = 20,
        noise_std: float = 0.1,
        seed: int = 42,
    ):
        self.n_users = n_users
        self.n_docs = n_docs
        self.n_topics = n_topics
        self.context_dim = context_dim
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.doc_topics = self.rng.dirichlet(
            np.ones(n_topics) * 0.5, size=n_docs
        )
        self.doc_quality = self.rng.uniform(0.5, 1.5, n_docs)

        self.user_interests = self.rng.dirichlet(
            np.ones(n_topics), size=n_users
        )

        self.proj_matrix = self.rng.standard_normal((n_topics, context_dim)) * 0.5

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        user_idx = self.rng.integers(self.n_users)
        interests = self.user_interests[user_idx]

        context = interests @ self.proj_matrix
        context += self.rng.standard_normal(self.context_dim) * 0.05
        context = context / (np.linalg.norm(context) + 1e-8)

        interest_match = self.doc_topics @ interests
        values = interest_match * self.doc_quality
        values += self.rng.standard_normal(self.n_docs) * self.noise_std
        values = np.clip(values, 0.0, 2.0)

        return context, values

    def get_config(self) -> Dict:
        return {
            'n_users': self.n_users,
            'n_docs': self.n_docs,
            'n_topics': self.n_topics,
            'context_dim': self.context_dim,
        }

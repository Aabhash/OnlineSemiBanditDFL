import numpy as np
from typing import Tuple


def sample_uniform_unit_ball(rng: np.random.Generator, dim: int) -> np.ndarray:
    x = rng.standard_normal(dim)
    x = x / np.linalg.norm(x)

    u = rng.uniform(0, 1)
    x = x * (u ** (1.0 / dim))

    return x


class CNSyntheticData:
    def __init__(
        self,
        n_arms: int = 20,
        n_features: int = 80,
        K: int = 4,
        score_type: str = 'linear',
        noise_std: float = 0.0,
        seed: int = 42,
    ):
        self.n_arms = n_arms
        self.n_features = n_features
        self.K = K
        self.score_type = score_type
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        a = self.rng.standard_normal(n_features)
        self.coefficient = a / np.linalg.norm(a)

        self.weights = np.ones(n_arms)
        self.capacity = float(K)

    def _compute_scores(self, arm_contexts: np.ndarray) -> np.ndarray:
        projections = arm_contexts @ self.coefficient

        if self.score_type == 'linear':
            scores = projections
        elif self.score_type == 'quadratic':
            scores = projections ** 2
        elif self.score_type == 'cosine':
            scores = np.cos(np.pi * projections)
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

        return scores

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        arm_contexts = np.array([
            sample_uniform_unit_ball(self.rng, self.n_features)
            for _ in range(self.n_arms)
        ])

        values = self._compute_scores(arm_contexts)

        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, self.n_arms)
            values = values + noise

        return arm_contexts, values

    def get_config(self) -> dict:
        return {
            'n_items': self.n_arms,
            'n_features': self.n_features,
            'K': self.K,
            'capacity': self.capacity,
            'score_type': self.score_type,
            'noise_std': self.noise_std,
            'weights': self.weights.copy(),
        }


class CNSyntheticEnv:
    def __init__(
        self,
        n_arms: int = 20,
        n_features: int = 80,
        K: int = 4,
        score_type: str = 'linear',
        noise_std: float = 0.0,
        seed: int = 42,
    ):
        self.data = CNSyntheticData(
            n_arms=n_arms,
            n_features=n_features,
            K=K,
            score_type=score_type,
            noise_std=noise_std,
            seed=seed,
        )
        self.n_arms = n_arms
        self.n_features = n_features
        self.context_dim = n_features
        self.K = K

        self.weights = self.data.weights
        self.capacity = self.data.capacity

        self._current_arm_contexts = None
        self._current_values = None

    def reset(self) -> np.ndarray:
        self._current_arm_contexts, self._current_values = self.data.sample()
        return self._current_arm_contexts

    def get_context(self) -> np.ndarray:
        self._current_arm_contexts, self._current_values = self.data.sample()
        return self._current_arm_contexts

    def step(self, selection: np.ndarray, full_feedback: bool = False) -> Tuple:
        selection = np.asarray(selection)

        true_values = self._current_values
        optimal_selection = np.zeros(self.n_arms)
        top_k_indices = np.argsort(-true_values)[:self.K]
        optimal_selection[top_k_indices] = 1

        reward = np.sum(true_values * selection)
        optimal_reward = np.sum(true_values * optimal_selection)
        regret = optimal_reward - reward

        if full_feedback:
            observed_values = true_values.copy()
        else:
            observed_values = np.zeros(self.n_arms)
            observed_values[selection > 0.5] = true_values[selection > 0.5]

        info = {
            'regret': regret,
            'optimal_reward': optimal_reward,
            'true_values': true_values.copy(),
            'optimal_selection': optimal_selection,
            'arm_contexts': self._current_arm_contexts.copy(),
        }

        return observed_values, reward, info

    def generate_test_set(self, n_samples: int = 100) -> Tuple:
        X_test = []
        Y_test = []

        for _ in range(n_samples):
            arm_contexts, values = self.data.sample()
            X_test.append(arm_contexts)
            Y_test.append(values)

        return np.array(X_test), np.array(Y_test)

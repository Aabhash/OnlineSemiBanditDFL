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
        n_features: int = None,
        context_dim: int = None,
        K: int = 4,
        score_type: str = 'linear',
        noise_std: float = 0.0,
        use_knapsack: bool = True,  # If True, use varied weights + knapsack
        weight_range: tuple = (1.0, 5.0),  # Weight range like Fantasy salaries
        seed: int = 42,
    ):
        self.n_arms = n_arms
        if context_dim is not None:
            self.n_features = context_dim
        elif n_features is not None:
            self.n_features = n_features
        else:
            self.n_features = 80  # default
        self.context_dim = self.n_features  # alias for consistency
        self.K = K
        self.score_type = score_type
        self.noise_std = noise_std
        self.use_knapsack = use_knapsack
        self.rng = np.random.default_rng(seed)

        self.coefficients = np.zeros((n_arms, self.n_features))
        for i in range(n_arms):
            a = self.rng.standard_normal(self.n_features)
            self.coefficients[i] = a / np.linalg.norm(a)

        if use_knapsack:
            self.weights = self.rng.uniform(weight_range[0], weight_range[1], n_arms)
            avg_weight = (weight_range[0] + weight_range[1]) / 2
            self.capacity = avg_weight * K * 1.1
        else:
            self.weights = np.ones(n_arms)
            self.capacity = float(K)

    def _compute_scores(self, context: np.ndarray) -> np.ndarray:
        projections = self.coefficients @ context

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
        context = sample_uniform_unit_ball(self.rng, self.n_features)
        values = self._compute_scores(context)

        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, self.n_arms)
            values = values + noise

        return context, values

    def sample_per_arm(self) -> Tuple[np.ndarray, np.ndarray]:
        contexts = np.zeros((self.n_arms, self.n_features))
        for i in range(self.n_arms):
            contexts[i] = sample_uniform_unit_ball(self.rng, self.n_features)

        projections = np.sum(contexts * self.coefficients, axis=1)

        if self.score_type == 'linear':
            values = projections
        elif self.score_type == 'quadratic':
            values = projections ** 2
        elif self.score_type == 'cosine':
            values = np.cos(np.pi * projections)
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, self.n_arms)
            values = values + noise

        return contexts, values

    def get_config(self) -> dict:
        return {
            'n_items': self.n_arms,
            'n_features': self.n_features,
            'context_dim': self.context_dim,
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
        n_features: int = None,
        context_dim: int = None,
        K: int = 4,
        score_type: str = 'linear',
        noise_std: float = 0.0,
        use_knapsack: bool = True,  # If True, use varied weights + knapsack
        weight_range: tuple = (1.0, 5.0),
        seed: int = 42,
        use_per_arm_context: bool = True,  # Default to per-arm for CN datasets
    ):
        if context_dim is not None:
            effective_dim = context_dim
        elif n_features is not None:
            effective_dim = n_features
        else:
            effective_dim = 80  # default

        self.data = CNSyntheticData(
            n_arms=n_arms,
            context_dim=effective_dim,
            K=K,
            score_type=score_type,
            noise_std=noise_std,
            use_knapsack=use_knapsack,
            weight_range=weight_range,
            seed=seed,
        )
        self.n_arms = n_arms
        self.n_features = effective_dim
        self.context_dim = effective_dim
        self.K = K
        self.use_knapsack = use_knapsack
        self._use_per_arm_context = use_per_arm_context

        self.weights = self.data.weights
        self.capacity = self.data.capacity

        self._current_context = None
        self._current_values = None

    @property
    def per_arm_context(self) -> bool:
        return self._use_per_arm_context

    def reset(self) -> np.ndarray:
        if self._use_per_arm_context:
            self._current_context, self._current_values = self.data.sample_per_arm()
        else:
            self._current_context, self._current_values = self.data.sample()
        return self._current_context

    def get_context(self) -> np.ndarray:
        if self._use_per_arm_context:
            self._current_context, self._current_values = self.data.sample_per_arm()
        else:
            self._current_context, self._current_values = self.data.sample()
        return self._current_context

    def step(self, selection: np.ndarray, full_feedback: bool = False) -> Tuple:
        selection = np.asarray(selection)

        true_values = self._current_values
        optimal_selection = self.solve(true_values)

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
            'context': self._current_context.copy(),
        }

        return observed_values, reward, info

    def generate_test_set(self, n_samples: int = 100) -> Tuple:
        X_test = []
        Y_test = []

        for _ in range(n_samples):
            if self._use_per_arm_context:
                context, values = self.data.sample_per_arm()
            else:
                context, values = self.data.sample()
            X_test.append(context)
            Y_test.append(values)

        return np.array(X_test), np.array(Y_test)

    def solve(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if self.use_knapsack:
            from src.solver import solve_knapsack
            return solve_knapsack(values, self.weights, self.capacity)
        else:
            selection = np.zeros(self.n_arms)
            top_k_indices = np.argsort(-values)[:self.K]
            selection[top_k_indices] = 1
            return selection

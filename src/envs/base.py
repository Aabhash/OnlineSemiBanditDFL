import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class ConstrainedBanditEnv(ABC):

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        weights: np.ndarray,
        capacity: float,
        seed: int = 42,
    ):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.weights = np.array(weights)
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

        self.current_context = None
        self.current_values = None
        self.t = 0

    @abstractmethod
    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def reset(self) -> np.ndarray:
        self.t = 0
        self.current_context, self.current_values = self._generate_sample()
        return self.current_context

    def step(
        self,
        action: np.ndarray,
        full_feedback: bool = False,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        reward = np.sum(action * self.current_values)

        optimal_action = self._solve_optimal(self.current_values)
        optimal_reward = np.sum(optimal_action * self.current_values)
        regret = optimal_reward - reward

        if full_feedback:
            observed_values = self.current_values.copy()
        else:
            observed_values = np.zeros(self.n_arms)
            selected = action > 0.5
            observed_values[selected] = self.current_values[selected]

        info = {
            'reward': reward,
            'optimal_reward': optimal_reward,
            'regret': regret,
            'action': action,
            'true_values': self.current_values.copy(),
            'selected_indices': np.where(action > 0.5)[0],
        }

        self.t += 1
        self.current_context, self.current_values = self._generate_sample()

        return observed_values, reward, info

    def _solve_optimal(self, values: np.ndarray) -> np.ndarray:
        from src.solver import solve_knapsack
        return solve_knapsack(values, self.weights, self.capacity)

    def get_context(self) -> np.ndarray:
        return self.current_context

    def get_config(self) -> Dict[str, Any]:
        return {
            'n_arms': self.n_arms,
            'context_dim': self.context_dim,
            'capacity': self.capacity,
            'weights': self.weights.tolist(),
        }

    def generate_test_set(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        X_test = []
        Y_test = []
        for _ in range(n_samples):
            context, values = self._generate_sample()
            X_test.append(context)
            Y_test.append(values)
        return np.array(X_test), np.array(Y_test)

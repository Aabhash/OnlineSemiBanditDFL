
import numpy as np
from typing import Tuple, Dict, Any

from src.envs.base import ConstrainedBanditEnv


class TopKEnv(ConstrainedBanditEnv):

    def __init__(
        self,
        n_arms: int = 8,
        k: int = 2,
        context_dim: int = 5,
        cost_fn: str = 'sigmoid',
        noise_width: float = 0.0,
        w_spread: float = 0.4,
        deg: int = 4,
        seed: int = 42,
    ):
        weights = np.ones(n_arms)
        capacity = float(k)

        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            weights=weights,
            capacity=capacity,
            seed=seed,
        )

        self.k = k
        self.cost_fn = cost_fn
        self.noise_width = noise_width
        self.w_spread = w_spread
        self.deg = deg


        self.W = (self.rng.standard_normal((1, context_dim))
                  + self.rng.standard_normal((n_arms, context_dim)) * w_spread)


        if cost_fn == 'polynomial':
            self.B = self.rng.integers(0, 2, size=(n_arms, context_dim)).astype(float)

        self.current_context, self.current_values = self._generate_sample()

    @property
    def minimize(self) -> bool:
        return True

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        x = self.rng.standard_normal(self.context_dim)
        costs = self._generate_costs(x)
        return x, costs

    def _generate_costs(self, x: np.ndarray) -> np.ndarray:
        if self.cost_fn == 'sigmoid':
            linear = self.W @ x
            costs = 1.0 / (1.0 + np.exp(-linear.flatten()))
        elif self.cost_fn == 'linear':
            costs = (self.W @ x).flatten()
        elif self.cost_fn == 'polynomial':
            linear = self.B @ x / np.sqrt(self.context_dim)
            poly = (linear + 3) ** self.deg + 1
            costs = poly * 5 / (3.5 ** self.deg)
        else:
            raise ValueError(f"Unknown cost_fn: {self.cost_fn}")

        if self.noise_width > 0:
            noise = self.rng.uniform(
                1 - self.noise_width,
                1 + self.noise_width,
                self.n_arms,
            )
            costs = costs * noise

        return costs

    def _solve_optimal(self, values: np.ndarray) -> np.ndarray:
        sol = np.zeros(self.n_arms)
        idx = np.argsort(values)[:self.k]
        sol[idx] = 1.0
        return sol

    def solve(self, values: np.ndarray) -> np.ndarray:
        return self._solve_optimal(values)

    def step(
        self,
        action: np.ndarray,
        full_feedback: bool = False,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:

        agent_cost = np.sum(action * self.current_values)
        optimal_action = self._solve_optimal(self.current_values)
        optimal_cost = np.sum(optimal_action * self.current_values)
        regret = agent_cost - optimal_cost

        if full_feedback:
            observed_values = self.current_values.copy()
        else:
            observed_values = np.zeros(self.n_arms)
            selected = action > 0.5
            observed_values[selected] = self.current_values[selected]

        info = {
            'reward': -agent_cost,
            'optimal_reward': -optimal_cost,
            'regret': regret,
            'action': action,
            'optimal_selection': optimal_action,
            'true_values': self.current_values.copy(),
            'selected_indices': np.where(action > 0.5)[0],
        }

        self.t += 1
        self.current_context, self.current_values = self._generate_sample()

        return observed_values, -agent_cost, info

    def _advance(self):
        self.t += 1
        self.current_context, self.current_values = self._generate_sample()

    def get_config(self) -> Dict[str, Any]:
        return {
            'n_arms': self.n_arms,
            'k': self.k,
            'context_dim': self.context_dim,
            'cost_fn': self.cost_fn,
            'noise_width': self.noise_width,
            'w_spread': self.w_spread,
            'capacity': self.capacity,
            'weights': self.weights.tolist(),
        }

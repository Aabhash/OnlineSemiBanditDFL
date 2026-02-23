import numpy as np
from typing import Tuple, Dict, Any

from src.solver import (
    n_edges_grid,
    solve_shortest_path_edges,
    compute_edge_slacks,
)


class SyntheticEdgeData:

    def __init__(
        self,
        grid_size: int = 5,
        context_dim: int = 10,
        deg: int = 4,
        noise_width: float = 0.5,
        seed: int = 42,
        nonstationarity: str = 'none',
        drift_rate: float = 0.01,
        shift_interval: int = 500,
        shift_fraction: float = 0.2,
    ):
        self.grid_size = grid_size
        self.n_edges = n_edges_grid(grid_size)
        self.context_dim = context_dim
        self.noise_width = noise_width
        self.nonstationarity = nonstationarity
        self.t = 0

        init_rng = np.random.default_rng(seed)
        n_hidden = context_dim * deg

        self.W1 = init_rng.standard_normal((context_dim, n_hidden)).astype(np.float32)
        self.W1 /= np.sqrt(context_dim)
        self.b1 = init_rng.standard_normal(n_hidden).astype(np.float32) * 0.5

        self.W2 = init_rng.standard_normal((n_hidden, self.n_edges)).astype(np.float32)
        self.W2 /= np.sqrt(n_hidden)

        self.bias = init_rng.uniform(1.0, 5.0, size=self.n_edges).astype(np.float32)

        self._sample_seed = seed + 7777
        self.sample_rng = np.random.default_rng(self._sample_seed)

    def reset(self):
        self.t = 0
        self.sample_rng = np.random.default_rng(self._sample_seed)

    def step(self):
        self.t += 1

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        x = self.sample_rng.standard_normal(self.context_dim).astype(np.float32)

        h = np.maximum(0.0, x @ self.W1 + self.b1)          # ReLU
        mean_costs = np.abs(h @ self.W2) + self.bias         # positive

        noise = self.sample_rng.uniform(
            -self.noise_width / 2, self.noise_width / 2,
            size=self.n_edges,
        ).astype(np.float32)

        costs = np.maximum(0.01, mean_costs + noise)         # safety clamp
        return x, costs


class ShortestPathEdgeEnv:

    def __init__(
        self,
        grid_size: int = 5,
        context_dim: int = 10,
        deg: int = 4,
        noise_width: float = 0.5,
        seed: int = 42,
        nonstationarity: str = 'none',
        drift_rate: float = 0.01,
        shift_interval: int = 500,
        shift_fraction: float = 0.2,
    ):
        self.grid_size = grid_size
        self.n_arms = n_edges_grid(grid_size)
        self.context_dim = context_dim

        self.data = SyntheticEdgeData(
            grid_size=grid_size,
            context_dim=context_dim,
            deg=deg,
            noise_width=noise_width,
            seed=seed,
            nonstationarity=nonstationarity,
            drift_rate=drift_rate,
            shift_interval=shift_interval,
            shift_fraction=shift_fraction,
        )

        # kept for interface compatibility (used by agent constructors)
        self.weights = np.ones(self.n_arms)
        self.capacity = float(2 * (grid_size - 1))   # path length in edges

        self._current_context = None
        self._current_costs = None
        self.t = 0

    # -- properties ---------------------------------------------------------

    @property
    def minimize(self) -> bool:
        return True

    @property
    def per_arm_context(self) -> bool:
        return False

    # -- core interface -----------------------------------------------------

    def reset(self) -> np.ndarray:
        self.t = 0
        self.data.reset()
        self._current_context, self._current_costs = self.data.sample()
        return self._current_context

    def get_context(self) -> np.ndarray:
        self._current_context, self._current_costs = self.data.sample()
        return self._current_context

    def solve(self, predicted_costs: np.ndarray) -> np.ndarray:
        """Solve shortest path given predicted edge costs."""
        return solve_shortest_path_edges(
            np.asarray(predicted_costs, dtype=np.float64), self.grid_size
        )

    def step(
        self,
        selection: np.ndarray,
        full_feedback: bool = False,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        selection = np.asarray(selection).flatten()
        true_costs = self._current_costs.flatten()

        path_cost = float(np.sum(true_costs * selection))

        optimal_path = solve_shortest_path_edges(true_costs, self.grid_size)
        optimal_cost = float(np.sum(true_costs * optimal_path))

        regret = path_cost - optimal_cost

        if full_feedback:
            observed_costs = true_costs.copy()
        else:
            observed_costs = np.zeros(self.n_arms, dtype=np.float32)
            mask = selection > 0.5
            observed_costs[mask] = true_costs[mask]

        info = {
            'regret': regret,
            'path_cost': path_cost,
            'optimal_cost': optimal_cost,
            'true_costs': self._current_costs.copy(),
            'optimal_path': optimal_path,
            'context': self._current_context.copy(),
            'selected_edges': np.where(selection > 0.5)[0],
        }

        reward = -path_cost

        self.t += 1
        self.data.step()
        self._current_context, self._current_costs = self.data.sample()

        return observed_costs, reward, info

    # -- evaluation helpers -------------------------------------------------

    def generate_test_set(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        X_test, Y_test = [], []
        for _ in range(n_samples):
            context, costs = self.data.sample()
            X_test.append(context)
            Y_test.append(costs)
        return np.array(X_test), np.array(Y_test)

    def get_config(self) -> Dict[str, Any]:
        return {
            'grid_size': self.grid_size,
            'n_arms': self.n_arms,
            'context_dim': self.context_dim,
            'capacity': self.capacity,
            'weights': self.weights.tolist(),
        }
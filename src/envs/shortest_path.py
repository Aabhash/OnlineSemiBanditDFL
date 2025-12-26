import numpy as np
from typing import Tuple, Dict, Any

from src.solver import solve_shortest_path_grid
from src.data.shortest_path import SyntheticGridData, WarcraftData


class ShortestPathEnv:
    """Shortest path environment for semi-bandit learning.

    Key differences from knapsack:
    - Objective: Minimize path cost (not maximize value)
    - Constraint: Path must connect source to target (not capacity)
    - Action space: Path through grid (not subset of items)
    - Feedback: Observe costs only for cells in chosen path

    The environment treats grid cells as "arms". Predictions are interpreted
    as cost estimates, and the solver finds the shortest path.
    """

    def __init__(
        self,
        grid_size: int = 5,
        data_type: str = 'synthetic',
        context_dim: int = 10,
        cost_type: str = 'linear',
        noise_std: float = 0.0,
        data_dir: str = 'data/warcraft',
        seed: int = 42,
    ):
        self.grid_size = grid_size
        self.data_type = data_type

        if data_type == 'synthetic':
            self.data = SyntheticGridData(
                grid_size=grid_size,
                context_dim=context_dim,
                cost_type=cost_type,
                noise_std=noise_std,
                seed=seed,
            )
            self.context_dim = context_dim
        elif data_type == 'warcraft':
            self.data = WarcraftData(
                grid_size=grid_size,
                data_dir=data_dir,
                seed=seed,
            )
            self.context_dim = self.data.context_dim
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        self.n_arms = grid_size * grid_size
        self.n_cells = self.n_arms

        self.weights = np.ones(self.n_arms)
        self.capacity = float(2 * grid_size)  # Approximate max path length

        self._current_context = None
        self._current_costs = None
        self.t = 0

    @property
    def minimize(self) -> bool:
        return True

    def reset(self) -> np.ndarray:
        self.t = 0
        self._current_context, self._current_costs = self.data.sample()
        return self._current_context

    def get_context(self) -> np.ndarray:
        self._current_context, self._current_costs = self.data.sample()
        return self._current_context

    def solve(self, predicted_costs: np.ndarray) -> np.ndarray:
        costs_grid = np.asarray(predicted_costs).reshape(self.grid_size, self.grid_size)
        costs_grid = np.maximum(costs_grid, 1e-6)
        path = solve_shortest_path_grid(costs_grid)
        return path.flatten()

    def step(
        self,
        selection: np.ndarray,
        full_feedback: bool = False,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        selection = np.asarray(selection).flatten()
        true_costs = self._current_costs.flatten()

        path_cost = np.sum(true_costs * selection)

        optimal_path = solve_shortest_path_grid(self._current_costs)
        optimal_cost = np.sum(true_costs * optimal_path.flatten())

        regret = path_cost - optimal_cost

        if full_feedback:
            observed_costs = true_costs.copy()
        else:
            observed_costs = np.zeros(self.n_arms)
            mask = selection > 0.5
            observed_costs[mask] = true_costs[mask]

        info = {
            'regret': regret,
            'path_cost': path_cost,
            'optimal_cost': optimal_cost,
            'true_costs': self._current_costs.copy(),
            'optimal_path': optimal_path,
            'context': self._current_context.copy(),
            'selected_cells': np.where(selection > 0.5)[0],
        }

        reward = -path_cost

        self.t += 1
        self._current_context, self._current_costs = self.data.sample()

        return observed_costs, reward, info

    def generate_test_set(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        X_test = []
        Y_test = []

        for _ in range(n_samples):
            context, costs = self.data.sample()
            X_test.append(context)
            Y_test.append(costs)

        return np.array(X_test), np.array(Y_test)

    def get_config(self) -> Dict[str, Any]:
        return {
            'grid_size': self.grid_size,
            'n_arms': self.n_arms,
            'n_cells': self.n_cells,
            'context_dim': self.context_dim,
            'data_type': self.data_type,
            'capacity': self.capacity,
            'weights': self.weights.tolist(),
        }

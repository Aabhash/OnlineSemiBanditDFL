import numpy as np
from typing import Tuple, Dict, Any

from src.solver import (
    n_edges_grid,
    solve_shortest_path_edges,
    compute_edge_slacks,
    ShortestPathSolver,
)


class SyntheticEdgeData:

    def __init__(
        self,
        grid_size: int = 5,
        context_dim: int = 10,
        deg: int = 6,
        noise_width: float = 0.5,
        seed: int = 42,
        **kwargs,  # absorb legacy nonstationarity params
    ):
        self.grid_size = grid_size
        self.n_edges = n_edges_grid(grid_size)
        self.context_dim = context_dim
        self.deg = deg
        self.noise_width = noise_width

        init_rng = np.random.default_rng(seed)

        # Binary feature mask (PyEPO convention)
        self.B = init_rng.integers(0, 2, size=(self.n_edges, context_dim)).astype(float)

        self._sample_seed = seed + 7777
        self.sample_rng = np.random.default_rng(self._sample_seed)

    def reset(self):
        self.sample_rng = np.random.default_rng(self._sample_seed)

    def step(self):
        pass  # stationary DGP

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        x = self.sample_rng.standard_normal(self.context_dim).astype(np.float32)

        linear = self.B @ x / np.sqrt(self.context_dim)
        mean_costs = (linear + 3) ** self.deg + 1
        mean_costs = mean_costs / (3.5 ** self.deg)

        noise = self.sample_rng.uniform(
            1 - self.noise_width,
            1 + self.noise_width,
            size=self.n_edges,
        )
        costs = (mean_costs * noise).astype(np.float32)

        return x, costs


class ShortestPathEdgeEnv:

    def __init__(
        self,
        grid_size: int = 5,
        context_dim: int = 10,
        deg: int = 6,
        noise_width: float = 0.5,
        seed: int = 42,
        **kwargs,  # absorb legacy nonstationarity params
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
        )

        self.sp_solver = ShortestPathSolver(grid_size)

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

    # -- core interface -----------------------------------------------------

    def reset(self) -> np.ndarray:
        self.t = 0
        self.data.reset()
        self._current_context, self._current_costs = self.data.sample()
        return self._current_context

    def get_context(self) -> np.ndarray:
        return self._current_context

    def solve(self, predicted_costs: np.ndarray) -> np.ndarray:
        return self.sp_solver.solve(
            np.asarray(predicted_costs, dtype=np.float64)
        )

    def _advance(self):
        self.t += 1
        self.data.step()
        self._current_context, self._current_costs = self.data.sample()

    def step(
        self,
        selection: np.ndarray,
        full_feedback: bool = False,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        selection = np.asarray(selection).flatten()
        true_costs = self._current_costs.flatten()

        path_cost = float(np.sum(true_costs * selection))

        optimal_path = self.sp_solver.solve(true_costs)
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

    def get_config(self) -> Dict[str, Any]:
        return {
            'grid_size': self.grid_size,
            'n_arms': self.n_arms,
            'context_dim': self.context_dim,
            'capacity': self.capacity,
            'weights': self.weights.tolist(),
        }

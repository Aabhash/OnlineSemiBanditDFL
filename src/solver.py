"""Backwards-compatibility shim — imports forwarded from src.solvers."""

from src.solvers import *  # noqa: F401,F403
from src.solvers import (  # explicit re-exports for type checkers
    solve_knapsack,
    solve_knapsack_gurobi,
    get_capacity,
    KnapsackSolver,
    solve_shortest_path_edges,
    compute_edge_slacks,
    extract_k_alternatives,
    n_edges_grid,
    right_edge_idx,
    down_edge_idx,
    edge_to_nodes,
    ShortestPathSolver,
    solve_fantasy_lineup,
)

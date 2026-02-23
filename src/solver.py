import numpy as np

import gurobipy as gp
from gurobipy import GRB


_GUROBI_ENV = None


def _get_gurobi_env():
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        _GUROBI_ENV = gp.Env(empty=True)
        _GUROBI_ENV.setParam('OutputFlag', 0)
        _GUROBI_ENV.start()
    return _GUROBI_ENV



def solve_knapsack_gurobi(values: np.ndarray, weights: np.ndarray, capacity: float) -> np.ndarray:
    n_items = len(values)

    values = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)

    env = _get_gurobi_env()
    with gp.Model(env=env) as model:
        x = model.addVars(n_items, vtype=GRB.BINARY, name="x")
        model.setObjective(
            gp.quicksum(values[i] * x[i] for i in range(n_items)),
            GRB.MAXIMIZE
        )
        model.addConstr(
            gp.quicksum(weights[i] * x[i] for i in range(n_items)) <= capacity,
            name="capacity"
        )
        model.optimize()

        solution = np.zeros(n_items)
        if model.Status == GRB.OPTIMAL:
            for i in range(n_items):
                solution[i] = x[i].X
        return solution


def solve_knapsack(values: np.ndarray, weights: np.ndarray, capacity: float,
                   tie_break_noise: float = 1e-6) -> np.ndarray:
    values = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)

    if tie_break_noise > 0 and np.std(values) < 1e-4:
        values = values + np.random.randn(len(values)) * tie_break_noise

    return solve_knapsack_gurobi(values, weights, capacity)


def get_capacity(weights: np.ndarray, capacity_fraction: float) -> float:
    return capacity_fraction * np.sum(weights)


def n_edges_grid(n: int) -> int:
    return 2 * n * (n - 1)


def right_edge_idx(i: int, j: int, n: int) -> int:
    return i * (n - 1) + j


def down_edge_idx(i: int, j: int, n: int) -> int:
    return n * (n - 1) + i * n + j


def edge_to_nodes(idx: int, n: int):
    n_right = n * (n - 1)
    if idx < n_right:
        i = idx // (n - 1)
        j = idx % (n - 1)
        return (i, j), (i, j + 1)
    else:
        k = idx - n_right
        i = k // n
        j = k % n
        return (i, j), (i + 1, j)



def _forward_dp(costs: np.ndarray, n: int) -> np.ndarray:
    d = np.full((n, n), np.inf)
    d[0, 0] = 0.0
    for j in range(1, n):
        d[0, j] = d[0, j - 1] + costs[right_edge_idx(0, j - 1, n)]
    for i in range(1, n):
        d[i, 0] = d[i - 1, 0] + costs[down_edge_idx(i - 1, 0, n)]
    for i in range(1, n):
        for j in range(1, n):
            d[i, j] = min(
                d[i - 1, j] + costs[down_edge_idx(i - 1, j, n)],
                d[i, j - 1] + costs[right_edge_idx(i, j - 1, n)],
            )
    return d


def _backward_dp(costs: np.ndarray, n: int) -> np.ndarray:
    d = np.full((n, n), np.inf)
    d[n - 1, n - 1] = 0.0
    for j in range(n - 2, -1, -1):
        d[n - 1, j] = d[n - 1, j + 1] + costs[right_edge_idx(n - 1, j, n)]
    for i in range(n - 2, -1, -1):
        d[i, n - 1] = d[i + 1, n - 1] + costs[down_edge_idx(i, n - 1, n)]
    for i in range(n - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            d[i, j] = min(
                d[i + 1, j] + costs[down_edge_idx(i, j, n)],
                d[i, j + 1] + costs[right_edge_idx(i, j, n)],
            )
    return d


# ---------------------------------------------------------------------------
#  Solver
# ---------------------------------------------------------------------------

def solve_shortest_path_edges(edge_costs: np.ndarray, grid_size: int) -> np.ndarray:
    n = grid_size
    costs = np.asarray(edge_costs, dtype=np.float64)
    d = _forward_dp(costs, n)

    path = np.zeros(n_edges_grid(n))
    i, j = n - 1, n - 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            path[right_edge_idx(i, j, n)] = 1.0
        elif j == 0:
            i -= 1
            path[down_edge_idx(i, j, n)] = 1.0
        else:
            cost_above = d[i - 1, j] + costs[down_edge_idx(i - 1, j, n)]
            cost_left  = d[i, j - 1] + costs[right_edge_idx(i, j - 1, n)]
            if cost_above <= cost_left:          # prefer up on tie
                i -= 1
                path[down_edge_idx(i, j, n)] = 1.0
            else:
                j -= 1
                path[right_edge_idx(i, j, n)] = 1.0

    return path


# ---------------------------------------------------------------------------
#  Slack computation
# ---------------------------------------------------------------------------

def compute_edge_slacks(edge_costs: np.ndarray, grid_size: int) -> np.ndarray:
    n = grid_size
    costs = np.asarray(edge_costs, dtype=np.float64)
    ne = n_edges_grid(n)

    d_f = _forward_dp(costs, n)
    d_b = _backward_dp(costs, n)
    opt = d_f[n - 1, n - 1]

    slacks = np.empty(ne)

    # Right edges: (i,j) -> (i,j+1)
    for i in range(n):
        for j in range(n - 1):
            idx = right_edge_idx(i, j, n)
            slacks[idx] = d_f[i, j] + costs[idx] + d_b[i, j + 1] - opt

    # Down edges: (i,j) -> (i+1,j)
    for i in range(n - 1):
        for j in range(n):
            idx = down_edge_idx(i, j, n)
            slacks[idx] = d_f[i, j] + costs[idx] + d_b[i + 1, j] - opt

    return np.maximum(slacks, 0.0)         # clamp numerical noise


def solve_fantasy_lineup(
    values: np.ndarray,
    salaries: np.ndarray,
    positions: np.ndarray,
    salary_cap: float = 50000,
) -> np.ndarray:
    n_players = len(values)

    values = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)

    env = _get_gurobi_env()
    with gp.Model(env=env) as model:
        x = model.addVars(n_players, vtype=GRB.BINARY, name="x")

        model.setObjective(
            gp.quicksum(values[i] * x[i] for i in range(n_players)),
            GRB.MAXIMIZE
        )

        model.addConstr(
            gp.quicksum(salaries[i] * x[i] for i in range(n_players)) <= salary_cap,
            name="salary_cap"
        )

        position_limits = {
            'QB': (1, 1),
            'RB': (2, 3),
            'WR': (3, 4),
            'TE': (1, 2),
            'DST': (1, 1),
        }

        for pos, (min_count, max_count) in position_limits.items():
            pos_indices = [i for i in range(n_players) if positions[i] == pos]
            if pos_indices:
                model.addConstr(
                    gp.quicksum(x[i] for i in pos_indices) >= min_count,
                    name=f"min_{pos}"
                )
                model.addConstr(
                    gp.quicksum(x[i] for i in pos_indices) <= max_count,
                    name=f"max_{pos}"
                )

        model.addConstr(
            gp.quicksum(x[i] for i in range(n_players)) == 9,
            name="total_players"
        )

        model.optimize()

        solution = np.zeros(n_players)
        if model.Status == GRB.OPTIMAL:
            for i in range(n_players):
                solution[i] = x[i].X
        else:
            return solve_knapsack(values, salaries, salary_cap)

        return solution




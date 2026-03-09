import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.solvers._gurobi import _get_gurobi_env
from src.solvers.knapsack import solve_knapsack


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

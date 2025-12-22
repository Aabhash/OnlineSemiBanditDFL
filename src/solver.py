import numpy as np

import gurobipy as gp
from gurobipy import GRB


def solve_knapsack_greedy(values: np.ndarray, weights: np.ndarray, capacity: float) -> np.ndarray:
    n_items = len(values)
    ratios = values / (weights + 1e-8)
    order = np.argsort(-ratios)

    solution = np.zeros(n_items)
    remaining_capacity = capacity

    for i in order:
        if weights[i] <= remaining_capacity:
            solution[i] = 1
            remaining_capacity -= weights[i]

    return solution


def solve_knapsack_gurobi(values: np.ndarray, weights: np.ndarray, capacity: float) -> np.ndarray:
    n_items = len(values)

    values = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

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


def solve_knapsack(values: np.ndarray, weights: np.ndarray, capacity: float) -> np.ndarray:
    values = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)

    return solve_knapsack_gurobi(values, weights, capacity)


def get_capacity(weights: np.ndarray, capacity_fraction: float) -> float:
    return capacity_fraction * np.sum(weights)


def solve_knapsack_with_sensitivity(values: np.ndarray, weights: np.ndarray,
                                     capacity: float) -> dict:
    n_items = len(values)

    values = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)

    solution = solve_knapsack(values, weights, capacity)

    value_to_weight = values / (weights + 1e-8)

    selected = solution > 0.5
    if selected.any():
        shadow_price = np.min(value_to_weight[selected])
    else:
        shadow_price = 0.0

    reduced_costs = values - shadow_price * weights

    lp_solution = np.zeros(n_items)
    order = np.argsort(-value_to_weight)
    remaining = capacity
    for i in order:
        if weights[i] <= remaining:
            lp_solution[i] = 1.0
            remaining -= weights[i]
        elif remaining > 0:
            lp_solution[i] = remaining / weights[i]
            remaining = 0

    decision_relevance = np.minimum(lp_solution, 1 - lp_solution)

    rc_magnitude = np.abs(reduced_costs)
    rc_relevance = 1.0 / (rc_magnitude + 0.1)
    rc_relevance = rc_relevance / (rc_relevance.sum() + 1e-8)

    return {
        "solution": solution,
        "lp_solution": lp_solution,
        "reduced_costs": reduced_costs,
        "shadow_price": shadow_price,
        "value_to_weight": value_to_weight,
        "decision_relevance": decision_relevance,
        "rc_relevance": rc_relevance,
    }

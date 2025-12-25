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


def solve_top_k(values: np.ndarray, K: int) -> np.ndarray:
    n_items = len(values)
    solution = np.zeros(n_items)
    top_k_indices = np.argsort(-values)[:K]
    solution[top_k_indices] = 1
    return solution


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


def soft_knapsack_lp(values: np.ndarray, weights: np.ndarray, capacity: float) -> np.ndarray:
    n_items = len(values)
    values = np.array(values, dtype=np.float64)
    
    value_to_weight = values / (weights + 1e-8)
    order = np.argsort(-value_to_weight)
    
    lp_solution = np.zeros(n_items)
    remaining = capacity
    
    for i in order:
        if weights[i] <= remaining:
            lp_solution[i] = 1.0
            remaining -= weights[i]
        elif remaining > 0:
            lp_solution[i] = remaining / weights[i]
            remaining = 0
        else:
            lp_solution[i] = 0.0
    
    return lp_solution


def perturbed_soft_knapsack(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    temperature: float = 0.5,
    n_samples: int = 10,
    rng: np.random.Generator = None
) -> tuple:
    if rng is None:
        rng = np.random.default_rng()
    
    n_items = len(values)
    samples = []
    
    for _ in range(n_samples):
        u = rng.uniform(0.001, 0.999, size=n_items)
        gumbel = -np.log(-np.log(u))
        perturbed_values = values + temperature * gumbel
        
        lp_sol = soft_knapsack_lp(perturbed_values, weights, capacity)
        samples.append(lp_sol)

    soft_selection = np.mean(samples, axis=0)
    
    return soft_selection, samples


def soft_knapsack_gradient(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    loss_grad: np.ndarray,
    temperature: float = 0.5,
    n_samples: int = 10,
    rng: np.random.Generator = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    
    n_items = len(values)
    value_grads = np.zeros(n_items)
    
    for _ in range(n_samples):
        u = rng.uniform(0.001, 0.999, size=n_items)
        epsilon = -np.log(-np.log(u)) * temperature
        
        perturbed_values = values + epsilon
        lp_sol = soft_knapsack_lp(perturbed_values, weights, capacity)

        sample_loss = np.dot(loss_grad, lp_sol)
        value_grads += epsilon * sample_loss / (temperature ** 2)
    
    return value_grads / n_samples


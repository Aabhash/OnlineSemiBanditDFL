import jax
import jax.numpy as jnp
import numpy as np
from src.solver import solve_knapsack


def mse_loss(y_pred, y_true) -> float:
    y_pred = jnp.asarray(y_pred)
    y_true = jnp.asarray(y_true)
    return float(((y_pred - y_true) ** 2).mean())


def mse_grad(y_pred, y_true) -> jnp.ndarray:
    y_pred = jnp.asarray(y_pred)
    y_true = jnp.asarray(y_true)
    return 2 * (y_pred - y_true) / len(y_true)


def spo_plus_loss(y_pred, y_true, weights, capacity: float) -> float:
    y_pred_np = np.asarray(y_pred)
    y_true_np = np.asarray(y_true)
    weights_np = np.asarray(weights)

    x_opt = solve_knapsack(y_true_np, weights_np, capacity)
    x_spo = solve_knapsack(2 * y_pred_np - y_true_np, weights_np, capacity)
    loss = y_true_np @ x_opt - y_true_np @ x_spo
    return max(0.0, float(loss))


def spo_plus_grad(y_pred, y_true, weights, capacity: float) -> jnp.ndarray:
    y_pred_np = np.asarray(y_pred)
    y_true_np = np.asarray(y_true)
    weights_np = np.asarray(weights)

    x_opt = solve_knapsack(y_true_np, weights_np, capacity)
    x_spo = solve_knapsack(2 * y_pred_np - y_true_np, weights_np, capacity)
    return jnp.asarray(2 * (x_spo - x_opt))


def spo_plus_grad_balanced(y_pred, y_true, weights, capacity: float) -> jnp.ndarray:
    y_pred_np = np.asarray(y_pred)
    y_true_np = np.asarray(y_true)
    weights_np = np.asarray(weights)

    x_opt = solve_knapsack(y_true_np, weights_np, capacity)
    x_spo = solve_knapsack(2 * y_pred_np - y_true_np, weights_np, capacity)
    raw_grad = 2 * (x_spo - x_opt)

    selected = x_opt > 0.5
    n_selected = max(selected.sum(), 1)
    n_unselected = max((~selected).sum(), 1)

    balanced_grad = raw_grad.copy()
    ratio = n_unselected / n_selected
    balanced_grad[~selected] *= ratio

    return jnp.asarray(balanced_grad)


def spo_plus_grad_bandit(y_pred, y_observed, selection, weights, capacity: float) -> jnp.ndarray:
    y_pred_np = np.asarray(y_pred)
    y_observed_np = np.asarray(y_observed)
    selection_np = np.asarray(selection)
    weights_np = np.asarray(weights)

    n_arms = len(y_pred_np)
    selected_idx = np.where(selection_np > 0.5)[0]

    if len(selected_idx) == 0:
        return jnp.zeros(n_arms)

    y_pred_sel = y_pred_np[selected_idx]
    y_true_sel = y_observed_np[selected_idx]
    weights_sel = weights_np[selected_idx]
    used_capacity = weights_sel.sum()

    x_opt_sel = solve_knapsack(y_true_sel, weights_sel, used_capacity)
    x_spo_sel = solve_knapsack(2 * y_pred_sel - y_true_sel, weights_sel, used_capacity)

    grad = np.zeros(n_arms)
    grad[selected_idx] = 2 * (x_spo_sel - x_opt_sel)

    return jnp.asarray(grad)


def nce_grad_bandit(y_pred, y_observed, selection, solution_pool: list, weights, capacity: float) -> jnp.ndarray:
    y_pred_np = np.asarray(y_pred)
    y_observed_np = np.asarray(y_observed)
    selection_np = np.asarray(selection)

    n_arms = len(y_pred_np)

    if len(solution_pool) == 0:
        return jnp.zeros(n_arms)

    obj_observed = y_observed_np @ selection_np
    obj_predicted = y_pred_np @ selection_np
    advantage = obj_observed - obj_predicted

    grad = np.zeros(n_arms)
    margin = 0.1
    n_violations = 0

    for sol in solution_pool:
        obj_pool = y_pred_np @ sol

        if obj_pool > obj_predicted - margin:
            weight = 1.0 + np.clip(advantage, -1, 1)
            grad -= weight * (sol - selection_np)
            n_violations += 1

    if n_violations > 0:
        grad /= n_violations

    return jnp.asarray(grad)

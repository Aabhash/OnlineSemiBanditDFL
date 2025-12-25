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


def contrastive_dfl_grad(
    y_pred,
    y_observed,
    selection,
    weights,
    capacity: float,
    n_negatives: int = 5,
    margin: float = 0.1,
    rng=None,
) -> jnp.ndarray:
    y_pred_np = np.asarray(y_pred)
    y_observed_np = np.asarray(y_observed)
    selection_np = np.asarray(selection)
    weights_np = np.asarray(weights)

    if rng is None:
        rng = np.random.default_rng()

    n_arms = len(y_pred_np)
    selected_idx = np.where(selection_np > 0.5)[0]

    if len(selected_idx) == 0:
        return jnp.zeros(n_arms)

    observed_value = y_observed_np[selected_idx].sum()

    grad = np.zeros(n_arms)

    for _ in range(n_negatives):
        noise = rng.standard_normal(n_arms) * 0.5
        y_alt = y_pred_np + noise
        alt_selection = solve_knapsack(y_alt, weights_np, capacity)
        alt_idx = np.where(alt_selection > 0.5)[0]

        alt_value = 0.0
        for i in alt_idx:
            if i in selected_idx:
                alt_value += y_observed_np[i]
            else:
                alt_value += y_pred_np[i]

        if alt_value > observed_value + margin:
            for i in alt_idx:
                if selection_np[i] < 0.5:
                    grad[i] -= 1.0 / n_negatives
            for i in selected_idx:
                if alt_selection[i] < 0.5:
                    grad[i] += 1.0 / n_negatives

    return jnp.asarray(grad)


def nce_loss_bandit(y_pred, y_observed, selection, solution_pool: list, weights, capacity: float) -> float:
    y_pred_np = np.asarray(y_pred)
    selection_np = np.asarray(selection)

    if len(solution_pool) == 0:
        return 0.0

    obj_selected = y_pred_np @ selection_np
    pool_objs = np.array([y_pred_np @ sol for sol in solution_pool])
    margins = pool_objs - obj_selected
    loss = np.maximum(0, margins + 0.1).mean()

    return float(loss)


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


def pairwise_ltr_grad_bandit(y_pred, y_observed, selection, solution_pool: list, weights, capacity: float) -> jnp.ndarray:
    y_pred_np = np.asarray(y_pred)
    y_observed_np = np.asarray(y_observed)
    selection_np = np.asarray(selection)

    n_arms = len(y_pred_np)

    if len(solution_pool) == 0:
        return jnp.zeros(n_arms)

    obj_observed = y_observed_np @ selection_np

    grad = np.zeros(n_arms)

    for sol in solution_pool:
        obj_pool_pred = y_pred_np @ sol
        margin = obj_pool_pred - obj_observed

        if margin > 0:
            grad -= (sol - selection_np)

    norm = np.abs(grad).sum()
    if norm > 1e-8:
        grad = grad / norm

    return jnp.asarray(grad)


def listwise_ltr_grad_bandit(
    y_pred,
    y_observed,
    selection,
    solution_pool: list,
    weights,
    capacity: float,
    temperature: float = 1.0,
) -> jnp.ndarray:
    y_pred_np = np.asarray(y_pred)
    y_observed_np = np.asarray(y_observed)
    selection_np = np.asarray(selection)

    n_arms = len(y_pred_np)

    if len(solution_pool) == 0:
        return jnp.zeros(n_arms)

    all_solutions = [selection_np] + list(solution_pool)
    n_solutions = len(all_solutions)

    pred_objs = np.array([y_pred_np @ sol for sol in all_solutions])

    obj_observed = y_observed_np @ selection_np
    target_scores = np.zeros(n_solutions)
    target_scores[0] = obj_observed
    for i in range(1, n_solutions):
        target_scores[i] = y_pred_np @ all_solutions[i]

    def softmax(x, temp):
        x = x / temp
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    pred_probs = softmax(pred_objs, temperature)
    target_probs = softmax(target_scores, temperature)

    prob_diff = pred_probs - target_probs

    grad = np.zeros(n_arms)
    for i, sol in enumerate(all_solutions):
        grad += prob_diff[i] * sol

    return jnp.asarray(grad)

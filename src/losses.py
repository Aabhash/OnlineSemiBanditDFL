import numpy as np
from src.solver import solve_knapsack


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return ((y_pred - y_true) ** 2).mean()


def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / len(y_true)


def spo_plus_loss(y_pred: np.ndarray, y_true: np.ndarray,
                  weights: np.ndarray, capacity: float) -> float:
    x_opt = solve_knapsack(y_true, weights, capacity)
    x_spo = solve_knapsack(2 * y_pred - y_true, weights, capacity)
    loss = y_true @ x_opt - y_true @ x_spo
    return max(0.0, loss)


def spo_plus_grad(y_pred: np.ndarray, y_true: np.ndarray,
                  weights: np.ndarray, capacity: float) -> np.ndarray:
    x_opt = solve_knapsack(y_true, weights, capacity)
    x_spo = solve_knapsack(2 * y_pred - y_true, weights, capacity)
    return 2 * (x_spo - x_opt)


def spo_plus_grad_bandit(
    y_pred: np.ndarray,
    y_observed: np.ndarray,
    selection: np.ndarray,
    weights: np.ndarray,
    capacity: float,
) -> np.ndarray:
    n_arms = len(y_pred)
    selected_idx = np.where(selection > 0.5)[0]

    if len(selected_idx) == 0:
        return np.zeros(n_arms)

    y_pred_sel = y_pred[selected_idx]
    y_true_sel = y_observed[selected_idx]
    weights_sel = weights[selected_idx]
    used_capacity = weights_sel.sum()

    x_opt_sel = solve_knapsack(y_true_sel, weights_sel, used_capacity)
    x_spo_sel = solve_knapsack(2 * y_pred_sel - y_true_sel, weights_sel, used_capacity)

    grad = np.zeros(n_arms)
    grad[selected_idx] = 2 * (x_spo_sel - x_opt_sel)

    return grad


def contrastive_dfl_grad(
    y_pred: np.ndarray,
    y_observed: np.ndarray,
    selection: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    n_negatives: int = 5,
    margin: float = 0.1,
    rng: np.random.Generator = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    n_arms = len(y_pred)
    selected_idx = np.where(selection > 0.5)[0]

    if len(selected_idx) == 0:
        return np.zeros(n_arms)

    observed_value = y_observed[selected_idx].sum()
    predicted_value = y_pred[selected_idx].sum()

    grad = np.zeros(n_arms)

    for _ in range(n_negatives):
        noise = rng.standard_normal(n_arms) * 0.5
        y_alt = y_pred + noise
        alt_selection = solve_knapsack(y_alt, weights, capacity)
        alt_idx = np.where(alt_selection > 0.5)[0]

        alt_value = 0.0
        for i in alt_idx:
            if i in selected_idx:
                alt_value += y_observed[i]
            else:
                alt_value += y_pred[i]

        if alt_value > observed_value + margin:
            for i in alt_idx:
                if selection[i] < 0.5:
                    grad[i] -= 1.0 / n_negatives
            for i in selected_idx:
                if alt_selection[i] < 0.5:
                    grad[i] += 1.0 / n_negatives

        elif observed_value > alt_value + margin:
            pass

    return grad


def nce_loss_bandit(
    y_pred: np.ndarray,
    y_observed: np.ndarray,
    selection: np.ndarray,
    solution_pool: list,
    weights: np.ndarray,
    capacity: float,
) -> float:
    if len(solution_pool) == 0:
        return 0.0

    obj_selected = y_pred @ selection
    pool_objs = np.array([y_pred @ sol for sol in solution_pool])
    margins = pool_objs - obj_selected
    loss = np.maximum(0, margins + 0.1).mean()

    return loss


def nce_grad_bandit(
    y_pred: np.ndarray,
    y_observed: np.ndarray,
    selection: np.ndarray,
    solution_pool: list,
    weights: np.ndarray,
    capacity: float,
) -> np.ndarray:
    n_arms = len(y_pred)

    if len(solution_pool) == 0:
        return np.zeros(n_arms)

    obj_observed = y_observed @ selection
    obj_predicted = y_pred @ selection
    advantage = obj_observed - obj_predicted

    grad = np.zeros(n_arms)
    margin = 0.1
    n_violations = 0

    for sol in solution_pool:
        obj_pool = y_pred @ sol

        if obj_pool > obj_predicted - margin:
            weight = 1.0 + np.clip(advantage, -1, 1)
            grad -= weight * (sol - selection)
            n_violations += 1

    if n_violations > 0:
        grad /= n_violations

    return grad


def pairwise_ltr_grad_bandit(
    y_pred: np.ndarray,
    y_observed: np.ndarray,
    selection: np.ndarray,
    solution_pool: list,
    weights: np.ndarray,
    capacity: float,
) -> np.ndarray:
    n_arms = len(y_pred)

    if len(solution_pool) == 0:
        return np.zeros(n_arms)

    obj_observed = y_observed @ selection

    grad = np.zeros(n_arms)

    for sol in solution_pool:
        obj_pool_pred = y_pred @ sol
        margin = obj_pool_pred - obj_observed

        if margin > 0:
            grad -= (sol - selection)

    norm = np.abs(grad).sum()
    if norm > 1e-8:
        grad = grad / norm

    return grad


def listwise_ltr_grad_bandit(
    y_pred: np.ndarray,
    y_observed: np.ndarray,
    selection: np.ndarray,
    solution_pool: list,
    weights: np.ndarray,
    capacity: float,
    temperature: float = 1.0,
) -> np.ndarray:
    n_arms = len(y_pred)

    if len(solution_pool) == 0:
        return np.zeros(n_arms)

    all_solutions = [selection] + list(solution_pool)
    n_solutions = len(all_solutions)

    pred_objs = np.array([y_pred @ sol for sol in all_solutions])

    obj_observed = y_observed @ selection
    target_scores = np.zeros(n_solutions)
    target_scores[0] = obj_observed
    for i in range(1, n_solutions):
        target_scores[i] = y_pred @ all_solutions[i]

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

    return grad

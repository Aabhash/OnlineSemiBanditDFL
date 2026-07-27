import warnings
import numpy as np
from typing import Dict, List
from scipy.stats import spearmanr, ConstantInputWarning
import src.solver as solver_module
from src.solver import solve_knapsack


def _compute_contested_mask(true_vals, co_solver=None, contested_frac=0.25):
    true_vals = np.asarray(true_vals).flatten()
    n = len(true_vals)

    if co_solver is not None and hasattr(co_solver, 'compute_slacks'):
        try:
            slacks = co_solver.compute_slacks(true_vals)
            threshold = np.percentile(slacks, contested_frac * 100)
            return slacks <= threshold
        except Exception:
            pass

    ranks = np.argsort(np.argsort(true_vals))
    lo = int(n * (0.5 - contested_frac / 2))
    hi = int(n * (0.5 + contested_frac / 2))
    return (ranks >= lo) & (ranks < hi)


def _compute_flip_set(pred, true_vals, solver):
    pred = np.asarray(pred).flatten()
    true_vals = np.asarray(true_vals).flatten()
    w_true = np.asarray(solver(true_vals)).flatten()
    c_bar = 2.0 * pred - true_vals
    w_spo = np.asarray(solver(c_bar)).flatten()
    flip_mask = np.abs(w_spo - w_true) > 0.5
    return flip_mask


def _compute_round_diagnostics(agent, X, true_vals, co_solver=None,
                                solver=None, optimal_action=None):
    true_vals = np.asarray(true_vals).flatten()
    if hasattr(agent, 'predict_mean_and_uncertainty'):
        pred, sigmas = agent.predict_mean_and_uncertainty(X)
    else:
        pred = agent.predict_mean(X)
        sigmas = agent.predict_uncertainty(X)
    contested = _compute_contested_mask(true_vals, co_solver)
    noncontested = ~contested

    result = {
        'overall_mae': np.nan,
        'contested_mae': np.nan,
        'noncontested_mae': np.nan,
        'rank_corr': np.nan,
        'mean_sigma': np.nan,
        'contested_sigma': np.nan,
        'flip_set_size': np.nan,
        'decision_correct': np.nan,
        'contested_sigma_sum': np.nan,
        'flip_contested_mae': np.nan,
    }

    if pred is not None:
        pred = np.asarray(pred).flatten()
        result['overall_mae'] = float(np.mean(np.abs(pred - true_vals)))

        if contested.any():
            result['contested_mae'] = float(np.mean(np.abs(pred[contested] - true_vals[contested])))
        if noncontested.any():
            result['noncontested_mae'] = float(np.mean(np.abs(pred[noncontested] - true_vals[noncontested])))
        if len(pred) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConstantInputWarning)
                corr, _ = spearmanr(pred, true_vals)
            result['rank_corr'] = float(corr) if np.isfinite(corr) else np.nan

        if solver is not None:
            try:
                flip_mask = _compute_flip_set(pred, true_vals, solver)
                result['flip_set_size'] = int(flip_mask.sum())

                if flip_mask.any():
                    result['flip_contested_mae'] = float(
                        np.mean(np.abs(pred[flip_mask] - true_vals[flip_mask])))

                    if sigmas is not None:
                        sigmas_arr = np.asarray(sigmas).flatten()
                        result['contested_sigma_sum'] = float(np.sum(sigmas_arr[flip_mask]))
            except Exception:
                pass

        if optimal_action is not None:
            try:
                agent_decision = np.asarray(solver(pred)).flatten()
                opt = np.asarray(optimal_action).flatten()
                result['decision_correct'] = int(np.array_equal(
                    (agent_decision > 0.5).astype(int),
                    (opt > 0.5).astype(int),
                ))
            except Exception:
                pass

    if sigmas is not None:
        sigmas = np.asarray(sigmas).flatten()
        result['mean_sigma'] = float(np.mean(sigmas))
        if contested.any():
            result['contested_sigma'] = float(np.mean(sigmas[contested]))

    return result


def _compute_mse(Y_pred, Y_true):
    if Y_pred is None:
        return np.nan
    Y_pred = np.asarray(Y_pred).flatten()
    Y_true = np.asarray(Y_true).flatten()
    return np.mean((Y_pred - Y_true) ** 2)


def _compute_spo_loss(Y_pred, Y_true, weights, capacity):
    if Y_pred is None:
        return np.nan
    Y_pred = np.asarray(Y_pred).flatten()
    Y_true = np.asarray(Y_true).flatten()
    x_opt = solve_knapsack(Y_true, weights, capacity)
    x_spo = solve_knapsack(2 * Y_pred - Y_true, weights, capacity)
    loss = Y_true @ x_opt - Y_true @ x_spo
    return max(0.0, loss)


def _compute_test_losses(agent, X_test, Y_test, weights, capacity):
    n_samples = len(X_test)
    mse_total = 0.0
    spo_total = 0.0
    valid_count = 0

    for i in range(n_samples):
        Y_pred = agent.predict_mean(X_test[i])
        if Y_pred is not None:
            mse_total += _compute_mse(Y_pred, Y_test[i])
            spo_total += _compute_spo_loss(Y_pred, Y_test[i], weights, capacity)
            valid_count += 1

    if valid_count == 0:
        return np.nan, np.nan

    return mse_total / valid_count, spo_total / valid_count


def _compute_test_regret(agent, X_test, Y_test, weights, capacity, solver=None, minimize=False):
    n_samples = len(X_test)
    total_regret = 0.0

    for i in range(n_samples):
        selection = agent.act(X_test[i])
        selection = np.asarray(selection).flatten()
        Y_true = np.asarray(Y_test[i]).flatten()

        if solver is not None:
            optimal_selection = solver(Y_test[i])
            optimal_selection = np.asarray(optimal_selection).flatten()
        else:
            optimal_selection = solve_knapsack(Y_true, weights, capacity)

        value = np.sum(Y_true * selection)
        optimal_value = np.sum(Y_true * optimal_selection)

        if minimize:

            total_regret += value - optimal_value
        else:

            total_regret += optimal_value - value

    return total_regret


def pregenerate_episode(
    env, n_rounds: int, full_feedback: bool = False,
) -> dict:
    solver_module._GUROBI_ENV = None

    solver = env.solve
    minimize = getattr(env, 'minimize', False)
    X = env.reset()

    rounds_data = []
    for t in range(n_rounds):
        if hasattr(env, '_current_costs'):
            true_vals = env._current_costs.copy()
        elif hasattr(env, 'current_values'):
            true_vals = env.current_values.copy()
        else:
            raise AttributeError('Cannot extract true values from env')

        optimal = solver(true_vals)
        optimal_val = float(np.sum(true_vals * np.asarray(optimal)))

        rd = {
            'context': X.copy(),
            'true_values': true_vals,
            'optimal_action': np.asarray(optimal).copy(),
            'optimal_value': optimal_val,
        }

        conditional_mean = getattr(env, 'conditional_mean', None)
        if callable(conditional_mean):
            mean_vals = np.asarray(conditional_mean(X)).copy()
            mean_optimal = np.asarray(solver(mean_vals)).copy()
            rd.update({
                'mean_values': mean_vals,
                'mean_optimal_action': mean_optimal,
                'mean_optimal_value': float(np.sum(mean_vals * mean_optimal)),
            })

        rounds_data.append(rd)

        env._advance()
        X = env.get_context()

    return {
        'rounds_data': rounds_data,
        'env': env,
        'minimize': minimize,
    }


def run_agent_on_episode(
    agent,
    episode: dict,
    n_rounds: int,
    lr: float = 0.01,
    full_feedback: bool = False,
    diagnostics: bool = False,
    diagnostics_interval: int = 1,
    co_solver=None,
) -> Dict[str, List[float]]:
    solver_module._GUROBI_ENV = None

    rounds_data = episode['rounds_data']
    minimize = episode['minimize']

    rewards = []
    optimal_rewards = []
    regrets = []
    cumulative_regret = []
    actions = []
    optimal_actions = []
    pseudo_regrets = []
    cumulative_pseudo_regret = []
    mean_agent_values = []
    mean_optimal_values = []
    total_regret = 0.0
    total_pseudo_regret = 0.0
    total_reward = 0.0
    total_optimal = 0.0
    n_arms = len(rounds_data[0]['true_values'])
    obs_counts = np.zeros(n_arms)

    diag_keys = ['overall_mae', 'contested_mae', 'noncontested_mae', 'rank_corr', 'mean_sigma',
                  'contested_sigma', 'flip_set_size', 'decision_correct',
                  'contested_sigma_sum', 'flip_contested_mae']
    diag_accum = {k: [] for k in diag_keys} if diagnostics else None
    diagnostic_rounds = [] if diagnostics else None
    diagnostics_interval = max(int(diagnostics_interval), 1)


    _diag_solver = None
    if diagnostics:
        if co_solver is not None and hasattr(co_solver, 'solve'):
            _diag_solver = co_solver.solve
        elif hasattr(agent, 'solver'):
            _diag_solver = agent.solver

    for t in range(n_rounds):
        rd = rounds_data[t]
        X = rd['context']
        true_vals = rd['true_values']
        optimal_action = rd['optimal_action']
        optimal_val = rd['optimal_value']

        selection = agent.act(X)
        selection = np.asarray(selection).flatten()

        if diagnostics and (
            t == 0
            or (t + 1) % diagnostics_interval == 0
            or t == n_rounds - 1
        ):
            rd_diag = _compute_round_diagnostics(
                agent, X, true_vals, co_solver,
                solver=_diag_solver, optimal_action=optimal_action,
            )
            diagnostic_rounds.append(t + 1)
            for k in diag_keys:
                diag_accum[k].append(rd_diag[k])

        agent_val = float(np.sum(true_vals * selection))
        if minimize:
            instant_regret = agent_val - optimal_val
            reward = -agent_val
        else:
            instant_regret = optimal_val - agent_val
            reward = agent_val

        total_regret += instant_regret
        total_reward += reward
        total_optimal += optimal_val

        rewards.append(reward)
        optimal_rewards.append(optimal_val if not minimize else -optimal_val)
        regrets.append(instant_regret)
        cumulative_regret.append(total_regret)
        actions.append(selection.copy())
        optimal_actions.append(optimal_action.flatten().copy())
        obs_counts += (selection > 0.5).astype(float)

        mean_vals = rd.get('mean_values')
        if mean_vals is not None:
            mean_agent_val = float(np.sum(mean_vals * selection))
            mean_optimal_val = float(rd['mean_optimal_value'])
            if minimize:
                instant_pseudo_regret = mean_agent_val - mean_optimal_val
            else:
                instant_pseudo_regret = mean_optimal_val - mean_agent_val

            total_pseudo_regret += instant_pseudo_regret
            pseudo_regrets.append(instant_pseudo_regret)
            cumulative_pseudo_regret.append(total_pseudo_regret)
            mean_agent_values.append(mean_agent_val)
            mean_optimal_values.append(mean_optimal_val)

        if full_feedback:
            observed = true_vals.copy()
        else:
            observed = np.zeros_like(true_vals)
            mask = selection > 0.5
            observed[mask] = true_vals[mask]


        agent.update(X, selection, observed, lr=lr, full_feedback=full_feedback)

    result = {
        'rewards': rewards,
        'optimal_rewards': optimal_rewards,
        'regrets': regrets,
        'cumulative_regret': cumulative_regret,
        'final_regret': total_regret,
        'final_reward': total_reward,
        'final_optimal': total_optimal,
        'actions': actions,
        'optimal_actions': optimal_actions,
        'obs_counts': obs_counts,
        'pseudo_regrets': pseudo_regrets,
        'cumulative_pseudo_regret': cumulative_pseudo_regret,
        'final_pseudo_regret': total_pseudo_regret if pseudo_regrets else None,
        'mean_agent_values': mean_agent_values,
        'mean_optimal_values': mean_optimal_values,
    }
    if diagnostics and diag_accum is not None:
        result.update(diag_accum)
        result['diagnostic_rounds'] = diagnostic_rounds
    return result


def run_experiment(
    env,
    agent,
    n_rounds: int,
    lr: float = 0.01,
    full_feedback: bool = True,
    eval_interval: int = 0,
) -> Dict[str, List[float]]:
    rewards = []
    optimal_rewards = []
    regrets = []
    cumulative_regret = []
    holdout_evals = []
    actions = []
    optimal_actions = []
    total_regret = 0.0
    total_reward = 0.0
    total_optimal = 0.0

    has_holdout = hasattr(env, 'evaluate_on_holdout') and hasattr(env, 'holdout_weeks') and env.holdout_weeks > 0
    X = env.reset()

    for t in range(n_rounds):
        selection = agent.act(X)
        observed_values, reward, info = env.step(selection, full_feedback=full_feedback)


        if has_holdout and eval_interval > 0 and (t == n_rounds - 1 or (t > 0 and t % eval_interval == 0)):
            holdout_result = env.evaluate_on_holdout(agent)
            holdout_evals.append((t, holdout_result))

        instant_regret = info['regret']
        optimal_reward = info.get('optimal_reward', reward + instant_regret)

        total_regret += instant_regret
        total_reward += reward
        total_optimal += optimal_reward

        rewards.append(reward)
        optimal_rewards.append(optimal_reward)
        regrets.append(instant_regret)
        cumulative_regret.append(total_regret)
        actions.append(np.asarray(selection).copy())

        opt_action = info.get('optimal_selection')
        if opt_action is None:
            opt_action = info.get('optimal_path')
        if opt_action is None:
            opt_action = selection
        optimal_actions.append(np.asarray(opt_action).flatten().copy())


        agent.update(X, selection, observed_values, lr=lr, full_feedback=full_feedback)

        X = env.get_context()

    return {
        'rewards': rewards,
        'optimal_rewards': optimal_rewards,
        'regrets': regrets,
        'cumulative_regret': cumulative_regret,
        'final_regret': total_regret,
        'final_reward': total_reward,
        'final_optimal': total_optimal,
        'holdout_evals': holdout_evals,
        'actions': actions,
        'optimal_actions': optimal_actions,
    }

import warnings
import numpy as np
import jax.numpy as jnp
from typing import Dict, List
from scipy.stats import spearmanr, ConstantInputWarning
from src.agents.bayesian_utils import compute_bayesian_predictions, compute_uncertainties


def _get_sigma(agent, X):
    if hasattr(agent, '_features') and hasattr(agent, 'A_inv'):
        h = agent._features(X)
        return np.asarray(compute_uncertainties(h, agent.A_inv))

    if hasattr(agent, 'model') and hasattr(agent.model, 'predict_ucb') and hasattr(agent.model, 'predict_mean'):
        minimize = getattr(agent, 'minimize', False)
        ucb = np.asarray(agent.model.predict_ucb(X, 1.0, minimize))
        mean = np.asarray(agent.model.predict_mean(X))
        return np.abs(ucb - mean)

    if hasattr(agent, 'A_inv') and hasattr(agent, 'b') and not hasattr(agent, '_features'):
        features = jnp.asarray(X)
        return np.asarray(compute_uncertainties(features, agent.A_inv))

    return None


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
    pred = _get_prediction(agent, X)
    sigmas = _get_sigma(agent, X)
    contested = _compute_contested_mask(true_vals, co_solver)
    noncontested = ~contested

    result = {
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


def _get_prediction(agent, X):
    if hasattr(agent, 'predict_mean') and callable(agent.predict_mean):
        pred = agent.predict_mean(X)
        return np.asarray(pred)

    if hasattr(agent, 'predict') and callable(agent.predict):
        pred = agent.predict(X)
        return np.asarray(pred)

    if hasattr(agent, 'bayesian_model'):
        pred = agent.bayesian_model.predict_mean(X)
        return np.asarray(pred)

    if hasattr(agent, 'A_inv') and hasattr(agent, 'b'):
        X_jax = jnp.asarray(X)
        if len(agent.A_inv.shape) == 3:
            pred = compute_bayesian_predictions(X_jax, agent.A_inv, agent.b)
            return np.asarray(pred)
        else:
            n_arms = len(agent.A_inv)
            predictions = np.zeros(n_arms)
            for i in range(n_arms):
                theta_i = agent.A_inv[i] @ agent.b[i]
                predictions[i] = theta_i @ X
            return predictions

    if hasattr(agent, 'A_inv') and hasattr(agent, 'b_bayes'):
        X_jax = jnp.asarray(X)
        if hasattr(agent, '_forward'):
            result = agent._forward(X_jax)
            features = result[1] if len(result) >= 2 else result
            pred = compute_bayesian_predictions(features, agent.A_inv, agent.b_bayes)
            return np.asarray(pred)

    if hasattr(agent, 'model'):
        model = agent.model
        if hasattr(model, 'predict'):
            pred = model.predict(X)
            return np.asarray(pred)
        elif hasattr(model, 'predict_mean'):
            pred = model.predict_mean(X)
            return np.asarray(pred)

    return None


def _compute_mse(Y_pred, Y_true):
    if Y_pred is None:
        return np.nan
    Y_pred = np.asarray(Y_pred).flatten()
    Y_true = np.asarray(Y_true).flatten()
    return np.mean((Y_pred - Y_true) ** 2)


def _compute_spo_loss(Y_pred, Y_true, weights, capacity):
    from src.solver import solve_knapsack
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
        Y_pred = _get_prediction(agent, X_test[i])
        if Y_pred is not None:
            mse_total += _compute_mse(Y_pred, Y_test[i])
            spo_total += _compute_spo_loss(Y_pred, Y_test[i], weights, capacity)
            valid_count += 1

    if valid_count == 0:
        return np.nan, np.nan

    return mse_total / valid_count, spo_total / valid_count


def _compute_test_regret(agent, X_test, Y_test, weights, capacity, solver=None, minimize=False):
    from src.solver import solve_knapsack
    n_samples = len(X_test)
    total_regret = 0.0

    for i in range(n_samples):
        selection = agent.act(X_test[i])
        selection = np.asarray(selection).flatten()
        Y_true = np.asarray(Y_test[i]).flatten()

        if solver is not None:
            optimal_selection = solver(Y_test[i])  # Pass original shape for solver
            optimal_selection = np.asarray(optimal_selection).flatten()
        else:
            optimal_selection = solve_knapsack(Y_true, weights, capacity)

        value = np.sum(Y_true * selection)
        optimal_value = np.sum(Y_true * optimal_selection)

        if minimize:
            # For minimization: regret = agent_cost - optimal_cost
            total_regret += value - optimal_value
        else:
            # For maximization: regret = optimal_reward - agent_reward
            total_regret += optimal_value - value

    return total_regret


def pregenerate_episode(
    env, n_rounds: int, full_feedback: bool = False,
) -> dict:
    import src.solver as _solver_module

    _solver_module._GUROBI_ENV = None

    solver = env.solve
    minimize = getattr(env, 'minimize', False)
    X = env.reset()

    rounds_data = []
    for t in range(n_rounds):
        if hasattr(env, '_current_costs'):          # ShortestPathEdgeEnv
            true_vals = env._current_costs.copy()
        elif hasattr(env, 'current_values'):        # Knapsack (ConstrainedBanditEnv)
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

        if getattr(env, 'bernoulli_feedback', False):
            rd['prob_values'] = true_vals  # XGBoost probabilities
            true_vals = env.rng.binomial(1, true_vals).astype(float)
            optimal = solver(true_vals)
            optimal_val = float(np.sum(true_vals * np.asarray(optimal)))
            rd['true_values'] = true_vals
            rd['optimal_action'] = np.asarray(optimal).copy()
            rd['optimal_value'] = optimal_val

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
    co_solver=None,
) -> Dict[str, List[float]]:
    import src.solver as _solver_module
    from src.agents.shared_context.baselines import RandomAgent
    from src.agents.shared_context.greedy_agents import GreedyAgent
    from src.agents.shared_context.ts_agents import ThompsonSamplingAgent
    from src.agents.shared_context.ucb_agents import LinUCBAgent, NeuralUCBAgent, UCBAgent
    from src.agents.shared_context.bayesian_spo import BayesianSPOAgent
    from src.agents.shared_context.bayesian_spo_dfe import BayesianSPO_DFE_Agent
    from src.agents.shared_context.bayesian_spo_weber import BayesianSPO_Weber_Agent

    _solver_module._GUROBI_ENV = None

    rounds_data = episode['rounds_data']
    minimize = episode['minimize']

    rewards = []
    optimal_rewards = []
    regrets = []
    cumulative_regret = []
    actions = []
    optimal_actions = []
    total_regret = 0.0
    total_reward = 0.0
    total_optimal = 0.0
    n_arms = len(rounds_data[0]['true_values'])
    obs_counts = np.zeros(n_arms)

    diag_keys = ['contested_mae', 'noncontested_mae', 'rank_corr', 'mean_sigma',
                  'contested_sigma', 'flip_set_size', 'decision_correct',
                  'contested_sigma_sum', 'flip_contested_mae']
    diag_accum = {k: [] for k in diag_keys} if diagnostics else None

    # Resolve solver for flip set computation
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

        if diagnostics:
            rd_diag = _compute_round_diagnostics(
                agent, X, true_vals, co_solver,
                solver=_diag_solver, optimal_action=optimal_action,
            )
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

        # Compute observed values (semi-bandit or full)
        if full_feedback:
            observed = true_vals.copy()
        else:
            observed = np.zeros_like(true_vals)
            mask = selection > 0.5
            observed[mask] = true_vals[mask]

        # Update agent
        if isinstance(agent, (ThompsonSamplingAgent, UCBAgent, LinUCBAgent,
                              NeuralUCBAgent, RandomAgent)):
            agent.update(X, selection, observed, full_feedback=full_feedback)
        elif isinstance(agent, (BayesianSPOAgent, BayesianSPO_DFE_Agent, BayesianSPO_Weber_Agent)):
            prob_vals = rd.get('prob_values')
            agent.update(X, selection, observed, lr, full_feedback=full_feedback,
                         prob_values=prob_vals)
        elif isinstance(agent, GreedyAgent):
            if full_feedback:
                Y_target = true_vals
            else:
                Y_target = np.asarray(agent.model.predict(X)).copy()
                Y_target[selection > 0.5] = observed[selection > 0.5]
            agent.update(X, Y_target, lr)
        else:
            try:
                agent.update(X, selection, observed, lr, full_feedback=full_feedback)
            except TypeError:
                try:
                    agent.update(X, selection, observed, lr)
                except TypeError:
                    agent.update(X, selection, observed, full_feedback=full_feedback)

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
    }
    if diagnostics and diag_accum is not None:
        result.update(diag_accum)
    return result


def run_experiment(
    env,
    agent,
    n_rounds: int,
    lr: float = 0.01,
    full_feedback: bool = True,
    eval_interval: int = 0,
) -> Dict[str, List[float]]:
    from src.agents.shared_context.baselines import RandomAgent
    from src.agents.shared_context.greedy_agents import GreedyAgent
    from src.agents.shared_context.ts_agents import ThompsonSamplingAgent
    from src.agents.shared_context.ucb_agents import LinUCBAgent, NeuralUCBAgent, UCBAgent
    from src.agents.shared_context.bayesian_spo import BayesianSPOAgent
    from src.agents.shared_context.bayesian_spo_weber import BayesianSPO_Weber_Agent

    rewards = []
    optimal_rewards = []
    regrets = []
    cumulative_regret = []
    holdout_evals = []  # (round, holdout_result) pairs
    actions = []  # Track selections/actions
    optimal_actions = []  # Track optimal selections
    total_regret = 0.0
    total_reward = 0.0
    total_optimal = 0.0

    has_holdout = hasattr(env, 'evaluate_on_holdout') and hasattr(env, 'holdout_weeks') and env.holdout_weeks > 0
    X = env.reset()

    for t in range(n_rounds):
        selection = agent.act(X)
        observed_values, reward, info = env.step(selection, full_feedback=full_feedback)

        # Holdout evaluation (for envs that have holdout data)
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
        # Handle different key names for optimal action across environments
        opt_action = info.get('optimal_selection')
        if opt_action is None:
            opt_action = info.get('optimal_path')
        if opt_action is None:
            opt_action = selection  # Fallback
        optimal_actions.append(np.asarray(opt_action).flatten().copy())

        if isinstance(agent, (ThompsonSamplingAgent, UCBAgent, LinUCBAgent,
                              NeuralUCBAgent, RandomAgent)):
            agent.update(X, selection, observed_values, full_feedback=full_feedback)

        elif isinstance(agent, (BayesianSPOAgent, BayesianSPO_Weber_Agent)):
            agent.update(X, selection, observed_values, lr, full_feedback=full_feedback)

        elif isinstance(agent, GreedyAgent):
            if full_feedback:
                Y_target = info['true_values']
            else:
                Y_target = np.asarray(agent.model.predict(X)).copy()
                Y_target[selection > 0.5] = observed_values[selection > 0.5]
            agent.update(X, Y_target, lr)

        else:
            try:
                agent.update(X, selection, observed_values, lr, full_feedback=full_feedback)
            except TypeError:
                try:
                    agent.update(X, selection, observed_values, lr)
                except TypeError:
                    agent.update(X, selection, observed_values, full_feedback=full_feedback)

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

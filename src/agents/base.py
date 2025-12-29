import numpy as np
import jax.numpy as jnp
from typing import Dict, List
from src.agents.bayesian_utils import compute_bayesian_predictions


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


def _compute_test_regret(agent, X_test, Y_test, weights, capacity, solver=None):
    from src.solver import solve_knapsack
    n_samples = len(X_test)
    total_regret = 0.0

    for i in range(n_samples):
        selection = agent.act(X_test[i])
        selection = np.asarray(selection)
        Y_true = Y_test[i]
        if solver is not None:
            optimal_selection = solver(Y_true)
        else:
            optimal_selection = solve_knapsack(Y_true, weights, capacity)
        reward = np.sum(Y_true * selection)
        optimal_reward = np.sum(Y_true * optimal_selection)
        total_regret += optimal_reward - reward

    return total_regret


def run_experiment(
    env,
    agent,
    n_rounds: int,
    lr: float = 0.01,
    full_feedback: bool = True,
    test_set: tuple = None,
    n_test_samples: int = 100,
) -> Dict[str, List[float]]:
    from src.agents.shared_context.baselines import RandomAgent
    from src.agents.shared_context.greedy_agents import GreedyAgent
    from src.agents.shared_context.ts_agents import ThompsonSamplingAgent
    from src.agents.shared_context.ucb_agents import LinUCBAgent, NeuralUCBAgent, UCBAgent
    from src.agents.shared_context.bayesian_spo import BayesianSPOAgent

    rewards = []
    optimal_rewards = []
    regrets = []
    cumulative_regret = []
    test_regrets = []
    holdout_evals = []  # (round, holdout_result) pairs
    total_regret = 0.0
    total_reward = 0.0
    total_optimal = 0.0

    if test_set is not None:
        X_test, Y_test = test_set
    else:
        X_test, Y_test = env.generate_test_set(n_test_samples)

    solver = getattr(env, 'solve', None)
    has_holdout = hasattr(env, 'evaluate_on_holdout') and hasattr(env, 'holdout_weeks') and env.holdout_weeks > 0
    X = env.reset()
    eval_interval = 5  # Evaluate holdout every 5 rounds

    for t in range(n_rounds):
        selection = agent.act(X)
        observed_values, reward, info = env.step(selection, full_feedback=full_feedback)

        if t == n_rounds - 1 or (t > 0 and t % eval_interval == 0):
            test_regret = _compute_test_regret(agent, X_test, Y_test, env.weights, env.capacity, solver)
            test_regrets.append(test_regret)

            # Evaluate on holdout if available
            if has_holdout:
                holdout_result = env.evaluate_on_holdout(agent)
                holdout_evals.append((t, holdout_result))
        else:
            test_regrets.append(np.nan)

        instant_regret = info['regret']
        optimal_reward = info.get('optimal_reward', reward + instant_regret)

        total_regret += instant_regret
        total_reward += reward
        total_optimal += optimal_reward

        rewards.append(reward)
        optimal_rewards.append(optimal_reward)
        regrets.append(instant_regret)
        cumulative_regret.append(total_regret)

        if isinstance(agent, (ThompsonSamplingAgent, UCBAgent, LinUCBAgent,
                              NeuralUCBAgent, RandomAgent)):
            agent.update(X, selection, observed_values, full_feedback=full_feedback)

        elif isinstance(agent, BayesianSPOAgent):
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
        'test_regrets': test_regrets,
        'holdout_evals': holdout_evals,
    }

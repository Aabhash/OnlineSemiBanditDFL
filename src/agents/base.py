import numpy as np
import jax.numpy as jnp
from typing import Dict, List
from src.agents.utils.bayesian import compute_bayesian_predictions


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
    return np.mean((np.asarray(Y_pred) - np.asarray(Y_true)) ** 2)


def _compute_spo_loss(Y_pred, Y_true, weights, capacity):
    from src.solver import solve_knapsack
    if Y_pred is None:
        return np.nan
    Y_pred = np.asarray(Y_pred)
    Y_true = np.asarray(Y_true)
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


def run_experiment(
    env,
    agent,
    n_rounds: int,
    lr: float = 0.01,
    full_feedback: bool = True,
    test_set: tuple = None,
    n_test_samples: int = 100,
) -> Dict[str, List[float]]:
    from src.agents.baselines import RandomAgent, GreedyAgent
    from src.agents.bayesian import ThompsonSamplingAgent, UCBAgent
    from src.agents.ucb import LinUCBAgent, NeuralUCBAgent
    from src.agents.spo import FullSPOAgent, BayesianSPOAgent

    rewards = []
    regrets = []
    cumulative_regret = []
    mse_losses = []
    spo_losses = []
    total_regret = 0.0

    if test_set is not None:
        X_test, Y_test = test_set
    else:
        X_test, Y_test = env.generate_test_set(n_test_samples)

    X = env.reset()

    for t in range(n_rounds):
        selection = agent.act(X)

        observed_values, reward, info = env.step(selection, full_feedback=full_feedback)

        if t == n_rounds - 1:
            mse, spo = _compute_test_losses(agent, X_test, Y_test, env.weights, env.capacity)
            mse_losses.append(mse)
            spo_losses.append(spo)
        else:
            mse_losses.append(np.nan)
            spo_losses.append(np.nan)

        instant_regret = info['regret']
        total_regret += instant_regret

        rewards.append(reward)
        regrets.append(instant_regret)
        cumulative_regret.append(total_regret)

        if isinstance(agent, (ThompsonSamplingAgent, UCBAgent, LinUCBAgent,
                              NeuralUCBAgent, RandomAgent)):
            agent.update(X, selection, observed_values, full_feedback=full_feedback)

        elif isinstance(agent, (FullSPOAgent, BayesianSPOAgent)):
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
        'regrets': regrets,
        'cumulative_regret': cumulative_regret,
        'final_regret': total_regret,
        'mse_losses': mse_losses,
        'spo_losses': spo_losses,
    }

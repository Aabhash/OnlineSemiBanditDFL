"""
Evaluate all agents on synthetic (noise variants) and assortment datasets.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from src.envs import create_env
from src.agents import run_experiment
from src.agents.ucb import LinUCBAgent, NeuralUCBAgent
from src.agents.bayesian import ThompsonSamplingAgent, UCBAgent
from src.agents.spo import BayesianSPOAgent, FullSPOAgent, DecisionUCBAgent
from src.agents.spo_experimental import SimplePKLAgent, InverseRegretAgent
from src.agents.unified import ConservativeUnifiedAgent
from src.models import BayesianLinearModel


DATASETS = {
    'synthetic_0.00': {'n_items': 32, 'n_features': 5, 'capacity': 20.0, 'noise_width': 0.0},
    'synthetic_0.25': {'n_items': 32, 'n_features': 5, 'capacity': 20.0, 'noise_width': 0.25},
    'synthetic_0.50': {'n_items': 32, 'n_features': 5, 'capacity': 20.0, 'noise_width': 0.5},
    'assortment': {'n_products': 50, 'n_slots': 5, 'context_dim': 10},
}


def get_agents(env, seed, lr=0.01, n_rounds=300):
    return {
        'UCB-Bayesian': UCBAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, beta=1.0
        ),
        'ThompsonSampling': ThompsonSamplingAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity
        ),
        'LinUCB': LinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=1.0, lambda_reg=1.0
        ),
        'NeuralUCB': NeuralUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=32, alpha=1.0, lambda_reg=1.0, lr=lr, seed=seed
        ),
        'BayesianSPO': BayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=32, beta=1.0, decay_rounds=n_rounds,
            alpha=0.0, lambda_reg=1.0, seed=seed
        ),
        'FullSPO': FullSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=32, beta_start=1.0, beta_end=1.0, decay_rounds=n_rounds,
            lambda_reg=1.0, seed=seed
        ),
        'DecisionUCB': DecisionUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=32, beta_pred_start=1.0, beta_pred_end=1.0,
            beta_decision=1.0, decay_rounds=n_rounds,
            alpha=0.0, lambda_reg=1.0, seed=seed
        ),
        'SimplePKL': SimplePKLAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=32, beta=1.0, temperature=0.1,
            n_perturbations=10, lambda_reg=1.0, seed=seed
        ),
        'IRM': InverseRegretAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=32, beta=1.0, bound_weight=1.0, mse_weight=1.0,
            lambda_reg=1.0, seed=seed
        ),
        'ConservativeUnified': ConservativeUnifiedAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=32, beta=1.0, diversity_threshold=1.0,
            temperature=0.1, n_perturbations=10, lambda_reg=1.0, seed=seed
        ),
    }


def run_evaluation(n_rounds=10000, n_seeds=5, lr=0.01):
    results = {}

    for dataset_name, config in DATASETS.items():
        print(f'\n{"=" * 60}')
        print(f'DATASET: {dataset_name.upper()}')
        print(f'{"=" * 60}')

        # Determine env type
        env_type = 'assortment' if 'assortment' in dataset_name else 'synthetic'

        results[dataset_name] = {'semi_bandit': {}, 'full': {}}

        for feedback_mode, full_feedback in [('semi_bandit', False), ('full', True)]:
            for seed in range(n_seeds):
                env_base = create_env(env_type, seed=seed, **config)
                agents = get_agents(env_base, seed, lr, n_rounds)

                for agent_name, agent in agents.items():
                    env = create_env(env_type, seed=seed, **config)
                    result = run_experiment(
                        env, agent, n_rounds,
                        lr=lr, full_feedback=full_feedback
                    )

                    if agent_name not in results[dataset_name][feedback_mode]:
                        results[dataset_name][feedback_mode][agent_name] = []
                    results[dataset_name][feedback_mode][agent_name].append(result['final_regret'])

                print(f'Seed {seed + 1}: done', flush=True)

        # Print summary for this dataset
        for feedback_mode in ['semi_bandit', 'full']:
            print(f'\n{feedback_mode.replace("_", " ").title()}:')
            agents_data = results[dataset_name][feedback_mode]
            sorted_agents = sorted(agents_data.items(), key=lambda x: np.mean(x[1]))
            for agent_name, regrets in sorted_agents:
                mean, std = np.mean(regrets), np.std(regrets)
                print(f'  {agent_name:20s}: {mean:8.1f} +/- {std:6.1f}')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_rounds', type=int, default=10000)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    results = run_evaluation(args.n_rounds, args.n_seeds, args.lr)

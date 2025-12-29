import argparse
import numpy as np
import sys
from datetime import datetime

sys.path.insert(0, '.')

from src.envs import create_env
from src.agents import run_experiment
from src.agents.per_arm_context import PerArmBayesianSPOAgent, PerArmRandomAgent
from experiments.ablations.agents import PerArmNTKSPOAgent, PerArmNTKDecisionUCBAgent


DATASETS = {
    'fantasy': {
        'n_players': None,
        'salary_cap': 50000,
        'context_dim': 14,
        'season': 2020,
        'use_position_constraint': False,
        'holdout_weeks': 5,
    },
}


def get_agents(env, seed, lr=0.01, n_rounds=300, hidden_dim=32):
    solver = env.solve

    return {
        'SPO-Bayesian': PerArmBayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=1.0, alpha=0.0, lambda_reg=1.0,
            lr=lr, decay_rounds=n_rounds, solver=solver, seed=seed
        ),
        'SPO-NTK': PerArmNTKSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, gamma=1.0,
            lambda_reg=1.0, lambda_ntk=1.0,
            lr=lr, solver=solver, seed=seed
        ),
        'SPO-NTK-Decision': PerArmNTKDecisionUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, gamma=1.0,
            beta_decision=1.0, lambda_reg=1.0,
            lambda_ntk=1.0, lr=lr, solver=solver, seed=seed
        ),
        'Random': PerArmRandomAgent(
            env.n_arms, env.weights, env.capacity, seed=seed
        ),
    }


def run_ablation(
    dataset: str = 'fantasy',
    n_rounds: int = 200,
    n_seeds: int = 3,
    lr: float = 0.01,
    hidden_dim: int = 32,
    verbose: bool = True,
):
    dataset_config = DATASETS.get(dataset, DATASETS['fantasy']).copy()
    results = {}

    if verbose:
        print(f'\n{"=" * 60}')
        print(f'UNCERTAINTY ESTIMATION ABLATION')
        print(f'{"=" * 60}')
        print(f'Dataset: {dataset}, Rounds: {n_rounds}, Seeds: {n_seeds}')

    for seed in range(n_seeds):
        if verbose:
            print(f'\nSeed {seed + 1}/{n_seeds}:', end=' ', flush=True)

        env = create_env(dataset, seed=seed, **dataset_config)
        test_set = env.generate_test_set(100)
        agents = get_agents(env, seed, lr, n_rounds, hidden_dim)

        seed_results = {}
        for agent_name, agent in agents.items():
            env = create_env(dataset, seed=seed, **dataset_config)
            result = run_experiment(
                env, agent, n_rounds,
                lr=lr, full_feedback=False,
                test_set=test_set
            )
            seed_results[agent_name] = result['final_regret']
            if agent_name not in results:
                results[agent_name] = []
            results[agent_name].append(result['final_regret'])

        if verbose:
            parts = [f'{k}={v:.0f}' for k, v in seed_results.items()]
            print(' '.join(parts))

    return results


def print_results(results):
    print(f'\n{"=" * 60}')
    print(f'RESULTS')
    print(f'{"=" * 60}')

    categories = {
        'SPO-Bayesian': 'Bayesian (main)',
        'SPO-NTK': 'NTK',
        'SPO-NTK-Decision': 'NTK + Decision',
        'Random': 'Baseline',
    }

    print(f'\n{"Agent":<20} {"Mean":>10} {"Std":>10} {"Method":>20}')
    print('-' * 65)

    sorted_agents = sorted(results.items(), key=lambda x: np.mean(x[1]))

    for agent_name, regrets in sorted_agents:
        mean = np.mean(regrets)
        std = np.std(regrets)
        category = categories.get(agent_name, 'Unknown')
        marker = ' **' if agent_name == sorted_agents[0][0] else ''
        print(f'{agent_name:<20} {mean:>10.0f} {std:>10.0f} {category:>20}{marker}')


def main():
    parser = argparse.ArgumentParser(description='Uncertainty estimation ablation')
    parser.add_argument('--dataset', type=str, default='fantasy')
    parser.add_argument('--n_rounds', type=int, default=200)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=32)

    args = parser.parse_args()

    print('=' * 60)
    print('UNCERTAINTY ESTIMATION ABLATION')
    print('=' * 60)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    results = run_ablation(
        dataset=args.dataset,
        n_rounds=args.n_rounds,
        n_seeds=args.n_seeds,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
    )

    print_results(results)
    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()

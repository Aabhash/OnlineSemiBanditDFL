import argparse
import numpy as np
import sys
import yaml
from datetime import datetime

sys.path.insert(0, '.')

from src.envs import create_env
from src.agents import run_experiment
from src.agents.per_arm_context import PerArmBayesianSPOAgent, PerArmRandomAgent
from experiments.ablations.agents import PerArmNTKSPOAgent, PerArmNTKDecisionUCBAgent


DEFAULT_CONFIG_PATH = 'experiments/configs/experiment.yaml'


def load_config(config_path: str = DEFAULT_CONFIG_PATH):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_agents(env, seed, config, n_rounds):
    defaults = config['agent_defaults']
    lr = defaults['lr']
    hidden_dim = defaults['hidden_dim_small']
    lambda_reg = defaults['lambda_reg']
    beta = defaults['beta']
    gamma = defaults['gamma']

    solver = env.solve

    return {
        'SPO-Bayesian': PerArmBayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=beta, alpha=0.0, lambda_reg=lambda_reg,
            lr=lr, decay_rounds=n_rounds, solver=solver, seed=seed
        ),
        'SPO-NTK': PerArmNTKSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, gamma=gamma,
            lambda_reg=lambda_reg, lambda_ntk=lambda_reg,
            lr=lr, solver=solver, seed=seed
        ),
        'SPO-NTK-Decision': PerArmNTKDecisionUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, gamma=gamma,
            beta_decision=beta, lambda_reg=lambda_reg,
            lambda_ntk=lambda_reg, lr=lr, solver=solver, seed=seed
        ),
        'Random': PerArmRandomAgent(
            env.n_arms, env.weights, env.capacity, seed=seed
        ),
    }


def run_ablation(
    dataset: str,
    n_rounds: int,
    n_seeds: int,
    config: dict,
    verbose: bool = True,
):
    dataset_config = config['datasets'][dataset].copy()
    dataset_config.pop('env_type', None)

    defaults = config['agent_defaults']
    lr = defaults['lr']

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
        agents = get_agents(env, seed, config, n_rounds)

        seed_results = {}
        for agent_name, agent in agents.items():
            env = create_env(dataset, seed=seed, **dataset_config)
            result = run_experiment(
                env, agent, n_rounds,
                lr=lr, full_feedback=False,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--dataset', type=str, default='fantasy')
    parser.add_argument('--n_rounds', type=int, default=None)
    parser.add_argument('--n_seeds', type=int, default=None)

    args = parser.parse_args()

    config = load_config(args.config)
    ablation_config = config['ablations']['uncertainty']

    n_rounds = args.n_rounds or ablation_config['n_rounds']
    n_seeds = args.n_seeds or ablation_config['n_seeds']

    print('=' * 60)
    print('UNCERTAINTY ESTIMATION ABLATION')
    print('=' * 60)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    results = run_ablation(
        dataset=args.dataset,
        n_rounds=n_rounds,
        n_seeds=n_seeds,
        config=config,
    )

    print_results(results)
    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()

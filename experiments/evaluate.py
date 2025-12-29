import argparse
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, '.')

from src.envs import create_env
from src.agents import run_experiment

from src.agents.shared_context import (
    BayesianSPOAgent,
    LinUCBAgent,
    NeuralUCBAgent,
    ThompsonSamplingAgent,
    UCBAgent,
    EpsilonGreedyAgent,
    RandomAgent,
)
from src.agents.per_arm_context import (
    PerArmBayesianSPOAgent,
    CNUCBAgent,
    CNTSAgent,
    PerArmLinUCBAgent,
    PerArmThompsonSamplingAgent,
    PerArmMSENeuralAgent,
    PerArmEpsilonGreedyAgent,
    PerArmRandomAgent,
)
from src.models import BayesianLinearModel



DATASETS = {
    'synthetic': {
        'n_items': 32,
        'context_dim': 5,
        'capacity': 20.0,
    },
    'shortest_path_5': {
        'grid_size': 5,
        'data_type': 'synthetic',
        'context_dim': 10,
    },
    'shortest_path_8': {
        'grid_size': 8,
        'data_type': 'synthetic',
        'context_dim': 20,
    },
    'criteo_ads': {
        'n_campaigns': 100,
        'n_select': 10,
        'context_dim': 50,
        'subsample_rate': 0.1,
        'env_fraction': 0.8,
        'noise_std': 0.05,
    },
    'fantasy': {
        'n_players': None,
        'salary_cap': 50000,
        'context_dim': 14,
        'season': 2020,
        'use_position_constraint': False,
        'holdout_weeks': 5,
    },
}

SHARED_CONTEXT_DATASETS = ['synthetic', 'shortest_path_5', 'shortest_path_8', 'criteo_ads']
PER_ARM_CONTEXT_DATASETS = ['fantasy']


def get_agents(env, seed, lr=0.01, n_rounds=300, hidden_dim=None):
    if hidden_dim is None:
        hidden_dim = 64 if env.context_dim > 20 else 32

    solver = env.solve

    minimize = getattr(env, 'minimize', False)

    return {
        'UCB-Bayesian': UCBAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, beta=1.0, minimize=minimize, solver=solver
        ),
        'ThompsonSampling': ThompsonSamplingAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, minimize=minimize, solver=solver
        ),
        'LinUCB': LinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=1.0, lambda_reg=1.0, minimize=minimize, solver=solver
        ),
        'NeuralUCB': NeuralUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, alpha=1.0, lambda_reg=1.0, lr=lr,
            minimize=minimize, solver=solver, seed=seed
        ),
        'BayesianSPO': BayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=1.0, decay_rounds=n_rounds,
            alpha=0.0, lambda_reg=1.0, solver=solver, seed=seed
        ),
        'EpsilonGreedy': EpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, epsilon=0.1, decay_epsilon=True,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'Random': RandomAgent(
            env.n_arms, env.weights, env.capacity, seed=seed
        ),
    }


def get_per_arm_agents(env, seed, lr=0.01, n_rounds=300, hidden_dim=None):
    if hidden_dim is None:
        hidden_dim = 64 if env.context_dim > 20 else 32

    solver = env.solve

    return {
        'PerArmBayesianSPO': PerArmBayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=1.0, alpha=0.0, lambda_reg=1.0,
            lr=lr, decay_rounds=n_rounds, solver=solver, seed=seed
        ),
        'PerArmLinUCB': PerArmLinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=1.0, lambda_reg=1.0, solver=solver
        ),
        'PerArmTS': PerArmThompsonSamplingAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lambda_reg=1.0, sigma=1.0, solver=solver, seed=seed
        ),

        'CN-UCB': CNUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, lambda_reg=1.0, gamma=1.0, lr=lr,
            per_arm_context=True, solver=solver, seed=seed
        ),
        'CN-TS': CNTSAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, lambda_reg=1.0, nu=1.0, lr=lr,
            per_arm_context=True, solver=solver, seed=seed
        ),
        'PerArmMSENeural': PerArmMSENeuralAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=1.0, lambda_reg=1.0,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'PerArmEpsilonGreedy': PerArmEpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, epsilon=0.1, decay_epsilon=True,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'PerArmGreedy': PerArmEpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, epsilon=0.0, decay_epsilon=False,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'PerArmRandom': PerArmRandomAgent(
            env.n_arms, env.weights, env.capacity, seed=seed
        ),
    }


def run_evaluation(
    dataset: str = 'synthetic',
    n_rounds: int = 300,
    n_seeds: int = 5,
    lr: float = 0.01,
    hidden_dim: int = None,
    output_dir: str = 'results',
    noise: float = 0.0,
    verbose: bool = True,
):
    dataset_config = DATASETS.get(dataset, DATASETS['synthetic']).copy()

    if dataset == 'synthetic' and noise != 0.0:
        dataset_config['noise_width'] = noise

    results = {'full': {}, 'semi_bandit': {}}
    trajectories = {}

    if verbose:
        print(f'\n{"=" * 60}')
        print(f'DATASET: {dataset.upper()}')
        print(f'{"=" * 60}')
        noise_info = f', Noise: {dataset_config.get("noise_width", 0.0)}' if dataset == 'synthetic' else ''
        print(f'Rounds: {n_rounds}, Seeds: {n_seeds}, LR: {lr}{noise_info}')

    for feedback_mode, full_feedback in [('semi_bandit', False), ('full', True)]:
        if verbose:
            print(f'\n--- {feedback_mode.replace("_", " ").title()} Feedback ---')

        for seed in range(n_seeds):
            if verbose:
                print(f'Seed {seed + 1}/{n_seeds}:', flush=True)

            env_base = create_env(dataset, seed=seed, **dataset_config)
            test_set = env_base.generate_test_set(100)

            if env_base.per_arm_context:
                agents = get_per_arm_agents(env_base, seed, lr, n_rounds, hidden_dim)
            else:
                agents = get_agents(env_base, seed, lr, n_rounds, hidden_dim)

            for agent_name, agent in agents.items():
                if verbose:
                    print(f'  {agent_name}...', end=' ', flush=True)

                env = create_env(dataset, seed=seed, **dataset_config)

                result = run_experiment(
                    env, agent, n_rounds,
                    lr=lr, full_feedback=full_feedback,
                    test_set=test_set
                )

                if agent_name not in results[feedback_mode]:
                    results[feedback_mode][agent_name] = []
                results[feedback_mode][agent_name].append(result['final_regret'])

                trajectories[(agent_name, dataset, feedback_mode, seed)] = {
                    'cumulative_regret': result['cumulative_regret'],
                    'regrets': result['regrets'],
                    'rewards': result['rewards'],
                    'optimal_rewards': result['optimal_rewards'],
                    'holdout_evals': result.get('holdout_evals', []),
                    'mse_losses': result.get('mse_losses', []),
                    'spo_losses': result.get('spo_losses', []),
                }

                if verbose:
                    holdout_evals = result.get('holdout_evals', [])
                    if holdout_evals:
                        final_holdout = holdout_evals[-1][1]
                        holdout_regret = final_holdout['regret']
                        print(f'regret={result["final_regret"]:.0f}, holdout_regret={holdout_regret:.0f}', flush=True)
                    else:
                        print(f'{result["final_regret"]:.0f}', flush=True)

    return results, trajectories


def print_summary(results, dataset='synthetic', trajectories=None):
    print(f'\n{"=" * 70}')
    print(f'RESULTS SUMMARY - {dataset.upper()}')
    print(f'{"=" * 70}')

    has_holdout = False
    if trajectories:
        for key, data in trajectories.items():
            if key[1] == dataset and data.get('holdout_evals'):
                has_holdout = True
                break

    for feedback_mode in ['semi_bandit', 'full']:
        print(f'\n{feedback_mode.replace("_", " ").title()} Feedback:')

        if has_holdout and feedback_mode == 'semi_bandit':
            print(f'{"Agent":<25} {"Regret":>10} {"Std":>8} {"Holdout":>10} {"Rank":>6}')
            print('-' * 65)
        else:
            print(f'{"Agent":<25} {"Regret":>10} {"Std":>8} {"Rank":>6}')
            print('-' * 55)

        agents = results[feedback_mode]
        sorted_agents = sorted(agents.items(), key=lambda x: np.mean(x[1]))

        holdout_data = {}
        if trajectories and has_holdout and feedback_mode == 'semi_bandit':
            for key, data in trajectories.items():
                agent_name, ds, mode, seed = key
                if ds == dataset and mode == feedback_mode:
                    holdout_evals = data.get('holdout_evals', [])
                    if holdout_evals:
                        final_regret = holdout_evals[-1][1]['regret']
                        if agent_name not in holdout_data:
                            holdout_data[agent_name] = []
                        holdout_data[agent_name].append(final_regret)

        for rank, (agent_name, regrets) in enumerate(sorted_agents, 1):
            mean = np.mean(regrets)
            std = np.std(regrets)
            marker = ' **' if rank == 1 else ''

            if has_holdout and feedback_mode == 'semi_bandit' and agent_name in holdout_data:
                holdout_regret = np.mean(holdout_data[agent_name])
                print(f'{agent_name:<25} {mean:>10.0f} {std:>8.0f} {holdout_regret:>10.0f} {rank:>5}{marker}')
            else:
                print(f'{agent_name:<25} {mean:>10.0f} {std:>8.0f} {rank:>5}{marker}')

    semi_bandit_best = min(results['semi_bandit'].items(), key=lambda x: np.mean(x[1]))
    full_best = min(results['full'].items(), key=lambda x: np.mean(x[1]))

    print(f'\n{"=" * 70}')
    print('WINNERS:')
    print(f'  Semi-bandit: {semi_bandit_best[0]} ({np.mean(semi_bandit_best[1]):.0f})')
    print(f'  Full feedback: {full_best[0]} ({np.mean(full_best[1]):.0f})')


def plot_trajectories(trajectories, dataset, output_dir, feedback_mode='semi_bandit'):
    from experiments.plot_results import plot_trajectories as _plot_trajectories
    _plot_trajectories(trajectories, dataset, output_dir, feedback_mode)


def save_results(results, trajectories, output_dir):
    import json
    import pickle

    os.makedirs(f'{output_dir}/metrics', exist_ok=True)

    results_json = {}
    for key, value in results.items():
        if isinstance(value, dict) and any(isinstance(v, dict) for v in value.values()):
            results_json[key] = {}
            for mode, agents in value.items():
                results_json[key][mode] = {
                    agent: [float(r) for r in regrets]
                    for agent, regrets in agents.items()
                }
        else:
            results_json[key] = {
                agent: [float(r) for r in regrets]
                for agent, regrets in value.items()
            }

    with open(f'{output_dir}/metrics/results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    with open(f'{output_dir}/metrics/trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

    print(f'\nSaved results to {output_dir}/metrics/')


def main():
    parser = argparse.ArgumentParser(description='Evaluate constrained bandits agents')
    parser.add_argument('--quick', action='store_true',
                        help='Quick evaluation (300 rounds, 3 seeds)')
    parser.add_argument('--full', action='store_true',
                        help='Full evaluation (500 rounds, 5 seeds)')
    parser.add_argument('--n_rounds', type=int, default=None,
                        help='Number of rounds (overrides --quick/--full)')
    parser.add_argument('--n_seeds', type=int, default=None,
                        help='Number of seeds (overrides --quick/--full)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='results/eval',
                        help='Output directory')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save results')
    parser.add_argument('--datasets', type=str, default='synthetic',
                        help='Comma-separated list of datasets (synthetic,portfolio,assortment)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise width for synthetic dataset (default: 0.0)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension for neural networks (default: auto based on context_dim)')

    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')]
    for d in datasets:
        if d not in DATASETS:
            print(f"Unknown dataset: {d}. Available: {list(DATASETS.keys())}")
            return

    if args.quick:
        n_rounds = args.n_rounds or 300
        n_seeds = args.n_seeds or 3
    elif args.full:
        n_rounds = args.n_rounds or 2000
        n_seeds = args.n_seeds or 10
    else:
        n_rounds = args.n_rounds or 300
        n_seeds = args.n_seeds or 3

    print('=' * 70)
    print('CONSTRAINED BANDITS EVALUATION')
    print('=' * 70)
    print(f'Mode: {"quick" if args.quick else "full" if args.full else "custom"}')
    print(f'Datasets: {datasets}')
    print(f'Rounds: {n_rounds}')
    print(f'Seeds: {n_seeds}')
    print(f'Learning rate: {args.lr}')
    print(f'Hidden dim: {args.hidden_dim or "auto"}')
    print(f'Noise: {args.noise}')
    print(f'Output: {args.output_dir}')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    all_results = {}
    all_trajectories = {}

    for dataset in datasets:
        results, trajectories = run_evaluation(
            dataset=dataset,
            n_rounds=n_rounds,
            n_seeds=n_seeds,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            output_dir=args.output_dir,
            noise=args.noise,
        )

        print_summary(results, dataset, trajectories)

        all_results[dataset] = results
        all_trajectories.update(trajectories)

    if not args.no_save:
        save_results(all_results, all_trajectories, args.output_dir)

        for dataset in datasets:
            plot_trajectories(all_trajectories, dataset, args.output_dir, 'semi_bandit')

    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()

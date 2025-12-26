import argparse
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, '.')

from src.envs import create_env
from src.agents import run_experiment
from src.agents.ucb import LinUCBAgent, NeuralUCBAgent
from src.agents.bayesian import ThompsonSamplingAgent, UCBAgent
from src.agents.baselines import RandomAgent, EpsilonGreedyAgent, MSENeuralAgent
from src.agents.spo import BayesianSPOAgent, FullSPOAgent, DecisionUCBAgent
from src.agents.pkl import SimplePKLAgent
from src.agents.irm import InverseRegretAgent
from src.agents.unified import ConservativeUnifiedAgent
from src.agents.cn_ucb import CNUCBAgent, CNTSAgent
from src.models import BayesianLinearModel


DATASETS = {
    'synthetic': {
        'n_items': 32,
        'context_dim': 5,
        'capacity': 20.0,
    },
    # REMOVED: 'portfolio' - predictions too good, decisions don't vary much
    'assortment': {
        'n_products': 100,
        'n_slots': 10,
        'context_dim': 10,
    },
    # MovieLens recommendation datasets
    'movielens': {
        'n_movies': 50,
        'n_slots': 5,
        'context_dim': 20,
    },
    'movielens_large': {
        'n_movies': 100,
        'n_slots': 10,
        'context_dim': 30,
    },
    # RecSim recommendation datasets (non-stationary user interests)
    'recsim': {
        'n_docs': 50,
        'slate_size': 5,
        'n_topics': 10,
        'context_dim': 20,
    },
    'recsim_stationary': {
        'n_docs': 50,
        'slate_size': 5,
        'n_topics': 10,
        'context_dim': 20,
        'stationary': True,
    },
    # CN datasets: score_type is extracted from dataset name (cn_linear -> linear)
    'cn_linear': {
        'n_arms': 20,
        'context_dim': 80,
        'K': 4,
    },
    'cn_quadratic': {
        'n_arms': 20,
        'context_dim': 80,
        'K': 4,
    },
    'cn_cosine': {
        'n_arms': 20,
        'context_dim': 80,
        'K': 4,
    },
    # Shortest path datasets (synthetic only - Warcraft requires CNN)
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
    # REMOVED: 'warcraft_12' - image data requires CNN, MLP insufficient
}


def get_agents(env, seed, lr=0.01, n_rounds=300, hidden_dim=None):
    # Scale hidden_dim based on input dimension for high-dim datasets if not specified
    # Use 64 for high-dim (CN datasets with 80 features), 32 for low-dim
    if hidden_dim is None:
        hidden_dim = 64 if env.context_dim > 20 else 32

    # Get solver from environment (handles knapsack, shortest path, etc.)
    solver = env.solve

    # Check if environment is a minimization problem (e.g., shortest path)
    # If so, UCB-based agents should use LCB instead
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
        # 'NeuralUCB': NeuralUCBAgent(
        #     env.context_dim, env.n_arms, env.weights, env.capacity,
        #     hidden_dim=hidden_dim, alpha=1.0, lambda_reg=1.0, lr=lr,
        #     minimize=minimize, solver=solver, seed=seed
        # ),
        'BayesianSPO': BayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=1.0, decay_rounds=n_rounds,
            alpha=0.0, lambda_reg=1.0, solver=solver, seed=seed
        ),
        'FullSPO': FullSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta_start=1.0, beta_end=1.0, decay_rounds=n_rounds,
            lambda_reg=1.0, solver=solver, seed=seed
        ),
        'DecisionUCB': DecisionUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta_pred_start=1.0, beta_pred_end=1.0,
            beta_decision=1.0, decay_rounds=n_rounds,
            alpha=0.0, lambda_reg=1.0, solver=solver, seed=seed
        ),
        # 'SimplePKL': SimplePKLAgent(
        #     env.context_dim, env.n_arms, env.weights, env.capacity,
        #     hidden_dim=hidden_dim, beta=1.0, temperature=0.1,
        #     n_perturbations=10, lambda_reg=1.0, solver=solver, seed=seed
        # ),
        # 'IRM': InverseRegretAgent(
        #     env.context_dim, env.n_arms, env.weights, env.capacity,
        #     hidden_dim=hidden_dim, beta=1.0, bound_weight=1.0, mse_weight=1.0,
        #     lambda_reg=1.0, solver=solver, seed=seed
        # ),
        # 'ConservativeUnified': ConservativeUnifiedAgent(
        #     env.context_dim, env.n_arms, env.weights, env.capacity,
        #     hidden_dim=hidden_dim, beta=1.0, diversity_threshold=1.0,
        #     temperature=0.1, n_perturbations=10, lambda_reg=1.0,
        #     solver=solver, seed=seed
        # ),

        # 'CN-UCB': CNUCBAgent(
        #     env.context_dim, env.n_arms, env.weights, env.capacity,
        #     hidden_dim=hidden_dim, lambda_reg=1.0, gamma=1.0, lr=lr,
        #     constraint_type='knapsack', per_arm_context=False,
        #     solver=solver, seed=seed
        # ),

        # IERM (Hu & Kallus, 2024) is NOT included - it's an offline method
        # See docs/why_not_ierm.md for details

        # Baselines
        'EpsilonGreedy': EpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, epsilon=0.1, decay_epsilon=True,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'MSENeural': MSENeuralAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=1.0, lambda_reg=1.0,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
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
                    'mse_losses': result.get('mse_losses', []),
                    'spo_losses': result.get('spo_losses', []),
                }

                if verbose:
                    print(f'{result["final_regret"]:.0f}', flush=True)

    return results, trajectories


def print_summary(results, dataset='synthetic'):
    print(f'\n{"=" * 70}')
    print(f'RESULTS SUMMARY - {dataset.upper()}')
    print(f'{"=" * 70}')

    for feedback_mode in ['semi_bandit', 'full']:
        print(f'\n{feedback_mode.replace("_", " ").title()} Feedback:')
        print(f'{"Agent":<20} {"Mean":>12} {"Std":>10} {"Rank":>6}')
        print('-' * 50)

        agents = results[feedback_mode]
        sorted_agents = sorted(agents.items(), key=lambda x: np.mean(x[1]))

        for rank, (agent_name, regrets) in enumerate(sorted_agents, 1):
            mean = np.mean(regrets)
            std = np.std(regrets)
            marker = ' **' if rank == 1 else ''
            print(f'{agent_name:<20} {mean:>12.1f} {std:>10.1f} {rank:>6}{marker}')

    semi_bandit_best = min(results['semi_bandit'].items(), key=lambda x: np.mean(x[1]))
    full_best = min(results['full'].items(), key=lambda x: np.mean(x[1]))

    print(f'\n{"=" * 70}')
    print('WINNERS:')
    print(f'  Semi-bandit: {semi_bandit_best[0]} ({np.mean(semi_bandit_best[1]):.1f})')
    print(f'  Full feedback: {full_best[0]} ({np.mean(full_best[1]):.1f})')


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
        n_rounds = args.n_rounds or 2000  # Paper-scale: 2000 rounds
        n_seeds = args.n_seeds or 10      # Paper-scale: 10 seeds
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

        print_summary(results, dataset)

        all_results[dataset] = results
        all_trajectories.update(trajectories)

    if not args.no_save:
        save_results(all_results, all_trajectories, args.output_dir)

    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()

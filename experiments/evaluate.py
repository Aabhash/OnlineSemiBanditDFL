import argparse
import numpy as np
import sys
import os
import yaml
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


DEFAULT_CONFIG_PATH = 'experiments/configs/experiment.yaml'


def load_config(config_path: str = DEFAULT_CONFIG_PATH):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_agents(env, seed, config, n_rounds):
    defaults = config['agent_defaults']
    lr = defaults['lr']
    hidden_dim = defaults['hidden_dim'] if env.context_dim > 20 else defaults['hidden_dim_small']
    lambda_reg = defaults['lambda_reg']
    alpha = defaults['alpha']
    beta = defaults['beta']
    epsilon = defaults['epsilon']
    decay_epsilon = defaults['decay_epsilon']

    solver = env.solve
    minimize = getattr(env, 'minimize', False)

    return {
        'Random': RandomAgent(
            env.n_arms, env.weights, env.capacity, seed=seed
        ),
        'EpsilonGreedy': EpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, epsilon=epsilon, decay_epsilon=decay_epsilon,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'UCB-Bayesian': UCBAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, beta=beta, minimize=minimize, solver=solver
        ),
        'TS': ThompsonSamplingAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, minimize=minimize, solver=solver
        ),
        'LinUCB': LinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=alpha, lambda_reg=lambda_reg, minimize=minimize, solver=solver
        ),
        'NeuralUCB': NeuralUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, alpha=alpha, lambda_reg=lambda_reg, lr=lr,
            minimize=minimize, solver=solver, seed=seed
        ),
        'BayesianSPO': BayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=beta, decay_rounds=n_rounds,
            alpha=0.0, lambda_reg=lambda_reg, solver=solver, seed=seed
        ),
    }


def get_per_arm_agents(env, seed, config, n_rounds):
    defaults = config['agent_defaults']
    lr = defaults['lr']
    hidden_dim = defaults['hidden_dim'] if env.context_dim > 20 else defaults['hidden_dim_small']
    lambda_reg = defaults['lambda_reg']
    alpha = defaults['alpha']
    beta = defaults['beta']
    gamma = defaults['gamma']
    nu = defaults['nu']
    epsilon = defaults['epsilon']
    decay_epsilon = defaults['decay_epsilon']

    solver = env.solve

    return {
        'PerArmRandom': PerArmRandomAgent(
            env.n_arms, env.weights, env.capacity, seed=seed
        ),
        'PerArmGreedy': PerArmEpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, epsilon=0.0, decay_epsilon=False,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'PerArmLinUCB': PerArmLinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=alpha, lambda_reg=lambda_reg, solver=solver
        ),
        'PerArmTS': PerArmThompsonSamplingAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lambda_reg=lambda_reg, sigma=1.0, solver=solver, seed=seed
        ),
        'CN-UCB': CNUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, lambda_reg=lambda_reg, gamma=gamma, lr=lr,
            per_arm_context=True, solver=solver, seed=seed
        ),
        'CN-TS': CNTSAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, lambda_reg=lambda_reg, nu=nu, lr=lr,
            per_arm_context=True, solver=solver, seed=seed
        ),
        'PerArmBayesianSPO': PerArmBayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=beta, alpha=0.0, lambda_reg=lambda_reg,
            lr=lr, decay_rounds=n_rounds, solver=solver, seed=seed
        ),
        'PerArmMSENeural': PerArmMSENeuralAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=beta, lambda_reg=lambda_reg,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'PerArmEpsilonGreedy': PerArmEpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, epsilon=epsilon, decay_epsilon=decay_epsilon,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
    }


def run_evaluation(
    dataset: str,
    n_rounds: int,
    n_seeds: int,
    config: dict,
    output_dir: str,
    noise: float = 0.0,
    verbose: bool = True,
):
    dataset_config = config['datasets'][dataset].copy()
    dataset_config.pop('env_type', None)

    if dataset == 'synthetic' and noise != 0.0:
        dataset_config['noise_width'] = noise

    results = {'full': {}, 'semi_bandit': {}}
    trajectories = {}

    if verbose:
        print(f'\n{"=" * 60}')
        print(f'DATASET: {dataset.upper()}')
        print(f'{"=" * 60}')
        noise_info = f', Noise: {dataset_config.get("noise_width", 0.0)}' if dataset == 'synthetic' else ''
        print(f'Rounds: {n_rounds}, Seeds: {n_seeds}{noise_info}')

    for feedback_mode, full_feedback in [('semi_bandit', False), ('full', True)]:
        if verbose:
            print(f'\n--- {feedback_mode.replace("_", " ").title()} Feedback ---')

        for seed in range(n_seeds):
            if verbose:
                print(f'Seed {seed + 1}/{n_seeds}:', flush=True)

            env_base = create_env(dataset, seed=seed, **dataset_config)
            test_set = env_base.generate_test_set(100)

            if env_base.per_arm_context:
                agents = get_per_arm_agents(env_base, seed, config, n_rounds)
            else:
                agents = get_agents(env_base, seed, config, n_rounds)

            for agent_name, agent in agents.items():
                if verbose:
                    print(f'  {agent_name}...', end=' ', flush=True)

                env = create_env(dataset, seed=seed, **dataset_config)
                lr = config['agent_defaults']['lr']

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


def print_summary(results, dataset, trajectories=None):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--n_rounds', type=int, default=None)
    parser.add_argument('--n_seeds', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--datasets', type=str, default='synthetic')
    parser.add_argument('--noise', type=float, default=0.0)

    args = parser.parse_args()

    config = load_config(args.config)

    datasets = [d.strip() for d in args.datasets.split(',')]
    for d in datasets:
        if d not in config['datasets']:
            raise ValueError(f"Unknown dataset: {d}. Available: {list(config['datasets'].keys())}")

    eval_config = config['evaluation']
    if args.quick:
        n_rounds = args.n_rounds or eval_config['quick']['n_rounds']
        n_seeds = args.n_seeds or eval_config['quick']['n_seeds']
    elif args.full:
        n_rounds = args.n_rounds or eval_config['full']['n_rounds']
        n_seeds = args.n_seeds or eval_config['full']['n_seeds']
    else:
        n_rounds = args.n_rounds or eval_config['quick']['n_rounds']
        n_seeds = args.n_seeds or eval_config['quick']['n_seeds']

    output_dir = args.output_dir or eval_config['output_dir']

    print('=' * 70)
    print('CONSTRAINED BANDITS EVALUATION')
    print('=' * 70)
    print(f'Mode: {"quick" if args.quick else "full" if args.full else "custom"}')
    print(f'Datasets: {datasets}')
    print(f'Rounds: {n_rounds}')
    print(f'Seeds: {n_seeds}')
    print(f'Noise: {args.noise}')
    print(f'Output: {output_dir}')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    all_results = {}
    all_trajectories = {}

    for dataset in datasets:
        results, trajectories = run_evaluation(
            dataset=dataset,
            n_rounds=n_rounds,
            n_seeds=n_seeds,
            config=config,
            output_dir=output_dir,
            noise=args.noise,
        )

        print_summary(results, dataset, trajectories)

        all_results[dataset] = results
        all_trajectories.update(trajectories)

    if not args.no_save:
        save_results(all_results, all_trajectories, output_dir)

        for dataset in datasets:
            plot_trajectories(all_trajectories, dataset, output_dir, 'semi_bandit')

    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()

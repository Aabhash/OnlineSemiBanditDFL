import argparse
import json
import numpy as np
import os
import sys
from itertools import product
from datetime import datetime

sys.path.insert(0, '.')

from src.envs import create_env
from src.agents import run_experiment
from src.agents.shared_context.ucb import LinUCBAgent, NeuralUCBAgent
from src.agents.shared_context.bayesian import ThompsonSamplingAgent, UCBAgent
from src.agents.shared_context.baselines import EpsilonGreedyAgent, MSENeuralAgent
from src.agents.shared_context.bayesian_spo import BayesianSPOAgent
from src.agents.shared_context.full_spo import FullSPOAgent
from src.agents.shared_context.decision_ucb import DecisionUCBAgent
from src.agents.per_arm_context.per_arm_bayesian_spo import PerArmBayesianSPOAgent
from src.agents.per_arm_context.per_arm_spo import PerArmFullSPOAgent
from src.agents.per_arm_context.per_arm_ntk_spo import PerArmNTKSPOAgent, PerArmNTKDecisionUCBAgent
from src.agents.per_arm_context.cn_ucb import CNUCBAgent, CNTSAgent
from src.agents.per_arm_context.per_arm_bayesian import PerArmLinUCBAgent
from src.agents.per_arm_context.per_arm_baselines import PerArmNeuralUCBAgent
from src.models import BayesianLinearModel


HYPERPARAMETER_GRIDS = {
    'BayesianSPO': {
        'beta': [0.5, 1.0, 2.0],
        'alpha': [0.0, 0.25, 0.5],
        'hidden_dim': [32, 64],
        'lambda_reg': [0.1, 1.0],
    },
    'FullSPO': {
        'beta_start': [0.5, 1.0, 2.0],
        'hidden_dim': [32, 64],
        'lambda_reg': [0.1, 1.0],
    },
    'DecisionUCB': {
        'beta_pred_start': [0.5, 1.0, 2.0],
        'beta_decision': [0.5, 1.0, 2.0],
        'hidden_dim': [32, 64],
    },
    'NeuralUCB': {
        'alpha': [0.5, 1.0, 2.0],
        'lambda_reg': [0.1, 1.0],
        'hidden_dim': [32, 64],
    },
    'LinUCB': {
        'alpha': [0.5, 1.0, 2.0],
        'lambda_reg': [0.1, 1.0, 10.0],
    },
    'EpsilonGreedy': {
        'epsilon': [0.05, 0.1, 0.2],
        'decay_epsilon': [True, False],
        'hidden_dim': [32, 64],
    },
    'MSENeural': {
        'beta': [0.5, 1.0, 2.0],
        'hidden_dim': [32, 64],
        'lambda_reg': [0.1, 1.0],
    },
    'UCB-Bayesian': {
        'beta': [0.5, 1.0, 2.0],
    },
    'ThompsonSampling': {},
    # Per-arm context agents
    'PerArmNTKSPO': {
        'gamma_start': [1.0, 2.0, 3.0],
        'gamma_end': [0.05, 0.1, 0.2],
        'lambda_ntk': [0.1, 1.0, 10.0],
        'lambda_reg': [0.1, 1.0],
        'hidden_dim': [32, 64],
    },
    'PerArmNTKDecisionUCB': {
        'gamma_start': [1.0, 2.0],
        'beta_decision': [0.5, 1.0],
        'lambda_ntk': [0.1, 1.0, 10.0],
        'hidden_dim': [32, 64],
    },
    'PerArmBayesianSPO': {
        'beta_start': [0.5, 1.0, 2.0],
        'hidden_dim': [32, 64],
        'lambda_reg': [0.1, 1.0],
    },
    'PerArmFullSPO': {
        'beta_start': [0.5, 1.0, 2.0],
        'hidden_dim': [32, 64],
    },
    'CN-UCB': {
        'alpha': [0.5, 1.0, 2.0],
        'lambda_reg': [0.1, 1.0],
        'hidden_dim': [32, 64],
    },
    'CN-TS': {
        'hidden_dim': [32, 64],
        'lambda_reg': [0.1, 1.0],
    },
    'PerArmLinUCB': {
        'alpha': [0.5, 1.0, 2.0],
        'lambda_reg': [0.1, 1.0, 10.0],
    },
    'PerArmNeuralUCB': {
        'alpha': [0.5, 1.0, 2.0],
        'lambda_reg': [0.1, 1.0],
        'hidden_dim': [32, 64],
    },
}


DATASET_CONFIGS = {
    'synthetic': {
        'n_items': 32,
        'context_dim': 5,
        'capacity': 20.0,
    },
    'movielens': {
        'n_movies': 50,
        'n_slots': 5,
        'context_dim': 20,
    },
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
    'assortment': {
        'n_products': 100,
        'n_slots': 10,
        'context_dim': 10,
    },
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
    # Real-world datasets
    'fantasy': {
        'n_players': 100,
        'salary_cap': 50000,
        'context_dim': 20,
    },
    'criteo_ads': {
        'n_campaigns': 100,
        'n_select': 10,
        'context_dim': 50,
    },
}

# Shared context datasets (use BayesianSPO, DecisionUCB, etc.)
SHARED_CONTEXT_DATASETS = ['synthetic', 'shortest_path_5', 'criteo_ads']

# Per-arm context datasets (use NTK-SPO, CN-UCB, etc.)
PER_ARM_CONTEXT_DATASETS = ['fantasy', 'cn_linear', 'cn_quadratic', 'cn_cosine']

# Shared context agents
SHARED_CONTEXT_AGENTS = ['BayesianSPO', 'DecisionUCB', 'FullSPO', 'LinUCB', 'NeuralUCB',
                         'MSENeural', 'UCB-Bayesian', 'ThompsonSampling', 'EpsilonGreedy']

# Per-arm context agents
PER_ARM_CONTEXT_AGENTS = ['PerArmNTKSPO', 'PerArmNTKDecisionUCB', 'PerArmBayesianSPO',
                          'PerArmFullSPO', 'CN-UCB', 'CN-TS', 'PerArmLinUCB', 'PerArmNeuralUCB']

DEFAULT_DATASETS = ['synthetic', 'fantasy', 'cn_linear', 'shortest_path_5']


def product_dict(d):
    if not d:
        yield {}
        return
    keys = list(d.keys())
    for values in product(*d.values()):
        yield dict(zip(keys, values))


def create_agent(agent_name, env, params, seed, n_rounds, lr):
    solver = env.solve
    minimize = getattr(env, 'minimize', False)

    if agent_name == 'BayesianSPO':
        return BayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed, **params
        )
    elif agent_name == 'FullSPO':
        beta_start = params.get('beta_start', 1.0)
        return FullSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            beta_start=beta_start, beta_end=beta_start,
            decay_rounds=n_rounds, solver=solver, seed=seed,
            hidden_dim=params.get('hidden_dim', 32),
            lambda_reg=params.get('lambda_reg', 1.0),
        )
    elif agent_name == 'DecisionUCB':
        beta_pred = params.get('beta_pred_start', 1.0)
        return DecisionUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            beta_pred_start=beta_pred, beta_pred_end=beta_pred,
            decay_rounds=n_rounds, solver=solver, seed=seed,
            hidden_dim=params.get('hidden_dim', 32),
            beta_decision=params.get('beta_decision', 1.0),
        )
    elif agent_name == 'NeuralUCB':
        return NeuralUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=lr, minimize=minimize, solver=solver, seed=seed, **params
        )
    elif agent_name == 'LinUCB':
        return LinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            minimize=minimize, solver=solver, **params
        )
    elif agent_name == 'EpsilonGreedy':
        return EpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed, **params
        )
    elif agent_name == 'MSENeural':
        return MSENeuralAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed, **params
        )
    elif agent_name == 'UCB-Bayesian':
        return UCBAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, minimize=minimize, solver=solver, **params
        )
    elif agent_name == 'ThompsonSampling':
        return ThompsonSamplingAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, minimize=minimize, solver=solver
        )
    # Per-arm context agents
    elif agent_name == 'PerArmNTKSPO':
        return PerArmNTKSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            gamma_start=params.get('gamma_start', 2.0),
            gamma_end=params.get('gamma_end', 0.1),
            decay_rounds=n_rounds,
            lambda_reg=params.get('lambda_reg', 1.0),
            lambda_ntk=params.get('lambda_ntk', 1.0),
            hidden_dim=params.get('hidden_dim', 64),
            lr=lr, solver=solver, seed=seed
        )
    elif agent_name == 'PerArmNTKDecisionUCB':
        return PerArmNTKDecisionUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            gamma_start=params.get('gamma_start', 2.0),
            gamma_end=params.get('gamma_end', 0.1),
            decay_rounds=n_rounds,
            lambda_reg=params.get('lambda_reg', 1.0),
            lambda_ntk=params.get('lambda_ntk', 1.0),
            hidden_dim=params.get('hidden_dim', 64),
            beta_decision=params.get('beta_decision', 1.0),
            lr=lr, solver=solver, seed=seed
        )
    elif agent_name == 'PerArmBayesianSPO':
        return PerArmBayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            beta_start=params.get('beta_start', 2.0),
            beta_end=params.get('beta_end', 0.1),
            decay_rounds=n_rounds,
            lambda_reg=params.get('lambda_reg', 1.0),
            hidden_dim=params.get('hidden_dim', 64),
            lr=lr, solver=solver, seed=seed
        )
    elif agent_name == 'PerArmFullSPO':
        return PerArmFullSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            beta_start=params.get('beta_start', 2.0),
            beta_end=params.get('beta_end', 0.1),
            decay_rounds=n_rounds,
            hidden_dim=params.get('hidden_dim', 64),
            lr=lr, solver=solver, seed=seed
        )
    elif agent_name == 'CN-UCB':
        return CNUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=params.get('alpha', 1.0),
            lambda_reg=params.get('lambda_reg', 1.0),
            hidden_dim=params.get('hidden_dim', 64),
            lr=lr, solver=solver, seed=seed
        )
    elif agent_name == 'CN-TS':
        return CNTSAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lambda_reg=params.get('lambda_reg', 1.0),
            hidden_dim=params.get('hidden_dim', 64),
            lr=lr, solver=solver, seed=seed
        )
    elif agent_name == 'PerArmLinUCB':
        return PerArmLinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=params.get('alpha', 1.0),
            lambda_reg=params.get('lambda_reg', 1.0),
            solver=solver
        )
    elif agent_name == 'PerArmNeuralUCB':
        return PerArmNeuralUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=params.get('alpha', 1.0),
            lambda_reg=params.get('lambda_reg', 1.0),
            hidden_dim=params.get('hidden_dim', 64),
            lr=lr, solver=solver, seed=seed
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def run_single_experiment(
    agent_name,
    dataset,
    params,
    seed,
    n_rounds,
    lr,
    full_feedback,
    config_idx=0,
    use_wandb=False,
    wandb_project=None,
):
    dataset_config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS['synthetic'])
    feedback_type = 'full' if full_feedback else 'semi_bandit'

    env = create_env(dataset, seed=seed, **dataset_config)
    agent = create_agent(agent_name, env, params, seed, n_rounds, lr)

    run_name = f"{feedback_type}_{dataset}_{agent_name}_c{config_idx}_s{seed}"

    run_config = {
        'agent': agent_name,
        'dataset': dataset,
        'feedback_type': feedback_type,
        'full_feedback': full_feedback,
        'seed': seed,
        'n_rounds': n_rounds,
        'lr': lr,
        'dataset_config': dataset_config,
        **params,
    }

    # Initialize wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project or 'constrained-bandits-tuning',
            name=run_name,
            config=run_config,
            group=f"{dataset}_{agent_name}",
            tags=[feedback_type, dataset, agent_name],
            reinit=True,
        )

    result = run_experiment(
        env, agent, n_rounds,
        lr=lr, full_feedback=full_feedback,
    )

    if use_wandb:
        import wandb
        for t in range(len(result['cumulative_regret'])):
            log_dict = {
                'round': t,
                'train/cumulative_regret': result['cumulative_regret'][t],
                'train/instantaneous_regret': result['regrets'][t],
            }
            if 'test_regrets' in result and not np.isnan(result['test_regrets'][t]):
                log_dict['test/regret'] = result['test_regrets'][t]
            wandb.log(log_dict)

        wandb.summary['train/final_regret'] = result['final_regret']
        if 'test_regrets' in result:
            non_nan = [r for r in result['test_regrets'] if not np.isnan(r)]
            if non_nan:
                wandb.summary['test/final_regret'] = non_nan[-1]

        wandb.finish()

    return result


def tune_agent(
    agent_name,
    dataset,
    n_rounds=500,
    n_seeds=3,
    lr=0.01,
    full_feedback=False,
    verbose=True,
    use_wandb=False,
    wandb_project=None,
):
    grid = HYPERPARAMETER_GRIDS.get(agent_name, {})
    feedback_type = 'full' if full_feedback else 'semi_bandit'

    if not grid:
        if verbose:
            print(f"No hyperparameters to tune for {agent_name}")
        grid = {'_default': [True]}

    best_regret = float('inf')
    best_params = None
    all_results = []

    param_combinations = list(product_dict(grid))
    if verbose:
        print(f"\nTuning {agent_name} on {dataset} ({feedback_type}): {len(param_combinations)} combinations")

    for i, params in enumerate(param_combinations):
        actual_params = {k: v for k, v in params.items() if k != '_default'}

        regrets = []
        test_regrets = []

        for seed in range(n_seeds):
            result = run_single_experiment(
                agent_name=agent_name,
                dataset=dataset,
                params=actual_params,
                seed=seed,
                n_rounds=n_rounds,
                lr=lr,
                full_feedback=full_feedback,
                config_idx=i,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
            )

            regrets.append(result['final_regret'])
            if 'test_regrets' in result:
                non_nan = [r for r in result['test_regrets'] if not np.isnan(r)]
                if non_nan:
                    test_regrets.append(non_nan[-1])

        mean_regret = np.mean(regrets)
        std_regret = np.std(regrets)
        mean_test_regret = np.mean(test_regrets) if test_regrets else np.nan

        all_results.append({
            'params': actual_params,
            'mean_train_regret': float(mean_regret),
            'std_train_regret': float(std_regret),
            'mean_test_regret': float(mean_test_regret) if not np.isnan(mean_test_regret) else None,
            'train_regrets': [float(r) for r in regrets],
            'test_regrets': [float(r) for r in test_regrets] if test_regrets else [],
        })

        if mean_regret < best_regret:
            best_regret = mean_regret
            best_params = actual_params

        if verbose:
            test_str = f", test={mean_test_regret:.1f}" if not np.isnan(mean_test_regret) else ""
            print(f"  [{i+1}/{len(param_combinations)}] {actual_params} -> train={mean_regret:.1f}+/-{std_regret:.1f}{test_str}")

    if verbose:
        print(f"\nBest params for {agent_name} on {dataset} ({feedback_type}): {best_params}")
        print(f"Best train regret: {best_regret:.1f}")

    return best_params, best_regret, all_results


def run_full_tuning(
    agents,
    datasets,
    n_rounds=500,
    n_seeds=3,
    lr=0.01,
    feedback_types=None,
    use_wandb=False,
    wandb_project=None,
    output_dir='experiments/tuning/results',
):
    """Run hyperparameter tuning for multiple agents across multiple datasets."""
    if feedback_types is None:
        feedback_types = [False]  # Default: semi-bandit only

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    best_hyperparams = {}

    for full_feedback in feedback_types:
        feedback_name = 'full' if full_feedback else 'semi_bandit'

        for dataset in datasets:
            print(f"\n{'='*70}")
            print(f"DATASET: {dataset.upper()} | FEEDBACK: {feedback_name.upper()}")
            print(f"{'='*70}")

            key = f"{dataset}_{feedback_name}"
            all_results[key] = {}
            best_hyperparams[key] = {}

            for agent_name in agents:
                try:
                    best_params, best_regret, agent_results = tune_agent(
                        agent_name, dataset,
                        n_rounds=n_rounds,
                        n_seeds=n_seeds,
                        lr=lr,
                        full_feedback=full_feedback,
                        use_wandb=use_wandb,
                        wandb_project=wandb_project,
                    )

                    all_results[key][agent_name] = {
                        'best_params': best_params,
                        'best_regret': float(best_regret),
                        'all_results': agent_results,
                    }

                    best_hyperparams[key][agent_name] = {
                        'params': best_params,
                        'train_regret': float(best_regret),
                    }

                except Exception as e:
                    print(f"  ERROR tuning {agent_name} on {dataset}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[key][agent_name] = {'error': str(e)}

            output_file = f'{output_dir}/tuning_{key}.json'
            with open(output_file, 'w') as f:
                json.dump(all_results[key], f, indent=2)
            print(f"Saved results to {output_file}")

    combined_file = f'{output_dir}/tuning_all.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    best_params_file = f'{output_dir}/best_hyperparams.json'
    with open(best_params_file, 'w') as f:
        json.dump(best_hyperparams, f, indent=2)
    print(f"\nSaved best hyperparameters to {best_params_file}")

    return all_results, best_hyperparams


def print_summary(results, best_hyperparams):
    """Print a summary table of best hyperparameters."""
    print('\n' + '=' * 70)
    print('TUNING SUMMARY - BEST HYPERPARAMETERS')
    print('=' * 70)

    for key, agents in best_hyperparams.items():
        print(f"\n{key.upper()}:")
        print(f"{'Agent':<20} {'Train Regret':>12} {'Best Params'}")
        print('-' * 70)

        sorted_agents = sorted(
            [(a, r) for a, r in agents.items() if 'params' in r],
            key=lambda x: x[1]['train_regret']
        )

        for agent_name, result in sorted_agents:
            params_str = ', '.join(f"{k}={v}" for k, v in (result['params'] or {}).items())
            if len(params_str) > 35:
                params_str = params_str[:32] + '...'
            print(f"{agent_name:<20} {result['train_regret']:>12.1f} {params_str}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for bandits agents')
    parser.add_argument('--agent', type=str, default=None,
                        help='Agent to tune (e.g., BayesianSPO, NeuralUCB)')
    parser.add_argument('--all', action='store_true',
                        help='Tune all agents')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset to use for tuning')
    parser.add_argument('--all_datasets', action='store_true',
                        help='Tune on all default datasets')
    parser.add_argument('--n_rounds', type=int, default=500,
                        help='Number of rounds per run')
    parser.add_argument('--n_seeds', type=int, default=3,
                        help='Number of seeds per configuration')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--both_feedback', action='store_true',
                        help='Run both full and semi-bandit feedback')
    parser.add_argument('--full_feedback', action='store_true',
                        help='Use full feedback (default: semi-bandit)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='constrained-bandits-tuning',
                        help='W&B project name')
    parser.add_argument('--output_dir', type=str, default='experiments/tuning/results',
                        help='Output directory for results')

    args = parser.parse_args()

    print('=' * 70)
    print('HYPERPARAMETER TUNING')
    print('=' * 70)
    print(f'Rounds: {args.n_rounds}')
    print(f'Seeds: {args.n_seeds}')
    print(f'Learning rate: {args.lr}')
    print(f'W&B logging: {args.wandb}')
    print(f'Output dir: {args.output_dir}')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    if args.all:
        agents_to_tune = list(HYPERPARAMETER_GRIDS.keys())
    elif args.agent:
        if args.agent not in HYPERPARAMETER_GRIDS:
            print(f"Unknown agent: {args.agent}")
            print(f"Available: {list(HYPERPARAMETER_GRIDS.keys())}")
            return
        agents_to_tune = [args.agent]
    else:
        print("Specify --agent <name> or --all")
        return

    if args.all_datasets:
        datasets_to_tune = DEFAULT_DATASETS
    elif args.dataset:
        if args.dataset not in DATASET_CONFIGS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(DATASET_CONFIGS.keys())}")
            return
        datasets_to_tune = [args.dataset]
    else:
        datasets_to_tune = ['synthetic']

    if args.both_feedback:
        feedback_types = [False, True]  # semi-bandit and full
    elif args.full_feedback:
        feedback_types = [True]
    else:
        feedback_types = [False]  # semi-bandit only

    feedback_names = ['full' if f else 'semi_bandit' for f in feedback_types]
    print(f'Agents: {agents_to_tune}')
    print(f'Datasets: {datasets_to_tune}')
    print(f'Feedback types: {feedback_names}')

    results, best_hyperparams = run_full_tuning(
        agents=agents_to_tune,
        datasets=datasets_to_tune,
        n_rounds=args.n_rounds,
        n_seeds=args.n_seeds,
        lr=args.lr,
        feedback_types=feedback_types,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir,
    )

    print_summary(results, best_hyperparams)
    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()

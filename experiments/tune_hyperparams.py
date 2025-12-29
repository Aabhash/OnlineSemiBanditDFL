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
from src.agents.shared_context import (
    BayesianSPOAgent,
    LinUCBAgent,
    NeuralUCBAgent,
    ThompsonSamplingAgent,
    UCBAgent,
    EpsilonGreedyAgent,
    MSENeuralAgent,
)
from src.agents.per_arm_context import (
    PerArmBayesianSPOAgent,
    CNUCBAgent,
    CNTSAgent,
    PerArmLinUCBAgent,
    PerArmThompsonSamplingAgent,
    PerArmMSENeuralAgent,
    PerArmNeuralUCBAgent,
)
from experiments.ablations.agents import (
    DecisionUCBAgent,
    FullSPOAgent,
    PerArmFullSPOAgent,
    PerArmNTKSPOAgent,
    PerArmNTKDecisionUCBAgent,
)
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
        'gamma': [0.5, 1.0, 2.0],
        'lambda_reg': [0.1, 1.0],
        'hidden_dim': [32, 64],
        'train_interval': [1, 5],
        'train_epochs': [5, 10],
    },
    'CN-TS': {
        'nu': [0.5, 1.0, 2.0],
        'lambda_reg': [0.1, 1.0],
        'hidden_dim': [32, 64],
        'train_interval': [1, 5],
        'train_epochs': [5, 10],
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
    'fantasy': {
        'n_players': 100,
        'salary_cap': 50000,
        'context_dim': 20,
        'holdout_weeks': 5,
    },
    'criteo_ads': {
        'n_campaigns': 100,
        'n_select': 10,
        'context_dim': 50,
    },
}

SHARED_CONTEXT_DATASETS = ['synthetic', 'shortest_path_5', 'criteo_ads']

PER_ARM_CONTEXT_DATASETS = ['fantasy', 'cn_linear', 'cn_quadratic', 'cn_cosine']

SHARED_CONTEXT_AGENTS = ['BayesianSPO', 'DecisionUCB', 'FullSPO', 'LinUCB', 'NeuralUCB',
                         'MSENeural', 'UCB-Bayesian', 'ThompsonSampling', 'EpsilonGreedy']

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
            gamma=params.get('gamma', 1.0),
            lambda_reg=params.get('lambda_reg', 1.0),
            hidden_dim=params.get('hidden_dim', 64),
            train_interval=params.get('train_interval', 1),
            train_epochs=params.get('train_epochs', 10),
            lr=lr, solver=solver, seed=seed
        )
    elif agent_name == 'CN-TS':
        return CNTSAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            nu=params.get('nu', 1.0),
            lambda_reg=params.get('lambda_reg', 1.0),
            hidden_dim=params.get('hidden_dim', 64),
            train_interval=params.get('train_interval', 1),
            train_epochs=params.get('train_epochs', 10),
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
    eval_interval=10,
):
    dataset_config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS['synthetic']).copy()
    feedback_type = 'full' if full_feedback else 'semi_bandit'

    env = create_env(dataset, seed=seed, **dataset_config)
    test_set = env.generate_test_set(100)
    agent = create_agent(agent_name, env, params, seed, n_rounds, lr)

    has_holdout = hasattr(env, 'evaluate_on_holdout') and hasattr(env, 'holdout_weeks') and env.holdout_weeks > 0

    run_name = f"{feedback_type}_{dataset}_{agent_name}_c{config_idx}_s{seed}"

    run_config = {
        'agent': agent_name,
        'dataset': dataset,
        'feedback_type': feedback_type,
        'full_feedback': full_feedback,
        'seed': seed,
        'n_rounds': n_rounds,
        'lr': lr,
        'has_holdout': has_holdout,
        'dataset_config': {k: v for k, v in dataset_config.items() if not isinstance(v, np.ndarray)},
        **params,
    }

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
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
        test_set=test_set,
    )

    if use_wandb and wandb_run is not None:
        import wandb

        for t in range(len(result['cumulative_regret'])):
            log_dict = {
                'round': t,
                'train/cumulative_regret': result['cumulative_regret'][t],
                'train/instantaneous_regret': result['regrets'][t],
                'train/reward': result['rewards'][t],
                'train/optimal_reward': result['optimal_rewards'][t],
            }
            if 'test_regrets' in result and t < len(result['test_regrets']):
                if not np.isnan(result['test_regrets'][t]):
                    log_dict['test/regret'] = result['test_regrets'][t]
            wandb.log(log_dict)

        holdout_evals = result.get('holdout_evals', [])
        for round_num, holdout_result in holdout_evals:
            holdout_log = {'round': round_num}
            if isinstance(holdout_result, dict):
                for key, value in holdout_result.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        holdout_log[f'holdout/{key}'] = value
            else:
                holdout_log['holdout/regret'] = holdout_result
            wandb.log(holdout_log)

        wandb.summary['train/final_regret'] = result['final_regret']
        wandb.summary['train/final_reward'] = result['final_reward']
        wandb.summary['train/final_optimal'] = result['final_optimal']

        if 'test_regrets' in result:
            non_nan = [r for r in result['test_regrets'] if not np.isnan(r)]
            if non_nan:
                wandb.summary['test/final_regret'] = non_nan[-1]

        if holdout_evals:
            final_holdout = holdout_evals[-1][1]
            if isinstance(final_holdout, dict):
                for key, value in final_holdout.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        wandb.summary[f'holdout/final_{key}'] = value
            else:
                wandb.summary['holdout/final_regret'] = final_holdout

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
    best_holdout_regret = float('inf')
    best_params = None
    all_results = []

    param_combinations = list(product_dict(grid))
    if verbose:
        print(f"\nTuning {agent_name} on {dataset} ({feedback_type}): {len(param_combinations)} combinations")

    for i, params in enumerate(param_combinations):
        actual_params = {k: v for k, v in params.items() if k != '_default'}

        train_regrets = []
        test_regrets = []
        holdout_regrets = []
        holdout_rewards = []
        seed_details = []

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

            train_regrets.append(result['final_regret'])

            seed_detail = {
                'seed': seed,
                'final_regret': float(result['final_regret']),
                'final_reward': float(result['final_reward']),
                'rewards_per_round': [float(r) for r in result['rewards']],
                'regrets_per_round': [float(r) for r in result['regrets']],
                'cumulative_regret': [float(r) for r in result['cumulative_regret']],
            }

            if 'test_regrets' in result:
                non_nan = [r for r in result['test_regrets'] if not np.isnan(r)]
                if non_nan:
                    test_regrets.append(non_nan[-1])
                    seed_detail['final_test_regret'] = float(non_nan[-1])

            holdout_evals = result.get('holdout_evals', [])
            if holdout_evals:
                final_holdout = holdout_evals[-1][1]
                if isinstance(final_holdout, dict):
                    if 'regret' in final_holdout:
                        holdout_regrets.append(final_holdout['regret'])
                    if 'avg_reward' in final_holdout:
                        holdout_rewards.append(final_holdout['avg_reward'])
                    elif 'total_reward' in final_holdout:
                        holdout_rewards.append(final_holdout['total_reward'])
                    seed_detail['holdout'] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                              for k, v in final_holdout.items()}
                else:
                    holdout_regrets.append(final_holdout)
                    seed_detail['holdout_regret'] = float(final_holdout)

            seed_details.append(seed_detail)

        mean_train = np.mean(train_regrets)
        std_train = np.std(train_regrets)
        mean_test = np.mean(test_regrets) if test_regrets else np.nan
        mean_holdout_regret = np.mean(holdout_regrets) if holdout_regrets else np.nan
        std_holdout_regret = np.std(holdout_regrets) if holdout_regrets else np.nan
        mean_holdout_reward = np.mean(holdout_rewards) if holdout_rewards else np.nan

        result_entry = {
            'params': actual_params,
            'mean_train_regret': float(mean_train),
            'std_train_regret': float(std_train),
            'mean_test_regret': float(mean_test) if not np.isnan(mean_test) else None,
            'final_train_regrets': [float(r) for r in train_regrets],
            'final_test_regrets': [float(r) for r in test_regrets] if test_regrets else [],
            'seed_details': seed_details,
        }

        if holdout_regrets:
            result_entry['mean_holdout_regret'] = float(mean_holdout_regret)
            result_entry['std_holdout_regret'] = float(std_holdout_regret)
            result_entry['holdout_regrets'] = [float(r) for r in holdout_regrets]
        if holdout_rewards:
            result_entry['mean_holdout_reward'] = float(mean_holdout_reward)
            result_entry['holdout_rewards'] = [float(r) for r in holdout_rewards]

        all_results.append(result_entry)

        if mean_train < best_regret:
            best_regret = mean_train
            best_params = actual_params
        if holdout_regrets and mean_holdout_regret < best_holdout_regret:
            best_holdout_regret = mean_holdout_regret

        if verbose:
            parts = [f"train={mean_train:.1f}±{std_train:.1f}"]
            if not np.isnan(mean_test):
                parts.append(f"test={mean_test:.1f}")
            if holdout_regrets:
                parts.append(f"holdout={mean_holdout_regret:.1f}±{std_holdout_regret:.1f}")
            print(f"  [{i+1}/{len(param_combinations)}] {actual_params} -> {', '.join(parts)}")

    if verbose:
        print(f"\nBest params for {agent_name} on {dataset} ({feedback_type}): {best_params}")
        print(f"Best train regret: {best_regret:.1f}")
        if not np.isinf(best_holdout_regret):
            print(f"Best holdout regret: {best_holdout_regret:.1f}")

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
    if feedback_types is None:
        feedback_types = [False]

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

                    best_config = None
                    for result in agent_results:
                        if result['params'] == best_params:
                            best_config = result
                            break

                    best_entry = {
                        'params': best_params,
                        'train_regret_mean': float(best_regret),
                        'train_regret_std': float(best_config['std_train_regret']) if best_config else 0.0,
                        'n_rounds': n_rounds,
                        'n_seeds': n_seeds,
                        'lr': lr,
                    }

                    if best_config:
                        if 'mean_holdout_regret' in best_config:
                            best_entry['holdout_regret_mean'] = best_config['mean_holdout_regret']
                            best_entry['holdout_regret_std'] = best_config.get('std_holdout_regret', 0.0)
                        if 'mean_holdout_reward' in best_config:
                            best_entry['holdout_reward_mean'] = best_config['mean_holdout_reward']
                        if 'mean_test_regret' in best_config and best_config['mean_test_regret'] is not None:
                            best_entry['test_regret_mean'] = best_config['mean_test_regret']

                    best_hyperparams[key][agent_name] = best_entry

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
    print('\n' + '=' * 90)
    print('TUNING SUMMARY - BEST HYPERPARAMETERS')
    print('=' * 90)

    for key, agents in best_hyperparams.items():
        print(f"\n{key.upper()}:")

        has_holdout = any('holdout_regret_mean' in r for r in agents.values() if isinstance(r, dict))

        if has_holdout:
            print(f"{'Agent':<20} {'Train':>12} {'Holdout':>12} {'Best Params'}")
        else:
            print(f"{'Agent':<20} {'Train Regret':>12} {'Best Params'}")
        print('-' * 90)

        sorted_agents = sorted(
            [(a, r) for a, r in agents.items() if 'params' in r],
            key=lambda x: x[1].get('holdout_regret_mean', x[1]['train_regret_mean'])
        )

        for agent_name, result in sorted_agents:
            params_str = ', '.join(f"{k}={v}" for k, v in (result['params'] or {}).items())
            if len(params_str) > 40:
                params_str = params_str[:37] + '...'

            train_str = f"{result['train_regret_mean']:.1f}±{result.get('train_regret_std', 0):.1f}"
            if has_holdout and 'holdout_regret_mean' in result:
                holdout_str = f"{result['holdout_regret_mean']:.1f}±{result.get('holdout_regret_std', 0):.1f}"
                print(f"{agent_name:<20} {train_str:>12} {holdout_str:>12} {params_str}")
            else:
                print(f"{agent_name:<20} {train_str:>12} {params_str}")


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
        feedback_types = [False, True]
    elif args.full_feedback:
        feedback_types = [True]
    else:
        feedback_types = [False]

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

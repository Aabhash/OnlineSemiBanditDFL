import numpy as np
from typing import Dict, Any, Callable, List, Optional
from pathlib import Path
import yaml
import optuna

from .timing import TrialTimer
from .results_store import TrialResult, ResultsStore
from .early_stopping import run_with_pruning
from .wandb_logger import WandbTrialLogger, WandbRunLogger


def load_search_spaces(search_space_dir: str) -> Dict[str, Dict[str, Any]]:
    search_spaces = {}
    search_space_path = Path(search_space_dir)

    for yaml_file in search_space_path.glob('*.yaml'):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        if 'agents' in data:
            search_spaces.update(data['agents'])

    return search_spaces


def has_tunable_params(search_space: Dict[str, Any]) -> bool:
    for name, spec in search_space.items():
        if name == 'tune_lr':
            continue
        if isinstance(spec, dict) and 'type' in spec:
            return True
    return False


def get_tunable_agents(search_spaces: Dict[str, Dict[str, Any]]) -> List[str]:
    """Get list of agents that have tunable hyperparameters."""
    return [agent for agent, space in search_spaces.items() if has_tunable_params(space)]


def get_non_tunable_agents(search_spaces: Dict[str, Dict[str, Any]]) -> List[str]:
    """Get list of agents that have NO tunable hyperparameters (e.g., Random, TS)."""
    return [agent for agent, space in search_spaces.items() if not has_tunable_params(space)]


def sample_hyperparameters(
    trial: 'optuna.Trial',
    search_space: Dict[str, Any],
) -> Dict[str, Any]:
    params = {}

    for name, spec in search_space.items():
        if name in ('tune_lr',):
            continue

        if not isinstance(spec, dict):
            continue

        param_type = spec['type']

        if param_type == 'float':
            params[name] = trial.suggest_float(
                name,
                spec['low'],
                spec['high'],
                log=spec.get('log', False),
            )
        elif param_type == 'int':
            params[name] = trial.suggest_int(
                name,
                spec['low'],
                spec['high'],
                log=spec.get('log', False),
            )
        elif param_type == 'categorical':
            choices = spec['choices']
            params[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return params


def create_agent_from_params(
    agent_name: str,
    env,
    params: Dict[str, Any],
    seed: int,
    n_rounds: int,
    lr: float,
):
    from src.agents.shared_context import (
        BayesianSPOAgent,
        LinUCBAgent,
        NeuralUCBAgent,
        ThompsonSamplingAgent,
        UCBAgent,
        EpsilonGreedyAgent,
        MSENeuralAgent,
        RandomAgent,
    )
    from src.agents.per_arm_context import (
        PerArmBayesianSPOAgent,
        CNUCBAgent,
        CNTSAgent,
        PerArmLinUCBAgent,
        PerArmThompsonSamplingAgent,
        PerArmMSENeuralAgent,
        PerArmNeuralUCBAgent,
        PerArmRandomAgent,
        PerArmEpsilonGreedyAgent,
    )
    from src.models import BayesianLinearModel

    solver = env.solve
    minimize = getattr(env, 'minimize', False)

    if agent_name == 'BayesianSPO':
        return BayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'LinUCB':
        return LinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            minimize=minimize, solver=solver,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'NeuralUCB':
        return NeuralUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=lr, minimize=minimize, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'UCB-Bayesian':
        return UCBAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, minimize=minimize, solver=solver,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'ThompsonSampling':
        return ThompsonSamplingAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, minimize=minimize, solver=solver
        )
    elif agent_name == 'EpsilonGreedy':
        return EpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'MSENeural':
        return MSENeuralAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'Greedy':
        # Greedy is EpsilonGreedy with epsilon=0 (no exploration)
        return EpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            epsilon=0.0, decay_epsilon=False,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k not in ['lr', 'epsilon', 'decay_epsilon']}
        )
    elif agent_name == 'Random':
        return RandomAgent(env.n_arms, env.weights, env.capacity, solver=solver)
    elif agent_name == 'PerArmBayesianSPO':
        return PerArmBayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'CN-UCB':
        return CNUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'CN-TS':
        return CNTSAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'PerArmLinUCB':
        return PerArmLinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            solver=solver,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'PerArmNeuralUCB':
        return PerArmNeuralUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'PerArmThompsonSampling':
        return PerArmThompsonSamplingAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            solver=solver
        )
    elif agent_name == 'PerArmMSENeural':
        return PerArmMSENeuralAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'PerArmEpsilonGreedy':
        return PerArmEpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'PerArmGreedy':
        # PerArmGreedy is PerArmEpsilonGreedy with epsilon=0 (no exploration)
        return PerArmEpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            epsilon=0.0, decay_epsilon=False,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k not in ['lr', 'epsilon', 'decay_epsilon']}
        )
    elif agent_name == 'PerArmRandom':
        return PerArmRandomAgent(env.n_arms, env.weights, env.capacity, solver=solver)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def create_env_for_tuning(
    dataset: str,
    seed: int,
    config: Dict[str, Any],
):
    from src.envs import create_env

    dataset_config = config['datasets'][dataset].copy()
    dataset_config.pop('env_type', None)

    return create_env(dataset, seed=seed, **dataset_config)


def get_n_rounds_for_dataset(config: Dict[str, Any], dataset: str) -> int:
    dataset_config = config['datasets'].get(dataset, {})
    if 'n_rounds' in dataset_config:
        return dataset_config['n_rounds']
    return config['tuning']['n_rounds']


def uses_holdout_for_selection(config: Dict[str, Any], dataset: str) -> bool:
    dataset_config = config['datasets'].get(dataset, {})
    return dataset_config.get('eval_metric') == 'holdout'


def is_online_dataset(config: Dict[str, Any], dataset: str) -> bool:
    return not uses_holdout_for_selection(config, dataset)


class ObjectiveFactory:

    @staticmethod
    def create_objective(
        agent_name: str,
        dataset: str,
        config: Dict[str, Any],
        search_space: Dict[str, Any],
        results_store: ResultsStore,
        feedback_type: str,
    ) -> Callable[['optuna.Trial'], float]:
        # Use dataset-specific n_rounds if available (e.g., nfl_online: 17, nfl_logged: 1000)
        n_rounds = get_n_rounds_for_dataset(config, dataset)
        seeds = config['seeds']
        eval_interval = config['validation']['eval_interval']
        full_feedback = feedback_type == 'full'
        default_lr = config['tuning']['default_lr']
        uses_lr = search_space.get('tune_lr', False) or 'lr' in search_space
        wandb_enabled = config.get('wandb', {}).get('enabled', False)
        use_holdout = uses_holdout_for_selection(config, dataset)
        is_online = is_online_dataset(config, dataset)

        def objective(trial: 'optuna.Trial') -> float:
            params = sample_hyperparameters(trial, search_space)

            if search_space.get('tune_lr', False):
                lr = params.get('lr', default_lr)
            else:
                lr = default_lr

            # Create wandb trial logger factory (creates per-seed loggers)
            trial_logger: Optional[WandbTrialLogger] = None
            if wandb_enabled:
                trial_logger = WandbTrialLogger(
                    config=config,
                    agent_name=agent_name,
                    dataset=dataset,
                    feedback_type=feedback_type,
                    trial_number=trial.number,
                    params=params,
                )

            # Track regrets for HP selection
            # Use holdout regret for datasets with eval_metric: holdout (e.g., nfl_logged)
            # Use cumulative training regret for others (standard in online bandit literature)
            selection_regrets = []
            all_pruned = True
            total_wall_time = 0.0
            n_pruned = 0

            n_seeds = len(seeds)
            for seed_idx, seed in enumerate(seeds):
                # Create per-seed wandb run
                seed_logger: Optional[WandbRunLogger] = None
                if trial_logger is not None:
                    seed_logger = trial_logger.create_seed_logger(seed)
                    seed_logger.start()

                try:
                    env = create_env_for_tuning(dataset, seed, config)

                    agent = create_agent_from_params(
                        agent_name, env, params, seed, n_rounds, lr
                    )

                    with TrialTimer() as timer:
                        result = run_with_pruning(
                            env=env,
                            agent=agent,
                            trial=trial,
                            n_rounds=n_rounds,
                            eval_interval=eval_interval,
                            validator=None,  # No validation for online datasets
                            config=config,
                            full_feedback=full_feedback,
                            lr=lr,
                            uses_lr=uses_lr,
                            seed_idx=seed_idx,
                            n_seeds=n_seeds,
                            wandb_logger=seed_logger,
                            is_online=is_online,
                        )

                    total_wall_time += timer.elapsed

                    # Log final summary to wandb
                    if seed_logger is not None:
                        seed_logger.log_final_summary(
                            final_train_regret=result.final_train_regret,
                            final_validation_regret=result.final_validation_regret,
                            holdout_regret=result.holdout_regret,
                            wall_time_seconds=timer.elapsed,
                            pruned=result.pruned,
                            pruned_at_round=result.pruned_at_round,
                            total_rounds=result.total_rounds,
                        )
                        seed_logger.finish(status='pruned' if result.pruned else 'completed')

                    trial_result = TrialResult(
                        trial_id=trial.number,
                        agent_name=agent_name,
                        dataset=dataset,
                        feedback_type=feedback_type,
                        seed=seed,
                        params=params,
                        validation_regrets=result.validation_regrets,
                        train_regrets=result.train_regrets,
                        final_validation_regret=result.final_validation_regret,
                        final_train_regret=result.final_train_regret,
                        holdout_regret=result.holdout_regret,
                        wall_time_seconds=timer.elapsed,
                        pruned=result.pruned,
                        pruned_at_round=result.pruned_at_round,
                        total_rounds=result.total_rounds,
                        cumulative_validation_regret=result.cumulative_validation_regret,
                        cumulative_holdout_regret=result.cumulative_holdout_regret,
                    )
                    results_store.save_trial(trial_result)

                    # Collect regret for HP selection based on dataset config
                    if use_holdout and result.cumulative_holdout_regret is not None:
                        selection_regrets.append(result.cumulative_holdout_regret)
                    else:
                        selection_regrets.append(result.final_train_regret)

                    if result.pruned:
                        n_pruned += 1
                    else:
                        all_pruned = False

                except Exception as e:
                    if seed_logger is not None:
                        seed_logger.finish(status='failed')
                    raise

            if all_pruned:
                raise optuna.TrialPruned()

            # HP selection based on average regret across seeds
            # Uses holdout regret for datasets with eval_metric: holdout (e.g., nfl_logged)
            # Uses cumulative training regret for others (standard in online bandit literature)
            return float(np.mean(selection_regrets))

        return objective

    @staticmethod
    def create_simple_objective(
        agent_name: str,
        dataset: str,
        config: Dict[str, Any],
        search_space: Dict[str, Any],
        feedback_type: str = 'semi_bandit',
    ) -> Callable[['optuna.Trial'], float]:
        # Use dataset-specific n_rounds if available
        n_rounds = get_n_rounds_for_dataset(config, dataset)
        seeds = config['seeds'][:3]
        eval_interval = config['validation']['eval_interval']
        full_feedback = feedback_type == 'full'
        default_lr = config['tuning']['default_lr']
        uses_lr = search_space.get('tune_lr', False) or 'lr' in search_space
        wandb_enabled = config.get('wandb', {}).get('enabled', False)
        use_holdout = uses_holdout_for_selection(config, dataset)
        is_online = is_online_dataset(config, dataset)

        def objective(trial: 'optuna.Trial') -> float:
            params = sample_hyperparameters(trial, search_space)
            lr = params.get('lr', default_lr) if search_space.get('tune_lr', False) else default_lr

            # Create wandb trial logger factory
            trial_logger: Optional[WandbTrialLogger] = None
            if wandb_enabled:
                trial_logger = WandbTrialLogger(
                    config=config,
                    agent_name=agent_name,
                    dataset=dataset,
                    feedback_type=feedback_type,
                    trial_number=trial.number,
                    params=params,
                )

            selection_regrets = []
            n_seeds = len(seeds)

            for seed_idx, seed in enumerate(seeds):
                # Create per-seed wandb run
                seed_logger: Optional[WandbRunLogger] = None
                if trial_logger is not None:
                    seed_logger = trial_logger.create_seed_logger(seed)
                    seed_logger.start()

                try:
                    env = create_env_for_tuning(dataset, seed, config)
                    agent = create_agent_from_params(agent_name, env, params, seed, n_rounds, lr)

                    result = run_with_pruning(
                        env=env,
                        agent=agent,
                        trial=trial,
                        n_rounds=n_rounds,
                        eval_interval=eval_interval,
                        validator=None,  # No validation for online datasets
                        config=config,
                        full_feedback=full_feedback,
                        lr=lr,
                        uses_lr=uses_lr,
                        seed_idx=seed_idx,
                        n_seeds=n_seeds,
                        wandb_logger=seed_logger,
                        is_online=is_online,
                    )

                    if seed_logger is not None:
                        seed_logger.log_final_summary(
                            final_train_regret=result.final_train_regret,
                            final_validation_regret=result.final_validation_regret,
                            holdout_regret=result.holdout_regret,
                            wall_time_seconds=0.0,
                            pruned=result.pruned,
                            pruned_at_round=result.pruned_at_round,
                            total_rounds=result.total_rounds,
                        )
                        seed_logger.finish(status='pruned' if result.pruned else 'completed')

                    if result.pruned:
                        raise optuna.TrialPruned()

                    # Collect regret for HP selection based on dataset config
                    if use_holdout and result.cumulative_holdout_regret is not None:
                        selection_regrets.append(result.cumulative_holdout_regret)
                    else:
                        selection_regrets.append(result.final_train_regret)

                except Exception as e:
                    if seed_logger is not None:
                        seed_logger.finish(status='failed')
                    raise

            # HP selection based on average regret across seeds
            return float(np.mean(selection_regrets))

        return objective

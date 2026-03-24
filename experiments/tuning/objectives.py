import numpy as np
from typing import Dict, Any, Callable, List, Optional, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import optuna

from .timing import TrialTimer
from .results_store import TrialResult, ResultsStore
from .wandb_logger import WandbTrialLogger, WandbRunLogger

if TYPE_CHECKING:
    from .wandb_logger import WandbRunLogger as WandbRunLoggerType


@dataclass
class ExperimentResult:
    final_validation_regret: float
    final_train_regret: float
    validation_regrets: List[float] = field(default_factory=list)
    train_regrets: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    optimal_rewards: List[float] = field(default_factory=list)
    total_rounds: int = 0
    holdout_regret: Optional[float] = None
    cumulative_validation_regret: float = 0.0
    cumulative_holdout_regret: Optional[float] = None


def run_tuning_experiment(
    env,
    agent,
    n_rounds: int,
    eval_interval: int,
    config: Dict[str, Any],
    full_feedback: bool,
    lr: float,
    uses_lr: bool,
    seed_idx: int = 0,
    n_seeds: int = 1,
    wandb_logger: Optional['WandbRunLoggerType'] = None,
    is_online: bool = True,
    log_interval: int = 1,
) -> ExperimentResult:
    rewards = []
    optimal_rewards = []
    regrets = []
    cumulative_regret = 0.0
    cumulative_reward = 0.0

    # Track cumulative holdout metrics (for datasets with holdout evaluation)
    cumulative_holdout_regret = 0.0
    cumulative_holdout_reward = 0.0

    # Use provided log interval for wandb
    wandb_log_interval = log_interval

    # Check if env supports holdout evaluation (only relevant for non-online datasets)
    has_holdout = hasattr(env, 'evaluate_on_holdout') and not is_online

    X = env.reset()

    for t in range(n_rounds):
        selection = agent.act(X)
        observed_values, reward, info = env.step(selection, full_feedback=full_feedback)

        rewards.append(reward)
        cumulative_reward += reward
        # Handle both maximization (optimal_reward) and minimization (optimal_cost)
        optimal_reward = info.get('optimal_reward', -info.get('optimal_cost', 0))
        optimal_rewards.append(optimal_reward)
        instant_regret = info.get('regret', optimal_reward - reward)
        regrets.append(instant_regret)
        cumulative_regret += instant_regret

        # Try to update agent - agents have varying signatures for update()
        try:
            if uses_lr:
                agent.update(X, selection, observed_values, lr=lr, full_feedback=full_feedback)
            else:
                agent.update(X, selection, observed_values, full_feedback=full_feedback)
        except TypeError:
            # Fallback chain for different agent signatures
            try:
                agent.update(X, selection, observed_values, full_feedback=full_feedback)
            except TypeError:
                try:
                    agent.update(X, selection, observed_values, lr)
                except TypeError:
                    agent.update(X, selection, observed_values)

        X = env.get_context()

        # Log training metrics to wandb at regular intervals
        should_log_wandb = (t + 1) % wandb_log_interval == 0 or t == n_rounds - 1
        if wandb_logger is not None and should_log_wandb:
            wandb_logger.log_round(
                round_num=t,
                regret=instant_regret,
                cumulative_regret=cumulative_regret,
                reward=reward,
                optimal_reward=optimal_reward,
                cumulative_reward=cumulative_reward,
            )

        should_eval = (t + 1) % eval_interval == 0 or t == n_rounds - 1
        if should_eval:
            # For holdout datasets, evaluate on holdout
            if has_holdout:
                holdout_result = env.evaluate_on_holdout(agent)
                if holdout_result:
                    holdout_regret = holdout_result['regret']
                    holdout_reward = holdout_result.get('total_reward', holdout_result.get('reward', 0.0))

                    # Update cumulative holdout metrics
                    cumulative_holdout_regret += holdout_regret
                    cumulative_holdout_reward += holdout_reward if holdout_reward else 0.0

                    # Log holdout metrics to wandb
                    if wandb_logger is not None:
                        wandb_logger.log_holdout(
                            round_num=t,
                            holdout_regret=holdout_regret,
                            cumulative_holdout_regret=cumulative_holdout_regret,
                            holdout_reward=holdout_reward,
                            cumulative_holdout_reward=cumulative_holdout_reward,
                        )

    # Get final holdout regret for the result (only for holdout datasets)
    holdout_regret = None
    if has_holdout:
        holdout_result = env.evaluate_on_holdout(agent)
        if holdout_result:
            holdout_regret = holdout_result['regret']

    return ExperimentResult(
        final_validation_regret=cumulative_regret,
        final_train_regret=cumulative_regret,
        validation_regrets=[],
        train_regrets=[sum(regrets[:i+1]) for i in range(len(regrets))],
        rewards=rewards,
        optimal_rewards=optimal_rewards,
        total_rounds=n_rounds,
        holdout_regret=holdout_regret,
        cumulative_validation_regret=0.0,
        cumulative_holdout_regret=cumulative_holdout_regret if has_holdout else None,
    )


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
        elif param_type == 'fixed':
            params[name] = spec['value']
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
        BayesianSPO_DFE_Agent,
        LinUCBAgent,
        NeuralUCBAgent,
        ThompsonSamplingAgent,
        UCBAgent,
        EpsilonGreedyAgent,
        BayesianMSEAgent,
        RandomAgent,
    )
    from src.models import BayesianLinearModel

    solver = env.solve
    minimize = getattr(env, 'minimize', False)

    if agent_name in ('BayesianSPO', 'BayesianSPO-PureSPO'):
        # Negate beta for minimization (LCB instead of UCB)
        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']
        # BayesianSPOAgent constructor uses 'lr' param name
        agent_lr = params.get('init_lr', 0.001)
        return BayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr,
            decay_rounds=n_rounds, minimize=minimize, solver=solver, seed=seed,
            mask_grad=True, nonlinear=False,
            **agent_params
        )
    elif agent_name == 'LinUCB':
        return LinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            minimize=minimize, solver=solver,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'NeuralUCB':
        agent_lr = params.get('init_lr', lr)
        return NeuralUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr, minimize=minimize, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        )
    elif agent_name == 'UCB-Bayesian':
        return UCBAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, minimize=minimize, solver=solver,
            **{k: v for k, v in params.items() if k != 'lr'}
        )
    elif agent_name == 'ThompsonSampling':
        prior_var = params.get('prior_var', 1.0)
        noise_var = params.get('noise_var', 0.1)
        return ThompsonSamplingAgent(
            BayesianLinearModel(env.context_dim, env.n_arms, prior_var=prior_var, noise_var=noise_var),
            env.weights, env.capacity, minimize=minimize, solver=solver
        )
    elif agent_name == 'EpsilonGreedy':
        agent_lr = params.get('init_lr', 0.001)
        return EpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr,
            decay_rounds=n_rounds, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        )
    elif agent_name == 'BayesianMSE':
        # Negate beta for minimization (LCB instead of UCB)
        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']
        agent_lr = params.get('init_lr', 0.001)
        return BayesianMSEAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr,
            decay_rounds=n_rounds, solver=solver, seed=seed,
            nonlinear=False,
            **agent_params
        )
    elif agent_name == 'Greedy':
        # Greedy is EpsilonGreedy with epsilon=0 (no exploration)
        agent_lr = params.get('init_lr', 0.001)
        return EpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr,
            epsilon=0.0, decay_epsilon=False,
            decay_rounds=n_rounds, solver=solver, seed=seed,
            **{k: v for k, v in params.items() if k not in ['lr', 'init_lr', 'epsilon', 'decay_epsilon']}
        )
    elif agent_name == 'Random':
        return RandomAgent(env.n_arms, env.weights, env.capacity, solver=solver)
    elif agent_name in ('BayesianSPO-DI', 'BayesianSPO-DFE'):
        from src.solver import ShortestPathSolver, KnapsackSolver
        grid_size = getattr(env, 'grid_size', None)
        if grid_size is not None:
            co_solver = ShortestPathSolver(grid_size)
        else:
            co_solver = KnapsackSolver(env.weights, env.capacity)
        # Pop keys handled explicitly so they don't duplicate in **kwargs
        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']

        # Map agent name to constructor kwargs
        if agent_name == 'BayesianSPO-DI':
            agent_params['imputation'] = 'disc_impute'
        elif agent_name == 'BayesianSPO-DFE':
            agent_params['imputation'] = 'disc_explore'
            agent_params['decision_focused_explore'] = True
            agent_params['adaptive_dfe'] = True

        agent_lr = params.get('init_lr', 0.001)
        return BayesianSPO_DFE_Agent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            init_lr=agent_lr,
            minimize=minimize, solver=solver, seed=seed,
            co_solver=co_solver, nonlinear=False,
            **agent_params,
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def create_env_for_tuning(
    dataset: str,
    seed: int,
    config: Dict[str, Any],
):
    from src.envs import create_env

    dataset_config = config['datasets'][dataset].copy()
    # Remove training loop parameters (not env constructor params)
    dataset_config.pop('env_type', None)
    dataset_config.pop('n_rounds', None)
    dataset_config.pop('eval_interval', None)
    dataset_config.pop('log_interval', None)
    dataset_config.pop('eval_metric', None)
    dataset_config.pop('skip_tuning', None)
    dataset_config.pop('tuning_year', None)

    return create_env(dataset, seed=seed, **dataset_config)


def get_n_rounds_for_dataset(config: Dict[str, Any], dataset: str) -> int:
    dataset_config = config['datasets'].get(dataset, {})
    if 'n_rounds' in dataset_config:
        return dataset_config['n_rounds']
    return config['tuning']['n_rounds']


def get_eval_interval_for_dataset(config: Dict[str, Any], dataset: str) -> int:
    dataset_config = config['datasets'].get(dataset, {})
    if 'eval_interval' in dataset_config:
        return dataset_config['eval_interval']
    return config['validation']['eval_interval']


def get_log_interval_for_dataset(config: Dict[str, Any], dataset: str) -> int:
    dataset_config = config['datasets'].get(dataset, {})
    if 'log_interval' in dataset_config:
        return dataset_config['log_interval']
    return config.get('wandb', {}).get('log_interval', 10)


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
        # Use dataset-specific n_rounds if available
        n_rounds = get_n_rounds_for_dataset(config, dataset)
        seeds = config.get('tuning_seeds', [0, 1, 2, 3, 4])
        eval_interval = get_eval_interval_for_dataset(config, dataset)
        log_interval = get_log_interval_for_dataset(config, dataset)
        full_feedback = feedback_type == 'full'
        default_lr = config['tuning']['default_lr']
        uses_lr = search_space.get('tune_lr', False) or 'lr' in search_space or 'init_lr' in search_space
        wandb_enabled = config.get('wandb', {}).get('enabled', False)
        use_holdout = uses_holdout_for_selection(config, dataset)
        is_online = is_online_dataset(config, dataset)

        # Pruning configuration
        pruning_config = config.get('tuning', {}).get('pruning', {})
        pruning_enabled = pruning_config.get('enabled', False)
        min_seeds_before_pruning = pruning_config.get('min_seeds_before_pruning', 1)
        warmup_trials = pruning_config.get('warmup_trials', 5)
        # Late-stage pruning: only consider last fraction of rounds for pruning decisions
        # This prevents pruning exploration-heavy configs before they can exploit
        use_late_stage = pruning_config.get('use_late_stage', False)
        late_stage_fraction = pruning_config.get('late_stage_fraction', 0.25)

        def objective(trial: 'optuna.Trial') -> float:
            params = sample_hyperparameters(trial, search_space)

            if search_space.get('tune_lr', False):
                lr = params.get('lr', params.get('init_lr', default_lr))
            else:
                lr = default_lr

            # Create wandb trial logger factory (creates per-seed loggers)
            # Get minimize flag from dataset config (defaults to False for maximization)
            is_minimize = config.get('datasets', {}).get(dataset, {}).get('minimize', False)

            trial_logger: Optional[WandbTrialLogger] = None
            if wandb_enabled:
                trial_logger = WandbTrialLogger(
                    config=config,
                    agent_name=agent_name,
                    dataset=dataset,
                    feedback_type=feedback_type,
                    trial_number=trial.number,
                    params=params,
                    minimize=is_minimize,
                )

            # Track regrets for HP selection
            # HP selection uses cumulative training regret (standard in online bandit literature)
            # Holdout regret is only for final evaluation reporting, not HP selection
            selection_regrets = []
            # Track late-stage regrets separately for pruning decisions
            # This allows exploration-heavy configs to not be pruned before they can exploit
            late_stage_regrets = []
            total_wall_time = 0.0

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
                        result = run_tuning_experiment(
                            env=env,
                            agent=agent,
                            n_rounds=n_rounds,
                            eval_interval=eval_interval,
                            config=config,
                            full_feedback=full_feedback,
                            lr=lr,
                            uses_lr=uses_lr,
                            seed_idx=seed_idx,
                            n_seeds=n_seeds,
                            wandb_logger=seed_logger,
                            is_online=is_online,
                            log_interval=log_interval,
                        )

                    total_wall_time += timer.elapsed

                    # Log final summary to wandb
                    if seed_logger is not None:
                        final_train_cumulative_reward = sum(result.rewards) if result.rewards else None
                        final_train_reward_per_round = result.rewards[-1] if result.rewards else None
                        # Per-round regret = optimal_reward - reward for last round
                        final_train_regret_per_round = None
                        if result.optimal_rewards and result.rewards:
                            final_train_regret_per_round = result.optimal_rewards[-1] - result.rewards[-1]

                        seed_logger.log_final_summary(
                            final_train_cumulative_regret=result.final_train_regret,
                            final_train_cumulative_reward=final_train_cumulative_reward,
                            final_train_regret_per_round=final_train_regret_per_round,
                            final_train_reward_per_round=final_train_reward_per_round,
                            final_holdout_cumulative_regret=result.cumulative_holdout_regret,
                            final_holdout_cumulative_reward=None,  # Not tracked in result
                            final_holdout_regret_per_round=result.holdout_regret,
                            final_holdout_reward_per_round=None,  # Not tracked in result
                            wall_time_seconds=timer.elapsed,
                            total_rounds=result.total_rounds,
                        )
                        seed_logger.finish(status='completed')

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

                    # Compute late-stage regret for pruning decisions
                    # This is the regret accumulated in the last fraction of rounds
                    # Allows exploration-heavy configs to not be pruned before they can exploit
                    if use_late_stage and result.train_regrets:
                        n_total = len(result.train_regrets)
                        late_start_idx = int(n_total * (1 - late_stage_fraction))
                        # Late-stage regret = cumulative regret in last fraction of rounds
                        late_regret = result.train_regrets[-1] - result.train_regrets[late_start_idx]
                        late_stage_regrets.append(late_regret)
                    else:
                        # Fall back to full cumulative regret if late-stage not enabled
                        late_stage_regrets.append(result.final_train_regret)

                    # Report intermediate value for pruning (average regret so far)
                    # Use late-stage regret for pruning to not penalize early exploration
                    pruning_metric = float(np.mean(late_stage_regrets))
                    trial.report(pruning_metric, seed_idx)

                    # Check for pruning after minimum seeds have completed
                    # Only prune if: (1) pruning enabled, (2) past warmup, (3) enough seeds run
                    should_check_pruning = (
                        pruning_enabled
                        and trial.number >= warmup_trials
                        and seed_idx >= min_seeds_before_pruning - 1
                        and seed_idx < n_seeds - 1  # Don't prune on last seed
                    )

                    if should_check_pruning and trial.should_prune():
                        # Clean up any remaining wandb loggers
                        if seed_logger is not None:
                            seed_logger.finish(status='pruned')
                        raise optuna.TrialPruned(
                            f"Pruned after {seed_idx + 1} seeds with late-stage regret {pruning_metric:.1f}"
                        )

                except optuna.TrialPruned:
                    raise  # Re-raise pruning exception
                except Exception as e:
                    if seed_logger is not None:
                        seed_logger.finish(status='failed')
                    raise

            # HP selection based on average cumulative training regret across seeds
            # (standard in online bandit literature - holdout is only for final reporting)
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
        seeds = config.get('tuning_seeds', [0, 1, 2])[:3]
        eval_interval = get_eval_interval_for_dataset(config, dataset)
        log_interval = get_log_interval_for_dataset(config, dataset)
        full_feedback = feedback_type == 'full'
        default_lr = config['tuning']['default_lr']
        uses_lr = search_space.get('tune_lr', False) or 'lr' in search_space or 'init_lr' in search_space
        wandb_enabled = config.get('wandb', {}).get('enabled', False)
        use_holdout = uses_holdout_for_selection(config, dataset)
        is_online = is_online_dataset(config, dataset)

        def objective(trial: 'optuna.Trial') -> float:
            params = sample_hyperparameters(trial, search_space)
            lr = params.get('lr', params.get('init_lr', default_lr)) if search_space.get('tune_lr', False) else default_lr

            # Create wandb trial logger factory
            # Get minimize flag from dataset config (defaults to False for maximization)
            is_minimize = config.get('datasets', {}).get(dataset, {}).get('minimize', False)

            trial_logger: Optional[WandbTrialLogger] = None
            if wandb_enabled:
                trial_logger = WandbTrialLogger(
                    config=config,
                    agent_name=agent_name,
                    dataset=dataset,
                    feedback_type=feedback_type,
                    trial_number=trial.number,
                    params=params,
                    minimize=is_minimize,
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

                    result = run_tuning_experiment(
                        env=env,
                        agent=agent,
                        n_rounds=n_rounds,
                        eval_interval=eval_interval,
                        config=config,
                        full_feedback=full_feedback,
                        lr=lr,
                        uses_lr=uses_lr,
                        seed_idx=seed_idx,
                        n_seeds=n_seeds,
                        wandb_logger=seed_logger,
                        is_online=is_online,
                        log_interval=log_interval,
                    )

                    if seed_logger is not None:
                        final_train_cumulative_reward = sum(result.rewards) if result.rewards else None
                        final_train_reward_per_round = result.rewards[-1] if result.rewards else None
                        # Per-round regret = optimal_reward - reward for last round
                        final_train_regret_per_round = None
                        if result.optimal_rewards and result.rewards:
                            final_train_regret_per_round = result.optimal_rewards[-1] - result.rewards[-1]

                        seed_logger.log_final_summary(
                            final_train_cumulative_regret=result.final_train_regret,
                            final_train_cumulative_reward=final_train_cumulative_reward,
                            final_train_regret_per_round=final_train_regret_per_round,
                            final_train_reward_per_round=final_train_reward_per_round,
                            final_holdout_cumulative_regret=result.cumulative_holdout_regret,
                            final_holdout_cumulative_reward=None,
                            final_holdout_regret_per_round=result.holdout_regret,
                            final_holdout_reward_per_round=None,
                            wall_time_seconds=0.0,
                            total_rounds=result.total_rounds,
                        )
                        seed_logger.finish(status='completed')

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

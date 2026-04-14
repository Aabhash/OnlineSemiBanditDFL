import numpy as np
from typing import Dict, Any, Callable, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import optuna

from .timing import TrialTimer
from .results_store import TrialResult, ResultsStore
from src.agents.contextual import (
    BayesianSPOAgent,
    BayesianPFYAgent,
    BayesianSPO_DFE_Agent,
    LinUCBAgent,
    NeuralUCBAgent,
    ThompsonSamplingAgent,
    UCBAgent,
    EpsilonGreedyAgent,
    BayesianMSEAgent,
    RandomAgent,
    WeberBLLAgent,
)
from src.envs import create_env
from src.models import BayesianLinearModel
from src.solver import ShortestPathSolver, KnapsackSolver


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
    is_online: bool = True,
    log_interval: int = 1,
) -> ExperimentResult:
    rewards = []
    optimal_rewards = []
    regrets = []
    cumulative_regret = 0.0
    cumulative_reward = 0.0


    cumulative_holdout_regret = 0.0
    cumulative_holdout_reward = 0.0




    has_holdout = hasattr(env, 'evaluate_on_holdout') and not is_online

    X = env.reset()

    for t in range(n_rounds):
        selection = agent.act(X)
        observed_values, reward, info = env.step(selection, full_feedback=full_feedback)

        rewards.append(reward)
        cumulative_reward += reward

        optimal_reward = info.get('optimal_reward', -info.get('optimal_cost', 0))
        optimal_rewards.append(optimal_reward)
        instant_regret = info.get('regret', optimal_reward - reward)
        regrets.append(instant_regret)
        cumulative_regret += instant_regret


        agent.update(X, selection, observed_values, lr=lr, full_feedback=full_feedback)

        X = env.get_context()


        should_eval = (t + 1) % eval_interval == 0 or t == n_rounds - 1
        if should_eval:

            if has_holdout:
                holdout_result = env.evaluate_on_holdout(agent)
                if holdout_result:
                    holdout_regret = holdout_result['regret']
                    holdout_reward = holdout_result.get('total_reward', holdout_result.get('reward', 0.0))


                    cumulative_holdout_regret += holdout_regret
                    cumulative_holdout_reward += holdout_reward if holdout_reward else 0.0


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


def load_search_spaces(search_space_path: str) -> Dict[str, Dict[str, Any]]:
    search_spaces = {}
    p = Path(search_space_path)

    yaml_files = [p] if p.is_file() else list(p.glob('*.yaml'))

    for yaml_file in yaml_files:
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
    return [agent for agent, space in search_spaces.items() if has_tunable_params(space)]


def get_non_tunable_agents(search_spaces: Dict[str, Dict[str, Any]]) -> List[str]:
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


PHASED_TO_ONLINE_BACKBONE = {
    'BayesianSPO-Phased': 'BayesianSPO',
    'BayesianPFY-Phased': 'BayesianPFY',
    'BayesianMSE-Phased': 'BayesianMSE',
    'NeuralUCB-Phased': 'NeuralUCB',
    'WeberBLL-Phased': 'WeberBLL-Online',
}


def load_backbone_params(agent_name, dataset, feedback_type, results_store):
    online_name = PHASED_TO_ONLINE_BACKBONE.get(agent_name)
    if online_name is None:
        return {}
    try:
        return results_store.get_best_params(online_name, dataset, feedback_type)
    except Exception:
        return {}


def merge_backbone_and_phase_params(backbone_params, phase_params):
    merged = dict(backbone_params)
    merged.update(phase_params)
    return merged


def create_agent_from_params(
    agent_name: str,
    env,
    params: Dict[str, Any],
    seed: int,
    n_rounds: int,
    lr: float,
    backbone_params: Dict[str, Any] = None,
):
    if backbone_params and agent_name in PHASED_TO_ONLINE_BACKBONE:
        params = merge_backbone_and_phase_params(backbone_params, params)

    solver = env.solve
    minimize = getattr(env, 'minimize', False)

    if agent_name in ('BayesianSPO', 'BayesianSPO-PureSPO'):

        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']

        agent_lr = params.get('init_lr', 0.001)
        return BayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr,
            decay_rounds=n_rounds, minimize=minimize, solver=solver, seed=seed,
            mask_grad=True, nonlinear=False,
            **agent_params
        )
    elif agent_name == 'BayesianPFY':
        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']

        agent_lr = params.get('init_lr', 0.001)
        return BayesianPFYAgent(
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
    elif agent_name in ('TS', 'ThompsonSampling'):
        prior_var = params.get('prior_var', 1.0)
        noise_var = params.get('noise_var', 0.1)
        return ThompsonSamplingAgent(
            BayesianLinearModel(env.context_dim, env.n_arms, prior_var=prior_var, noise_var=noise_var),
            env.weights, env.capacity, minimize=minimize, solver=solver, seed=seed
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
        grid_size = getattr(env, 'grid_size', None)
        if grid_size is not None:
            co_solver = ShortestPathSolver(grid_size)
        else:
            co_solver = KnapsackSolver(env.weights, env.capacity)

        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']


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
    elif agent_name == 'WeberBLL-Online':
        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']
        agent_lr = params.get('init_lr', 0.001)
        return WeberBLLAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr,
            decay_rounds=n_rounds, minimize=minimize, solver=solver, seed=seed,
            mask_grad=True, nonlinear=False, phased=False,
            **agent_params
        )

    elif agent_name == 'BayesianSPO-Phased':
        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']
        agent_lr = params.get('init_lr', 0.001)
        return BayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr,
            decay_rounds=n_rounds, minimize=minimize, solver=solver, seed=seed,
            mask_grad=True, nonlinear=False, phased=True,
            **agent_params
        )
    elif agent_name == 'BayesianPFY-Phased':
        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']
        agent_lr = params.get('init_lr', 0.001)
        return BayesianPFYAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr,
            decay_rounds=n_rounds, minimize=minimize, solver=solver, seed=seed,
            mask_grad=True, nonlinear=False, phased=True,
            **agent_params
        )
    elif agent_name == 'BayesianMSE-Phased':
        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']
        agent_lr = params.get('init_lr', 0.001)
        return BayesianMSEAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr,
            decay_rounds=n_rounds, solver=solver, seed=seed,
            nonlinear=False, phased=True,
            **agent_params
        )
    elif agent_name == 'NeuralUCB-Phased':
        agent_lr = params.get('init_lr', lr)
        phase_params = {k: v for k, v in params.items() if k in ('phase_length', 'phase_epochs')}
        return NeuralUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=params.get('hidden_dim', 16),
            alpha=params.get('alpha', 1.0),
            lambda_reg=params.get('lambda_reg', 1.0),
            lr=agent_lr,
            minimize=minimize, solver=solver, seed=seed,
            phased=True, **phase_params
        )
    elif agent_name == 'WeberBLL-Phased':
        agent_params = {k: v for k, v in params.items() if k not in ('lr', 'init_lr')}
        if minimize and 'beta' in agent_params:
            agent_params['beta'] = -agent_params['beta']
        agent_lr = params.get('init_lr', 0.001)
        return WeberBLLAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lr=agent_lr,
            decay_rounds=n_rounds, minimize=minimize, solver=solver, seed=seed,
            mask_grad=True, nonlinear=False, phased=True,
            **agent_params
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def create_env_for_tuning(
    dataset: str,
    seed: int,
    config: Dict[str, Any],
):
    dataset_config = config['datasets'][dataset].copy()

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
    return 10


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
        backbone_results_store: ResultsStore = None,
    ) -> Callable[['optuna.Trial'], float]:

        n_rounds = get_n_rounds_for_dataset(config, dataset)
        seeds = config.get('tuning_seeds', [0, 1, 2, 3, 4])
        eval_interval = get_eval_interval_for_dataset(config, dataset)
        log_interval = get_log_interval_for_dataset(config, dataset)
        full_feedback = feedback_type == 'full'
        default_lr = config['tuning']['default_lr']
        uses_lr = search_space.get('tune_lr', False) or 'lr' in search_space or 'init_lr' in search_space
        use_holdout = uses_holdout_for_selection(config, dataset)
        is_online = is_online_dataset(config, dataset)


        bb_store = backbone_results_store if backbone_results_store is not None else results_store
        backbone = load_backbone_params(agent_name, dataset, feedback_type, bb_store)
        if backbone:
            print(f"  Loaded stage-1 backbone from {PHASED_TO_ONLINE_BACKBONE[agent_name]}: {list(backbone.keys())}")

        pruning_config = config.get('tuning', {}).get('pruning', {})
        pruning_enabled = pruning_config.get('enabled', False)
        min_seeds_before_pruning = pruning_config.get('min_seeds_before_pruning', 1)
        warmup_trials = pruning_config.get('warmup_trials', 5)

        use_late_stage = pruning_config.get('use_late_stage', False)
        late_stage_fraction = pruning_config.get('late_stage_fraction', 0.25)
        trial_logger = None

        def objective(trial: 'optuna.Trial') -> float:
            params = sample_hyperparameters(trial, search_space)

            if search_space.get('tune_lr', False):
                lr = params.get('lr', params.get('init_lr', default_lr))
            else:
                lr = default_lr


            is_minimize = config.get('datasets', {}).get(dataset, {}).get('minimize', False)

            selection_regrets = []


            late_stage_regrets = []
            total_wall_time = 0.0

            n_seeds = len(seeds)
            for seed_idx, seed in enumerate(seeds):
                seed_logger = None

                if trial_logger is not None:
                    seed_logger = trial_logger.create_seed_logger(seed)
                    seed_logger.start()

                try:
                    env = create_env_for_tuning(dataset, seed, config)

                    agent = create_agent_from_params(
                        agent_name, env, params, seed, n_rounds, lr,
                        backbone_params=backbone,
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
                            is_online=is_online,
                            log_interval=log_interval,
                        )

                    total_wall_time += timer.elapsed


                    if seed_logger is not None:
                        final_train_cumulative_reward = sum(result.rewards) if result.rewards else None
                        final_train_reward_per_round = result.rewards[-1] if result.rewards else None

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


                    if use_holdout and result.cumulative_holdout_regret is not None:
                        selection_regrets.append(result.cumulative_holdout_regret)
                    else:
                        selection_regrets.append(result.final_train_regret)


                    if use_late_stage and result.train_regrets:
                        n_total = len(result.train_regrets)
                        late_start_idx = int(n_total * (1 - late_stage_fraction))

                        late_regret = result.train_regrets[-1] - result.train_regrets[late_start_idx]
                        late_stage_regrets.append(late_regret)
                    else:

                        late_stage_regrets.append(result.final_train_regret)


                    pruning_metric = float(np.mean(late_stage_regrets))
                    trial.report(pruning_metric, seed_idx)


                    should_check_pruning = (
                        pruning_enabled
                        and trial.number >= warmup_trials
                        and seed_idx >= min_seeds_before_pruning - 1
                        and seed_idx < n_seeds - 1
                    )

                    if should_check_pruning and trial.should_prune():

                        if seed_logger is not None:
                            seed_logger.finish(status='pruned')
                        raise optuna.TrialPruned(
                            f"Pruned after {seed_idx + 1} seeds with late-stage regret {pruning_metric:.1f}"
                        )

                except optuna.TrialPruned:
                    raise
                except Exception as e:
                    if seed_logger is not None:
                        seed_logger.finish(status='failed')
                    raise


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

        n_rounds = get_n_rounds_for_dataset(config, dataset)
        seeds = config.get('tuning_seeds', [0, 1, 2])[:3]
        eval_interval = get_eval_interval_for_dataset(config, dataset)
        log_interval = get_log_interval_for_dataset(config, dataset)
        full_feedback = feedback_type == 'full'
        default_lr = config['tuning']['default_lr']
        uses_lr = search_space.get('tune_lr', False) or 'lr' in search_space or 'init_lr' in search_space
        use_holdout = uses_holdout_for_selection(config, dataset)
        is_online = is_online_dataset(config, dataset)
        trial_logger = None

        def objective(trial: 'optuna.Trial') -> float:
            params = sample_hyperparameters(trial, search_space)
            lr = params.get('lr', params.get('init_lr', default_lr)) if search_space.get('tune_lr', False) else default_lr


            is_minimize = config.get('datasets', {}).get(dataset, {}).get('minimize', False)

            selection_regrets = []
            n_seeds = len(seeds)

            for seed_idx, seed in enumerate(seeds):
                seed_logger = None

                if trial_logger is not None:
                    seed_logger = trial_logger.create_seed_logger(seed)
                    seed_logger.start()

                try:
                    env = create_env_for_tuning(dataset, seed, config)
                    agent = create_agent_from_params(
                        agent_name, env, params, seed, n_rounds, lr)

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
                        is_online=is_online,
                        log_interval=log_interval,
                    )

                    if seed_logger is not None:
                        final_train_cumulative_reward = sum(result.rewards) if result.rewards else None
                        final_train_reward_per_round = result.rewards[-1] if result.rewards else None

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


                    if use_holdout and result.cumulative_holdout_regret is not None:
                        selection_regrets.append(result.cumulative_holdout_regret)
                    else:
                        selection_regrets.append(result.final_train_regret)

                except Exception as e:
                    if seed_logger is not None:
                        seed_logger.finish(status='failed')
                    raise


            return float(np.mean(selection_regrets))

        return objective

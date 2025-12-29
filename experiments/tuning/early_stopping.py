import numpy as np
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
import optuna

from .validation import BaseValidator, create_validator

if TYPE_CHECKING:
    from .wandb_logger import WandbRunLogger


@dataclass
class ExperimentResult:
    final_validation_regret: float
    final_train_regret: float
    validation_regrets: List[float] = field(default_factory=list)
    train_regrets: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    optimal_rewards: List[float] = field(default_factory=list)
    pruned: bool = False
    pruned_at_round: Optional[int] = None
    total_rounds: int = 0
    holdout_regret: Optional[float] = None
    # Cumulative metrics for online bandit evaluation
    cumulative_validation_regret: float = 0.0
    cumulative_holdout_regret: Optional[float] = None


def run_with_pruning(
    env,
    agent,
    trial: Optional['optuna.Trial'],
    n_rounds: int,
    eval_interval: int,
    validator: Optional[BaseValidator],
    config: Dict[str, Any],
    full_feedback: bool,
    lr: float,
    uses_lr: bool,
    seed_idx: int = 0,
    n_seeds: int = 1,
    wandb_logger: Optional['WandbRunLogger'] = None,
    is_online: bool = True,
) -> ExperimentResult:
    rewards = []
    optimal_rewards = []
    regrets = []
    cumulative_regret = 0.0

    # Track cumulative holdout metrics (only for non-online datasets like nfl_logged)
    cumulative_holdout_regret = 0.0
    cumulative_holdout_reward = 0.0

    # Determine log interval for wandb (every N rounds to avoid too many logs)
    wandb_log_interval = config.get('wandb', {}).get('log_interval', 10)

    # Check if env supports holdout evaluation (only relevant for non-online datasets)
    has_holdout = hasattr(env, 'evaluate_on_holdout') and not is_online

    X = env.reset()

    for t in range(n_rounds):
        selection = agent.act(X)
        observed_values, reward, info = env.step(selection, full_feedback=full_feedback)

        rewards.append(reward)
        optimal_reward = info['optimal_reward']
        optimal_rewards.append(optimal_reward)
        instant_regret = optimal_reward - reward
        regrets.append(instant_regret)
        cumulative_regret += instant_regret

        if uses_lr:
            agent.update(X, selection, observed_values, lr=lr, full_feedback=full_feedback)
        else:
            agent.update(X, selection, observed_values, full_feedback=full_feedback)

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
            )

        should_eval = (t + 1) % eval_interval == 0 or t == n_rounds - 1
        if should_eval:
            # For holdout datasets (nfl_logged), evaluate on holdout
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

            if trial is not None:
                # Offset step by seed index to avoid "step already reported" warnings
                unique_step = seed_idx * n_rounds + t

                # For online datasets: use cumulative training regret for pruning
                # For holdout datasets: use cumulative holdout regret for pruning
                if is_online:
                    pruning_metric = cumulative_regret
                else:
                    pruning_metric = cumulative_holdout_regret

                trial.report(pruning_metric, step=unique_step)

                if trial.should_prune():
                    return ExperimentResult(
                        final_validation_regret=cumulative_regret,  # Use training regret
                        final_train_regret=cumulative_regret,
                        validation_regrets=[],  # No validation for online
                        train_regrets=[sum(regrets[:i+1]) for i in range(len(regrets))],
                        rewards=rewards,
                        optimal_rewards=optimal_rewards,
                        pruned=True,
                        pruned_at_round=t,
                        total_rounds=t + 1,
                        cumulative_validation_regret=0.0,  # Not used for online
                        cumulative_holdout_regret=cumulative_holdout_regret if has_holdout else None,
                    )

    # Get final holdout regret for the result (only for holdout datasets)
    holdout_regret = None
    if has_holdout:
        holdout_result = env.evaluate_on_holdout(agent)
        if holdout_result:
            holdout_regret = holdout_result['regret']

    return ExperimentResult(
        final_validation_regret=cumulative_regret,  # Use training regret (no validation for online)
        final_train_regret=cumulative_regret,
        validation_regrets=[],  # No validation for online datasets
        train_regrets=[sum(regrets[:i+1]) for i in range(len(regrets))],
        rewards=rewards,
        optimal_rewards=optimal_rewards,
        pruned=False,
        pruned_at_round=None,
        total_rounds=n_rounds,
        holdout_regret=holdout_regret,
        cumulative_validation_regret=0.0,  # Not used for online
        cumulative_holdout_regret=cumulative_holdout_regret if has_holdout else None,
    )


def run_without_pruning(
    env,
    agent,
    n_rounds: int,
    eval_interval: int,
    validator: Optional[BaseValidator],
    config: Dict[str, Any],
    full_feedback: bool,
    lr: float,
    uses_lr: bool,
    seed_idx: int = 0,
    n_seeds: int = 1,
    wandb_logger: Optional['WandbRunLogger'] = None,
    is_online: bool = True,
) -> ExperimentResult:
    return run_with_pruning(
        env=env,
        agent=agent,
        trial=None,
        n_rounds=n_rounds,
        eval_interval=eval_interval,
        validator=validator,
        config=config,
        full_feedback=full_feedback,
        lr=lr,
        uses_lr=uses_lr,
        seed_idx=seed_idx,
        n_seeds=n_seeds,
        wandb_logger=wandb_logger,
        is_online=is_online,
    )


class PruningCallback:

    def __init__(self, patience: int, min_improvement: float, min_rounds: int):
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_rounds = min_rounds
        self._best_metric = float('inf')
        self._rounds_without_improvement = 0

    def should_prune(self, metric: float, round_num: int) -> bool:
        if round_num < self.min_rounds:
            return False

        if metric < self._best_metric * (1 - self.min_improvement):
            self._best_metric = metric
            self._rounds_without_improvement = 0
            return False

        self._rounds_without_improvement += 1

        if self._rounds_without_improvement >= self.patience:
            return True

        return False

    def reset(self):
        self._best_metric = float('inf')
        self._rounds_without_improvement = 0


def create_optuna_pruner(config: Dict[str, Any]) -> 'optuna.pruners.BasePruner':
    es_config = config['early_stopping']
    method = es_config['method']

    if method.upper() == 'ASHA':
        return optuna.pruners.HyperbandPruner(
            min_resource=es_config['min_resource'],
            max_resource=es_config['max_resource'],
            reduction_factor=es_config['reduction_factor'],
        )
    elif method.upper() == 'MEDIAN':
        return optuna.pruners.MedianPruner(
            n_startup_trials=es_config['n_startup_trials'],
            n_warmup_steps=es_config['n_warmup_steps'],
        )
    elif method.upper() == 'PERCENTILE':
        return optuna.pruners.PercentilePruner(
            percentile=es_config['percentile'],
            n_startup_trials=es_config['n_startup_trials'],
            n_warmup_steps=es_config['n_warmup_steps'],
        )
    elif method.upper() == 'NONE':
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruning method: {method}")

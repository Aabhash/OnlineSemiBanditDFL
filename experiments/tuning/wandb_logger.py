"""Wandb logging for hyperparameter tuning experiments."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


class WandbRunLogger:
    def __init__(
        self,
        config: Dict[str, Any],
        agent_name: str,
        dataset: str,
        feedback_type: str,
        trial_number: int,
        seed: int,
        params: Dict[str, Any],
    ):
        self.config = config
        self.agent_name = agent_name
        self.dataset = dataset
        self.feedback_type = feedback_type
        self.trial_number = trial_number
        self.seed = seed
        self.params = params

        self._wandb = None
        self._run = None
        self._enabled = config.get('wandb', {}).get('enabled', False)

        if self._enabled:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                self._enabled = False

    def start(self):
        if not self._enabled:
            return

        wandb_config = self.config.get('wandb', {})
        # Get tuning-specific config (nested under wandb.tuning)
        tuning_config = wandb_config.get('tuning', {})

        run_name = f"{self.dataset}_{self.feedback_type}_{self.agent_name}_trial{self.trial_number}_seed{self.seed}"

        run_config = {
            'agent': self.agent_name,
            'dataset': self.dataset,
            'feedback_type': self.feedback_type,
            'trial_number': self.trial_number,
            'seed': self.seed,
            'n_rounds': self.config['tuning']['n_rounds'],
            **{f'hp/{k}': v for k, v in self.params.items()},
        }

        # Get tags from tuning config, fallback to top-level tags
        base_tags = tuning_config.get('tags', wandb_config.get('tags', []))
        tags = base_tags + [
            self.agent_name,
            self.dataset,
            self.feedback_type,
            f'trial_{self.trial_number}',
            f'seed_{self.seed}',
        ]

        group = f"{self.dataset}_{self.feedback_type}_{self.agent_name}"

        # Get project from tuning config, fallback to top-level project
        project = tuning_config.get('project', wandb_config.get('project', 'constrained-bandits-tuning'))

        self._run = self._wandb.init(
            project=project,
            entity=wandb_config.get('entity'),
            name=run_name,
            group=group,
            job_type=f'trial_{self.trial_number}',
            config=run_config,
            tags=tags,
            reinit=True,
        )

    def log_round(
        self,
        round_num: int,
        regret: float,
        cumulative_regret: float,
        reward: float,
        optimal_reward: float,
        cumulative_reward: Optional[float] = None,
    ):
        if not self._enabled or self._run is None:
            return

        metrics = {
            'round': round_num,
            'train/regret': regret,
            'train/cumulative_regret': cumulative_regret,
            'train/reward': reward,
            'train/optimal_reward': optimal_reward,
        }
        if cumulative_reward is not None:
            metrics['train/cumulative_reward'] = cumulative_reward

        self._wandb.log(metrics, step=round_num)

    def log_validation(
        self,
        round_num: int,
        validation_regret: float,
        cumulative_validation_regret: float,
        validation_reward: Optional[float] = None,
        cumulative_validation_reward: Optional[float] = None,
    ):
        if not self._enabled or self._run is None:
            return

        metrics = {
            'validation/regret': validation_regret,
            'validation/cumulative_regret': cumulative_validation_regret,
        }
        if validation_reward is not None:
            metrics['validation/reward'] = validation_reward
        if cumulative_validation_reward is not None:
            metrics['validation/cumulative_reward'] = cumulative_validation_reward

        self._wandb.log(metrics, step=round_num)

    def log_holdout(
        self,
        round_num: int,
        holdout_regret: float,
        cumulative_holdout_regret: float,
        holdout_reward: Optional[float] = None,
        cumulative_holdout_reward: Optional[float] = None,
    ):
        if not self._enabled or self._run is None:
            return

        metrics = {
            'holdout/regret': holdout_regret,
            'holdout/cumulative_regret': cumulative_holdout_regret,
        }
        if holdout_reward is not None:
            metrics['holdout/reward'] = holdout_reward
        if cumulative_holdout_reward is not None:
            metrics['holdout/cumulative_reward'] = cumulative_holdout_reward

        self._wandb.log(metrics, step=round_num)

    def log_final_summary(
        self,
        final_train_cumulative_regret: float,
        final_train_cumulative_reward: Optional[float] = None,
        final_train_regret_per_round: Optional[float] = None,
        final_train_reward_per_round: Optional[float] = None,
        final_holdout_cumulative_regret: Optional[float] = None,
        final_holdout_cumulative_reward: Optional[float] = None,
        final_holdout_regret_per_round: Optional[float] = None,
        final_holdout_reward_per_round: Optional[float] = None,
        wall_time_seconds: float = 0.0,
        total_rounds: int = 0,
    ):
        if not self._enabled or self._run is None:
            return

        # Train metrics
        self._wandb.summary['final_train_cumulative_regret'] = final_train_cumulative_regret
        if final_train_cumulative_reward is not None:
            self._wandb.summary['final_train_cumulative_reward'] = final_train_cumulative_reward
        if final_train_regret_per_round is not None:
            self._wandb.summary['final_train_regret_per_round'] = final_train_regret_per_round
        if final_train_reward_per_round is not None:
            self._wandb.summary['final_train_reward_per_round'] = final_train_reward_per_round

        # Holdout metrics (same pattern as train)
        if final_holdout_cumulative_regret is not None:
            self._wandb.summary['final_holdout_cumulative_regret'] = final_holdout_cumulative_regret
        if final_holdout_cumulative_reward is not None:
            self._wandb.summary['final_holdout_cumulative_reward'] = final_holdout_cumulative_reward
        if final_holdout_regret_per_round is not None:
            self._wandb.summary['final_holdout_regret_per_round'] = final_holdout_regret_per_round
        if final_holdout_reward_per_round is not None:
            self._wandb.summary['final_holdout_reward_per_round'] = final_holdout_reward_per_round

        self._wandb.summary['wall_time_seconds'] = wall_time_seconds
        self._wandb.summary['total_rounds'] = total_rounds

    def finish(self, status: str = 'completed'):
        if not self._enabled or self._run is None:
            return

        self._wandb.summary['status'] = status
        self._wandb.finish()
        self._run = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = 'failed' if exc_type is not None else 'completed'
        self.finish(status)
        return False


class WandbTrialLogger:

    def __init__(
        self,
        config: Dict[str, Any],
        agent_name: str,
        dataset: str,
        feedback_type: str,
        trial_number: int,
        params: Dict[str, Any],
    ):
        self.config = config
        self.agent_name = agent_name
        self.dataset = dataset
        self.feedback_type = feedback_type
        self.trial_number = trial_number
        self.params = params
        self._enabled = config.get('wandb', {}).get('enabled', False)

    def create_seed_logger(self, seed: int) -> WandbRunLogger:
        """Create a logger for a specific seed."""
        return WandbRunLogger(
            config=self.config,
            agent_name=self.agent_name,
            dataset=self.dataset,
            feedback_type=self.feedback_type,
            trial_number=self.trial_number,
            seed=seed,
            params=self.params,
        )


class WandbStudyLogger:

    def __init__(
        self,
        config: Dict[str, Any],
        agent_name: str,
        dataset: str,
        feedback_type: str,
    ):
        self.config = config
        self.agent_name = agent_name
        self.dataset = dataset
        self.feedback_type = feedback_type

        self._wandb = None
        self._enabled = config.get('wandb', {}).get('enabled', False)

        if self._enabled:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                self._enabled = False

    def create_trial_logger(
        self,
        trial_number: int,
        params: Dict[str, Any],
    ) -> WandbTrialLogger:
        return WandbTrialLogger(
            config=self.config,
            agent_name=self.agent_name,
            dataset=self.dataset,
            feedback_type=self.feedback_type,
            trial_number=trial_number,
            params=params,
        )

    def log_study_summary(
        self,
        best_params: Dict[str, Any],
        best_value: float,
        n_trials: int,
        total_time_minutes: float,
    ):
        if not self._enabled:
            return

        wandb_config = self.config.get('wandb', {})
        run_name = f"{self.dataset}_{self.feedback_type}_{self.agent_name}_summary"

        run = self._wandb.init(
            project=wandb_config.get('project', 'constrained-bandits'),
            entity=wandb_config.get('entity'),
            name=run_name,
            group=f"{self.dataset}_{self.feedback_type}_{self.agent_name}",
            job_type='summary',
            config={
                'agent': self.agent_name,
                'dataset': self.dataset,
                'feedback_type': self.feedback_type,
                'type': 'study_summary',
            },
            tags=wandb_config.get('tags', []) + [
                self.agent_name,
                self.dataset,
                self.feedback_type,
                'summary',
            ],
            reinit=True,
        )

        self._wandb.summary['best_value'] = best_value
        self._wandb.summary['best_params'] = best_params
        self._wandb.summary['n_trials'] = n_trials
        self._wandb.summary['total_time_minutes'] = total_time_minutes

        self._wandb.finish()

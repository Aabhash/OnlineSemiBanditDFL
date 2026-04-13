import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import PercentilePruner

from .objectives import (
    ObjectiveFactory,
    load_search_spaces,
    sample_hyperparameters,
    create_agent_from_params,
    create_env_for_tuning,
    get_n_rounds_for_dataset,
    uses_holdout_for_selection,
)
from .results_store import ResultsStore
from .timing import ComputeTimer


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


class BanditHyperparameterTuner:

    def __init__(
        self,
        config_path: str,
        search_space_dir: str,
        output_dir: Optional[str] = None,
        backbone_tuning_dir: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        self.search_spaces = load_search_spaces(search_space_dir)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_store = ResultsStore(str(self.output_dir))
        if backbone_tuning_dir:
            self.backbone_results_store = ResultsStore(backbone_tuning_dir)
        else:
            self.backbone_results_store = self.results_store
        self.timer = ComputeTimer(use_gpu=self.config['compute']['use_gpu'])

        db_path = self.output_dir / 'optuna.db'
        self.storage = f"sqlite:///{db_path}"

        print(f"Initialized tuner with config from {config_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Loaded search spaces for {len(self.search_spaces)} agents")

    def _get_study_name(
        self,
        agent_name: str,
        dataset: str,
        feedback_type: str,
    ) -> str:
        return f"{dataset}_{agent_name}_{feedback_type}"

    def tune_agent(
        self,
        agent_name: str,
        dataset: str,
        feedback_type: str,
        n_trials: Optional[int] = None,
        resume: bool = True,
    ) -> 'optuna.Study':
        if agent_name not in self.search_spaces:
            raise ValueError(
                f"No search space defined for {agent_name}. "
                f"Available: {list(self.search_spaces.keys())}"
            )

        search_space = self.search_spaces[agent_name]
        study_name = self._get_study_name(agent_name, dataset, feedback_type)

        sampler_seed = self.config['tuning']['sampler_seed']


        pruning_config = self.config.get('tuning', {}).get('pruning', {})
        pruning_enabled = pruning_config.get('enabled', False)

        if pruning_enabled:


            percentile = pruning_config.get('percentile_threshold', 75)
            warmup_trials = pruning_config.get('warmup_trials', 5)
            min_seeds = pruning_config.get('min_seeds_before_pruning', 1)

            pruner = PercentilePruner(
                percentile=percentile,
                n_startup_trials=warmup_trials,
                n_warmup_steps=min_seeds,
            )
        else:
            pruner = optuna.pruners.NopPruner()

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=TPESampler(seed=sampler_seed),
            pruner=pruner,
            storage=self.storage,
            load_if_exists=resume,
        )

        objective = ObjectiveFactory.create_objective(
            agent_name=agent_name,
            dataset=dataset,
            config=self.config,
            search_space=search_space,
            results_store=self.results_store,
            feedback_type=feedback_type,
            backbone_results_store=self.backbone_results_store,
        )

        if n_trials is None:
            n_trials = self.config['tuning']['n_trials']

        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = max(0, n_trials - completed_trials)

        if remaining_trials == 0:
            print(f"Study {study_name} already has {completed_trials} completed trials")
            return study

        print(f"\nTuning {agent_name} on {dataset} ({feedback_type})")
        print(f"  Trials: {remaining_trials} remaining (target: {n_trials})")
        print(f"  Search space: {list(search_space.keys())}")
        print(f"  Pruning: {'enabled' if pruning_enabled else 'disabled'}")
        if pruning_enabled:
            print(f"    - Percentile threshold: {pruning_config.get('percentile_threshold', 75)}%")
            print(f"    - Min seeds before pruning: {pruning_config.get('min_seeds_before_pruning', 1)}")
            print(f"    - Warmup trials: {pruning_config.get('warmup_trials', 5)}")

        self.timer.start()
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=self.config['tuning']['timeout_per_trial'] * remaining_trials,
            n_jobs=self.config['compute']['max_parallel_trials'],
            show_progress_bar=True,
        )
        elapsed = self.timer.stop()


        print(f"\nCompleted {agent_name} tuning in {elapsed/60:.1f} minutes")
        print(f"  Best trial: {study.best_trial.number}")
        print(f"  Best value: {study.best_value:.2f}")
        print(f"  Best params: {study.best_params}")

        return study

    def tune_all_agents(
        self,
        dataset: str,
        agents: Optional[List[str]] = None,
        feedback_type: str = 'semi_bandit',
    ) -> Dict[str, 'optuna.Study']:
        if agents is None:
            agents = list(self.search_spaces.keys())

        effective_n_rounds = get_n_rounds_for_dataset(self.config, dataset)
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TUNING")
        print(f"{'='*70}")
        print(f"Dataset: {dataset}")
        print(f"Feedback: {feedback_type}")
        print(f"Agents: {agents}")
        print(f"Trials per agent: {self.config['tuning']['n_trials']}")
        print(f"Tuning seeds: {self.config.get('tuning_seeds', [0, 1, 2, 3, 4])}")
        print(f"Max rounds: {effective_n_rounds}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        studies = {}
        for agent_name in agents:
            studies[agent_name] = self.tune_agent(
                agent_name, dataset, feedback_type
            )

        self._save_tuning_summary(dataset, feedback_type, agents)

        print(f"\n{'='*70}")
        print(f"TUNING COMPLETE")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        return studies

    def _save_tuning_summary(
        self,
        dataset: str,
        feedback_type: str,
        agents: List[str],
    ):
        use_holdout = uses_holdout_for_selection(self.config, dataset)
        best_params = self.results_store.save_best_hyperparams(
            agents, dataset, feedback_type, use_holdout=use_holdout
        )

        compute_summary = self.results_store.save_compute_summary()

        print(f"\n{'='*70}")
        print("BEST HYPERPARAMETERS")
        print(f"{'='*70}")
        print(f"{'Agent':<25} {'Sel. Regret':>15} {'Params'}")
        print("-" * 70)

        sorted_agents = sorted(
            best_params.items(),
            key=lambda x: x[1]['mean_validation_regret']
        )

        for agent_name, result in sorted_agents:
            val_regret = result['mean_validation_regret']
            std = result['std_validation_regret']
            params_str = ', '.join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in result['params'].items())
            if len(params_str) > 35:
                params_str = params_str[:32] + "..."
            print(f"{agent_name:<25} {val_regret:>8.1f}±{std:<5.1f} {params_str}")

        print(f"\n{'='*70}")
        print("COMPUTE SUMMARY")
        print(f"{'='*70}")
        print(f"Total trials: {compute_summary['n_trials']}")
        print(f"Total wall time: {compute_summary['total_wall_time_hours']:.2f} hours")

    def get_best_params(
        self,
        agent_name: str,
        dataset: str,
        feedback_type: str,
    ) -> Dict[str, Any]:
        use_holdout = uses_holdout_for_selection(self.config, dataset)
        return self.results_store.get_best_params(agent_name, dataset, feedback_type, use_holdout=use_holdout)

    def load_study(
        self,
        agent_name: str,
        dataset: str,
        feedback_type: str,
    ) -> 'optuna.Study':
        study_name = self._get_study_name(agent_name, dataset, feedback_type)
        return optuna.load_study(study_name=study_name, storage=self.storage)


def quick_tune(
    agent_name: str,
    dataset: str,
    config_path: str = 'experiments/configs/experiment.yaml',
    search_space_dir: str = 'experiments/configs/search_spaces',
    n_trials: int = 10,
    n_seeds: int = 3,
) -> Dict[str, Any]:
    config = load_config(config_path)
    config['tuning']['n_trials'] = n_trials
    config['tuning_seeds'] = list(range(n_seeds))

    search_spaces = load_search_spaces(search_space_dir)

    if agent_name not in search_spaces:
        raise ValueError(f"Agent {agent_name} not found in search spaces")

    sampler_seed = config['tuning']['sampler_seed']

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=sampler_seed),
    )

    objective = ObjectiveFactory.create_simple_objective(
        agent_name=agent_name,
        dataset=dataset,
        config=config,
        search_space=search_spaces[agent_name],
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params

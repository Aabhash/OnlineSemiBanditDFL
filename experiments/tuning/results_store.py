import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np


@dataclass
class TrialResult:
    trial_id: int
    agent_name: str
    dataset: str
    feedback_type: str
    seed: int
    params: Dict[str, Any]
    validation_regrets: List[float]
    train_regrets: List[float]
    final_validation_regret: float
    final_train_regret: float
    holdout_regret: Optional[float] = None
    wall_time_seconds: float = 0.0
    gpu_time_seconds: Optional[float] = None
    total_rounds: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Cumulative metrics for online bandit evaluation
    cumulative_validation_regret: float = 0.0
    cumulative_holdout_regret: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                d[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, list):
                d[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrialResult':
        # Filter out legacy fields that no longer exist
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class AgentSummary:
    agent_name: str
    dataset: str
    feedback_type: str
    best_params: Dict[str, Any]
    mean_validation_regret: float
    std_validation_regret: float
    mean_train_regret: float
    std_train_regret: float
    mean_holdout_regret: Optional[float] = None
    std_holdout_regret: Optional[float] = None
    total_wall_time_seconds: float = 0.0
    total_gpu_time_seconds: Optional[float] = None
    n_trials: int = 0
    validation_ci_lower: Optional[float] = None
    validation_ci_upper: Optional[float] = None


class ResultsStore:

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir = self.output_dir / 'trials'
        self.trials_dir.mkdir(exist_ok=True)
        self.summaries_dir = self.output_dir / 'summaries'
        self.summaries_dir.mkdir(exist_ok=True)
        self._trials_cache: Dict[str, TrialResult] = {}

    def _get_trial_key(self, result: TrialResult) -> str:
        return f"{result.agent_name}_{result.dataset}_{result.feedback_type}_t{result.trial_id}_s{result.seed}"

    def _get_trial_filename(self, result: TrialResult) -> Path:
        key = self._get_trial_key(result)
        return self.trials_dir / f"{key}.json"

    def save_trial(self, result: TrialResult):
        filepath = self._get_trial_filename(result)
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        key = self._get_trial_key(result)
        self._trials_cache[key] = result

    def load_trial(self, filepath: Path) -> TrialResult:
        with open(filepath) as f:
            data = json.load(f)
        return TrialResult.from_dict(data)

    def load_all_trials(
        self,
        agent_name: Optional[str] = None,
        dataset: Optional[str] = None,
        feedback_type: Optional[str] = None,
    ) -> List[TrialResult]:
        trials = []
        for path in self.trials_dir.glob('*.json'):
            result = self.load_trial(path)
            if agent_name and result.agent_name != agent_name:
                continue
            if dataset and result.dataset != dataset:
                continue
            if feedback_type and result.feedback_type != feedback_type:
                continue
            trials.append(result)
        return trials

    def get_best_params(
        self,
        agent_name: str,
        dataset: str,
        feedback_type: str,
        use_holdout: bool = False,
    ) -> Dict[str, Any]:
        trials = self.load_all_trials(agent_name, dataset, feedback_type)
        if not trials:
            raise ValueError(f"No trials found for {agent_name} on {dataset} ({feedback_type})")

        param_metrics: Dict[str, List[float]] = {}
        param_map: Dict[str, Dict[str, Any]] = {}

        for trial in trials:
            key = json.dumps(trial.params, sort_keys=True)
            if key not in param_metrics:
                param_metrics[key] = []
                param_map[key] = trial.params

            # Use the appropriate metric based on use_holdout flag
            if use_holdout and trial.cumulative_holdout_regret is not None:
                param_metrics[key].append(trial.cumulative_holdout_regret)
            else:
                param_metrics[key].append(trial.final_train_regret)

        best_key = min(param_metrics, key=lambda k: np.mean(param_metrics[k]))
        return param_map[best_key]

    def get_results_for_params(
        self,
        agent_name: str,
        dataset: str,
        params: Dict[str, Any],
        feedback_type: str,
    ) -> List[TrialResult]:
        trials = self.load_all_trials(agent_name, dataset, feedback_type)
        return [t for t in trials if t.params == params]

    def compute_agent_summary(
        self,
        agent_name: str,
        dataset: str,
        feedback_type: str,
        use_holdout: bool = False,
    ) -> AgentSummary:
        trials = self.load_all_trials(agent_name, dataset, feedback_type)
        if not trials:
            raise ValueError(f"No trials found for {agent_name} on {dataset}")

        best_params = self.get_best_params(agent_name, dataset, feedback_type, use_holdout=use_holdout)
        best_trials = self.get_results_for_params(agent_name, dataset, best_params, feedback_type)

        # Use final_train_regret (cumulative regret) for summary
        train_regrets = [t.final_train_regret for t in best_trials]
        holdout_regrets = [t.holdout_regret for t in best_trials if t.holdout_regret is not None]

        total_wall_time = sum(t.wall_time_seconds for t in trials)
        total_gpu_time = sum(t.gpu_time_seconds for t in trials if t.gpu_time_seconds is not None)

        mean_train = float(np.mean(train_regrets))
        std_train = float(np.std(train_regrets))

        return AgentSummary(
            agent_name=agent_name,
            dataset=dataset,
            feedback_type=feedback_type,
            best_params=best_params,
            mean_validation_regret=mean_train,  # Same as train (no separate validation)
            std_validation_regret=std_train,
            mean_train_regret=mean_train,
            std_train_regret=std_train,
            mean_holdout_regret=float(np.mean(holdout_regrets)) if holdout_regrets else None,
            std_holdout_regret=float(np.std(holdout_regrets)) if holdout_regrets else None,
            total_wall_time_seconds=total_wall_time,
            total_gpu_time_seconds=total_gpu_time if total_gpu_time > 0 else None,
            n_trials=len(trials),
        )

    def save_summary(self, summary: AgentSummary):
        filename = f"{summary.agent_name}_{summary.dataset}_{summary.feedback_type}_summary.json"
        filepath = self.summaries_dir / filename
        with open(filepath, 'w') as f:
            json.dump(asdict(summary), f, indent=2)

    def save_best_hyperparams(
        self,
        agents: List[str],
        dataset: str,
        feedback_type: str,
        use_holdout: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        best_params = {}
        for agent in agents:
            params = self.get_best_params(agent, dataset, feedback_type, use_holdout=use_holdout)
            summary = self.compute_agent_summary(agent, dataset, feedback_type, use_holdout=use_holdout)
            best_params[agent] = {
                'params': params,
                'mean_validation_regret': summary.mean_validation_regret,
                'std_validation_regret': summary.std_validation_regret,
                'n_trials': summary.n_trials,
            }

        filepath = self.output_dir / f"best_hyperparams_{dataset}_{feedback_type}.json"
        with open(filepath, 'w') as f:
            json.dump(best_params, f, indent=2)

        return best_params

    def get_compute_summary(self) -> Dict[str, Any]:
        trials = self.load_all_trials()
        if not trials:
            raise ValueError("No trials found")

        agent_times: Dict[str, float] = {}
        agent_counts: Dict[str, int] = {}

        total_wall_time = 0.0
        total_gpu_time = 0.0

        for trial in trials:
            agent = trial.agent_name
            if agent not in agent_times:
                agent_times[agent] = 0.0
                agent_counts[agent] = 0

            agent_times[agent] += trial.wall_time_seconds
            agent_counts[agent] += 1
            total_wall_time += trial.wall_time_seconds

            if trial.gpu_time_seconds:
                total_gpu_time += trial.gpu_time_seconds

        return {
            'total_wall_time_seconds': total_wall_time,
            'total_wall_time_hours': total_wall_time / 3600,
            'total_gpu_time_seconds': total_gpu_time if total_gpu_time > 0 else None,
            'total_gpu_time_hours': total_gpu_time / 3600 if total_gpu_time > 0 else None,
            'n_trials': len(trials),
            'per_agent_wall_time_hours': {k: v / 3600 for k, v in agent_times.items()},
            'per_agent_n_trials': agent_counts,
        }

    def save_compute_summary(self):
        summary = self.get_compute_summary()
        filepath = self.output_dir / 'compute_summary.json'
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        return summary

    def get_trials_for_sensitivity(
        self,
        agent_name: str,
        dataset: str,
        hyperparameter: str,
        feedback_type: str,
    ) -> Dict[Any, List[float]]:
        trials = self.load_all_trials(agent_name, dataset, feedback_type)

        hp_values: Dict[Any, List[float]] = {}

        for trial in trials:
            if hyperparameter not in trial.params:
                continue

            value = trial.params[hyperparameter]
            if value not in hp_values:
                hp_values[value] = []
            hp_values[value].append(trial.final_train_regret)

        return hp_values

    def clear_trials(self, confirm: bool = False):
        if not confirm:
            raise ValueError("Set confirm=True to clear all trials")

        for path in self.trials_dir.glob('*.json'):
            path.unlink()
        self._trials_cache.clear()

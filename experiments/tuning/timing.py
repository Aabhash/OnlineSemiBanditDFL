import time
import subprocess
from typing import Optional, Dict
from dataclasses import dataclass, field
import jax


@dataclass
class TimingRecord:
    wall_time_seconds: float = 0.0
    gpu_time_seconds: Optional[float] = None
    n_rounds: int = 0


class ComputeTimer:

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._start_time: Optional[float] = None
        self._total_wall_time: float = 0.0
        self._total_gpu_time: float = 0.0
        self._n_trials: int = 0
        self._trial_times: list = []
        self._gpu_available = False

        if use_gpu:
            self._gpu_available = self._check_gpu_available()
            if not self._gpu_available:
                raise RuntimeError(
                    "GPU tracking requested but no GPU available. "
                    "Ensure CUDA is installed and nvidia-smi is accessible, "
                    "or set use_gpu=False in config."
                )

    def _check_gpu_available(self) -> bool:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return False
            devices = jax.devices()
            return any(d.platform == 'gpu' for d in devices)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def start(self):
        self._start_time = time.perf_counter()
        if self.use_gpu:
            jax.block_until_ready(jax.numpy.array([0.0]))

    def stop(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Timer was not started")

        if self.use_gpu:
            jax.block_until_ready(jax.numpy.array([0.0]))

        elapsed = time.perf_counter() - self._start_time
        self._total_wall_time += elapsed
        self._trial_times.append(elapsed)
        self._n_trials += 1
        self._start_time = None

        return elapsed

    def record_trial(self, record: TimingRecord):
        self._total_wall_time += record.wall_time_seconds
        if record.gpu_time_seconds is not None:
            self._total_gpu_time += record.gpu_time_seconds
        self._trial_times.append(record.wall_time_seconds)
        self._n_trials += 1

    def get_summary(self) -> Dict[str, float]:
        if self._n_trials == 0:
            raise RuntimeError("No trials recorded")

        return {
            'total_wall_time_seconds': self._total_wall_time,
            'total_wall_time_hours': self._total_wall_time / 3600,
            'total_gpu_time_seconds': self._total_gpu_time if self.use_gpu else None,
            'total_gpu_time_hours': self._total_gpu_time / 3600 if self.use_gpu else None,
            'total_cpu_time_seconds': self._total_wall_time - self._total_gpu_time if self.use_gpu else self._total_wall_time,
            'total_cpu_time_hours': (self._total_wall_time - self._total_gpu_time) / 3600 if self.use_gpu else self._total_wall_time / 3600,
            'n_trials': self._n_trials,
            'avg_time_per_trial_seconds': self._total_wall_time / self._n_trials,
            'avg_time_per_trial_minutes': (self._total_wall_time / self._n_trials) / 60,
        }

    def reset(self):
        self._start_time = None
        self._total_wall_time = 0.0
        self._total_gpu_time = 0.0
        self._n_trials = 0
        self._trial_times = []


class TrialTimer:

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def __enter__(self):
        if self.use_gpu:
            jax.block_until_ready(jax.numpy.array([0.0]))
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_gpu:
            jax.block_until_ready(jax.numpy.array([0.0]))
        self._end_time = time.perf_counter()
        return False

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return end - self._start_time

    def get_record(self, n_rounds: int) -> TimingRecord:
        return TimingRecord(
            wall_time_seconds=self.elapsed,
            gpu_time_seconds=None,
            n_rounds=n_rounds,
        )


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def compute_cost_summary(
    agent_times: Dict[str, float],
    use_gpu: bool = False
) -> Dict[str, any]:
    if not agent_times:
        raise ValueError("No agent times provided")

    total_time = sum(agent_times.values())

    return {
        'total_wall_time_seconds': total_time,
        'total_wall_time_hours': total_time / 3600,
        'per_agent_seconds': agent_times,
        'per_agent_hours': {k: v / 3600 for k, v in agent_times.items()},
        'used_gpu': use_gpu,
        'avg_per_agent_seconds': total_time / len(agent_times),
        'avg_per_agent_hours': (total_time / len(agent_times)) / 3600,
    }

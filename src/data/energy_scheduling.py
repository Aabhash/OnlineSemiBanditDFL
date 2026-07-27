import shlex
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


ENERGY_FEATURES = (
    "HolidayFlag",
    "DayOfWeek",
    "WeekOfYear",
    "Month",
    "ForecastWindProduction",
    "SystemLoadEA",
    "SMPEA",
    "CO2Intensity",
)


@dataclass(frozen=True)
class EnergySchedulingInstance:
    q_minutes: int
    machine_capacities: np.ndarray
    task_usage: np.ndarray
    durations: np.ndarray
    earliest: np.ndarray
    latest: np.ndarray
    powers: np.ndarray

    def __post_init__(self):
        capacities = np.asarray(self.machine_capacities, dtype=float)
        usage = np.asarray(self.task_usage, dtype=float)
        durations = np.asarray(self.durations, dtype=int)
        earliest = np.asarray(self.earliest, dtype=int)
        latest = np.asarray(self.latest, dtype=int)
        powers = np.asarray(self.powers, dtype=float)

        if 1440 % int(self.q_minutes):
            raise ValueError("q_minutes must divide one day exactly")
        if capacities.ndim != 2 or usage.ndim != 2:
            raise ValueError("machine capacities and task usage must be matrices")
        if usage.shape[1] != capacities.shape[1]:
            raise ValueError(
                "task usage and machine capacities must use the same resources"
            )
        n_tasks = usage.shape[0]
        if any(
            len(values) != n_tasks for values in (durations, earliest, latest, powers)
        ):
            raise ValueError("task arrays must have one entry per task")
        if np.any(durations <= 0) or np.any(powers < 0):
            raise ValueError("durations must be positive and powers nonnegative")
        if np.any(earliest < 0) or np.any(latest > self.horizon):
            raise ValueError("task windows must lie inside the scheduling horizon")
        if np.any(earliest + durations > latest):
            raise ValueError("every task must have at least one feasible start")

        object.__setattr__(self, "machine_capacities", capacities)
        object.__setattr__(self, "task_usage", usage)
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "earliest", earliest)
        object.__setattr__(self, "latest", latest)
        object.__setattr__(self, "powers", powers)

    @property
    def horizon(self):
        return 1440 // int(self.q_minutes)

    @property
    def n_machines(self):
        return self.machine_capacities.shape[0]

    @property
    def n_resources(self):
        return self.machine_capacities.shape[1]

    @property
    def n_tasks(self):
        return self.task_usage.shape[0]


@dataclass(frozen=True)
class EnergyPriceDataset:
    contexts: np.ndarray
    slot_costs: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray
    normalization_start: int
    normalization_stop: int


def _interpolate_nans(values):
    values = np.asarray(values, dtype=float).copy()
    missing = np.isnan(values)
    if not missing.any():
        return values
    observed = np.flatnonzero(~missing)
    if not observed.size:
        raise ValueError("cannot interpolate a vector containing only missing values")
    values[missing] = np.interp(np.flatnonzero(missing), observed, values[observed])
    return values


def _interpolate_nans_by_split(values, split_stop):
    values = np.asarray(values, dtype=float).copy()
    values[:split_stop] = _interpolate_nans(values[:split_stop])
    values[split_stop:] = _interpolate_nans(values[split_stop:])
    return values


@lru_cache(maxsize=None)
def load_energy_price_dataset(data_root, validation_fraction=0.5):
    path = Path(data_root).expanduser().resolve() / "prices2013.dat"
    if not path.exists():
        raise FileNotFoundError(f"energy price data not found: {path}")

    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        columns = {name: i for i, name in enumerate(header)}
        required = (*ENERGY_FEATURES, "SMPEP2")
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"energy price file is missing columns: {missing}")
        rows = [shlex.split(line) for line in handle if line.strip()]

    if len(rows) % 48:
        raise ValueError(f"expected a multiple of 48 half-hour rows, got {len(rows)}")

    features = np.asarray(
        [[float(row[columns[name]]) for name in ENERGY_FEATURES] for row in rows],
        dtype=float,
    )
    slot_costs = np.asarray(
        [float(row[columns["SMPEP2"]]) for row in rows], dtype=float
    )

    co2_column = ENERGY_FEATURES.index("CO2Intensity")
    features[features[:, co2_column] == 0, co2_column] = np.nan
    _, normalization_stop = energy_split_indices(
        len(rows) // 48,
        split="validation",
        validation_fraction=float(validation_fraction),
    )
    features[:, co2_column] = _interpolate_nans_by_split(
        features[:, co2_column],
        split_stop=48 * normalization_stop,
    )

    contexts, costs = [], []
    for start in range(0, len(rows), 48):
        day_context = features[start : start + 48]
        day_costs = slot_costs[start : start + 48]
        if np.isnan(day_context).any() or np.isnan(day_costs).any():
            continue
        contexts.append(day_context)
        costs.append(day_costs)

    contexts = np.asarray(contexts, dtype=float)
    costs = np.asarray(costs, dtype=float)
    normalization_start, normalization_stop = energy_split_indices(
        len(contexts),
        split="validation",
        validation_fraction=float(validation_fraction),
    )
    fit_contexts = contexts[normalization_start:normalization_stop].reshape(
        -1,
        len(ENERGY_FEATURES),
    )
    mean, std = fit_contexts.mean(axis=0), fit_contexts.std(axis=0)
    std[std < 1e-8] = 1.0
    contexts = (contexts - mean) / std
    return EnergyPriceDataset(
        contexts=contexts,
        slot_costs=costs,
        context_mean=mean,
        context_std=std,
        normalization_start=normalization_start,
        normalization_stop=normalization_stop,
    )


def energy_split_indices(n_days, split="experiment", validation_fraction=0.5):
    split = str(split).lower()
    if split == "all":
        return 0, n_days
    if split not in {"validation", "experiment"}:
        raise ValueError("split must be 'all', 'validation', or 'experiment'")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")

    cut = min(max(int(round(n_days * validation_fraction)), 1), n_days - 1)
    return (0, cut) if split == "validation" else (cut, n_days)


def load_energy_instance(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"energy scheduling instance not found: {path}")
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    q_minutes = int(lines[0])
    n_resources = int(lines[1])
    n_machines = int(lines[2])
    capacities = np.asarray(
        [
            list(map(float, lines[2 * (machine + 2)].split()))
            for machine in range(n_machines)
        ],
        dtype=float,
    )
    if capacities.shape != (n_machines, n_resources):
        raise ValueError(
            f"expected machine capacities with shape {(n_machines, n_resources)}, "
            f"got {capacities.shape}"
        )

    offset = 2 * n_machines + 2
    n_tasks = int(lines[offset + 1])
    usage, durations, earliest, latest, powers = [], [], [], [], []
    for task in range(n_tasks):
        task_row = lines[2 * task + offset + 2].split()
        durations.append(int(task_row[1]))
        earliest.append(int(task_row[2]))
        latest.append(int(task_row[3]))
        powers.append(float(task_row[4]))
        usage.append(list(map(float, lines[2 * task + offset + 3].split())))

    return EnergySchedulingInstance(
        q_minutes=q_minutes,
        machine_capacities=capacities,
        task_usage=np.asarray(usage, dtype=float),
        durations=np.asarray(durations),
        earliest=np.asarray(earliest),
        latest=np.asarray(latest),
        powers=np.asarray(powers),
    )

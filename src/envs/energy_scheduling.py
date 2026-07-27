from pathlib import Path

import numpy as np

from src.data.energy_scheduling import (
    ENERGY_FEATURES,
    energy_split_indices,
    load_energy_instance,
    load_energy_price_dataset,
)
from src.solvers.energy_scheduling import EnergySchedulingSolver


class EnergySchedulingEnv:
    def __init__(
        self,
        data_root,
        energy_instance="load3",
        dataset_split="experiment",
        validation_fraction=0.5,
        cost_scale=None,
        seed=42,
        **kwargs,
    ):
        del kwargs
        self.seed = int(seed)
        self.data_root = str(Path(data_root).expanduser().resolve())
        self.energy_instance = str(energy_instance)
        self.dataset_split = str(dataset_split)
        self.validation_fraction = float(validation_fraction)

        dataset = load_energy_price_dataset(
            self.data_root,
            validation_fraction=self.validation_fraction,
        )
        self.context_mean = dataset.context_mean.copy()
        self.context_std = dataset.context_std.copy()
        self.normalization_start = int(dataset.normalization_start)
        self.normalization_stop = int(dataset.normalization_stop)
        instance_path = Path(self.data_root) / self.energy_instance / "day01.txt"
        self.instance = load_energy_instance(instance_path)
        self.energy_solver = EnergySchedulingSolver(self.instance)
        all_raw_costs = dataset.slot_costs @ self.energy_solver.component_load_matrix.T

        if cost_scale is None:
            validation_start, validation_stop = energy_split_indices(
                len(dataset.contexts),
                "validation",
                self.validation_fraction,
            )
            nonzero = np.abs(all_raw_costs[validation_start:validation_stop])
            nonzero = nonzero[nonzero > 0]
            cost_scale = float(np.median(nonzero))
        self.cost_scale = float(cost_scale)
        if not np.isfinite(self.cost_scale) or self.cost_scale <= 0:
            raise ValueError("cost_scale must be finite and positive")

        start, stop = energy_split_indices(
            len(dataset.contexts),
            self.dataset_split,
            self.validation_fraction,
        )
        self.contexts = dataset.contexts[start:stop].reshape(stop - start, -1)
        self.slot_costs = dataset.slot_costs[start:stop]
        self.raw_component_costs = all_raw_costs[start:stop]
        self.component_costs = self.raw_component_costs / self.cost_scale

        self.n_arms = self.energy_solver.n_components
        self.context_dim = self.contexts.shape[1]
        self.weights = np.ones(self.n_arms, dtype=float)
        self.capacity = float(self.instance.n_tasks)
        self.n_days = len(self.contexts)
        self.t = 0
        self.current_context = None
        self.current_values = None
        self._current_costs = None
        self.current_slot_costs = None
        self._context_indices = {
            np.asarray(context, dtype=float).tobytes(): i
            for i, context in enumerate(self.contexts)
        }
        self.reset()

    @property
    def minimize(self):
        return True

    def _set_round(self):
        index = self.t % self.n_days
        self.current_context = self.contexts[index].copy()
        self.current_values = self.component_costs[index].copy()
        self._current_costs = self.current_values
        self.current_slot_costs = self.slot_costs[index].copy()

    def reset(self):
        self.t = 0
        self._set_round()
        return self.current_context

    def _advance(self):
        self.t += 1
        self._set_round()

    def get_context(self):
        return self.current_context

    def solve(self, costs):
        return self.energy_solver.solve(costs)

    def conditional_mean(self, context):
        key = np.asarray(context, dtype=float).reshape(-1).tobytes()
        if key not in self._context_indices:
            raise ValueError("context is not part of this fixed empirical stream")
        return self.component_costs[self._context_indices[key]].copy()

    def step(self, action, full_feedback=False):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (self.n_arms,):
            raise ValueError(
                f"expected action shape {(self.n_arms,)}, got {action.shape}"
            )

        cost = float(self.current_values @ action)
        optimal_action = self.solve(self.current_values)
        optimal_cost = float(self.current_values @ optimal_action)
        regret = cost - optimal_cost
        selected = action > 0.5
        observed = (
            self.current_values.copy()
            if full_feedback
            else np.where(
                selected,
                self.current_values,
                0.0,
            )
        )
        info = {
            "regret": regret,
            "cost": cost,
            "optimal_cost": optimal_cost,
            "raw_cost": cost * self.cost_scale,
            "raw_optimal_cost": optimal_cost * self.cost_scale,
            "action": action.copy(),
            "optimal_action": optimal_action,
            "true_values": self.current_values.copy(),
            "selected_indices": np.flatnonzero(selected),
        }
        self._advance()
        return observed, -cost, info

    def objective_error(self, day_index=0):
        day_index = int(day_index) % self.n_days
        raw_costs = self.raw_component_costs[day_index]
        action = self.solve(raw_costs)
        component_objective = float(raw_costs @ action)
        slot = float(
            self.slot_costs[day_index] @ self.energy_solver.load_profile(action)
        )
        return abs(component_objective - slot)

    def get_config(self):
        return {
            "data_root": self.data_root,
            "energy_instance": self.energy_instance,
            "dataset_split": self.dataset_split,
            "validation_fraction": self.validation_fraction,
            "n_days": self.n_days,
            "n_arms": self.n_arms,
            "context_dim": self.context_dim,
            "n_tasks": self.instance.n_tasks,
            "n_machines": self.instance.n_machines,
            "n_resources": self.instance.n_resources,
            "component_definition": "task_start",
            "n_internal_assignments": len(self.energy_solver.assignments),
            "cost_scale": self.cost_scale,
            "context_normalization": {
                "method": "z_score",
                "fit_split": "validation",
                "fit_start": self.normalization_start,
                "fit_stop": self.normalization_stop,
                "fit_days": self.normalization_stop - self.normalization_start,
                "feature_names": list(ENERGY_FEATURES),
                "mean": self.context_mean.tolist(),
                "std": self.context_std.tolist(),
            },
        }

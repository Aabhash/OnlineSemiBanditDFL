import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.data.energy_scheduling import EnergySchedulingInstance
from src.solvers._gurobi import _get_gurobi_env


class EnergySchedulingSolver:
    def __init__(self, instance: EnergySchedulingInstance):
        self.instance = instance
        self.components = tuple(
            (task, start)
            for task in range(instance.n_tasks)
            for start in range(
                int(instance.earliest[task]),
                int(instance.latest[task] - instance.durations[task] + 1),
            )
        )
        self.assignments = tuple(
            (task, machine, start)
            for task in range(instance.n_tasks)
            for machine in range(instance.n_machines)
            for start in range(
                int(instance.earliest[task]),
                int(instance.latest[task] - instance.durations[task] + 1),
            )
        )
        self.n_components = len(self.components)
        self.component_load_matrix = self._build_load_matrix()
        component_index = {component: i for i, component in enumerate(self.components)}
        self._assignment_components = np.asarray(
            [component_index[(task, start)] for task, _, start in self.assignments]
        )
        self._task_assignments = [
            [
                i
                for i, assignment in enumerate(self.assignments)
                if assignment[0] == task
            ]
            for task in range(instance.n_tasks)
        ]
        self.selected_assignments = ()
        self._build_model()

    def _build_load_matrix(self):
        matrix = np.zeros((self.n_components, self.instance.horizon), dtype=float)
        hours_per_slot = self.instance.q_minutes / 60.0
        for i, (task, start) in enumerate(self.components):
            stop = start + int(self.instance.durations[task])
            matrix[i, start:stop] = self.instance.powers[task] * hours_per_slot
        return matrix

    def _build_model(self):
        self._model = gp.Model("energy_scheduling", env=_get_gurobi_env())
        self._model.Params.OutputFlag = 0
        self._model.Params.Threads = 1
        self._x = self._model.addVars(
            len(self.assignments),
            vtype=GRB.BINARY,
            name="x",
        )
        self._vars = [self._x[i] for i in range(len(self.assignments))]
        self._model.ModelSense = GRB.MINIMIZE

        for task, indices in enumerate(self._task_assignments):
            self._model.addConstr(
                gp.quicksum(self._x[i] for i in indices) == 1,
                name=f"assign_{task}",
            )

        for resource in range(self.instance.n_resources):
            for machine in range(self.instance.n_machines):
                for slot in range(self.instance.horizon):
                    terms = [
                        self.instance.task_usage[task, resource] * self._x[i]
                        for i, (task, assigned_machine, start) in enumerate(
                            self.assignments
                        )
                        if assigned_machine == machine
                        and start <= slot < start + self.instance.durations[task]
                    ]
                    if terms:
                        self._model.addConstr(
                            gp.quicksum(terms)
                            <= self.instance.machine_capacities[machine, resource],
                            name=f"capacity_{resource}_{machine}_{slot}",
                        )
        self._model.update()

    def component_costs(self, slot_costs):
        slot_costs = np.asarray(slot_costs, dtype=float).reshape(-1)
        if slot_costs.shape != (self.instance.horizon,):
            raise ValueError(
                f"expected {self.instance.horizon} slot costs, got {slot_costs.shape}"
            )
        return self.component_load_matrix @ slot_costs

    def load_profile(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (self.n_components,):
            raise ValueError(
                f"expected action shape {(self.n_components,)}, got {action.shape}"
            )
        return action @ self.component_load_matrix

    def solve(self, costs):
        costs = np.asarray(costs, dtype=float).reshape(-1)
        if costs.shape != (self.n_components,):
            raise ValueError(
                f"expected cost shape {(self.n_components,)}, got {costs.shape}"
            )
        costs = np.nan_to_num(costs, nan=0.0, posinf=1e12, neginf=-1e12)
        assignment_costs = costs[self._assignment_components]
        self._model.setAttr(GRB.Attr.Obj, self._vars, assignment_costs.tolist())
        self._model.optimize()
        if self._model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"energy scheduling solve failed with status {self._model.Status}"
            )
        assignment_action = np.asarray([var.X for var in self._vars], dtype=float)
        selected = np.flatnonzero(assignment_action > 0.5)
        self.selected_assignments = tuple(self.assignments[i] for i in selected)
        action = np.zeros(self.n_components, dtype=float)
        action[self._assignment_components[selected]] = 1.0
        return action

    def __del__(self):
        model = getattr(self, "_model", None)
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

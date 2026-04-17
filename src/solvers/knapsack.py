import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.solvers._gurobi import _get_gurobi_env


def solve_knapsack_gurobi(values: np.ndarray, weights: np.ndarray, capacity: float) -> np.ndarray:
    n_items = len(values)

    values = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)


    values = values + np.arange(n_items) * 1e-10

    env = _get_gurobi_env()
    with gp.Model(env=env) as model:
        x = model.addVars(n_items, vtype=GRB.BINARY, name="x")
        model.setObjective(
            gp.quicksum(values[i] * x[i] for i in range(n_items)),
            GRB.MAXIMIZE
        )
        model.addConstr(
            gp.quicksum(weights[i] * x[i] for i in range(n_items)) <= capacity,
            name="capacity"
        )
        model.optimize()

        solution = np.zeros(n_items)
        if model.Status == GRB.OPTIMAL:
            for i in range(n_items):
                solution[i] = x[i].X
        return solution


def solve_knapsack(values: np.ndarray, weights: np.ndarray, capacity: float,
                   tie_break_noise: float = 1e-6) -> np.ndarray:
    values = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)

    if tie_break_noise > 0 and np.std(values) < 1e-4:


        values = values + np.arange(len(values)) * 1e-8

    return solve_knapsack_gurobi(values, weights, capacity)


def get_capacity(weights: np.ndarray, capacity_fraction: float) -> float:
    return capacity_fraction * np.sum(weights)


class KnapsackSolver:
    def __init__(self, weights, capacity):
        self.weights = np.array(weights, dtype=np.float64)
        self.capacity = capacity
        self.n_items = len(self.weights)
        self._env = _get_gurobi_env()
        self._model = gp.Model(env=self._env)
        self._model.Params.OutputFlag = 0
        self._model.Params.Threads = 1
        self._x = self._model.addVars(self.n_items, vtype=GRB.BINARY, name="x")
        self._model.setObjective(
            gp.quicksum(0.0 * self._x[i] for i in range(self.n_items)),
            GRB.MAXIMIZE,
        )
        self._model.addConstr(
            gp.quicksum(self.weights[i] * self._x[i] for i in range(self.n_items))
            <= self.capacity,
            name="capacity",
        )
        self._var_list = [self._x[i] for i in range(self.n_items)]
        self._model.update()

    def _sanitize_values(self, values):
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)
        return values

    def solve(self, values):
        values = self._sanitize_values(values)
        values = values + np.arange(self.n_items) * 1e-10
        self._model.setAttr(GRB.Attr.Obj, self._var_list, values.tolist())
        self._model.optimize()

        solution = np.zeros(self.n_items)
        if self._model.Status == GRB.OPTIMAL:
            for i in range(self.n_items):
                solution[i] = self._x[i].X
        return solution

    def extract_k_alternatives(self, values, K=5):
        values = self._sanitize_values(values)


        values = values + np.arange(len(values)) * 1e-10

        env = _get_gurobi_env()
        with gp.Model(env=env) as model:
            x = model.addVars(self.n_items, vtype=GRB.BINARY, name="x")
            model.setObjective(
                gp.quicksum(values[i] * x[i] for i in range(self.n_items)),
                GRB.MAXIMIZE,
            )
            model.addConstr(
                gp.quicksum(self.weights[i] * x[i] for i in range(self.n_items))
                <= self.capacity,
                name="capacity",
            )


            model.setParam('PoolSearchMode', 2)
            model.setParam('PoolSolutions', K + 1)
            model.optimize()

            if model.Status != GRB.OPTIMAL:
                opt_sol = np.zeros(self.n_items)
                return opt_sol, []


            model.setParam('SolutionNumber', 0)
            opt_sol = np.array([x[i].Xn for i in range(self.n_items)])
            opt_obj = model.PoolObjVal

            alternatives = []
            seen = set()
            seen.add(tuple(np.where(opt_sol > 0.5)[0]))

            for s in range(1, model.SolCount):
                model.setParam('SolutionNumber', s)
                alt_sol = np.array([x[i].Xn for i in range(self.n_items)])
                alt_obj = model.PoolObjVal

                key = tuple(np.where(alt_sol > 0.5)[0])
                if key in seen:
                    continue
                seen.add(key)

                obj_diff = opt_obj - alt_obj
                sym_diff = np.abs(opt_sol - alt_sol) > 0.5
                alternatives.append((alt_sol, obj_diff, sym_diff))

            alternatives.sort(key=lambda x: x[1])
            return opt_sol, alternatives[:K]

    def compute_slacks(self, values):
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            values = np.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)

        env = _get_gurobi_env()
        with gp.Model(env=env) as model:

            x = model.addVars(
                self.n_items, lb=0.0, ub=1.0,
                vtype=GRB.CONTINUOUS, name="x",
            )
            model.setObjective(
                gp.quicksum(values[i] * x[i] for i in range(self.n_items)),
                GRB.MAXIMIZE,
            )
            model.addConstr(
                gp.quicksum(self.weights[i] * x[i] for i in range(self.n_items))
                <= self.capacity,
                name="capacity",
            )
            model.optimize()

            if model.Status != GRB.OPTIMAL:
                return np.zeros(self.n_items)

            slacks = np.array([abs(x[i].RC) for i in range(self.n_items)])
            return slacks

    def __del__(self):
        model = getattr(self, '_model', None)
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.solvers._gurobi import _get_gurobi_env


def solve_portfolio_gurobi(
    returns: np.ndarray,
    covariance: np.ndarray,
    gamma: float,
    max_assets: int,
    time_limit: float | None = None,
) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)

    n_assets = returns.shape[0]
    if covariance.shape != (n_assets, n_assets):
        raise ValueError(
            f'covariance must have shape ({n_assets}, {n_assets}), got {covariance.shape}'
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError('covariance contains infinite values')

    if max_assets <= 0 or max_assets > n_assets:
        raise ValueError(
            f'max_assets must be in [1, {n_assets}], got {max_assets}'
        )

    # For binary subset selection, quadratic risk is dominated by the sum of
    # selected asset variances. Using the average diagonal variance scaled by
    # the subset size produces a feasible, portfolio-like risk budget while
    # keeping the problem nontrivial.
    risk_level = float(gamma) * float(np.mean(np.diag(covariance))) * float(max_assets)

    if not np.all(np.isfinite(returns)):
        raise ValueError('returns contains infinite values')
   

    env = _get_gurobi_env()
    with gp.Model('portfolio', env=env) as model:
        x = model.addMVar(n_assets, vtype=GRB.BINARY, name='x')
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        if time_limit is not None and time_limit > 0:
            model.Params.TimeLimit = float(time_limit)
        model.setObjective(returns @ x, GRB.MAXIMIZE)
        model.addConstr(x.sum() == float(max_assets), name='cardinality')
        model.addConstr(x.T @ covariance @ x <= risk_level, name='risk_limit')

        # Allow non-convex QP if covariance is not PSD.
        model.setParam('NonConvex', 2)
        model.optimize()

        if model.SolCount > 0:
            return np.asarray(x.X, dtype=np.float64)
        return np.zeros(n_assets, dtype=np.float64)

def solve_portfolio(
    returns: np.ndarray,
    covariance: np.ndarray,
    gamma: float,
    max_assets: int,
    time_limit: float | None = None,
    tie_break_noise: float = 1e-6,
) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    if tie_break_noise > 0 and np.std(returns) < 1e-4:
        returns = returns + np.arange(len(returns)) * float(tie_break_noise)
    return solve_portfolio_gurobi(
        returns,
        covariance,
        gamma,
        max_assets,
        time_limit=time_limit,
    )


class PortfolioSolver:
    def __init__(
        self,
        covariance: np.ndarray,
        gamma: float,
        max_assets: int,
        factor_loadings: np.ndarray | None = None,
        idiosyncratic_var: float | None = None,
        time_limit: float | None = None,
    ):
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.gamma = float(gamma)
        self.max_assets = int(max_assets)
        self.n_assets = self.covariance.shape[0]
        if self.covariance.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f'covariance must be square, got {self.covariance.shape}'
            )
        if self.max_assets <= 0 or self.max_assets > self.n_assets:
            raise ValueError(
                f'max_assets must be in [1, {self.n_assets}], got {self.max_assets}'
            )

        self.factor_loadings = None if factor_loadings is None else np.asarray(factor_loadings, dtype=np.float64)
        self.idiosyncratic_var = None if idiosyncratic_var is None else float(idiosyncratic_var)
        self.time_limit = None if time_limit is None else float(time_limit)
        self._use_low_rank = (
            self.factor_loadings is not None
            and self.idiosyncratic_var is not None
        )

        if self._use_low_rank:
            if self.factor_loadings.shape[0] != self.n_assets:
                raise ValueError(
                    f'factor_loadings must have {self.n_assets} rows, got {self.factor_loadings.shape}'
                )
            if self.idiosyncratic_var < 0.0:
                raise ValueError('idiosyncratic_var must be nonnegative')

        self.risk_level = (
            float(self.gamma)
            * float(np.mean(np.diag(self.covariance)))
            * float(self.max_assets)
        )

        self._env = _get_gurobi_env()
        self._model = gp.Model('portfolio', env=self._env)
        self._model.Params.OutputFlag = 0
        self._model.Params.Threads = 1
        if self.time_limit is not None and self.time_limit > 0:
            self._model.Params.TimeLimit = self.time_limit
        self._x = self._model.addVars(self.n_assets, vtype=GRB.BINARY, name='x')
        self._var_list = [self._x[i] for i in range(self.n_assets)]
        self._model.setObjective(
            gp.quicksum(0.0 * self._x[i] for i in range(self.n_assets)),
            GRB.MAXIMIZE,
        )
        self._model.addConstr(
            gp.quicksum(self._x[i] for i in range(self.n_assets)) == float(self.max_assets),
            name='cardinality',
        )

        if self._use_low_rank:
            factor_dim = int(self.factor_loadings.shape[1])
            self._z = self._model.addVars(
                factor_dim,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name='z',
            )
            for j in range(factor_dim):
                self._model.addConstr(
                    self._z[j] == gp.quicksum(
                        float(self.factor_loadings[i, j]) * self._x[i]
                        for i in range(self.n_assets)
                    ),
                    name=f'factor_{j}',
                )

            residual_budget = self.risk_level - self.idiosyncratic_var * float(self.max_assets)
            residual_budget = max(float(residual_budget), 0.0)
            self._model.addQConstr(
                gp.quicksum(self._z[j] * self._z[j] for j in range(factor_dim))
                <= residual_budget,
                name='risk_limit',
            )
        else:
            self._model.addQConstr(
                gp.quicksum(
                    float(self.covariance[i, j]) * self._x[i] * self._x[j]
                    for i in range(self.n_assets)
                    for j in range(self.n_assets)
                ) <= self.risk_level,
                name='risk_limit',
            )
        self._model.setParam('NonConvex', 2)
        self._model.update()
        self._last_solution = None

    def _sanitize_returns(self, returns: np.ndarray) -> np.ndarray:
        returns = np.asarray(returns, dtype=np.float64)
        if not np.all(np.isfinite(returns)):
            returns = np.nan_to_num(returns, nan=0.0, posinf=100.0, neginf=-100.0)
        return returns

    def solve(self, returns: np.ndarray, tie_break_noise: float = 1e-6) -> np.ndarray:
        returns = self._sanitize_returns(returns)
        if tie_break_noise > 0 and np.std(returns) < 1e-4:
            returns = returns + np.arange(self.n_assets) * float(tie_break_noise)
        self._model.setAttr(GRB.Attr.Obj, self._var_list, returns.tolist())
        if self._last_solution is not None:
            self._model.setAttr(GRB.Attr.Start, self._var_list, self._last_solution.tolist())
        self._model.optimize()

        solution = np.zeros(self.n_assets, dtype=np.float64)
        if self._model.SolCount > 0:
            for i in range(self.n_assets):
                solution[i] = self._x[i].X
            self._last_solution = solution.copy()
        return solution

    def __del__(self):
        model = getattr(self, '_model', None)
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

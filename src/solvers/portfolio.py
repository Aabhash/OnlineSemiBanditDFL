import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.solvers._gurobi import _get_gurobi_env


def solve_portfolio_gurobi(
    returns: np.ndarray,
    covariance: np.ndarray,
    gamma: float,
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

    risk_level = float(gamma) * float(np.mean(covariance))

    if not np.all(np.isfinite(returns)):
        raise ValueError('returns contains infinite values')
   

    env = _get_gurobi_env()
    with gp.Model('portfolio', env=env) as model:
        x = model.addMVar(n_assets, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
        model.setObjective(returns @ x, GRB.MAXIMIZE)
        model.addConstr(x.sum() == 1.0, name='budget')
        model.addConstr(x.T @ covariance @ x <= risk_level, name='risk_limit')

        # Allow non-convex QP if covariance is not PSD.
        model.setParam('NonConvex', 2)
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            return np.asarray(x.X, dtype=np.float64)
        return np.zeros(n_assets, dtype=np.float64)

def solve_portfolio(
    returns: np.ndarray,
    covariance: np.ndarray,
    gamma: float,
    tie_break_noise: float = 1e-6,
) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    if tie_break_noise > 0 and np.std(returns) < 1e-4:
        returns = returns + np.arange(len(returns)) * float(tie_break_noise)
    return solve_portfolio_gurobi(returns, covariance, gamma)


class PortfolioSolver:
    def __init__(self, covariance: np.ndarray, gamma: float):
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.gamma = float(gamma)
        self.n_assets = self.covariance.shape[0]
        if self.covariance.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f'covariance must be square, got {self.covariance.shape}'
            )

    def solve(self, returns: np.ndarray, tie_break_noise: float = 1e-6) -> np.ndarray:
        return solve_portfolio(
            returns,
            self.covariance,
            self.gamma,
            tie_break_noise=tie_break_noise,
        )

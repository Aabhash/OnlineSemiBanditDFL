import gurobipy as gp

_GUROBI_ENV = None


def _get_gurobi_env():
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        _GUROBI_ENV = gp.Env(empty=True)
        _GUROBI_ENV.setParam('OutputFlag', 0)
        _GUROBI_ENV.setParam('Threads', 1)
        _GUROBI_ENV.setParam('Seed', 0)
        _GUROBI_ENV.start()
    return _GUROBI_ENV

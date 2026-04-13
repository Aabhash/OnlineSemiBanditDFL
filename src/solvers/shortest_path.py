import numpy as np

def n_edges_grid(n: int) -> int:
    return 2 * n * (n - 1)


def right_edge_idx(i: int, j: int, n: int) -> int:
    return i * (n - 1) + j


def down_edge_idx(i: int, j: int, n: int) -> int:
    return n * (n - 1) + i * n + j


def edge_to_nodes(idx: int, n: int):
    n_right = n * (n - 1)
    if idx < n_right:
        i = idx // (n - 1)
        j = idx % (n - 1)
        return (i, j), (i, j + 1)
    else:
        k = idx - n_right
        i = k // n
        j = k % n
        return (i, j), (i + 1, j)


def _forward_dp(costs: np.ndarray, n: int) -> np.ndarray:
    d = np.full((n, n), np.inf)
    d[0, 0] = 0.0
    for j in range(1, n):
        d[0, j] = d[0, j - 1] + costs[right_edge_idx(0, j - 1, n)]
    for i in range(1, n):
        d[i, 0] = d[i - 1, 0] + costs[down_edge_idx(i - 1, 0, n)]
    for i in range(1, n):
        for j in range(1, n):
            d[i, j] = min(
                d[i - 1, j] + costs[down_edge_idx(i - 1, j, n)],
                d[i, j - 1] + costs[right_edge_idx(i, j - 1, n)],
            )
    return d


def _backward_dp(costs: np.ndarray, n: int) -> np.ndarray:
    d = np.full((n, n), np.inf)
    d[n - 1, n - 1] = 0.0
    for j in range(n - 2, -1, -1):
        d[n - 1, j] = d[n - 1, j + 1] + costs[right_edge_idx(n - 1, j, n)]
    for i in range(n - 2, -1, -1):
        d[i, n - 1] = d[i + 1, n - 1] + costs[down_edge_idx(i, n - 1, n)]
    for i in range(n - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            d[i, j] = min(
                d[i + 1, j] + costs[down_edge_idx(i, j, n)],
                d[i, j + 1] + costs[right_edge_idx(i, j, n)],
            )
    return d


def solve_shortest_path_edges(edge_costs: np.ndarray, grid_size: int) -> np.ndarray:
    n = grid_size
    costs = np.asarray(edge_costs, dtype=np.float64)
    d = _forward_dp(costs, n)

    path = np.zeros(n_edges_grid(n))
    i, j = n - 1, n - 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            path[right_edge_idx(i, j, n)] = 1.0
        elif j == 0:
            i -= 1
            path[down_edge_idx(i, j, n)] = 1.0
        else:
            cost_above = d[i - 1, j] + costs[down_edge_idx(i - 1, j, n)]
            cost_left  = d[i, j - 1] + costs[right_edge_idx(i, j - 1, n)]
            if cost_above <= cost_left:
                i -= 1
                path[down_edge_idx(i, j, n)] = 1.0
            else:
                j -= 1
                path[right_edge_idx(i, j, n)] = 1.0

    return path


def compute_edge_slacks(edge_costs: np.ndarray, grid_size: int) -> np.ndarray:
    n = grid_size
    costs = np.asarray(edge_costs, dtype=np.float64)
    ne = n_edges_grid(n)

    d_f = _forward_dp(costs, n)
    d_b = _backward_dp(costs, n)
    opt = d_f[n - 1, n - 1]

    slacks = np.empty(ne)


    for i in range(n):
        for j in range(n - 1):
            idx = right_edge_idx(i, j, n)
            slacks[idx] = d_f[i, j] + costs[idx] + d_b[i, j + 1] - opt


    for i in range(n - 1):
        for j in range(n):
            idx = down_edge_idx(i, j, n)
            slacks[idx] = d_f[i, j] + costs[idx] + d_b[i + 1, j] - opt

    return np.maximum(slacks, 0.0)


def extract_k_alternatives(pred_costs, grid_size, K=5):
    costs = np.asarray(pred_costs, dtype=np.float64)
    n = grid_size

    opt_path = solve_shortest_path_edges(costs, n)
    opt_cost = float(costs @ opt_path)
    opt_edges = np.where(opt_path > 0.5)[0]

    alternatives = []
    seen = set()

    for e in opt_edges:
        mod = costs.copy()
        mod[e] = 1e10
        alt = solve_shortest_path_edges(mod, n)
        key = tuple(np.where(alt > 0.5)[0])
        if key in seen:
            continue
        seen.add(key)

        cost_diff = float(costs @ alt) - opt_cost
        sym_diff = np.abs(opt_path - alt) > 0.5
        alternatives.append((alt, cost_diff, sym_diff))

    alternatives.sort(key=lambda x: x[1])
    return opt_path, alternatives[:K]


class ShortestPathSolver:

    def __init__(self, grid_size: int):
        n = grid_size
        self.n = n
        self.ne = 2 * n * (n - 1)


        self.right_idx = np.empty((n, n - 1), dtype=np.intp)
        for i in range(n):
            for j in range(n - 1):
                self.right_idx[i, j] = i * (n - 1) + j

        self.down_idx = np.empty((n - 1, n), dtype=np.intp)
        for i in range(n - 1):
            for j in range(n):
                self.down_idx[i, j] = n * (n - 1) + i * n + j

    def _forward_dp(self, costs: np.ndarray) -> np.ndarray:
        n = self.n
        ri, di = self.right_idx, self.down_idx
        d = np.full((n, n), np.inf)
        d[0, 0] = 0.0
        for j in range(1, n):
            d[0, j] = d[0, j - 1] + costs[ri[0, j - 1]]
        for i in range(1, n):
            d[i, 0] = d[i - 1, 0] + costs[di[i - 1, 0]]
        for i in range(1, n):
            for j in range(1, n):
                d[i, j] = min(
                    d[i - 1, j] + costs[di[i - 1, j]],
                    d[i, j - 1] + costs[ri[i, j - 1]],
                )
        return d

    def _backward_dp(self, costs: np.ndarray) -> np.ndarray:
        n = self.n
        ri, di = self.right_idx, self.down_idx
        d = np.full((n, n), np.inf)
        d[n - 1, n - 1] = 0.0
        for j in range(n - 2, -1, -1):
            d[n - 1, j] = d[n - 1, j + 1] + costs[ri[n - 1, j]]
        for i in range(n - 2, -1, -1):
            d[i, n - 1] = d[i + 1, n - 1] + costs[di[i, n - 1]]
        for i in range(n - 2, -1, -1):
            for j in range(n - 2, -1, -1):
                d[i, j] = min(
                    d[i + 1, j] + costs[di[i, j]],
                    d[i, j + 1] + costs[ri[i, j]],
                )
        return d

    def solve(self, edge_costs: np.ndarray) -> np.ndarray:
        n = self.n
        ri, di = self.right_idx, self.down_idx
        costs = np.asarray(edge_costs, dtype=np.float64)
        d = self._forward_dp(costs)

        path = np.zeros(self.ne)
        i, j = n - 1, n - 1
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
                path[ri[i, j]] = 1.0
            elif j == 0:
                i -= 1
                path[di[i, j]] = 1.0
            else:
                cost_above = d[i - 1, j] + costs[di[i - 1, j]]
                cost_left  = d[i, j - 1] + costs[ri[i, j - 1]]
                if cost_above <= cost_left:
                    i -= 1
                    path[di[i, j]] = 1.0
                else:
                    j -= 1
                    path[ri[i, j]] = 1.0
        return path

    def compute_slacks(self, edge_costs: np.ndarray) -> np.ndarray:
        n = self.n
        ri, di = self.right_idx, self.down_idx
        costs = np.asarray(edge_costs, dtype=np.float64)

        d_f = self._forward_dp(costs)
        d_b = self._backward_dp(costs)
        opt = d_f[n - 1, n - 1]

        slacks = np.empty(self.ne)
        for i in range(n):
            for j in range(n - 1):
                idx = ri[i, j]
                slacks[idx] = d_f[i, j] + costs[idx] + d_b[i, j + 1] - opt
        for i in range(n - 1):
            for j in range(n):
                idx = di[i, j]
                slacks[idx] = d_f[i, j] + costs[idx] + d_b[i + 1, j] - opt
        return np.maximum(slacks, 0.0)

    def extract_k_alternatives(self, pred_costs, K=5):
        costs = np.asarray(pred_costs, dtype=np.float64)

        opt_path = self.solve(costs)
        opt_cost = float(costs @ opt_path)
        opt_edges = np.where(opt_path > 0.5)[0]

        alternatives = []
        seen = set()

        for e in opt_edges:
            mod = costs.copy()
            mod[e] = 1e10
            alt = self.solve(mod)
            key = tuple(np.where(alt > 0.5)[0])
            if key in seen:
                continue
            seen.add(key)

            cost_diff = float(costs @ alt) - opt_cost
            sym_diff = np.abs(opt_path - alt) > 0.5
            alternatives.append((alt, cost_diff, sym_diff))

        alternatives.sort(key=lambda x: x[1])
        return opt_path, alternatives[:K]

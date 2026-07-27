from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments/configs/experiment.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/rebuttal/pretraining_structure_validation"
DEFAULT_DATASETS = [
    "knapsack_32",
    "knapsack_100",
    "shortest_path_8x8",
    "shortest_path_20x20",
    "portfolio_32",
]
DEFAULT_SEEDS = list(range(100, 110))
DEFAULT_DESIGN_SEED_BASE = 20_260_725
DEFAULT_N_Q0_ACTIONS = 5_000
DESIGN_RNG_NAMESPACE = "pretraining-structure-design-v1"
OBJECTIVE_CLASS_RELATIVE_TOLERANCE = 1e-6
DATASET_CONFIG_KEYS = {
    "env_type",
    "n_rounds",
    "eval_interval",
    "log_interval",
    "eval_metric",
    "skip_tuning",
    "tuning_year",
}


def _binary_action(action: np.ndarray) -> np.ndarray:
    return (np.asarray(action, dtype=float).reshape(-1) > 0.5).astype(float)


def design_rng_seed(
    dataset: str,
    environment_seed: int,
    design_seed_base: int = DEFAULT_DESIGN_SEED_BASE,
) -> int:
    key = (
        f"{DESIGN_RNG_NAMESPACE}:{int(design_seed_base)}:"
        f"{dataset}:{int(environment_seed)}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _objective_gap(
    mean: np.ndarray,
    optimal_action: np.ndarray,
    alternative_action: np.ndarray,
    minimize: bool,
) -> float:
    orientation = -1.0 if minimize else 1.0
    return float(
        orientation
        * np.asarray(mean, dtype=float)
        @ (_binary_action(optimal_action) - _binary_action(alternative_action))
    )


def _nonnegative_gap(
    gap: float,
    mean: np.ndarray,
    optimal_action: np.ndarray,
    alternative_action: np.ndarray,
) -> float:
    scale = max(
        1.0,
        abs(float(np.asarray(mean) @ _binary_action(optimal_action))),
        abs(float(np.asarray(mean) @ _binary_action(alternative_action))),
    )
    tolerance = 1e-8 * scale
    if gap < -tolerance:
        raise RuntimeError(
            "The distinct-action oracle returned an action better than the "
            f"reported optimum (gap={gap:.6g}, tolerance={tolerance:.6g})."
        )
    return max(float(gap), 0.0)


def _objective_class_tolerance(
    optimal_objective: float,
    relative_tolerance: float,
) -> float:
    scale = max(abs(float(optimal_objective)), np.finfo(float).tiny)
    return float(relative_tolerance) * scale


def _first_positive_gap_mip(
    mean: np.ndarray,
    oracle: Any,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray | None, float, bool]:
    from gurobipy import GRB, quicksum

    if oracle.kind not in {"knapsack", "portfolio"}:
        raise ValueError(f"MIP positive-gap routine cannot handle {oracle.kind}.")

    mean = np.asarray(mean, dtype=float).reshape(-1)
    model = oracle.solver._model
    variables = oracle.solver._var_list
    model.setAttr(GRB.Attr.Obj, variables, mean.tolist())
    model.ModelSense = GRB.MAXIMIZE
    previous_feasibility_tolerance = float(model.Params.FeasibilityTol)
    previous_integrality_tolerance = float(model.Params.IntFeasTol)
    model.Params.FeasibilityTol = min(previous_feasibility_tolerance, 1e-9)
    model.Params.IntFeasTol = min(previous_integrality_tolerance, 1e-9)
    model.optimize()
    if int(model.Status) != int(GRB.OPTIMAL):
        raise RuntimeError(
            f"{oracle.kind} optimum was not proven: status={model.Status}."
        )
    optimal = _binary_action([variable.X for variable in variables])
    optimal_objective = float(mean @ optimal)
    class_tolerance = _objective_class_tolerance(
        optimal_objective,
        relative_tolerance,
    )

    no_good = model.addConstr(
        quicksum(
            (1.0 - variables[i]) if optimal[i] > 0.5 else variables[i]
            for i in range(len(optimal))
        )
        >= 1.0,
        name="pretraining_distinct_optimum_check",
    )
    try:
        model.optimize()
        if model.SolCount <= 0:
            has_equivalent_optimal_action = False
        elif int(model.Status) != int(GRB.OPTIMAL):
            raise RuntimeError(
                f"{oracle.kind} distinct-optimum check was not proven: "
                f"status={model.Status}."
            )
        else:
            distinct = _binary_action([variable.X for variable in variables])
            distinct_gap = float(optimal_objective - mean @ distinct)
            has_equivalent_optimal_action = bool(distinct_gap <= class_tolerance)
    finally:
        model.remove(no_good)
        model.update()

    cutoff = model.addConstr(
        quicksum(float(mean[i]) * variables[i] for i in range(len(mean)))
        <= optimal_objective - class_tolerance,
        name="pretraining_first_positive_objective_class",
    )
    try:
        model.optimize()
        if model.SolCount <= 0:
            return (
                optimal,
                None,
                class_tolerance,
                has_equivalent_optimal_action,
            )
        if int(model.Status) != int(GRB.OPTIMAL):
            raise RuntimeError(
                f"{oracle.kind} positive-gap solve was not proven: "
                f"status={model.Status}."
            )
        alternative = _binary_action([variable.X for variable in variables])
    finally:
        model.remove(cutoff)
        model.update()
        model.Params.FeasibilityTol = previous_feasibility_tolerance
        model.Params.IntFeasTol = previous_integrality_tolerance

    gap = float(optimal_objective - mean @ alternative)
    if gap < 0.5 * class_tolerance:
        raise RuntimeError(
            "Objective-cutoff solve returned an action inside the numerical "
            f"Bayes-optimal class (gap={gap:.6g}, "
            f"class_tolerance={class_tolerance:.6g})."
        )
    return (
        optimal,
        alternative,
        class_tolerance,
        has_equivalent_optimal_action,
    )


def _first_positive_gap_shortest_path(
    mean: np.ndarray,
    grid_size: int,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray | None, float, bool]:
    from src.solvers.shortest_path import (
        down_edge_idx,
        n_edges_grid,
        right_edge_idx,
    )

    mean = np.asarray(mean, dtype=float).reshape(-1)
    grid_size = int(grid_size)
    n_edges = n_edges_grid(grid_size)
    if mean.shape != (n_edges,):
        raise ValueError(
            f"Expected {n_edges} edge costs for a {grid_size}x{grid_size} grid."
        )

    best_cost = np.full((grid_size, grid_size), np.inf, dtype=float)
    best_cost[0, 0] = 0.0
    for row in range(grid_size):
        for column in range(grid_size):
            if row == 0 and column == 0:
                continue
            candidates: list[float] = []
            if column > 0:
                candidates.append(
                    best_cost[row, column - 1]
                    + mean[right_edge_idx(row, column - 1, grid_size)]
                )
            if row > 0:
                candidates.append(
                    best_cost[row - 1, column]
                    + mean[down_edge_idx(row - 1, column, grid_size)]
                )
            best_cost[row, column] = min(candidates)
    optimal_objective = float(best_cost[-1, -1])
    class_tolerance = _objective_class_tolerance(
        optimal_objective,
        relative_tolerance,
    )

    optimal_class_counts = np.zeros(
        (grid_size, grid_size),
        dtype=np.int8,
    )
    optimal_class_counts[0, 0] = 1
    for row in range(grid_size):
        for column in range(grid_size):
            if row == 0 and column == 0:
                continue
            count = 0
            if column > 0:
                candidate_cost = (
                    best_cost[row, column - 1]
                    + mean[right_edge_idx(row, column - 1, grid_size)]
                )
                if candidate_cost <= best_cost[row, column] + class_tolerance:
                    count += int(optimal_class_counts[row, column - 1])
            if row > 0:
                candidate_cost = (
                    best_cost[row - 1, column]
                    + mean[down_edge_idx(row - 1, column, grid_size)]
                )
                if candidate_cost <= best_cost[row, column] + class_tolerance:
                    count += int(optimal_class_counts[row - 1, column])
            optimal_class_counts[row, column] = min(count, 2)

    states: list[list[list[tuple[float, np.ndarray]]]] = [
        [[] for _ in range(grid_size)] for _ in range(grid_size)
    ]
    states[0][0] = [(0.0, np.zeros(n_edges, dtype=float))]
    for row in range(grid_size):
        for column in range(grid_size):
            if row == 0 and column == 0:
                continue
            candidates: list[tuple[float, np.ndarray]] = []
            if column > 0:
                edge = right_edge_idx(row, column - 1, grid_size)
                for cost, action in states[row][column - 1]:
                    extended = action.copy()
                    extended[edge] = 1.0
                    candidates.append((float(cost + mean[edge]), extended))
            if row > 0:
                edge = down_edge_idx(row - 1, column, grid_size)
                for cost, action in states[row - 1][column]:
                    extended = action.copy()
                    extended[edge] = 1.0
                    candidates.append((float(cost + mean[edge]), extended))
            candidates.sort(key=lambda item: item[0])
            lowest = candidates[0][0]
            distinct_states = [candidates[0]]
            for candidate in candidates[1:]:
                if candidate[0] > lowest + class_tolerance:
                    distinct_states.append(candidate)
                    break
            states[row][column] = distinct_states

    terminal_states = states[-1][-1]
    optimal = terminal_states[0][1]
    alternative = terminal_states[1][1] if len(terminal_states) > 1 else None
    has_equivalent_optimal_action = bool(optimal_class_counts[-1, -1] > 1)
    return (
        optimal,
        alternative,
        class_tolerance,
        has_equivalent_optimal_action,
    )


def exact_decision_geometry(
    mean: np.ndarray,
    oracle: Any,
    minimize: bool,
    relative_tolerance: float = OBJECTIVE_CLASS_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    if oracle.kind == "portfolio":
        if minimize:
            raise ValueError("Portfolio diagnostics expect maximization.")
        (
            optimal,
            alternative,
            class_tolerance,
            has_equivalent,
        ) = _first_positive_gap_enumerated_portfolio(
            mean=mean,
            oracle=oracle,
            relative_tolerance=relative_tolerance,
        )
    elif oracle.kind == "shortest_path":
        if not minimize:
            raise ValueError("Shortest-path diagnostics expect minimization.")
        optimal, alternative, class_tolerance, has_equivalent = (
            _first_positive_gap_shortest_path(
                mean=mean,
                grid_size=oracle.solver.n,
                relative_tolerance=relative_tolerance,
            )
        )
    else:
        if minimize:
            raise ValueError(f"{oracle.kind} diagnostics expect maximization.")
        (
            optimal,
            alternative,
            class_tolerance,
            has_equivalent,
        ) = _first_positive_gap_mip(
            mean=mean,
            oracle=oracle,
            relative_tolerance=relative_tolerance,
        )

    optimal = _binary_action(optimal)
    if alternative is None:
        return {
            "optimal_action": optimal,
            "first_strictly_suboptimal_action": None,
            "first_strictly_positive_objective_gap": float("inf"),
            "near_alternative_hamming_size": 0,
            "second_best_per_change_margin": float("inf"),
            "objective_class_tolerance": class_tolerance,
            "has_equivalent_optimal_action": has_equivalent,
        }

    alternative = _binary_action(alternative)
    hamming = int(np.abs(alternative - optimal).sum())
    if hamming <= 0:
        raise RuntimeError("Positive-gap oracle returned the optimal action.")
    gap = _nonnegative_gap(
        _objective_gap(mean, optimal, alternative, minimize),
        mean,
        optimal,
        alternative,
    )
    if gap < 0.5 * class_tolerance:
        raise RuntimeError(
            "Positive-gap oracle returned an action inside the numerical "
            f"Bayes-optimal class (gap={gap:.6g}, "
            f"class_tolerance={class_tolerance:.6g})."
        )
    return {
        "optimal_action": optimal,
        "first_strictly_suboptimal_action": alternative,
        "first_strictly_positive_objective_gap": gap,
        "near_alternative_hamming_size": hamming,
        "second_best_per_change_margin": float(gap / hamming),
        "objective_class_tolerance": class_tolerance,
        "has_equivalent_optimal_action": has_equivalent,
    }


def shortest_path_incidence_row_basis(grid_size: int) -> np.ndarray:
    from src.solvers.shortest_path import edge_to_nodes, n_edges_grid

    grid_size = int(grid_size)
    n_nodes = grid_size * grid_size
    n_edges = n_edges_grid(grid_size)
    incidence = np.zeros((n_nodes, n_edges), dtype=float)
    for edge in range(n_edges):
        tail, head = edge_to_nodes(edge, grid_size)
        tail_index = int(tail[0]) * grid_size + int(tail[1])
        head_index = int(head[0]) * grid_size + int(head[1])
        incidence[tail_index, edge] = -1.0
        incidence[head_index, edge] = 1.0

    _, singular_values, right_vectors = np.linalg.svd(
        incidence,
        full_matrices=False,
    )
    tolerance = max(incidence.shape) * np.finfo(float).eps * float(singular_values[0])
    rank = int(np.sum(singular_values > tolerance))
    expected_rank = n_nodes - 1
    if rank != expected_rank:
        raise RuntimeError(
            "Unexpected grid incidence rank: "
            f"observed={rank}, expected={expected_rank}."
        )
    return right_vectors[:rank].T


def global_invariant_basis(
    kind: str,
    n_arms: int,
    grid_size: int | None = None,
) -> np.ndarray:
    kind = str(kind)
    n_arms = int(n_arms)
    if kind == "shortest_path":
        if grid_size is None:
            raise ValueError("grid_size is required for shortest_path.")
        basis = shortest_path_incidence_row_basis(int(grid_size))
        if basis.shape[0] != n_arms:
            raise ValueError(
                f"Incidence basis has {basis.shape[0]} rows, expected {n_arms}."
            )
        return basis
    if kind == "portfolio":
        return np.ones((n_arms, 1), dtype=float) / np.sqrt(float(n_arms))
    if kind == "knapsack":
        return np.empty((n_arms, 0), dtype=float)
    raise ValueError(f"Unsupported benchmark kind: {kind}")


def exact_global_invariant_basis(
    oracle: Any,
    n_arms: int,
    grid_size: int | None = None,
) -> np.ndarray:
    if oracle.kind == "portfolio":
        actions = oracle.feasible_actions.astype(float)
        differences = actions - actions[:1]
        gram = differences.T @ differences
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        scale = max(float(np.max(eigenvalues)), 1.0)
        tolerance = max(differences.shape) * np.finfo(float).eps * scale
        basis = eigenvectors[:, eigenvalues <= tolerance]
        if not basis.shape[1]:
            raise RuntimeError(
                "Fixed-cardinality Portfolio must have an invariant "
                "all-ones direction."
            )
        return basis
    return global_invariant_basis(
        kind=oracle.kind,
        n_arms=n_arms,
        grid_size=grid_size,
    )


def invariant_variance_fraction(
    conditional_means: np.ndarray,
    invariant_basis: np.ndarray,
) -> float:
    conditional_means = np.asarray(conditional_means, dtype=float)
    basis = np.asarray(invariant_basis, dtype=float)
    if conditional_means.ndim != 2:
        raise ValueError("conditional_means must be a two-dimensional matrix.")
    if basis.ndim != 2 or basis.shape[0] != conditional_means.shape[1]:
        raise ValueError(
            "invariant_basis must have shape "
            f"({conditional_means.shape[1]}, k), got {basis.shape}."
        )
    if basis.shape[1]:
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(basis.shape[1]), atol=1e-9):
            raise ValueError("invariant_basis columns must be orthonormal.")

    centered = conditional_means - conditional_means.mean(axis=0, keepdims=True)
    total_energy = float(np.square(centered).sum())
    invariant_energy = (
        float(np.square(centered @ basis).sum()) if basis.shape[1] else 0.0
    )
    fraction = (
        float(np.clip(invariant_energy / total_energy, 0.0, 1.0))
        if total_energy > np.finfo(float).eps
        else float("nan")
    )
    return fraction


def ranking_relevant_coefficient_scale(
    mean: np.ndarray,
    invariant_basis: np.ndarray,
) -> float:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    basis = np.asarray(invariant_basis, dtype=float)
    if basis.shape[0] != len(mean):
        raise ValueError("invariant_basis and mean must have matching rows.")
    ranking_part = mean - basis @ (basis.T @ mean) if basis.shape[1] else mean
    return float(np.linalg.norm(ranking_part) / np.sqrt(len(mean)))


def conditional_direction_noise_variance(
    env: Any,
    kind: str,
    conditional_mean: np.ndarray,
    action_difference: np.ndarray,
) -> float:
    mean = np.asarray(conditional_mean, dtype=float).reshape(-1)
    difference = np.asarray(action_difference, dtype=float).reshape(-1)
    if mean.shape != difference.shape:
        raise ValueError("conditional_mean and action_difference must align.")

    if kind in {"knapsack", "shortest_path"}:
        width = float(env.data.noise_width)
        component_variances = np.square(mean * width) / 3.0
        return float(np.square(difference) @ component_variances)
    if kind == "portfolio":
        covariance = np.asarray(env.data.covariance, dtype=float)
        return float(difference @ covariance @ difference)
    raise ValueError(f"Unsupported benchmark kind: {kind}")


def enumerate_portfolio_actions(env: Any) -> np.ndarray:
    n_arms = int(env.n_arms)
    combinations = list(itertools.combinations(range(n_arms), int(env.max_assets)))
    supports = np.zeros((len(combinations), n_arms), dtype=np.uint8)
    for row, support in enumerate(combinations):
        supports[row, list(support)] = 1
    covariance = np.asarray(env.data.covariance, dtype=float)
    risks = np.einsum(
        "bi,ij,bj->b",
        supports,
        covariance,
        supports,
        optimize=True,
    )
    feasible = supports[risks <= float(env._solver.risk_level) + 1e-12]
    if not len(feasible):
        raise RuntimeError("Portfolio enumeration found no feasible support.")
    return feasible


def _first_positive_gap_enumerated_portfolio(
    mean: np.ndarray,
    oracle: Any,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray | None, float, bool]:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    scores = oracle.feasible_actions @ mean
    optimal_index = int(np.argmax(scores))
    optimal = oracle.feasible_actions[optimal_index].astype(float)
    optimal_objective = float(scores[optimal_index])
    class_tolerance = _objective_class_tolerance(
        optimal_objective,
        relative_tolerance,
    )
    equivalent = scores >= optimal_objective - class_tolerance
    has_equivalent = bool(np.sum(equivalent) > 1)
    strictly_worse = scores < optimal_objective - class_tolerance
    if not np.any(strictly_worse):
        return optimal, None, class_tolerance, has_equivalent
    candidate_indices = np.flatnonzero(strictly_worse)
    alternative_index = int(candidate_indices[np.argmax(scores[candidate_indices])])
    alternative = oracle.feasible_actions[alternative_index].astype(float)
    return optimal, alternative, class_tolerance, has_equivalent


def sample_policy_independent_q0_actions(
    env: Any,
    kind: str,
    rng: np.random.Generator,
    n_actions: int,
) -> np.ndarray:
    n_actions = int(n_actions)
    n_arms = int(env.n_arms)
    actions = np.zeros((n_actions, n_arms), dtype=float)

    if kind == "knapsack":
        weights = np.asarray(env.weights, dtype=float)
        capacity = float(env.capacity)
        for draw in range(n_actions):
            remaining = capacity
            for arm in rng.permutation(n_arms):
                if weights[arm] <= remaining + 1e-12:
                    actions[draw, arm] = 1.0
                    remaining -= float(weights[arm])
        return actions

    if kind == "shortest_path":
        from src.solvers.shortest_path import down_edge_idx, right_edge_idx

        grid_size = int(env.grid_size)
        n_down = grid_size - 1
        n_right = grid_size - 1
        base_steps = np.concatenate(
            [
                np.zeros(n_right, dtype=np.int8),
                np.ones(n_down, dtype=np.int8),
            ]
        )
        for draw in range(n_actions):
            row = 0
            column = 0
            for step in rng.permutation(base_steps):
                if step == 0:
                    edge = right_edge_idx(row, column, grid_size)
                    column += 1
                else:
                    edge = down_edge_idx(row, column, grid_size)
                    row += 1
                actions[draw, edge] = 1.0
        return actions

    raise ValueError(f"Unsupported benchmark kind: {kind}")


def directional_feedback_exposure(
    action_difference: np.ndarray,
    component_exposure: np.ndarray,
) -> dict[str, float]:
    difference_sq = np.square(np.asarray(action_difference, dtype=float).reshape(-1))
    exposure = np.asarray(component_exposure, dtype=float).reshape(-1)
    if difference_sq.shape != exposure.shape:
        raise ValueError("action_difference and component_exposure must align.")
    mass = float(difference_sq.sum())
    if mass <= 0.0:
        return {
            "q0_directional_effective_exposure": float("nan"),
            "q0_directional_observation_coverage": float("nan"),
        }
    active = difference_sq > 0.0
    if np.any(exposure[active] <= 0.0):
        effective = 0.0
    else:
        effective = float(mass / np.sum(difference_sq[active] / exposure[active]))
    retention = float(difference_sq @ exposure / mass)
    return {
        "q0_directional_effective_exposure": effective,
        "q0_directional_observation_coverage": retention,
    }


def _finite_values(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _finite_mean(values: list[float]) -> float:
    finite = _finite_values(values)
    return float(np.mean(finite)) if len(finite) else float("nan")


def _finite_median(values: list[float]) -> float:
    finite = _finite_values(values)
    return float(np.median(finite)) if len(finite) else float("nan")


def analyze_seed(
    dataset: str,
    environment_seed: int,
    config: dict[str, Any],
    n_design_contexts: int,
    n_q0_actions: int = DEFAULT_N_Q0_ACTIONS,
    design_seed_base: int = DEFAULT_DESIGN_SEED_BASE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.envs import create_env

    if int(n_design_contexts) < 2:
        raise ValueError("n_design_contexts must be at least 2.")
    if int(n_q0_actions) < 1:
        raise ValueError("n_q0_actions must be at least 1.")

    environment_seed = int(environment_seed)
    dataset_config = dict(config["datasets"][dataset])
    for key in DATASET_CONFIG_KEYS:
        dataset_config.pop(key, None)
    env = create_env(dataset, seed=environment_seed, **dataset_config)
    if dataset.startswith("portfolio"):
        oracle = SimpleNamespace(
            kind="portfolio",
            feasible_actions=enumerate_portfolio_actions(env),
        )
    elif getattr(env, "grid_size", None) is not None:
        from src.solvers.shortest_path import ShortestPathSolver

        oracle = SimpleNamespace(
            kind="shortest_path",
            solver=ShortestPathSolver(env.grid_size),
        )
    else:
        from src.solvers.knapsack import KnapsackSolver

        oracle = SimpleNamespace(
            kind="knapsack",
            solver=KnapsackSolver(env.weights, env.capacity),
        )
    solver = getattr(oracle, "solver", None)
    if hasattr(solver, "_model"):
        solver._model.Params.MIPGap = 0.0
        solver._model.Params.MIPGapAbs = 0.0
        solver._model.Params.FeasibilityTol = 1e-9
        solver._model.Params.IntFeasTol = 1e-9
    minimize = bool(getattr(env, "minimize", False))
    n_arms = int(env.n_arms)
    context_dim = int(env.context_dim)
    derived_design_seed = design_rng_seed(
        dataset,
        environment_seed,
        design_seed_base,
    )
    stream_sequence = np.random.SeedSequence(derived_design_seed)
    context_sequence, q0_sequence = stream_sequence.spawn(2)
    context_rng = np.random.default_rng(context_sequence)
    q0_rng = np.random.default_rng(q0_sequence)
    contexts = context_rng.standard_normal(size=(int(n_design_contexts), context_dim))
    if oracle.kind == "portfolio":
        q0_indices = q0_rng.integers(
            0,
            len(oracle.feasible_actions),
            size=int(n_q0_actions),
        )
        q0_actions = oracle.feasible_actions[q0_indices].astype(float)
    else:
        q0_actions = sample_policy_independent_q0_actions(
            env=env,
            kind=oracle.kind,
            rng=q0_rng,
            n_actions=int(n_q0_actions),
        )
    q0_exposure = q0_actions.mean(axis=0)
    invariant_basis = exact_global_invariant_basis(
        oracle=oracle,
        n_arms=n_arms,
        grid_size=getattr(env, "grid_size", None),
    )
    means: list[np.ndarray] = []
    strict_action_differences: list[np.ndarray] = []
    context_rows: list[dict[str, Any]] = []

    for design_index, context in enumerate(contexts):
        mean = np.asarray(env.conditional_mean(context), dtype=float).reshape(-1)
        if mean.shape != (n_arms,):
            raise ValueError(
                f"{dataset} conditional_mean returned {mean.shape}, "
                f"expected ({n_arms},)."
            )
        geometry = exact_decision_geometry(
            mean=mean,
            oracle=oracle,
            minimize=minimize,
        )
        optimal = geometry["optimal_action"]
        strict_alternative = geometry["first_strictly_suboptimal_action"]
        ranking_scale = ranking_relevant_coefficient_scale(
            mean,
            invariant_basis,
        )
        directional_exposure = (
            {
                "q0_directional_effective_exposure": float("nan"),
                "q0_directional_observation_coverage": float("nan"),
            }
            if strict_alternative is None
            else directional_feedback_exposure(
                strict_alternative - optimal,
                q0_exposure,
            )
        )
        if strict_alternative is None:
            strict_difference = np.zeros(n_arms, dtype=float)
            normalized_per_change_margin = float("nan")
            noise_variance = float("nan")
            noise_snr = float("nan")
            new_arm_burden = 0
            removed_arm_burden = 0
            one_exchange = False
        else:
            strict_difference = _binary_action(strict_alternative) - optimal
            normalized_per_change_margin = float(
                geometry["second_best_per_change_margin"]
                / max(ranking_scale, np.finfo(float).tiny)
            )
            noise_variance = conditional_direction_noise_variance(
                env=env,
                kind=oracle.kind,
                conditional_mean=mean,
                action_difference=strict_difference,
            )
            noise_standard_deviation = float(np.sqrt(max(noise_variance, 0.0)))
            noise_snr = (
                float(
                    geometry["first_strictly_positive_objective_gap"]
                    / noise_standard_deviation
                )
                if noise_standard_deviation > np.finfo(float).tiny
                else float("inf")
            )
            new_arm_burden = int(np.sum((optimal < 0.5) & (strict_alternative > 0.5)))
            removed_arm_burden = int(
                np.sum((optimal > 0.5) & (strict_alternative < 0.5))
            )
            one_exchange = bool(new_arm_burden == 1 and removed_arm_burden == 1)
        means.append(mean)
        strict_action_differences.append(strict_difference)

        context_rows.append(
            {
                "dataset": dataset,
                "environment_seed": environment_seed,
                "derived_design_seed": derived_design_seed,
                "design_index": int(design_index),
                "minimize": minimize,
                "n_arms": n_arms,
                "context_dim": context_dim,
                "context_l2_norm": float(np.linalg.norm(context)),
                "conditional_mean_l2_norm": float(np.linalg.norm(mean)),
                "optimal_objective": float(mean @ optimal),
                "objective_class_relative_tolerance": (
                    OBJECTIVE_CLASS_RELATIVE_TOLERANCE
                ),
                "objective_class_absolute_tolerance": geometry[
                    "objective_class_tolerance"
                ],
                "has_equivalent_optimal_action": geometry[
                    "has_equivalent_optimal_action"
                ],
                "first_strictly_positive_objective_gap": geometry[
                    "first_strictly_positive_objective_gap"
                ],
                "near_alternative_hamming_size": geometry[
                    "near_alternative_hamming_size"
                ],
                "second_best_per_change_margin": geometry[
                    "second_best_per_change_margin"
                ],
                "ranking_relevant_coefficient_rms": ranking_scale,
                "normalized_per_change_margin": (normalized_per_change_margin),
                "strict_gap_direction_noise_variance": noise_variance,
                "strict_gap_noise_snr": noise_snr,
                "strict_alternative_new_arm_burden": new_arm_burden,
                "strict_alternative_removed_arm_burden": (removed_arm_burden),
                "strict_alternative_is_one_exchange": one_exchange,
                **directional_exposure,
                "optimal_selected_count": int(optimal.sum()),
                "optimal_selected_indices": ";".join(
                    str(index) for index in np.flatnonzero(optimal > 0.5)
                ),
                "first_strictly_suboptimal_selected_indices": (
                    ""
                    if strict_alternative is None
                    else ";".join(
                        str(index) for index in np.flatnonzero(strict_alternative > 0.5)
                    )
                ),
            }
        )

    means_array = np.asarray(means, dtype=float)
    strict_difference_array = np.asarray(
        strict_action_differences,
        dtype=float,
    )
    invariant_fraction = invariant_variance_fraction(
        means_array,
        invariant_basis,
    )
    component_contest_probability = np.mean(
        np.abs(strict_difference_array) > 0.5,
        axis=0,
    )
    component_predictable_variance = np.var(
        means_array,
        axis=0,
        ddof=1,
    )
    predictable_variance_total = float(component_predictable_variance.sum())
    variance_weighted_noncontest_fraction = (
        float(
            component_predictable_variance
            @ (1.0 - component_contest_probability)
            / predictable_variance_total
        )
        if predictable_variance_total > np.finfo(float).eps
        else float("nan")
    )

    nonzero_differences = strict_difference_array[
        np.linalg.norm(strict_difference_array, axis=1) > 0.0
    ]
    if len(nonzero_differences):
        normalized_differences = nonzero_differences / np.linalg.norm(
            nonzero_differences, axis=1, keepdims=True
        )
        _, singular_values, right_vectors = np.linalg.svd(
            normalized_differences,
            full_matrices=False,
        )
        span_tolerance = (
            max(normalized_differences.shape)
            * np.finfo(float).eps
            * float(singular_values[0])
        )
        local_basis = right_vectors[: int(np.sum(singular_values > span_tolerance))].T
        centered_means = means_array - means_array.mean(
            axis=0,
            keepdims=True,
        )
        local_energy = float(np.square(centered_means @ local_basis).sum())
        total_energy = float(np.square(centered_means).sum())
        local_predictable_fraction = (
            local_energy / total_energy
            if total_energy > np.finfo(float).eps
            else float("nan")
        )
    else:
        local_predictable_fraction = float("nan")

    summary = {
        "dataset": dataset,
        "environment_seed": environment_seed,
        "normalized_per_change_margin_median": _finite_median(
            [row["normalized_per_change_margin"] for row in context_rows]
        ),
        "strict_gap_noise_snr_median": _finite_median(
            [row["strict_gap_noise_snr"] for row in context_rows]
        ),
        "global_invariant_predictable_variance_fraction": invariant_fraction,
        "variance_weighted_noncontest_fraction": (
            variance_weighted_noncontest_fraction
        ),
        "component_contest_probability_mean": _finite_mean(
            component_contest_probability.tolist()
        ),
        "q0_directional_effective_exposure_mean": _finite_mean(
            [row["q0_directional_effective_exposure"] for row in context_rows]
        ),
        "q0_directional_observation_coverage_mean": _finite_mean(
            [row["q0_directional_observation_coverage"] for row in context_rows]
        ),
        "near_alternative_hamming_mean": _finite_mean(
            [row["near_alternative_hamming_size"] for row in context_rows]
        ),
        "strict_alternative_new_arm_burden_mean": _finite_mean(
            [row["strict_alternative_new_arm_burden"] for row in context_rows]
        ),
        "strict_alternative_one_exchange_rate": float(
            np.mean([row["strict_alternative_is_one_exchange"] for row in context_rows])
        ),
        "local_near_alternative_predictable_variance_fraction": (
            local_predictable_fraction
        ),
        "equivalent_optimal_action_rate": float(
            np.mean([row["has_equivalent_optimal_action"] for row in context_rows])
        ),
    }
    return summary, context_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_context_rows(
    output_dir: Path,
    summary: dict[str, Any],
    context_rows: list[dict[str, Any]],
) -> Path:
    stem = f"{summary['dataset']}_seed_{summary['environment_seed']}"
    per_seed = Path(output_dir) / "per_seed"
    csv_path = per_seed / f"{stem}.csv"
    _write_csv(csv_path, context_rows)
    return csv_path


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--n-design-contexts",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--n-q0-actions",
        type=int,
        default=DEFAULT_N_Q0_ACTIONS,
    )
    parser.add_argument(
        "--design-seed-base",
        type=int,
        default=DEFAULT_DESIGN_SEED_BASE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = _load_config(config_path)
    datasets = list(args.datasets)
    seeds = [int(seed) for seed in args.seeds]
    n_design_contexts = int(args.n_design_contexts)
    n_q0_actions = int(args.n_q0_actions)
    if args.smoke:
        datasets = ["knapsack_32", "shortest_path_8x8", "portfolio_32"]
        seeds = [100]
        n_design_contexts = 2
        n_q0_actions = 50

    summary_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for environment_seed in seeds:
            print(
                f"[run] {dataset} seed={environment_seed} "
                f"design_contexts={n_design_contexts}",
                flush=True,
            )
            summary, context_rows = analyze_seed(
                dataset=dataset,
                environment_seed=environment_seed,
                config=config,
                n_design_contexts=n_design_contexts,
                n_q0_actions=n_q0_actions,
                design_seed_base=int(args.design_seed_base),
            )
            csv_path = write_context_rows(output_dir, summary, context_rows)
            summary_rows.append(summary)
            print(f"[done] csv={csv_path}", flush=True)

    _write_csv(output_dir / "summary.csv", summary_rows)
    print(f"[complete] output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()

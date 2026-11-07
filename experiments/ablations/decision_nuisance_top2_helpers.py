"""Helpers for the Decision-Nuisance Top-2 feature-mechanism ablation.

This module keeps the paper figure generator independent of notebook/archive
code. The toy problem has two decision coordinates and three nuisance
coordinates: nuisance features explain cost variance, but do not alter the
oracle top-2 action.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.agents.contextual.bayesian_spo import BayesianSPOAgent
from src.agents.contextual.neural_agents import BayesianMSEAgent


ORACLE_W1 = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
)


class DecisionNuisanceTop2DGP:
    """Top-2 DGP with known decision and nuisance coordinates."""

    def __init__(
        self,
        seed: int = 42,
        noise_width: float = 0.1,
        nuisance_strength: float = 2.0,
    ):
        if nuisance_strength < 0:
            raise ValueError("nuisance_strength must be non-negative")

        self.n_items = 4
        self.n_features = 5
        self.context_dim = 5
        self.capacity = 2.0
        self.noise_width = noise_width
        self.nuisance_strength = float(nuisance_strength)
        self.rng = np.random.default_rng(seed)
        self.weights = np.ones(4)
        nuisance = self.nuisance_strength

        self.B = np.array(
            [
                [1.0, 0.0, nuisance, nuisance, nuisance],
                [-1.0, 0.0, nuisance, nuisance, nuisance],
                [0.0, 1.0, nuisance, nuisance, nuisance],
                [0.0, -1.0, nuisance, nuisance, nuisance],
            ]
        )

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        context = self.rng.standard_normal(self.n_features)
        return context, self._generate_values(context)

    def step(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def _generate_values(self, x: np.ndarray) -> np.ndarray:
        linear = self.B @ x / np.sqrt(self.n_features)
        values = linear + 3.0
        noise = self.rng.uniform(
            1 - self.noise_width,
            1 + self.noise_width,
            self.n_items,
        )
        return values * noise


class Top2Solver:
    """Select exactly the top-2 items by value."""

    def solve(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        action = np.zeros_like(values)
        action[np.argsort(values)[-2:]] = 1.0
        return action


class DecisionNuisanceTop2Env:
    """Decision-Nuisance Top-2 diagnostic environment."""

    def __init__(
        self,
        seed: int = 42,
        noise_width: float = 0.0,
        nuisance_strength: float = 2.0,
    ):
        self.data = DecisionNuisanceTop2DGP(
            seed=seed,
            noise_width=noise_width,
            nuisance_strength=nuisance_strength,
        )
        self.n_arms = 4
        self.context_dim = 5
        self.weights = np.ones(4, dtype=float)
        self.capacity = 2.0
        self.knapsack_solver = Top2Solver()
        self.current_context, self.current_values = self._generate_sample()

    @property
    def minimize(self) -> bool:
        return False

    def _generate_sample(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.sample()

    def reset(self) -> np.ndarray:
        self.data.reset()
        self.current_context, self.current_values = self._generate_sample()
        return self.current_context

    def _advance(self) -> None:
        self.data.step()
        self.current_context, self.current_values = self._generate_sample()

    def get_context(self) -> np.ndarray:
        return self.current_context

    def solve(self, values: np.ndarray) -> np.ndarray:
        return self.knapsack_solver.solve(values)


def decision_label(selection: np.ndarray) -> str:
    indices = np.where(np.asarray(selection) > 0.5)[0]
    return ",".join(str(i) for i in indices)


def feature_capacity_shares(W1: np.ndarray) -> np.ndarray:
    row_norms = np.linalg.norm(np.asarray(W1, dtype=float), axis=1)
    total = float(np.sum(row_norms))
    if total <= 1e-12:
        return np.zeros_like(row_norms)
    return row_norms / total


def create_agent_pair(
    env: DecisionNuisanceTop2Env,
    hidden_dim: int = 2,
    beta: float = 1.0,
    lambda_reg: float = 1.0,
    lr: float = 0.001,
    seed: int = 0,
    mask_spo: bool = False,
    mask_mse: bool = False,
    posterior_imputed_spo: bool = False,
    action_gap_imputation_samples: int = 1,
    action_gap_imputation_scale: float = 1.0,
    gradient_mode: str = 'current',
    **_unused,
) -> tuple[BayesianSPOAgent, BayesianMSEAgent]:
    """Create matched BayesianSPO and BayesianMSE agents for this diagnostic."""

    common = dict(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        minimize=env.minimize,
        nonlinear=False,
        seed=seed,
        solver=env.knapsack_solver.solve,
        gradient_mode=gradient_mode,
    )

    spo = BayesianSPOAgent(
        **common,
        mask_grad=mask_spo,
        imputation_samples=(
            action_gap_imputation_samples if posterior_imputed_spo else 1
        ),
        imputation_scale=action_gap_imputation_scale,
    )
    mse = BayesianMSEAgent(**common, mask_grad=mask_mse)
    return spo, mse


def train_with_checkpoints(
    agent,
    episode: dict,
    n_rounds: int,
    checkpoint_interval: int = 50,
    full_feedback: bool = False,
    eval_contexts: Optional[np.ndarray] = None,
    eval_values: Optional[np.ndarray] = None,
) -> dict[str, list]:
    """Train an agent on a pregenerated episode and keep light diagnostics."""

    rounds_data = episode["rounds_data"]
    w1_checkpoints = [(0, np.array(agent.W1), np.array(agent.b1))]
    regrets = []
    cumulative_regret = []
    actions = []
    prediction_error_checkpoints = []
    total_regret = 0.0

    def _prediction_mse() -> float:
        pred_matrix = np.vstack(
            [np.asarray(agent.predict_mean(x), dtype=float) for x in eval_contexts]
        )
        return float(np.mean((pred_matrix - eval_values) ** 2))

    if eval_contexts is not None and eval_values is not None:
        prediction_error_checkpoints.append((0, _prediction_mse()))

    for t in range(n_rounds):
        rd = rounds_data[t]
        x = rd["context"]
        true_values = rd["true_values"]
        optimal_value = rd["optimal_value"]

        selection = np.asarray(agent.act(x), dtype=float).reshape(-1)
        actions.append(selection.copy())
        instant_regret = float(optimal_value - np.sum(true_values * selection))
        total_regret += instant_regret
        regrets.append(instant_regret)
        cumulative_regret.append(total_regret)

        if full_feedback:
            observed = true_values.copy()
        else:
            observed = np.zeros_like(true_values)
            observed[selection > 0.5] = true_values[selection > 0.5]

        agent.update(
            x,
            selection,
            observed,
            full_feedback=full_feedback,
            true_values=true_values,
        )

        if (t + 1) % checkpoint_interval == 0:
            w1_checkpoints.append((t + 1, np.array(agent.W1), np.array(agent.b1)))
            if eval_contexts is not None and eval_values is not None:
                prediction_error_checkpoints.append((t + 1, _prediction_mse()))

    return {
        "W1_checkpoints": w1_checkpoints,
        "cumulative_regret": cumulative_regret,
        "regrets": regrets,
        "actions": actions,
        "prediction_error_checkpoints": prediction_error_checkpoints,
    }


def extract_features(agent, x_test: np.ndarray) -> np.ndarray:
    return np.stack([np.array(agent._features(x)) for x in x_test])


def generate_test_contexts(
    env: DecisionNuisanceTop2Env,
    n_samples: int = 500,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_test = rng.standard_normal((n_samples, env.context_dim))
    decisions = []
    values = []
    for x in x_test:
        value = env.data._generate_values(x)
        decisions.append(env.solve(value))
        values.append(value)
    return x_test, np.array(decisions), np.array(values)


def _convex_hull_df(h1, h2, group_label):
    from scipy.spatial import ConvexHull

    points = np.column_stack([h1, h2])
    if len(points) < 3:
        return []
    try:
        hull = ConvexHull(points)
        vertices = hull.vertices.tolist()
        vertices.append(vertices[0])
        return [
            {
                "h1": float(points[v, 0]),
                "h2": float(points[v, 1]),
                "decision": group_label,
            }
            for v in vertices
        ]
    except Exception:
        return []


def _decision_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, features, labels, cv=5, scoring="accuracy")
    return float(np.mean(scores))

import numpy as np
from typing import Tuple


def sherman_morrison_update(A_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
    Ax = A_inv @ x
    denom = 1.0 + x @ Ax
    return A_inv - np.outer(Ax, Ax) / denom


def per_arm_bayesian_update(
    A: np.ndarray,
    b: np.ndarray,
    A_inv: np.ndarray,
    features: np.ndarray,
    selection: np.ndarray,
    rewards: np.ndarray,
    full_feedback: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_arms = len(selection)
    for i in range(n_arms):
        if full_feedback or selection[i] > 0.5:
            A[i] += np.outer(features, features)
            b[i] += features * rewards[i]
            A_inv[i] = sherman_morrison_update(A_inv[i], features)

    return A, b, A_inv


def init_per_arm_regression(
    n_arms: int,
    feature_dim: int,
    lambda_reg: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.array([np.eye(feature_dim) * lambda_reg for _ in range(n_arms)])
    b = np.zeros((n_arms, feature_dim))
    A_inv = np.array([np.eye(feature_dim) / lambda_reg for _ in range(n_arms)])
    return A, b, A_inv


def compute_ucb_values(
    features: np.ndarray,
    A_inv: np.ndarray,
    b: np.ndarray,
    beta: float,
) -> np.ndarray:
    n_arms = len(A_inv)
    ucb_values = np.zeros(n_arms)
    for i in range(n_arms):
        theta_i = A_inv[i] @ b[i]
        mean = theta_i @ features
        std = np.sqrt(features @ A_inv[i] @ features)
        ucb_values[i] = mean + beta * std
    return ucb_values

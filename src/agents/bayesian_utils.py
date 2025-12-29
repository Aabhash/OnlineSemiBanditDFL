import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def mlp_forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h = jnp.maximum(0, z1)
    out = h @ W2 + b2
    return out, h, z1


@jax.jit
def mlp_backward(grad_out, h, z1, X, W2):
    d_W2 = jnp.outer(h, grad_out)
    d_b2 = grad_out
    d_h = W2 @ grad_out
    d_h = d_h * (z1 > 0)
    d_W1 = jnp.outer(X, d_h)
    d_b1 = d_h
    return {'W1': d_W1, 'b1': d_b1, 'W2': d_W2, 'b2': d_b2}


@jax.jit
def sherman_morrison_update(A_inv: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    Ax = A_inv @ x
    denom = 1.0 + x @ Ax
    return A_inv - jnp.outer(Ax, Ax) / denom


@jax.jit
def sherman_morrison_update_batched(A_inv: jnp.ndarray, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    Ax = jnp.einsum('nij,j->ni', A_inv, x)
    denom = 1.0 + jnp.einsum('i,ni->n', x, Ax)
    update = jnp.einsum('ni,nj->nij', Ax, Ax) / denom[:, None, None]
    mask_3d = mask[:, None, None]
    return jnp.where(mask_3d, A_inv - update, A_inv)


def init_per_arm_regression(
    n_arms: int,
    feature_dim: int,
    lambda_reg: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    eye = jnp.eye(feature_dim)
    A = jnp.broadcast_to(eye * lambda_reg, (n_arms, feature_dim, feature_dim)).copy()
    b = jnp.zeros((n_arms, feature_dim))
    A_inv = jnp.broadcast_to(eye / lambda_reg, (n_arms, feature_dim, feature_dim)).copy()
    return A, b, A_inv


def compute_ucb_values(
    features: jnp.ndarray,
    A_inv: jnp.ndarray,
    b: jnp.ndarray,
    beta: float,
    minimize: bool = False,
) -> jnp.ndarray:
    theta = jnp.einsum('nij,nj->ni', A_inv, b)
    means = theta @ features
    Ax = jnp.einsum('nij,j->ni', A_inv, features)
    variances = jnp.einsum('i,ni->n', features, Ax)
    stds = jnp.sqrt(jnp.maximum(variances, 0.0))
    if minimize:
        return means - beta * stds
    return means + beta * stds


@jax.jit
def compute_bayesian_predictions(
    features: jnp.ndarray,
    A_inv: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    theta = jnp.einsum('nij,nj->ni', A_inv, b)
    return theta @ features


@jax.jit
def compute_uncertainties(
    features: jnp.ndarray,
    A_inv: jnp.ndarray,
) -> jnp.ndarray:
    Ax = jnp.einsum('nij,j->ni', A_inv, features)
    variances = jnp.einsum('i,ni->n', features, Ax)
    return jnp.sqrt(jnp.maximum(variances, 0.0))


@jax.jit
def update_bayesian_stats(
    A: jnp.ndarray,
    b: jnp.ndarray,
    A_inv: jnp.ndarray,
    features: jnp.ndarray,
    rewards: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    outer = jnp.outer(features, features)
    mask_3d = mask[:, None, None]
    A_new = jnp.where(mask_3d, A + outer, A)
    mask_2d = mask[:, None]
    b_update = features * rewards[:, None]
    b_new = jnp.where(mask_2d, b + b_update, b)
    A_inv_new = sherman_morrison_update_batched(A_inv, features, mask)
    return A_new, b_new, A_inv_new


def init_shared_regression(
    feature_dim: int,
    lambda_reg: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    eye = jnp.eye(feature_dim)
    A = eye * lambda_reg
    b = jnp.zeros(feature_dim)
    A_inv = eye / lambda_reg
    return A, b, A_inv


def compute_shared_predictions(
    H: jnp.ndarray,
    A_inv: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    theta = A_inv @ b
    return H @ theta


def compute_shared_ucb_values(
    H: jnp.ndarray,
    A_inv: jnp.ndarray,
    b: jnp.ndarray,
    beta: float,
    minimize: bool = False,
) -> jnp.ndarray:
    theta = A_inv @ b
    means = H @ theta
    Ah = H @ A_inv
    variances = jnp.sum(H * Ah, axis=1)
    stds = jnp.sqrt(jnp.maximum(variances, 0.0))
    if minimize:
        return means - beta * stds
    return means + beta * stds


def compute_shared_uncertainties(
    H: jnp.ndarray,
    A_inv: jnp.ndarray,
) -> jnp.ndarray:
    Ah = H @ A_inv
    variances = jnp.sum(H * Ah, axis=1)
    return jnp.sqrt(jnp.maximum(variances, 0.0))


def update_shared_bayesian_stats(
    A: jnp.ndarray,
    b: jnp.ndarray,
    A_inv: jnp.ndarray,
    H: jnp.ndarray,
    rewards: jnp.ndarray,
    mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    H_obs = H[mask]
    r_obs = rewards[mask]
    A_new = A + H_obs.T @ H_obs
    b_new = b + H_obs.T @ r_obs
    A_inv_new = A_inv
    for i in range(H_obs.shape[0]):
        h = H_obs[i]
        A_inv_new = sherman_morrison_update(A_inv_new, h)
    return A_new, b_new, A_inv_new



import jax
import jax.numpy as jnp
from typing import Tuple


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


@jax.jit
def compute_ucb_values(
    features: jnp.ndarray,
    A_inv: jnp.ndarray,
    b: jnp.ndarray,
    beta: float,
) -> jnp.ndarray:
    theta = jnp.einsum('nij,nj->ni', A_inv, b)
    means = theta @ features
    Ax = jnp.einsum('nij,j->ni', A_inv, features)
    variances = jnp.einsum('i,ni->n', features, Ax)
    stds = jnp.sqrt(jnp.maximum(variances, 0.0))
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

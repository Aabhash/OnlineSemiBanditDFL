import numpy as np
from typing import Tuple, Dict


def init_mlp_weights(
    context_dim: int,
    hidden_dim: int,
    n_arms: int,
    seed: int = None,
    scale: float = 0.1,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        'W1': rng.standard_normal((context_dim, hidden_dim)) * scale,
        'b1': np.zeros(hidden_dim),
        'W2': rng.standard_normal((hidden_dim, n_arms)) * scale,
        'b2': np.zeros(n_arms),
    }


def mlp_forward(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)
    out = h @ W2 + b2
    return out, h


def mlp_backward(
    grad_output: np.ndarray,
    X: np.ndarray,
    h: np.ndarray,
    W2: np.ndarray,
    clip_value: float = 5.0,
) -> Dict[str, np.ndarray]:
    d_W2 = np.outer(h, grad_output)
    d_b2 = grad_output

    d_h = W2 @ grad_output
    d_h = d_h * (h > 0)

    d_W1 = np.outer(X, d_h)
    d_b1 = d_h

    d_W1 = np.clip(d_W1, -clip_value, clip_value)
    d_W2 = np.clip(d_W2, -clip_value, clip_value)
    d_b1 = np.clip(d_b1, -clip_value, clip_value)
    d_b2 = np.clip(d_b2, -clip_value, clip_value)

    return {
        'd_W1': d_W1,
        'd_b1': d_b1,
        'd_W2': d_W2,
        'd_b2': d_b2,
    }


def update_mlp_weights(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    grads: Dict[str, np.ndarray],
    lr: float,
    weight_clip: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1 = W1 - lr * grads['d_W1']
    b1 = b1 - lr * grads['d_b1']
    W2 = W2 - lr * grads['d_W2']
    b2 = b2 - lr * grads['d_b2']

    W1 = np.clip(W1, -weight_clip, weight_clip)
    W2 = np.clip(W2, -weight_clip, weight_clip)

    return W1, b1, W2, b2

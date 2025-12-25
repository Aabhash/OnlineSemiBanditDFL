import jax
import jax.numpy as jnp


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

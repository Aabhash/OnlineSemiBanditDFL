from functools import partial

import jax
import jax.numpy as jnp


RESIDUAL_ACTIVATIONS = ("relu", "silu", "square")
RESIDUAL_WEIGHT_INITIALIZATIONS = ("random", "identity")


def validate_residual_activation(activation):
    normalized = str(activation).lower()
    if normalized not in RESIDUAL_ACTIVATIONS:
        expected = ", ".join(RESIDUAL_ACTIVATIONS)
        raise ValueError(
            f"Unknown residual_activation={activation!r}; "
            f"expected one of {expected}."
        )
    return normalized


def validate_residual_weight_init(initialization):
    normalized = str(initialization).lower()
    if normalized not in RESIDUAL_WEIGHT_INITIALIZATIONS:
        expected = ", ".join(RESIDUAL_WEIGHT_INITIALIZATIONS)
        raise ValueError(
            f"Unknown residual_weight_init={initialization!r}; "
            f"expected one of {expected}."
        )
    return normalized


def _apply_residual_activation(features, activation):
    if activation == "relu":
        return jax.nn.relu(features)
    if activation == "silu":
        return jax.nn.silu(features)
    return jnp.square(features)


def init_gated_residual_params(
    context_dim,
    feature_dim,
    depth,
    seed,
    weight_init="random",
):
    if depth < 0:
        raise ValueError("depth must be non-negative")
    weight_init = validate_residual_weight_init(weight_init)

    input_key = jax.random.PRNGKey(seed)
    blocks = []
    for layer in range(depth):
        block_key = jax.random.PRNGKey(seed + 104_729 + layer)
        if weight_init == "identity":
            weight = jnp.eye(feature_dim)
        else:
            weight = (
                jax.random.normal(
                    block_key,
                    (feature_dim, feature_dim),
                )
                * 0.1
            )
        blocks.append(
            {
                "weight": weight,
                "bias": jnp.zeros(feature_dim),
                "gate": jnp.asarray(0.0),
            }
        )

    return {
        "W1": (
            jax.random.normal(
                input_key,
                (context_dim, feature_dim),
            )
            * 0.1
        ),
        "b1": jnp.zeros(feature_dim),
        "blocks": tuple(blocks),
    }


@partial(jax.jit, static_argnames=("activation",))
def gated_residual_forward(
    params,
    X,
    gate_scale,
    activation="relu",
):
    features = X @ params["W1"] + params["b1"]
    for block in params["blocks"]:
        residual = (
            _apply_residual_activation(features, activation) @ block["weight"]
            + block["bias"]
        )
        features = features + gate_scale * jnp.tanh(block["gate"]) * residual
    return features


@partial(jax.jit, static_argnames=("activation",))
def gated_residual_backward(
    params,
    X,
    grad_out,
    gate_scale,
    activation="relu",
):
    def feature_dot(current):
        features = gated_residual_forward(
            current,
            X,
            gate_scale,
            activation,
        )
        return jnp.vdot(features, grad_out)

    return jax.grad(feature_dot)(params)


def clip_gated_residual_params(params):
    blocks = tuple(
        {
            "weight": jnp.clip(block["weight"], -10, 10),
            "bias": block["bias"],
            "gate": jnp.clip(block["gate"], -3.0, 3.0),
        }
        for block in params["blocks"]
    )
    return {
        "W1": jnp.clip(params["W1"], -10, 10),
        "b1": params["b1"],
        "blocks": blocks,
    }

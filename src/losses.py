import jax
import jax.numpy as jnp
import numpy as np
from src.solver import solve_knapsack

_GUMBEL = 'gumbel'
_NORMAL = 'normal'


def mse_loss(y_pred, y_true) -> float:
    y_pred = jnp.asarray(y_pred)
    y_true = jnp.asarray(y_true)
    return float(((y_pred - y_true) ** 2).mean())


def mse_grad(y_pred, y_true) -> jnp.ndarray:
    y_pred = jnp.asarray(y_pred)
    y_true = jnp.asarray(y_true)
    return 2 * (y_pred - y_true) / len(y_true)


def spo_plus_grad(y_pred, y_true, weights, capacity: float) -> jnp.ndarray:
    y_pred_np = np.asarray(y_pred)
    y_true_np = np.asarray(y_true)
    weights_np = np.asarray(weights)

    x_opt = solve_knapsack(y_true_np, weights_np, capacity)
    x_spo = solve_knapsack(2 * y_pred_np - y_true_np, weights_np, capacity)
    return jnp.asarray(2 * (x_spo - x_opt))


def spo_plus_grad_generic(y_pred, y_true, solver, minimize=False):
    cp = np.asarray(y_pred, dtype=np.float64)
    ct = np.asarray(y_true, dtype=np.float64)
    n = cp.shape[0]

    w_true = solver(ct)
    w_spo = solver(2 * cp - ct)

    if minimize:
        return jnp.asarray(2.0 * (w_true - w_spo) / n)
    else:
        return jnp.asarray(2.0 * (w_spo - w_true) / n)


def make_spo_grad_fn(solver, minimize=False):
    def loss_grad_fn(y_pred, y_true):
        return spo_plus_grad_generic(y_pred, y_true, solver, minimize)
    return loss_grad_fn


def pfy_grad_generic(y_pred, y_true, solver, minimize=False,
    n_samples: int = 1,
    sigma: float = 1.,
    noise: str = _GUMBEL,
    rng=None,
):
    cp = np.asarray(y_pred, dtype=np.float64)
    ct = np.asarray(y_true, dtype=np.float64)
    n = cp.shape[0]

    w_true = np.asarray(solver(ct), dtype=np.float64)
    local_rng = rng if rng is not None else np.random.default_rng()

    noise_lower = noise.lower()
    if noise_lower == _GUMBEL:
        noises_vector = local_rng.gumbel(0.0, 1.0, size=(n_samples, cp.shape[0]))
    elif noise_lower == _NORMAL:
        noises_vector = local_rng.normal(0.0, 1.0, size=(n_samples, cp.shape[0]))
    else:
        raise ValueError(
            f"Invalid noise '{noise}'. Must be '{_GUMBEL}' or '{_NORMAL}'."
        )

    perturbed_costs = cp[None, :] + sigma * noises_vector
    ptb_solutions = np.stack(
        [np.asarray(solver(c), dtype=np.float64) for c in perturbed_costs],
        axis=0,
    )
    avg_solution = ptb_solutions.mean(axis=0)

    if minimize:
        grad = (w_true - avg_solution) / n
    else:
        grad = (avg_solution - w_true) / n
    return jnp.asarray(grad)


def make_pfy_grad_fn(
    solver,
    minimize=False,
    n_samples: int = 1,
    sigma: float = 1.,
    noise: str = _GUMBEL,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    def loss_grad_fn(y_pred, y_true):
        return pfy_grad_generic(
            y_pred,
            y_true,
            solver=solver,
            minimize=minimize,
            n_samples=n_samples,
            sigma=sigma,
            noise=noise,
            rng=rng,
        )

    return loss_grad_fn



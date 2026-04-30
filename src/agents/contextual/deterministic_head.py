import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.agents.agent_base import AgentBase
from src.agents.bayesian_utils import linear_forward, mlp_forward
from src.losses import make_spo_grad_fn
from src.solver import solve_knapsack


class DeterministicHeadAgent(AgentBase):
    """Shared feature extractor with a jointly trained deterministic linear head."""

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        loss_type: str = "mse",
        lr: float = 0.001,
        epsilon: float = 0.1,
        decay_epsilon: bool = True,
        decay_rounds: int = 10000,
        max_grad_norm: float = 5.0,
        mask_grad: bool = True,
        missing_grad_weight: float = 0.0,
        nonlinear: bool = False,
        minimize: bool = False,
        solver=None,
        seed: int = None,
    ):
        loss_type = loss_type.lower()
        if loss_type not in {"mse", "spo"}:
            raise ValueError(f"Unknown loss_type={loss_type!r}; expected 'mse' or 'spo'.")

        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.loss_type = loss_type
        self.epsilon = float(epsilon)
        self.decay_epsilon = bool(decay_epsilon)
        self.decay_rounds = int(decay_rounds)
        self.mask_grad = bool(mask_grad)
        self.missing_grad_weight = float(missing_grad_weight)
        self.nonlinear = bool(nonlinear)
        self.minimize = bool(minimize)
        self.t = 0

        if solver is None:
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver

        seed_value = seed if seed is not None else 0
        self.rng = np.random.default_rng(seed_value + 104729)
        key = jax.random.PRNGKey(seed_value)
        if self.nonlinear:
            k1, k2, k3 = jax.random.split(key, 3)
            self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            self.W2 = jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1
            self.b2 = jnp.zeros(hidden_dim)
            self.theta = jax.random.normal(k3, (n_arms, hidden_dim)) * 0.1
            init_params = {
                "W1": self.W1,
                "b1": self.b1,
                "W2": self.W2,
                "b2": self.b2,
                "theta": self.theta,
            }
        else:
            k1, k2 = jax.random.split(key)
            self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            self.theta = jax.random.normal(k2, (n_arms, hidden_dim)) * 0.1
            init_params = {"W1": self.W1, "b1": self.b1, "theta": self.theta}

        schedule = lambda count: lr / jnp.sqrt(count + 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule),
        )
        self.opt_state = self.optimizer.init(init_params)
        self.spo_grad_fn = make_spo_grad_fn(self.solver, self.minimize)

    def _get_params(self):
        if self.nonlinear:
            return {
                "W1": self.W1,
                "b1": self.b1,
                "W2": self.W2,
                "b2": self.b2,
                "theta": self.theta,
            }
        return {"W1": self.W1, "b1": self.b1, "theta": self.theta}

    def _set_params(self, params):
        self.W1 = jnp.clip(params["W1"], -10, 10)
        self.b1 = params["b1"]
        self.theta = jnp.clip(params["theta"], -10, 10)
        if self.nonlinear:
            self.W2 = jnp.clip(params["W2"], -10, 10)
            self.b2 = params["b2"]

    def _features(self, X):
        X = jnp.asarray(X)
        if self.nonlinear:
            h, _, _ = mlp_forward(X, self.W1, self.b1, self.W2, self.b2)
            return h
        return linear_forward(X, self.W1, self.b1)

    def _forward(self, X):
        h = self._features(X)
        return self.theta @ h, h

    def _current_epsilon(self):
        if not self.decay_epsilon:
            return self.epsilon
        return self.epsilon * max(0.0, 1.0 - self.t / max(self.decay_rounds, 1))

    def predict_mean(self, X):
        pred, _ = self._forward(X)
        return pred

    def predict_uncertainty(self, X):
        return jnp.zeros(self.n_arms)

    def predict_mean_and_uncertainty(self, X):
        return self.predict_mean(X), self.predict_uncertainty(X)

    def act(self, X):
        self.t += 1
        if self.rng.random() < self._current_epsilon():
            return self.solver(self.rng.standard_normal(self.n_arms))
        pred = self.predict_mean(X)
        return self.solver(np.asarray(pred))

    def _grad_pred(self, pred, target, selection_mask, full_feedback):
        if self.loss_type == "mse":
            grad_pred = 2.0 * (pred - target) / self.n_arms
        else:
            grad_pred = self.spo_grad_fn(pred, target)

        if self.mask_grad and not full_feedback:
            missing_weight = jnp.asarray(self.missing_grad_weight, dtype=grad_pred.dtype)
            grad_weight = jnp.where(selection_mask, 1.0, missing_weight)
            grad_pred = grad_pred * grad_weight
        return grad_pred

    def _backprop_and_step(self, X, h, grad_h, grad_theta):
        X = jnp.asarray(X)
        if self.nonlinear:
            z = X @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            grad_a = grad_h @ self.W2.T
            grad_z = grad_a * (z > 0)
            grads = {
                "W1": jnp.outer(X, grad_z),
                "b1": grad_z,
                "W2": jnp.outer(a, grad_h),
                "b2": grad_h,
                "theta": grad_theta,
            }
        else:
            grads = {
                "W1": jnp.outer(X, grad_h),
                "b1": grad_h,
                "theta": grad_theta,
            }

        params = self._get_params()
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, params)
        self._set_params(optax.apply_updates(params, updates))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False, **kwargs):
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)
        selection_mask = selection > 0.5

        pred, h = self._forward(X)
        if full_feedback:
            target = rewards
        else:
            target = jnp.where(selection_mask, rewards, pred)

        grad_pred = self._grad_pred(pred, target, selection_mask, full_feedback)
        grad_theta = jnp.outer(grad_pred, h)
        grad_h = grad_pred @ self.theta
        self._backprop_and_step(X, h, grad_h, grad_theta)


class DeterministicSPOAgent(DeterministicHeadAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, loss_type="spo", **kwargs)


class DeterministicMSEAgent(DeterministicHeadAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, loss_type="mse", **kwargs)

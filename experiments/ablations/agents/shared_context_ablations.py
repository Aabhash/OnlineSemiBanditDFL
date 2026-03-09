import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.losses import spo_plus_grad
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    compute_uncertainties,
    update_bayesian_stats,
)


class DecoupledSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        beta_start: float = None,
        beta_end: float = None,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        decision_lr: float = 0.01,
        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        key = jax.random.PRNGKey(seed if seed else 0)
        key1, key2 = jax.random.split(key)

        # Feature extractor (trained with MSE)
        self.W1 = jax.random.normal(key1, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)

        # Decision layer: transforms predictions to decision values (trained with SPO+)
        # Initialized near identity: decision_value = pred * W_dec + b_dec
        self.W_dec = jnp.ones(n_arms)  # Scale per arm
        self.b_dec = jnp.zeros(n_arms)  # Bias per arm

        # Separate optimizers for feature extractor and decision layer
        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.feature_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=schedule)
            )
            decision_schedule = optax.cosine_decay_schedule(decision_lr, decay_rounds, alpha=min_lr_ratio)
            self.decision_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=decision_schedule)
            )
        else:
            self.feature_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr)
            )
            self.decision_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=decision_lr)
            )

        self.feature_opt_state = self.feature_optimizer.init({'W1': self.W1, 'b1': self.b1})
        self.decision_opt_state = self.decision_optimizer.init({'W_dec': self.W_dec, 'b_dec': self.b_dec})

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        z1 = X @ self.W1 + self.b1
        return jnp.maximum(0, z1), z1

    def predict_mean(self, X):
        h, _ = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def _get_decision_values(self, pred, ucb_bonus):
        """Transform predictions to decision values using decision layer."""
        # Decision values = scaled predictions + uncertainty bonus
        return pred * self.W_dec + self.b_dec + self.beta * ucb_bonus

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h, _ = self._features(X)
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

        # Get uncertainty bonus for exploration
        ucb_bonus = compute_uncertainties(h, self.A_inv)

        # Use decision layer to transform predictions
        decision_values = self._get_decision_values(pred, ucb_bonus)
        return self.solver(np.asarray(decision_values))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h, z1 = self._features(X)

        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, selection_mask
        )

        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

        # Get uncertainty for exploration bonus
        ucb_bonus = compute_uncertainties(h, self.A_inv)

        if full_feedback:
            Y_target = rewards
        else:
            # Impute with UCB values for unobserved arms
            ucb_values = pred + self.beta * ucb_bonus
            Y_target = jnp.where(selection > 0.5, rewards, ucb_values)

        # 1. Train feature extractor with MSE (robust to imputation)
        mse_grad = 2 * (pred - Y_target) / self.n_arms
        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        grad_h_mse = jnp.einsum('n,ni->i', mse_grad, theta)
        grad_z1 = grad_h_mse * (z1 > 0)

        feature_grads = {'W1': jnp.outer(X, grad_z1), 'b1': grad_z1}
        updates, self.feature_opt_state = self.feature_optimizer.update(
            feature_grads, self.feature_opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']

        # 2. Train decision layer with SPO+
        # Decision values used for SPO+ optimization
        decision_values = pred * self.W_dec + self.b_dec  # Without UCB bonus for training

        # SPO+ gradient w.r.t. decision values
        spo_grad = spo_plus_grad(decision_values, Y_target, self.weights, self.capacity)

        # Gradient w.r.t. W_dec: d(loss)/d(W_dec) = spo_grad * pred
        grad_W_dec = spo_grad * pred
        # Gradient w.r.t. b_dec: d(loss)/d(b_dec) = spo_grad
        grad_b_dec = spo_grad

        decision_grads = {'W_dec': grad_W_dec, 'b_dec': grad_b_dec}
        dec_updates, self.decision_opt_state = self.decision_optimizer.update(
            decision_grads, self.decision_opt_state, {'W_dec': self.W_dec, 'b_dec': self.b_dec}
        )
        dec_params = optax.apply_updates({'W_dec': self.W_dec, 'b_dec': self.b_dec}, dec_updates)
        self.W_dec = jnp.clip(dec_params['W_dec'], 0.1, 10.0)  # Keep scales positive
        self.b_dec = jnp.clip(dec_params['b_dec'], -5.0, 5.0)


class DABSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        beta_start: float = None,
        beta_end: float = None,
        alpha: float = 1.0,  # Decision relevance weight
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.alpha = alpha  # Decision relevance amplification

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        key = jax.random.PRNGKey(seed if seed else 0)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)

        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=schedule)
            )
        else:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr)
            )
        self.opt_state = self.optimizer.init({'W1': self.W1, 'b1': self.b1})

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        z1 = X @ self.W1 + self.b1
        return jnp.maximum(0, z1), z1

    def predict_mean(self, X):
        h, _ = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def _compute_decision_relevance(self, values):
        """Compute decision relevance from LP relaxation.

        Returns relevance_i = min(lp_i, 1 - lp_i) for each arm.
        Arms with lp ≈ 0.5 have highest relevance (decision boundary).
        Arms with lp ≈ 0 or lp ≈ 1 have low relevance (clearly in/out).
        """
        from src.solver import solve_knapsack_with_sensitivity
        info = solve_knapsack_with_sensitivity(
            np.asarray(values), self.weights_np, self.capacity
        )
        return info["decision_relevance"]

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h, _ = self._features(X)
        mu = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        sigma = compute_uncertainties(h, self.A_inv)

        # Decision-weighted exploration:
        # Amplify exploration for arms at decision boundaries
        relevance = self._compute_decision_relevance(mu)
        relevance = jnp.asarray(relevance)

        # UCB_i = μ_i + β * σ_i * (1 + α * relevance_i)
        ucb_values = mu + self.beta * sigma * (1 + self.alpha * relevance)

        return self.solver(np.asarray(ucb_values))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h, z1 = self._features(X)

        # Update Bayesian stats for observed arms
        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, selection_mask
        )

        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        sigma = compute_uncertainties(h, self.A_inv)

        if full_feedback:
            Y_target = rewards
        else:
            # UCB imputation for unobserved arms (optimistic)
            ucb_values = pred + self.beta * sigma
            Y_target = jnp.where(selection > 0.5, rewards, ucb_values)

        # Decoupled MSE training for features (robust to imputation errors)
        mse_grad = 2 * (pred - Y_target) / self.n_arms

        # Backprop through Bayesian layer to features
        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        grad_h = jnp.einsum('n,ni->i', mse_grad, theta)
        grad_z1 = grad_h * (z1 > 0)

        grads = {'W1': jnp.outer(X, grad_z1), 'b1': grad_z1}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']


class ExperienceReplaySPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        beta_start: float = None,
        beta_end: float = None,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        buffer_size: int = 200,
        batch_size: int = 16,
        min_buffer: int = 50,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.min_buffer = min_buffer

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        self.seed = seed if seed else 0
        key = jax.random.PRNGKey(self.seed)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)
        self.rng_key = jax.random.PRNGKey(self.seed + 1)

        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=schedule)
            )
        else:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr)
            )
        self.opt_state = self.optimizer.init({'W1': self.W1, 'b1': self.b1})

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

        # Experience replay buffer
        self.buffer_X = []
        self.buffer_selection = []
        self.buffer_rewards = []

    def _features(self, X):
        X = jnp.asarray(X)
        z1 = X @ self.W1 + self.b1
        return jnp.maximum(0, z1), z1

    def predict_mean(self, X):
        h, _ = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h, _ = self._features(X)
        ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
        return self.solver(np.asarray(ucb_values))

    def _add_to_buffer(self, X, selection, rewards):
        """Add experience to replay buffer."""
        self.buffer_X.append(np.asarray(X))
        self.buffer_selection.append(np.asarray(selection))
        self.buffer_rewards.append(np.asarray(rewards))

        # Maintain buffer size
        if len(self.buffer_X) > self.buffer_size:
            self.buffer_X.pop(0)
            self.buffer_selection.pop(0)
            self.buffer_rewards.pop(0)

    def _sample_batch(self):
        """Sample a mini-batch from the replay buffer."""
        buffer_len = len(self.buffer_X)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        indices = jax.random.choice(
            subkey, buffer_len, shape=(min(self.batch_size, buffer_len),), replace=False
        )
        indices = np.asarray(indices)

        batch_X = [self.buffer_X[i] for i in indices]
        batch_selection = [self.buffer_selection[i] for i in indices]
        batch_rewards = [self.buffer_rewards[i] for i in indices]

        return batch_X, batch_selection, batch_rewards

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h, z1 = self._features(X)

        # Update Bayesian stats with current observation
        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, selection_mask
        )

        # Add to replay buffer
        self._add_to_buffer(X, selection, rewards)

        # If buffer too small, use MSE on current observation only
        if len(self.buffer_X) < self.min_buffer:
            pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

            if full_feedback:
                Y_target = rewards
            else:
                ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
                Y_target = jnp.where(selection > 0.5, rewards, ucb_values)

            # Pure MSE gradient for early training
            mse_grad = 2 * (pred - Y_target) / self.n_arms
            theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
            grad_h = jnp.einsum('n,ni->i', mse_grad, theta)
            grad_z1 = grad_h * (z1 > 0)

            grads = {'W1': jnp.outer(X, grad_z1), 'b1': grad_z1}

        else:
            # Sample batch from buffer and compute SPO+ gradient
            batch_X, batch_selection, batch_rewards = self._sample_batch()

            # Accumulate gradients over batch
            grad_W1_acc = jnp.zeros_like(self.W1)
            grad_b1_acc = jnp.zeros_like(self.b1)

            for bX, bsel, brew in zip(batch_X, batch_selection, batch_rewards):
                bX = jnp.asarray(bX)
                bsel = jnp.asarray(bsel)
                brew = jnp.asarray(brew)

                bh, bz1 = self._features(bX)
                bpred = compute_bayesian_predictions(bh, self.A_inv, self.b_bayes)

                if full_feedback:
                    bY_target = brew
                else:
                    bucb_values = compute_ucb_values(bh, self.A_inv, self.b_bayes, self.beta)
                    bY_target = jnp.where(bsel > 0.5, brew, bucb_values)

                # SPO+ gradient on batch sample
                spo_grad = spo_plus_grad(bpred, bY_target, self.weights, self.capacity)
                theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
                grad_h = jnp.einsum('n,ni->i', spo_grad, theta)
                grad_z1 = grad_h * (bz1 > 0)

                grad_W1_acc += jnp.outer(bX, grad_z1)
                grad_b1_acc += grad_z1

            # Average gradients
            batch_size_actual = len(batch_X)
            grads = {
                'W1': grad_W1_acc / batch_size_actual,
                'b1': grad_b1_acc / batch_size_actual
            }

        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']


class ConservativeSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        beta_start: float = None,
        beta_end: float = None,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        confidence_threshold: float = 0.5,
        blend_by_confidence: bool = True,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.confidence_threshold = confidence_threshold
        self.blend_by_confidence = blend_by_confidence

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        key = jax.random.PRNGKey(seed if seed else 0)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)

        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=schedule)
            )
        else:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=lr)
            )
        self.opt_state = self.optimizer.init({'W1': self.W1, 'b1': self.b1})

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        z1 = X @ self.W1 + self.b1
        return jnp.maximum(0, z1), z1

    def predict_mean(self, X):
        h, _ = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1

        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h, _ = self._features(X)
        ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
        return self.solver(np.asarray(ucb_values))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h, z1 = self._features(X)

        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, selection_mask
        )

        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

        # Get per-arm uncertainty (standard deviation)
        uncertainties = compute_uncertainties(h, self.A_inv)

        if full_feedback:
            Y_target = rewards
        else:
            ucb_values = compute_ucb_values(h, self.A_inv, self.b_bayes, self.beta)
            Y_target = jnp.where(selection > 0.5, rewards, ucb_values)

        # Compute MSE gradient
        mse_grad = 2 * (pred - Y_target) / self.n_arms

        # Compute SPO+ gradient
        spo_grad = spo_plus_grad(pred, Y_target, self.weights, self.capacity)

        if self.blend_by_confidence:
            # Smooth blending: confidence = 1 / (1 + uncertainty/threshold)
            # High uncertainty -> low confidence -> more MSE
            # Low uncertainty -> high confidence -> more SPO+
            confidence = 1.0 / (1.0 + uncertainties / self.confidence_threshold)
            # Per-arm blend: SPO+ weighted by confidence, MSE weighted by (1-confidence)
            hybrid_grad = confidence * spo_grad + (1 - confidence) * mse_grad
        else:
            # Hard switch: use SPO+ only for confident arms
            confident_mask = uncertainties < self.confidence_threshold
            hybrid_grad = jnp.where(confident_mask, spo_grad, mse_grad)

        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        grad_h = jnp.einsum('n,ni->i', hybrid_grad, theta)

        grad_z1 = grad_h * (z1 > 0)

        grads = {'W1': jnp.outer(X, grad_z1), 'b1': grad_z1}
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']


class DecoupledConservativeSPOAgent:

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        beta_start: float = None,
        beta_end: float = None,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        decision_lr: float = 0.01,
        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        confidence_threshold: float = 0.5,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.confidence_threshold = confidence_threshold

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        key = jax.random.PRNGKey(seed if seed else 0)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)
        self.W_dec = jnp.ones(n_arms)
        self.b_dec = jnp.zeros(n_arms)

        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.feature_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=schedule)
            )
            dec_schedule = optax.cosine_decay_schedule(decision_lr, decay_rounds, alpha=min_lr_ratio)
            self.decision_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=dec_schedule)
            )
        else:
            self.feature_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=lr)
            )
            self.decision_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=decision_lr)
            )

        self.feature_opt_state = self.feature_optimizer.init({'W1': self.W1, 'b1': self.b1})
        self.decision_opt_state = self.decision_optimizer.init({'W_dec': self.W_dec, 'b_dec': self.b_dec})
        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        z1 = X @ self.W1 + self.b1
        return jnp.maximum(0, z1), z1

    def predict_mean(self, X):
        h, _ = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1
        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h, _ = self._features(X)
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        ucb_bonus = compute_uncertainties(h, self.A_inv)
        decision_values = pred * self.W_dec + self.b_dec + self.beta * ucb_bonus
        return self.solver(np.asarray(decision_values))

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h, z1 = self._features(X)
        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, selection_mask
        )

        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        ucb_bonus = compute_uncertainties(h, self.A_inv)

        if full_feedback:
            Y_target = rewards
        else:
            Y_target = jnp.where(selection > 0.5, rewards, pred + self.beta * ucb_bonus)

        # Features with MSE
        mse_grad = 2 * (pred - Y_target) / self.n_arms
        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        grad_h = jnp.einsum('n,ni->i', mse_grad, theta)
        grad_z1 = grad_h * (z1 > 0)

        feature_grads = {'W1': jnp.outer(X, grad_z1), 'b1': grad_z1}
        updates, self.feature_opt_state = self.feature_optimizer.update(
            feature_grads, self.feature_opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']

        # Decision layer with confidence-weighted SPO+
        dec_values = pred * self.W_dec + self.b_dec
        confidence = 1.0 / (1.0 + ucb_bonus / self.confidence_threshold)
        spo_grad = spo_plus_grad(dec_values, Y_target, self.weights, self.capacity)
        weighted_grad = confidence * spo_grad

        decision_grads = {'W_dec': weighted_grad * pred, 'b_dec': weighted_grad}
        dec_updates, self.decision_opt_state = self.decision_optimizer.update(
            decision_grads, self.decision_opt_state, {'W_dec': self.W_dec, 'b_dec': self.b_dec}
        )
        dec_params = optax.apply_updates({'W_dec': self.W_dec, 'b_dec': self.b_dec}, dec_updates)
        self.W_dec = jnp.clip(dec_params['W_dec'], 0.1, 10.0)
        self.b_dec = jnp.clip(dec_params['b_dec'], -5.0, 5.0)


class DecoupledReplaySPOAgent:
    """Decoupled + Replay: MSE features + batch-trained SPO+ decision layer (no confidence weighting)."""

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        beta_start: float = None,
        beta_end: float = None,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        decision_lr: float = 0.01,
        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        buffer_size: int = 200,
        batch_size: int = 16,
        min_buffer: int = 50,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.min_buffer = min_buffer

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        self.seed = seed if seed else 0
        key = jax.random.PRNGKey(self.seed)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)
        self.W_dec = jnp.ones(n_arms)
        self.b_dec = jnp.zeros(n_arms)
        self.rng_key = jax.random.PRNGKey(self.seed + 1)

        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.feature_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=schedule)
            )
            dec_schedule = optax.cosine_decay_schedule(decision_lr, decay_rounds, alpha=min_lr_ratio)
            self.decision_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=dec_schedule)
            )
        else:
            self.feature_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=lr)
            )
            self.decision_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=decision_lr)
            )

        self.feature_opt_state = self.feature_optimizer.init({'W1': self.W1, 'b1': self.b1})
        self.decision_opt_state = self.decision_optimizer.init({'W_dec': self.W_dec, 'b_dec': self.b_dec})
        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)
        self.buffer_X = []
        self.buffer_selection = []
        self.buffer_rewards = []

    def _features(self, X):
        X = jnp.asarray(X)
        z1 = X @ self.W1 + self.b1
        return jnp.maximum(0, z1), z1

    def predict_mean(self, X):
        h, _ = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1
        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h, _ = self._features(X)
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        ucb_bonus = compute_uncertainties(h, self.A_inv)
        decision_values = pred * self.W_dec + self.b_dec + self.beta * ucb_bonus
        return self.solver(np.asarray(decision_values))

    def _add_to_buffer(self, X, selection, rewards):
        self.buffer_X.append(np.asarray(X))
        self.buffer_selection.append(np.asarray(selection))
        self.buffer_rewards.append(np.asarray(rewards))
        if len(self.buffer_X) > self.buffer_size:
            self.buffer_X.pop(0)
            self.buffer_selection.pop(0)
            self.buffer_rewards.pop(0)

    def _sample_batch(self):
        buffer_len = len(self.buffer_X)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        indices = np.asarray(jax.random.choice(
            subkey, buffer_len, shape=(min(self.batch_size, buffer_len),), replace=False
        ))
        return ([self.buffer_X[i] for i in indices],
                [self.buffer_selection[i] for i in indices],
                [self.buffer_rewards[i] for i in indices])

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h, z1 = self._features(X)
        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, selection_mask
        )
        self._add_to_buffer(X, selection, rewards)

        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        ucb_bonus = compute_uncertainties(h, self.A_inv)
        if full_feedback:
            Y_target = rewards
        else:
            Y_target = jnp.where(selection > 0.5, rewards, pred + self.beta * ucb_bonus)

        # Features with MSE
        mse_grad = 2 * (pred - Y_target) / self.n_arms
        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        grad_h = jnp.einsum('n,ni->i', mse_grad, theta)
        grad_z1 = grad_h * (z1 > 0)

        feature_grads = {'W1': jnp.outer(X, grad_z1), 'b1': grad_z1}
        updates, self.feature_opt_state = self.feature_optimizer.update(
            feature_grads, self.feature_opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']

        # Decision layer: SPO+ on batch (no confidence weighting)
        if len(self.buffer_X) >= self.min_buffer:
            batch_X, batch_selection, batch_rewards = self._sample_batch()
            grad_W_acc = jnp.zeros_like(self.W_dec)
            grad_b_acc = jnp.zeros_like(self.b_dec)

            for bX, bsel, brew in zip(batch_X, batch_selection, batch_rewards):
                bh, _ = self._features(jnp.asarray(bX))
                bpred = compute_bayesian_predictions(bh, self.A_inv, self.b_bayes)
                bucb = compute_uncertainties(bh, self.A_inv)
                if full_feedback:
                    bY = jnp.asarray(brew)
                else:
                    bY = jnp.where(jnp.asarray(bsel) > 0.5, jnp.asarray(brew), bpred + self.beta * bucb)

                bdec = bpred * self.W_dec + self.b_dec
                spo_grad = spo_plus_grad(bdec, bY, self.weights, self.capacity)
                grad_W_acc += spo_grad * bpred
                grad_b_acc += spo_grad

            n_batch = len(batch_X)
            decision_grads = {'W_dec': grad_W_acc / n_batch, 'b_dec': grad_b_acc / n_batch}
            dec_updates, self.decision_opt_state = self.decision_optimizer.update(
                decision_grads, self.decision_opt_state, {'W_dec': self.W_dec, 'b_dec': self.b_dec}
            )
            dec_params = optax.apply_updates({'W_dec': self.W_dec, 'b_dec': self.b_dec}, dec_updates)
            self.W_dec = jnp.clip(dec_params['W_dec'], 0.1, 10.0)
            self.b_dec = jnp.clip(dec_params['b_dec'], -5.0, 5.0)


class CombinedSPOAgent:
    """All three: Decoupled + Conservative + Replay."""

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        beta_start: float = None,
        beta_end: float = None,
        lambda_reg: float = 1.0,
        lr: float = 0.01,
        decision_lr: float = 0.01,
        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        min_lr_ratio: float = 0.1,
        confidence_threshold: float = 0.5,
        buffer_size: int = 200,
        batch_size: int = 16,
        min_buffer: int = 50,
        solver=None,
        seed: int = None,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.min_buffer = min_buffer

        if solver is None:
            from src.solver import solve_knapsack
            self.solver = lambda v: solve_knapsack(np.asarray(v), self.weights_np, capacity)
        else:
            self.solver = solver
        self.hidden_dim = hidden_dim
        self.lambda_reg = lambda_reg
        self.decay_rounds = decay_rounds
        self.t = 0

        if beta_start is not None and beta_end is not None:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta = beta_start
        else:
            self.beta_start = beta
            self.beta_end = beta
            self.beta = beta

        self.seed = seed if seed else 0
        key = jax.random.PRNGKey(self.seed)
        self.W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
        self.b1 = jnp.zeros(hidden_dim)
        self.W_dec = jnp.ones(n_arms)
        self.b_dec = jnp.zeros(n_arms)
        self.rng_key = jax.random.PRNGKey(self.seed + 1)

        if decay_rounds > 0:
            schedule = optax.cosine_decay_schedule(lr, decay_rounds, alpha=min_lr_ratio)
            self.feature_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=schedule)
            )
            dec_schedule = optax.cosine_decay_schedule(decision_lr, decay_rounds, alpha=min_lr_ratio)
            self.decision_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=dec_schedule)
            )
        else:
            self.feature_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=lr)
            )
            self.decision_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=decision_lr)
            )

        self.feature_opt_state = self.feature_optimizer.init({'W1': self.W1, 'b1': self.b1})
        self.decision_opt_state = self.decision_optimizer.init({'W_dec': self.W_dec, 'b_dec': self.b_dec})
        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)
        self.buffer_X = []
        self.buffer_selection = []
        self.buffer_rewards = []

    def _features(self, X):
        X = jnp.asarray(X)
        z1 = X @ self.W1 + self.b1
        return jnp.maximum(0, z1), z1

    def predict_mean(self, X):
        h, _ = self._features(X)
        return compute_bayesian_predictions(h, self.A_inv, self.b_bayes)

    def act(self, X):
        self.t += 1
        if self.beta_start != self.beta_end:
            progress = min(self.t / self.decay_rounds, 1.0)
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        h, _ = self._features(X)
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        ucb_bonus = compute_uncertainties(h, self.A_inv)
        decision_values = pred * self.W_dec + self.b_dec + self.beta * ucb_bonus
        return self.solver(np.asarray(decision_values))

    def _add_to_buffer(self, X, selection, rewards):
        self.buffer_X.append(np.asarray(X))
        self.buffer_selection.append(np.asarray(selection))
        self.buffer_rewards.append(np.asarray(rewards))
        if len(self.buffer_X) > self.buffer_size:
            self.buffer_X.pop(0)
            self.buffer_selection.pop(0)
            self.buffer_rewards.pop(0)

    def _sample_batch(self):
        buffer_len = len(self.buffer_X)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        indices = np.asarray(jax.random.choice(
            subkey, buffer_len, shape=(min(self.batch_size, buffer_len),), replace=False
        ))
        return ([self.buffer_X[i] for i in indices],
                [self.buffer_selection[i] for i in indices],
                [self.buffer_rewards[i] for i in indices])

    def update(self, X, selection, rewards, lr: float = None, full_feedback: bool = False):
        X = jnp.asarray(X)
        selection = jnp.asarray(selection)
        rewards = jnp.asarray(rewards)

        h, z1 = self._features(X)
        selection_mask = selection > 0.5
        self.A, self.b_bayes, self.A_inv = update_bayesian_stats(
            self.A, self.b_bayes, self.A_inv, h, rewards, selection_mask
        )
        self._add_to_buffer(X, selection, rewards)

        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        ucb_bonus = compute_uncertainties(h, self.A_inv)
        if full_feedback:
            Y_target = rewards
        else:
            Y_target = jnp.where(selection > 0.5, rewards, pred + self.beta * ucb_bonus)

        # Features with MSE
        mse_grad = 2 * (pred - Y_target) / self.n_arms
        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        grad_h = jnp.einsum('n,ni->i', mse_grad, theta)
        grad_z1 = grad_h * (z1 > 0)

        feature_grads = {'W1': jnp.outer(X, grad_z1), 'b1': grad_z1}
        updates, self.feature_opt_state = self.feature_optimizer.update(
            feature_grads, self.feature_opt_state, {'W1': self.W1, 'b1': self.b1}
        )
        params = optax.apply_updates({'W1': self.W1, 'b1': self.b1}, updates)
        self.W1 = jnp.clip(params['W1'], -10, 10)
        self.b1 = params['b1']

        # Decision layer: confidence-weighted SPO+ on batch
        if len(self.buffer_X) >= self.min_buffer:
            batch_X, batch_selection, batch_rewards = self._sample_batch()
            grad_W_acc = jnp.zeros_like(self.W_dec)
            grad_b_acc = jnp.zeros_like(self.b_dec)

            for bX, bsel, brew in zip(batch_X, batch_selection, batch_rewards):
                bh, _ = self._features(jnp.asarray(bX))
                bpred = compute_bayesian_predictions(bh, self.A_inv, self.b_bayes)
                bucb = compute_uncertainties(bh, self.A_inv)
                if full_feedback:
                    bY = jnp.asarray(brew)
                else:
                    bY = jnp.where(jnp.asarray(bsel) > 0.5, jnp.asarray(brew), bpred + self.beta * bucb)

                bdec = bpred * self.W_dec + self.b_dec
                confidence = 1.0 / (1.0 + bucb / self.confidence_threshold)
                spo_grad = spo_plus_grad(bdec, bY, self.weights, self.capacity)
                weighted = confidence * spo_grad
                grad_W_acc += weighted * bpred
                grad_b_acc += weighted

            n_batch = len(batch_X)
            decision_grads = {'W_dec': grad_W_acc / n_batch, 'b_dec': grad_b_acc / n_batch}
            dec_updates, self.decision_opt_state = self.decision_optimizer.update(
                decision_grads, self.decision_opt_state, {'W_dec': self.W_dec, 'b_dec': self.b_dec}
            )
            dec_params = optax.apply_updates({'W_dec': self.W_dec, 'b_dec': self.b_dec}, dec_updates)
            self.W_dec = jnp.clip(dec_params['W_dec'], 0.1, 10.0)
            self.b_dec = jnp.clip(dec_params['b_dec'], -5.0, 5.0)

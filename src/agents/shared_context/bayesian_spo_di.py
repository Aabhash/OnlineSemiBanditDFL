import jax
import jax.numpy as jnp
import numpy as np
import optax
from src.losses import make_spo_grad_fn
from src.solver import ShortestPathSolver, KnapsackSolver
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_bayesian_predictions,
    update_bayesian_stats,
)


@jax.jit
def _fused_bayesian_core(X, W1, b1, A, b_bayes, A_inv, rewards, mask):
    h = X @ W1 + b1
    A_new, b_new, A_inv_new = update_bayesian_stats(
        A, b_bayes, A_inv, h, rewards, mask)
    theta = jnp.einsum('nij,nj->ni', A_inv_new, b_new)
    pred = theta @ h
    return h, pred, theta, A_new, b_new, A_inv_new


@jax.jit
def _fused_bayesian_core_nl(X, W1, b1, W2, b2, A, b_bayes, A_inv, rewards, mask):
    z = X @ W1 + b1
    a = jnp.maximum(0, z)  # ReLU
    h = a @ W2 + b2
    A_new, b_new, A_inv_new = update_bayesian_stats(
        A, b_bayes, A_inv, h, rewards, mask)
    theta = jnp.einsum('nij,nj->ni', A_inv_new, b_new)
    pred = theta @ h
    return h, pred, theta, A_new, b_new, A_inv_new


@jax.jit
def _bayesian_grads(X, theta, grad_pred):
    grad_h = jnp.einsum('ni,n->i', theta, grad_pred)
    return {'W1': jnp.outer(X, grad_h), 'b1': grad_h}


@jax.jit
def _bayesian_grads_nl(X, W1, b1, W2, theta, grad_pred):
    grad_h = jnp.einsum('ni,n->i', theta, grad_pred)
    z = X @ W1 + b1
    a = jnp.maximum(0, z)
    grad_a = grad_h @ W2.T
    grad_z = grad_a * (z > 0)
    return {
        'W1': jnp.outer(X, grad_z), 'b1': grad_z,
        'W2': jnp.outer(a, grad_h), 'b2': grad_h,
    }


def _make_step_fn(optimizer):
    @jax.jit
    def step(grads, opt_state, params):
        updates, new_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state
    return step


@jax.jit
def _fused_predict(X, W1, b1, A_inv, b_bayes):
    h = X @ W1 + b1
    theta = jnp.einsum('nij,nj->ni', A_inv, b_bayes)
    return theta @ h


@jax.jit
def _fused_predict_nl(X, W1, b1, W2, b2, A_inv, b_bayes):
    z = X @ W1 + b1
    a = jnp.maximum(0, z)
    h = a @ W2 + b2
    theta = jnp.einsum('nij,nj->ni', A_inv, b_bayes)
    return theta @ h


# ---------------------------------------------------------------------------
# Standalone helpers for posterior-sample-based diagnostics
# ---------------------------------------------------------------------------

def compute_decision_disagreement(means_np, stds_np, solver, rng, M=10):
    """Compute decision disagreement rate delta_t.

    Draw M posterior-predictive samples, solve each, count pairwise disagreements.

    Args:
        means_np: (n_arms,) posterior means
        stds_np: (n_arms,) posterior stds
        solver: combinatorial solver callable
        rng: numpy random generator (np.random.Generator)
        M: number of posterior samples

    Returns:
        delta_t: float in [0, 1], decision disagreement rate
        solutions: list of M solution vectors (for diagnostics)
    """
    n_arms = len(means_np)
    solutions = []

    for m in range(M):
        z = rng.standard_normal(n_arms)
        c_sample = means_np + stds_np * z
        w_m = solver(c_sample)
        solutions.append((w_m > 0.5).astype(np.int8))

    # Count pairwise agreements
    n_pairs = 0
    n_agree = 0
    for i in range(M):
        for j in range(i + 1, M):
            n_pairs += 1
            if np.array_equal(solutions[i], solutions[j]):
                n_agree += 1

    delta_t = 1.0 - n_agree / max(n_pairs, 1)
    return delta_t, solutions


def compute_propensities(means_np, stds_np, solver, rng, M=10, p_min=0.05):
    """Estimate per-arm selection propensity from posterior samples.

    p_i = P(arm i in optimal solution) estimated by solving M posterior samples.

    Args:
        means_np: (n_arms,) posterior means
        stds_np: (n_arms,) posterior stds
        solver: combinatorial solver callable
        rng: numpy random generator (np.random.Generator)
        M: number of posterior samples
        p_min: minimum propensity clamp to avoid explosion

    Returns:
        propensities: (n_arms,) array in [p_min, 1.0]
    """
    n_arms = len(means_np)
    counts = np.zeros(n_arms)

    for m in range(M):
        z = rng.standard_normal(n_arms)
        c_sample = means_np + stds_np * z
        w_m = solver(c_sample)
        counts += (np.asarray(w_m) > 0.5).astype(float)

    propensities = counts / M
    propensities = np.clip(propensities, p_min, 1.0)
    return propensities


class BayesianSPO_DI_Agent:
    """BayesianSPO agent with discriminative exploration, imputation, and DFE.

    Imputation modes:
      'model'         -- impute unobserved with model prediction (+ optional mask_grad)
      'disc_explore'  (DX) -- masked SPO+ gradient (zero on unobserved)
      'disc_impute'   (DI) -- adversarial imputation via alternatives
      'ips_dr'        -- Doubly Robust: IPS-weighted observed gradients

    DFE (decision_focused_explore=True):
      Asymmetric exploration in act(): on-solution penalty, contested off-solution boost.

    Adaptive DFE (adaptive_dfe=True):
      gamma_t = gamma_0 * delta_t modulates exploration intensity per-context.

    Gated DI (di_gate_threshold > 0):
      Only apply adversarial imputation when delta_t < threshold.

    LR schedule: init_lr / (t+1)^lr_power  (default lr_power=0.5 = inv-sqrt).
    """

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        grid_size: int = None,
        hidden_dim: int = 32,
        beta: float = 1.0,
        lambda_reg: float = 1.0,
        init_lr: float = 0.001,

        max_grad_norm: float = 5.0,
        imputation: str = 'disc_explore',
        minimize: bool = True,
        solver=None,
        seed: int = None,
        loss_grad_fn=None,
        disc_K: int = 5,
        disc_lambda: float = 0.1,
        contested_alpha: float = 0.5,
        contested_tau: float = 2.0,
        contested_eps: float = 1e-6,
        co_solver=None,
        # --- New parameters (all default to off for backward compat) ---
        mask_grad: bool = False,
        decision_focused_explore: bool = False,
        adaptive_dfe: bool = False,
        dfe_gamma_0: float = 1.0,
        n_disagreement_samples: int = 10,
        adaptive_lambda: bool = False,
        di_gate_threshold: float = 0.0,
        ips_M: int = 10,
        ips_p_min: float = 0.05,
        ips_clip_max: float = 5.0,
        lr_power: float = 0.5,
        lr_schedule=None,
        nonlinear: bool = False,
    ):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.weights = jnp.asarray(weights)
        self.weights_np = np.asarray(weights)
        self.capacity = capacity
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.imputation = imputation
        self.minimize = minimize
        self.t = 0

        # Discriminative parameters
        self.disc_K = disc_K
        self.disc_lambda_0 = disc_lambda
        self.disc_lambda = disc_lambda
        self.contested_alpha = contested_alpha
        self.contested_tau = contested_tau
        self.contested_eps = contested_eps

        # New feature flags
        self.mask_grad = mask_grad
        self.decision_focused_explore = decision_focused_explore
        self.adaptive_dfe = adaptive_dfe
        self.dfe_gamma_0 = dfe_gamma_0
        self.n_disagreement_samples = n_disagreement_samples
        self.adaptive_lambda = adaptive_lambda
        self.di_gate_threshold = di_gate_threshold

        # IPS / DR parameters
        self.ips_M = ips_M
        self.ips_p_min = ips_p_min
        self.ips_clip_max = ips_clip_max

        # Diagnostic histories
        self.delta_history = []
        self.gamma_history = []
        self.lambda_history = []
        self.di_gated_history = []
        self.propensity_history = []
        self.contested_history = []
        self.sym_diff_sizes_history = []

        # Validation
        if imputation in ('disc_explore', 'disc_impute') and co_solver is None \
                and grid_size is None:
            raise ValueError(
                'co_solver or grid_size required for disc_explore/disc_impute')
        if decision_focused_explore and co_solver is None and grid_size is None:
            raise ValueError(
                'co_solver or grid_size required for decision_focused_explore')

        # Combinatorial solver for alternatives/slacks
        if co_solver is not None:
            self.co_solver = co_solver
        elif grid_size is not None:
            self.co_solver = ShortestPathSolver(grid_size)
        else:
            self.co_solver = None

        if solver is None:
            if self.co_solver is not None:
                self.solver = lambda v: self.co_solver.solve(
                    np.asarray(v, dtype=np.float64))
            else:
                raise ValueError(
                    "BayesianSPO_DI_Agent requires solver, co_solver, or grid_size"
                )
        else:
            self.solver = solver

        if loss_grad_fn is not None:
            self.loss_grad_fn = loss_grad_fn
        else:
            self.loss_grad_fn = make_spo_grad_fn(self.solver, minimize=minimize)

        # RNG
        key = jax.random.PRNGKey(seed if seed is not None else 0)
        key, init_key = jax.random.split(key)
        self.rng_key = key
        self.nonlinear = nonlinear
        self.rng_np = np.random.default_rng(seed if seed is not None else 0)

        if nonlinear:
            k1, k2 = jax.random.split(init_key)
            self.W1 = jax.random.normal(k1, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            self.W2 = jax.random.normal(k2, (hidden_dim, hidden_dim)) * 0.1
            self.b2 = jnp.zeros(hidden_dim)
            init_params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        else:
            self.W1 = jax.random.normal(init_key, (context_dim, hidden_dim)) * 0.1
            self.b1 = jnp.zeros(hidden_dim)
            init_params = {'W1': self.W1, 'b1': self.b1}

        # LR schedule: init_lr / (t+1)^lr_power
        if lr_schedule is not None:
            schedule = lr_schedule
        else:
            schedule = lambda count: init_lr / jnp.power(count + 1, lr_power)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(schedule),
        )
        self.opt_state = self.optimizer.init(init_params)
        self._step = _make_step_fn(self.optimizer)

        self.A, self.b_bayes, self.A_inv = init_per_arm_regression(
            n_arms, hidden_dim, lambda_reg)

    def _features(self, X):
        X = jnp.asarray(X)
        if self.nonlinear:
            z = X @ self.W1 + self.b1
            a = jnp.maximum(0, z)
            h = a @ self.W2 + self.b2
            return h
        else:
            h = X @ self.W1 + self.b1
            return h

    def _compute_stds(self, h):
        """Compute posterior standard deviations from features h."""
        Ax = jnp.einsum('nij,j->ni', self.A_inv, h)
        return jnp.sqrt(jnp.maximum(jnp.einsum('i,ni->n', h, Ax), 0.0))

    def _compute_delta_t(self, means_np, stds_np):
        """Compute decision disagreement delta_t from posterior samples."""
        delta_t, _ = compute_decision_disagreement(
            means_np, stds_np, self.solver, self.rng_np,
            M=self.n_disagreement_samples)
        return delta_t

    def predict_mean(self, X):
        X_j = jnp.asarray(X, dtype=jnp.float32)
        if self.nonlinear:
            return np.asarray(_fused_predict_nl(X_j, self.W1, self.b1,
                                                self.W2, self.b2,
                                                self.A_inv, self.b_bayes))
        else:
            return np.asarray(_fused_predict(X_j, self.W1, self.b1,
                                             self.A_inv, self.b_bayes))

    def act(self, X):
        self.t += 1
        X_j = jnp.asarray(X, dtype=jnp.float32)

        h = self._features(X_j)
        theta = jnp.einsum('nij,nj->ni', self.A_inv, self.b_bayes)
        means = theta @ h
        stds = self._compute_stds(h)

        # --- Path 1: Decision-focused exploration (DFE) ---
        if self.decision_focused_explore:
            means_np = np.asarray(means, dtype=np.float64)
            stds_np = np.asarray(stds, dtype=np.float64)

            if self.minimize:
                slacks = self.co_solver.compute_slacks(means_np)
                phi = np.exp(-slacks / (stds_np + self.contested_eps))

                if self.adaptive_dfe:
                    delta_t = self._compute_delta_t(means_np, stds_np)
                    gamma_t = self.dfe_gamma_0 * delta_t
                    self.delta_history.append(delta_t)
                    self.gamma_history.append(gamma_t)
                else:
                    gamma_t = self.dfe_gamma_0

                # On-solution: penalty (1 - gamma_t) to encourage departure
                on_solution = slacks < 1e-8
                phi[on_solution] = 1.0 - gamma_t

                phi_j = jnp.array(phi, dtype=jnp.float32)
                values = means - jnp.abs(self.beta) * stds * phi_j
            else:
                # Maximization: additive bonus on contested off-solution
                opt_sol, alts = self.co_solver.extract_k_alternatives(
                    means_np, self.disc_K)
                bonus = np.ones(self.n_arms)
                if alts:
                    disc_edges = np.zeros(self.n_arms, dtype=bool)
                    for _, _, sym_diff in alts:
                        disc_edges |= sym_diff
                    on_solution = opt_sol > 0.5
                    disc_off_sol = disc_edges & (~on_solution)
                    if disc_off_sol.any():
                        slacks = self.co_solver.compute_slacks(means_np)
                        bonus[disc_off_sol] = 1.0 + self.contested_alpha * np.exp(
                            -slacks[disc_off_sol] / (stds_np[disc_off_sol] + self.contested_eps))
                bonus_j = jnp.array(bonus, dtype=jnp.float32)
                values = means + self.beta * stds * bonus_j

            return self.solver(np.asarray(values))

        # --- Path 2: DI/DX targeted exploration (off-path contested boost) ---
        if self.imputation in ('disc_explore', 'disc_impute'):
            means_np = np.asarray(means, dtype=np.float64)
            stds_np = np.asarray(stds, dtype=np.float64)

            opt_sol, alts = self.co_solver.extract_k_alternatives(
                means_np, self.disc_K)

            bonus = np.ones(self.n_arms)
            if alts:
                disc_edges = np.zeros(self.n_arms, dtype=bool)
                for _, _, sym_diff in alts:
                    disc_edges |= sym_diff
                on_solution = opt_sol > 0.5
                disc_off_sol = disc_edges & (~on_solution)
                if disc_off_sol.any():
                    slacks = self.co_solver.compute_slacks(means_np)
                    bonus[disc_off_sol] = 1.0 + self.contested_alpha * np.exp(
                        -slacks[disc_off_sol] / (stds_np[disc_off_sol] + self.contested_eps))

            bonus_j = jnp.array(bonus, dtype=jnp.float32)
            if self.minimize:
                values = means - jnp.abs(self.beta) * stds * bonus_j
            else:
                values = means + self.beta * stds * bonus_j
            return self.solver(np.asarray(values))

        # --- Path 3: Standard UCB (model / ips_dr modes) ---
        if self.minimize:
            values = means - jnp.abs(self.beta) * stds
        else:
            values = means + self.beta * stds
        return self.solver(np.asarray(values))

    def update(self, X, selection, rewards, lr=None, full_feedback=False):
        X_j = jnp.asarray(X, dtype=jnp.float32)
        sel_j = jnp.asarray(selection, dtype=jnp.float32)
        rew_j = jnp.asarray(rewards, dtype=jnp.float32)
        sel_mask = sel_j > 0.5

        if full_feedback:
            obs_mask = jnp.ones(self.n_arms, dtype=jnp.bool_)
        else:
            obs_mask = sel_mask

        if self.nonlinear:
            h, pred, theta, self.A, self.b_bayes, self.A_inv = \
                _fused_bayesian_core_nl(X_j, self.W1, self.b1, self.W2, self.b2,
                                        self.A, self.b_bayes, self.A_inv,
                                        rew_j, obs_mask)
        else:
            h, pred, theta, self.A, self.b_bayes, self.A_inv = \
                _fused_bayesian_core(X_j, self.W1, self.b1,
                                     self.A, self.b_bayes, self.A_inv,
                                     rew_j, obs_mask)

        # Compute stds once for branches that need it
        stds = None
        if self.imputation in ('disc_impute', 'ips_dr') or self.adaptive_lambda:
            stds = self._compute_stds(h)

        # --- Adaptive lambda for imputation ---
        if self.adaptive_lambda and self.imputation == 'disc_impute':
            means_np = np.asarray(pred, dtype=np.float64)
            stds_np = np.asarray(stds, dtype=np.float64)
            delta_t = self._compute_delta_t(means_np, stds_np)
            self.disc_lambda = self.disc_lambda_0 * delta_t
            self.lambda_history.append(self.disc_lambda)

        # --- Confidence-gated DI ---
        if self.imputation == 'disc_impute' and self.di_gate_threshold > 0:
            if stds is None:
                stds = self._compute_stds(h)
            means_np = np.asarray(pred, dtype=np.float64)
            stds_np = np.asarray(stds, dtype=np.float64)
            delta_t = self._compute_delta_t(means_np, stds_np)
            self.delta_history.append(delta_t)

            if delta_t < self.di_gate_threshold:
                grad_pred = self._grad_disc_impute(pred, rew_j, obs_mask, stds)
                self.di_gated_history.append(1)
            else:
                grad_pred = self._grad_disc_explore(pred, rew_j, obs_mask)
                self.di_gated_history.append(0)

        elif self.imputation == 'ips_dr':
            grad_pred = self._grad_ips_dr(pred, rew_j, obs_mask, stds)

        elif self.imputation == 'disc_explore':
            grad_pred = self._grad_disc_explore(pred, rew_j, obs_mask)

        elif self.imputation == 'disc_impute':
            grad_pred = self._grad_disc_impute(pred, rew_j, obs_mask, stds)

        elif self.imputation == 'model':
            Y_target = jnp.where(obs_mask, rew_j, pred)
            grad_pred = self.loss_grad_fn(pred, Y_target)
            if self.mask_grad:
                grad_pred = jnp.where(obs_mask, grad_pred, 0.0)

        else:
            raise ValueError(f"Unknown imputation mode: {self.imputation}")

        if self.nonlinear:
            grads = _bayesian_grads_nl(X_j, self.W1, self.b1, self.W2, theta, grad_pred)
            params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        else:
            grads = _bayesian_grads(X_j, theta, grad_pred)
            params = {'W1': self.W1, 'b1': self.b1}

        new_params, self.opt_state = self._step(grads, self.opt_state, params)
        self.W1 = jnp.clip(new_params['W1'], -10, 10)
        self.b1 = new_params['b1']
        if self.nonlinear:
            self.W2 = jnp.clip(new_params['W2'], -10, 10)
            self.b2 = new_params['b2']

    # --- Gradient methods ---

    def _grad_disc_explore(self, pred, rew_j, obs_mask):
        """DX: Strictly masked gradient -- no imputed gradients."""
        Y_target = jnp.where(obs_mask, rew_j, pred)
        grad_pred = self.loss_grad_fn(pred, Y_target)
        return jnp.where(obs_mask, grad_pred, 0.0)

    def _grad_disc_impute(self, pred, rew_j, obs_mask, stds):
        """DI: Adversarial imputation on discriminating edges per alternative."""
        pred_np = np.asarray(pred, dtype=np.float64)
        stds_np = np.asarray(stds, dtype=np.float64)

        opt_sol, alts = self.co_solver.extract_k_alternatives(
            pred_np, self.disc_K)

        if not alts:
            self.sym_diff_sizes_history.append([])
            return self._grad_disc_explore(pred, rew_j, obs_mask)

        grad_pred = jnp.zeros(self.n_arms)
        all_disc = np.zeros(self.n_arms, dtype=bool)
        round_sym_sizes = []

        for alt_sol, _, sym_diff in alts:
            round_sym_sizes.append(int(sym_diff.sum()))

            imputed = pred_np.copy()
            opt_only = (opt_sol > 0.5) & (alt_sol < 0.5)
            alt_only = (alt_sol > 0.5) & (opt_sol < 0.5)

            if self.minimize:
                imputed[opt_only] += self.contested_alpha * stds_np[opt_only]
                imputed[alt_only] -= self.contested_alpha * stds_np[alt_only]
            else:
                imputed[opt_only] -= self.contested_alpha * stds_np[opt_only]
                imputed[alt_only] += self.contested_alpha * stds_np[alt_only]

            Y_k = np.where(np.asarray(obs_mask), np.asarray(rew_j), imputed)
            grad_k = self.loss_grad_fn(pred, jnp.array(Y_k, dtype=jnp.float32))

            disc_k = jnp.array(sym_diff)
            grad_k = jnp.where(
                obs_mask, grad_k,
                jnp.where(disc_k, self.disc_lambda * grad_k, 0.0))

            grad_pred = grad_pred + grad_k
            all_disc |= sym_diff

        self.contested_history.append(int(all_disc.sum()))
        self.sym_diff_sizes_history.append(round_sym_sizes)
        return grad_pred / len(alts)

    def _grad_ips_dr(self, pred, rew_j, obs_mask, stds):
        """Doubly Robust gradient: IPS-weighted observed + model baseline.

        DR target: Y_DR_i = pred_i + w_i * (reward_i - pred_i)
        For observed arms: w_i = 1/p_i (self-normalized)
        For unobserved arms: w_i = 0 (falls back to model prediction)
        """
        means_np = np.asarray(pred, dtype=np.float64)
        stds_np = np.asarray(stds, dtype=np.float64)

        # Estimate per-arm propensities from posterior samples
        propensities = compute_propensities(
            means_np, stds_np, self.solver, self.rng_np,
            M=self.ips_M, p_min=self.ips_p_min)

        # Track diagnostics
        obs_np = np.asarray(obs_mask)
        if obs_np.any():
            self.propensity_history.append({
                'mean_p_obs': float(propensities[obs_np].mean()),
                'min_p_obs': float(propensities[obs_np].min()),
                'mean_p_all': float(propensities.mean()),
                'n_low_p': int((propensities < 0.2).sum()),
            })

        # IPS weights: w_i = 1/p_i for observed, 0 for unobserved
        ips_weights = np.where(obs_np, 1.0 / propensities, 0.0)
        ips_weights = np.clip(ips_weights, 0.0, self.ips_clip_max)

        # Self-normalize: scale so observed weights sum to n_observed
        n_obs = obs_np.sum()
        if n_obs > 0:
            weight_sum = ips_weights[obs_np].sum()
            if weight_sum > 0:
                ips_weights[obs_np] *= n_obs / weight_sum

        # DR target: Y_DR = pred + ips_weight * (reward - pred)
        ips_w_j = jnp.array(ips_weights, dtype=jnp.float32)
        Y_dr = pred + ips_w_j * (rew_j - pred)

        # SPO+ gradient with DR target, masked to observed
        grad_pred = self.loss_grad_fn(pred, Y_dr)
        grad_pred = jnp.where(obs_mask, grad_pred, 0.0)

        return grad_pred

"""Helpers for feature representation analysis notebook (Claude version)."""

import numpy as np
import jax.numpy as jnp
import pandas as pd
from typing import Tuple, Dict, List, Optional
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from lets_plot import (
    aes,
    geom_point,
    geom_line,
    geom_hline,
    geom_bar,
    ggplot,
    ggsize,
    labs,
    scale_color_manual,
    scale_fill_manual,
    coord_fixed,
    theme,
    element_text,
    theme_minimal,
    facet_wrap,
)
from lets_plot.mapping import as_discrete

from src.data.knapsack import KnapsackData
from src.envs.knapsack import KnapsackEnv
from src.solvers.knapsack import KnapsackSolver
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.agents.contextual.bayesian_spo import BayesianSPOAgent
from src.agents.contextual.neural_agents import BayesianMSEAgent


# ── Task 1: Custom DGP and Oracle ──────────────────────────────────────────


class DecisionFocusedDGP(KnapsackData):
    """Custom DGP where decision-relevant and nuisance dimensions are known.

    4-arm unit-weight knapsack, capacity=2 (select exactly 2 of 4).
    Context x in R^5.

    B matrix design:
    - Items 0,1: costs depend on x[0] (decision dim 1)
    - Items 2,3: costs depend on x[1] (decision dim 2)
    - All items: costs shift by x[2:5] (nuisance dims) equally,
      so nuisance dims change magnitudes but never flip the ranking
      within {0,1} or within {2,3}.
    """

    def __init__(self, seed: int = 42, noise_width: float = 0.1):
        self.n_items = 4
        self.n_features = 5
        self.context_dim = 5
        self.capacity = 2.0
        self.deg = 1
        self.noise_width = noise_width
        self.rng = np.random.default_rng(seed)

        self.weights = np.ones(4)

        # Decision dims (0-1): opposite signs for paired items -> flip rankings
        # Nuisance dims (2-4): large equal coefficients -> high MSE impact but
        # never change relative ordering within pairs, so zero decision impact.
        # This creates a strong incentive for MSE to learn nuisance dims
        # (they explain a lot of variance) while SPO should ignore them.
        self.B = np.array([
            [ 1.0,  0.0,  2.0,  2.0,  2.0],
            [-1.0,  0.0,  2.0,  2.0,  2.0],
            [ 0.0,  1.0,  2.0,  2.0,  2.0],
            [ 0.0, -1.0,  2.0,  2.0,  2.0],
        ])

    def _generate_values(self, x: np.ndarray) -> np.ndarray:
        """Linear costs: v_i = B_i @ x / sqrt(d) + offset, with small noise."""
        linear = self.B @ x / np.sqrt(self.n_features)
        values = linear + 3.0
        noise = self.rng.uniform(
            1 - self.noise_width,
            1 + self.noise_width,
            self.n_items,
        )
        return values * noise


class DecisionFocusedEnv(KnapsackEnv):
    """Knapsack environment using the custom DGP."""

    def __init__(self, seed: int = 42, noise_width: float = 0.1):
        from src.envs.base import ConstrainedBanditEnv

        self.data = DecisionFocusedDGP(seed=seed, noise_width=noise_width)

        ConstrainedBanditEnv.__init__(
            self,
            n_arms=4,
            context_dim=5,
            weights=self.data.weights,
            capacity=2.0,
            seed=seed,
        )
        self.knapsack_solver = KnapsackSolver(self.weights, self.capacity)
        self.current_context, self.current_values = self._generate_sample()


def compute_oracle_decision(x: np.ndarray) -> np.ndarray:
    """Return the optimal 2-of-4 selection for context x under the noiseless DGP.

    Decision logic (maximization, unit weights, capacity=2):
    - Within pair (0,1): item 0 wins when x[0] > 0, item 1 when x[0] < 0
    - Within pair (2,3): item 2 wins when x[1] > 0, item 3 when x[1] < 0
    - Pick the winner from each pair -> always exactly 2 items.
    """
    sel = np.zeros(4)
    sel[0 if x[0] > 0 else 1] = 1.0
    sel[2 if x[1] > 0 else 3] = 1.0
    return sel


def compute_oracle_decisions_batch(X: np.ndarray) -> np.ndarray:
    """Batch version: X is (n_samples, 5), returns (n_samples, 4)."""
    return np.array([compute_oracle_decision(x) for x in X])


def decision_label(sel: np.ndarray) -> str:
    """Convert binary selection to a readable label like '0,2'."""
    indices = np.where(np.asarray(sel) > 0.5)[0]
    return ",".join(str(i) for i in indices)


ORACLE_W1 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
])


# ── Task 2: Training Utilities and Feature Extraction ──────────────────────


def create_agent_pair(
    env,
    hidden_dim: int = 2,
    beta: float = 1.0,
    lambda_reg: float = 1.0,
    lr: float = 0.001,
    seed: int = 0,
    mask_spo: bool = False,
    mask_mse: bool = False,
) -> Tuple[BayesianSPOAgent, BayesianMSEAgent]:
    """Create matched SPO and MSE agents with identical initialization."""
    common = dict(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        nonlinear=False,
        seed=seed,
        solver=env.knapsack_solver.solve,
    )
    spo = BayesianSPOAgent(**common, mask_grad=mask_spo)
    mse = BayesianMSEAgent(**common, mask_grad=mask_mse)
    return spo, mse


def train_with_checkpoints(
    agent,
    episode: dict,
    n_rounds: int,
    checkpoint_interval: int = 50,
    full_feedback: bool = False,
) -> Dict[str, list]:
    """Train an agent on a pregenerated episode, saving W1/b1 checkpoints.

    Returns dict with keys:
    - 'W1_checkpoints': list of (round, W1_array, b1_array) tuples
    - 'cumulative_regret': list of floats
    - 'regrets': list of floats
    """
    rounds_data = episode['rounds_data']
    W1_checkpoints = [(0, np.array(agent.W1), np.array(agent.b1))]
    regrets = []
    cumulative_regret = []
    total_regret = 0.0

    for t in range(n_rounds):
        rd = rounds_data[t]
        X = rd['context']
        true_vals = rd['true_values']
        optimal_val = rd['optimal_value']

        selection = agent.act(X)
        selection = np.asarray(selection).flatten()
        agent_val = float(np.sum(true_vals * selection))
        instant_regret = optimal_val - agent_val
        total_regret += instant_regret
        regrets.append(instant_regret)
        cumulative_regret.append(total_regret)

        if full_feedback:
            observed = true_vals.copy()
        else:
            observed = np.zeros_like(true_vals)
            mask = selection > 0.5
            observed[mask] = true_vals[mask]

        agent.update(X, selection, observed, full_feedback=full_feedback)

        if (t + 1) % checkpoint_interval == 0:
            W1_checkpoints.append((t + 1, np.array(agent.W1), np.array(agent.b1)))

    return {
        'W1_checkpoints': W1_checkpoints,
        'cumulative_regret': cumulative_regret,
        'regrets': regrets,
    }


def extract_features(agent, X_test: np.ndarray) -> np.ndarray:
    """Extract features h = X @ W1 + b1 for a batch of test contexts.

    Args:
        agent: BayesianSPOAgent or BayesianMSEAgent
        X_test: (n_samples, context_dim) array

    Returns:
        (n_samples, hidden_dim) array of features
    """
    features = []
    for x in X_test:
        h = agent._features(x)
        features.append(np.array(h))
    return np.stack(features)


def generate_test_contexts(env, n_samples: int = 500, seed: int = 99) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test contexts with oracle decisions and true values.

    Returns:
        X_test: (n_samples, context_dim)
        decisions: (n_samples, n_arms) binary optimal selections
        values: (n_samples, n_arms) true cost values
    """
    rng = np.random.default_rng(seed)
    X_test = rng.standard_normal((n_samples, env.context_dim))
    decisions = []
    values = []
    for x in X_test:
        v = env.data._generate_values(x)
        d = env.solve(v)
        decisions.append(d)
        values.append(v)
    return X_test, np.array(decisions), np.array(values)


# ── Task 3: Subspace Alignment Metric ─────────────────────────────────────


def subspace_angle(W_learned: np.ndarray, W_oracle: np.ndarray = None) -> float:
    """Compute the principal angle (degrees) between two 2D subspaces.

    Both W_learned and W_oracle are (d, 2) matrices whose columns span the subspace.
    Returns angle in [0, 90] degrees. 0 = perfectly aligned, 90 = orthogonal.

    If W_oracle is None, uses ORACLE_W1 (projection onto dims 0-1).
    """
    if W_oracle is None:
        W_oracle = ORACLE_W1

    Q1, _ = np.linalg.qr(W_learned)
    Q2, _ = np.linalg.qr(W_oracle)

    M = Q1.T @ Q2
    singular_values = np.linalg.svd(M, compute_uv=False)

    cos_angles = np.clip(singular_values, -1.0, 1.0)
    angles_rad = np.arccos(cos_angles)

    return float(np.degrees(np.max(angles_rad)))


def alignment_over_training(
    checkpoints,
) -> Tuple[List[int], List[float]]:
    """Compute subspace alignment angle at each checkpoint.

    Args:
        checkpoints: list of (round, W1, b1) from train_with_checkpoints

    Returns:
        rounds: list of round numbers
        angles: list of alignment angles in degrees
    """
    rounds = [ckpt[0] for ckpt in checkpoints]
    angles = [subspace_angle(ckpt[1]) for ckpt in checkpoints]
    return rounds, angles


def features_at_checkpoint(W1, b1, X_test: np.ndarray) -> np.ndarray:
    """Compute features h = X @ W1 + b1 using a saved checkpoint."""
    return X_test @ W1 + b1


# ── Trajectory (Sensitivity) Analysis ─────────────────────────────────────


def compute_trajectory_sensitivity(
    agent,
    dim_vary: int,
    n_steps: int = 50,
    n_traces: int = 4,
    sweep_range: Tuple[float, float] = (-2.5, 2.5),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep one context dimension while holding the rest fixed.

    Generates n_traces random base contexts, then varies dim_vary across
    n_steps values. Returns feature trajectories.

    Returns:
        sweep_vals: (n_steps,) the swept values
        trajectories: (n_traces, n_steps, hidden_dim) feature trajectories
    """
    rng = np.random.default_rng(seed)
    sweep_vals = np.linspace(sweep_range[0], sweep_range[1], n_steps)
    trajectories = []

    for _ in range(n_traces):
        base = rng.standard_normal(5)
        trace = []
        for v in sweep_vals:
            x = base.copy()
            x[dim_vary] = v
            h = np.array(agent._features(x))
            trace.append(h)
        trajectories.append(np.stack(trace))

    return sweep_vals, np.array(trajectories)


def trajectory_sensitivity_score(trajectories: np.ndarray) -> float:
    """Mean L2 displacement per step across all traces."""
    # trajectories: (n_traces, n_steps, hidden_dim)
    diffs = np.diff(trajectories, axis=1)  # (n_traces, n_steps-1, hidden_dim)
    step_norms = np.linalg.norm(diffs, axis=2)  # (n_traces, n_steps-1)
    return float(np.mean(step_norms))


def plot_trajectories(
    agent_spo,
    agent_mse,
    seed: int = 0,
) -> 'ggplot':
    """2x2 grid: rows = decision dim (x₀) / nuisance dim (x₂), cols = SPO / MSE.

    Each panel shows feature-space trajectories as lines. If the agent is sensitive
    to that dimension, lines spread across the feature space. If insensitive,
    lines are flat/short.

    Annotates each panel with mean sensitivity (L2 displacement per step).
    """
    from lets_plot import geom_path, geom_text, gggrid

    panels = []
    for dim_idx, dim_name in [(0, 'x₀ (decision)'), (2, 'x₂ (nuisance)')]:
        for agent, agent_name, color in [
            (agent_spo, 'BayesianSPO', COLORS['BayesianSPO']),
            (agent_mse, 'BayesianMSE', COLORS['BayesianMSE']),
        ]:
            sweep, trajs = compute_trajectory_sensitivity(
                agent, dim_vary=dim_idx, n_steps=50, n_traces=6, seed=seed)
            sens = trajectory_sensitivity_score(trajs)

            rows = []
            for t in range(trajs.shape[0]):
                for s in range(trajs.shape[1]):
                    rows.append({
                        'h1': float(trajs[t, s, 0]),
                        'h2': float(trajs[t, s, 1]),
                        'trace': str(t),
                    })
            df = pd.DataFrame(rows)

            # Annotation
            x_ann = float(df['h1'].min() + 0.02 * (df['h1'].max() - df['h1'].min()))
            y_ann = float(df['h2'].max() - 0.02 * (df['h2'].max() - df['h2'].min()))
            df_ann = pd.DataFrame([{'h1': x_ann, 'h2': y_ann,
                                    'text': f'sensitivity = {sens:.3f}'}])

            p = (
                ggplot(df, aes('h1', 'h2'))
                + geom_path(aes(group='trace'), color=color, size=0.8, alpha=0.6)
                + geom_point(size=0.5, color=color, alpha=0.3)
                + geom_text(aes('h1', 'h2', label='text'), data=df_ann,
                            color='black', size=9, hjust=0, fontface='bold')
                + labs(x='Feature dim 1', y='Feature dim 2',
                       title=f'{agent_name} — vary {dim_name}')
                + theme_minimal()
                + ggsize(350, 300)
            )
            panels.append(p)

    return gggrid(panels, ncol=2)


# ── Decoded Latent Analysis ────────────────────────────────────────────────


def decode_latents(features: np.ndarray, X_test: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit linear probes to predict each context dimension from features.

    Returns dict mapping latent name to predicted values.
    """
    from sklearn.linear_model import LinearRegression
    predictions = {}
    latent_names = ['x₀ (decision)', 'x₁ (decision)',
                    'x₂ (nuisance)', 'x₃ (nuisance)', 'x₄ (nuisance)']
    for i, name in enumerate(latent_names):
        reg = LinearRegression().fit(features, X_test[:, i])
        predictions[name] = reg.predict(features)
    return predictions


def plot_decoded_latents(
    features_spo: np.ndarray,
    features_mse: np.ndarray,
    X_test: np.ndarray,
    decisions: np.ndarray,
) -> 'ggplot':
    """Scatter: true latent vs decoded (predicted) latent from features.

    2x2 grid: rows = decision latent x₀ / nuisance latent x₂,
    cols = SPO / MSE. Diagonal = perfect recovery.
    """
    from lets_plot import (
        geom_abline, gggrid, scale_color_manual,
    )

    dec_labels = [decision_label(d) for d in decisions]
    decoded_spo = decode_latents(features_spo, X_test)
    decoded_mse = decode_latents(features_mse, X_test)

    panels = []
    for latent_idx, latent_name, short, dim_type in [
        (0, 'x₀ (decision)', 'x₀', 'DECISION'),
        (2, 'x₂ (nuisance)', 'x₂', 'NUISANCE'),
    ]:
        for agent_name, decoded in [('BayesianSPO', decoded_spo), ('BayesianMSE', decoded_mse)]:
            true_vals = X_test[:, latent_idx]
            pred_vals = decoded[latent_name]
            r2 = float(1 - np.sum((true_vals - pred_vals)**2) / np.sum((true_vals - true_vals.mean())**2))

            rows = [{'true': float(true_vals[i]), 'predicted': float(pred_vals[i]),
                      'decision': dec_labels[i]}
                    for i in range(len(true_vals))]
            df = pd.DataFrame(rows)

            p = (
                ggplot(df, aes('true', 'predicted', color='decision'))
                + geom_point(size=1.5, alpha=0.4)
                + geom_abline(slope=1, intercept=0, linetype='dashed', color='gray', size=0.8)
                + scale_color_manual(values=DECISION_COLORS)
                + labs(
                    x=f'True {short}',
                    y=f'Predicted {short}',
                    title=f'[{dim_type}] {agent_name} → {short} (R²={r2:.2f})',
                )
                + theme_minimal()
                + ggsize(350, 300)
            )
            panels.append(p)

    return gggrid(panels, ncol=2)


# ── Task 4: Plotting Functions (Tiny Problem) ─────────────────────────────


COLORS = {
    'BayesianSPO': '#1f77b4',
    'BayesianMSE': '#d62728',
    'Oracle': '#2ca02c',
}

DECISION_COLORS = {
    '0,2': '#1f77b4',
    '0,3': '#ff7f0e',
    '1,2': '#2ca02c',
    '1,3': '#d62728',
}


def _convex_hull_df(h1, h2, group_label):
    """Compute convex hull vertices for a set of 2D points."""
    from scipy.spatial import ConvexHull
    points = np.column_stack([h1, h2])
    if len(points) < 3:
        return []
    try:
        hull = ConvexHull(points)
        verts = hull.vertices.tolist()
        verts.append(verts[0])  # close the polygon
        return [{'h1': float(points[v, 0]), 'h2': float(points[v, 1]),
                 'decision': group_label}
                for v in verts]
    except Exception:
        return []


def _decision_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    """Linear classification accuracy for predicting decision from features."""
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    scores = cross_val_score(clf, features, labels, cv=5, scoring='accuracy')
    return float(np.mean(scores))


def _cost_correlation(features: np.ndarray, total_cost: np.ndarray) -> float:
    """Max absolute Pearson correlation between any feature dim and total cost."""
    max_corr = 0.0
    for j in range(features.shape[1]):
        r = np.corrcoef(features[:, j], total_cost)[0, 1]
        if np.isfinite(r):
            max_corr = max(max_corr, abs(r))
    return max_corr


def plot_feature_grid(
    features_spo: np.ndarray,
    features_mse: np.ndarray,
    X_test: np.ndarray,
    decisions: np.ndarray,
    values: np.ndarray,
) -> 'ggplot':
    """Main 2x3 figure: Oracle | SPO | MSE, colored by decision (row 1) and cost (row 2).

    Row 1: colored by decision with convex hulls + classification accuracy.
    Row 2: colored by total cost with cost-correlation annotation.
    """
    from lets_plot import (
        scale_color_gradient, gggrid, geom_vline, geom_text, geom_path,
    )

    oracle_h = X_test[:, :2]
    total_cost = values.sum(axis=1)

    dec_labels = np.array([decision_label(d) for d in decisions])

    # --- Build row 1: colored by decision + convex hulls + accuracy ---
    panels_row1 = []
    for label, h in [('Oracle (x₀, x₁)', oracle_h),
                      ('BayesianSPO', features_spo),
                      ('BayesianMSE', features_mse)]:
        acc = _decision_accuracy(h, dec_labels)

        rows = []
        for i in range(len(decisions)):
            dec = decision_label(decisions[i])
            rows.append({'h1': float(h[i, 0]), 'h2': float(h[i, 1]),
                         'decision': dec})
        df = pd.DataFrame(rows)

        # Convex hulls per decision
        hull_rows = []
        for dec in DECISION_COLORS:
            mask = df['decision'] == dec
            if mask.sum() >= 3:
                hull_rows.extend(_convex_hull_df(
                    df.loc[mask, 'h1'].values,
                    df.loc[mask, 'h2'].values,
                    dec))
        df_hull = pd.DataFrame(hull_rows) if hull_rows else None

        # Annotation
        x_ann = float(df['h1'].min() + 0.02 * (df['h1'].max() - df['h1'].min()))
        y_ann = float(df['h2'].max() - 0.02 * (df['h2'].max() - df['h2'].min()))
        df_ann = pd.DataFrame([{'h1': x_ann, 'h2': y_ann,
                                'text': f'accuracy = {acc:.0%}'}])

        p = (
            ggplot(df, aes('h1', 'h2', color='decision'))
            + geom_point(size=2.0, alpha=0.4)
            + scale_color_manual(values=DECISION_COLORS)
            + geom_text(aes('h1', 'h2', label='text'), data=df_ann,
                        color='black', size=9, hjust=0, fontface='bold')
            + labs(x='Feature dim 1', y='Feature dim 2', title=label)
            + theme_minimal()
            + ggsize(350, 320)
        )
        if df_hull is not None and len(df_hull) > 0:
            p = p + geom_path(aes('h1', 'h2', color='decision', group='decision'),
                              data=df_hull, size=1.0, alpha=0.7)
        # Add crosshair on oracle panel
        if label.startswith('Oracle'):
            p = (p
                 + geom_vline(xintercept=0, linetype='dashed', color='gray', size=0.8)
                 + geom_hline(yintercept=0, linetype='dashed', color='gray', size=0.8))
        panels_row1.append(p)

    # --- Build row 2: colored by cost magnitude + correlation ---
    panels_row2 = []
    for label, h in [('Oracle (x₀, x₁)', oracle_h),
                      ('BayesianSPO', features_spo),
                      ('BayesianMSE', features_mse)]:
        corr = _cost_correlation(h, total_cost)

        rows = []
        for i in range(len(decisions)):
            rows.append({'h1': float(h[i, 0]), 'h2': float(h[i, 1]),
                         'cost': float(total_cost[i])})
        df = pd.DataFrame(rows)

        x_ann = float(df['h1'].min() + 0.02 * (df['h1'].max() - df['h1'].min()))
        y_ann = float(df['h2'].max() - 0.02 * (df['h2'].max() - df['h2'].min()))
        df_ann = pd.DataFrame([{'h1': x_ann, 'h2': y_ann,
                                'text': f'|r| = {corr:.2f}'}])

        p = (
            ggplot(df, aes('h1', 'h2', color='cost'))
            + geom_point(size=2.0, alpha=0.6)
            + scale_color_gradient(low='#2166ac', high='#b2182b')
            + geom_text(aes('h1', 'h2', label='text'), data=df_ann,
                        color='black', size=9, hjust=0, fontface='bold')
            + labs(x='Feature dim 1', y='Feature dim 2', title=label)
            + theme_minimal()
            + ggsize(350, 320)
        )
        panels_row2.append(p)

    return gggrid(panels_row1 + panels_row2, ncol=3)


def _rolling_mean(x, window=50):
    """Simple rolling average for smoothing per-round regret."""
    cs = np.cumsum(np.insert(x, 0, 0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = (cs[i + 1] - cs[lo]) / (i - lo + 1)
    return out


def create_feature_evolution_animation(
    checkpoints_spo,
    checkpoints_mse,
    X_test: np.ndarray,
    decisions: np.ndarray,
    regrets_spo: List[float],
    regrets_mse: List[float],
    interval_ms: int = 500,
    smooth_window: int = 50,
):
    """Create animation: feature scatter (top) + per-instance regret (bottom).

    Top row: Oracle | SPO | MSE scatter with convex hulls.
    Bottom row: three aligned regret subplots (Oracle=0, SPO, MSE) revealing
    only past rounds — future regret is hidden.

    Returns an IPython HTML object for inline display.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    from scipy.spatial import ConvexHull

    dec_labels = np.array([decision_label(d) for d in decisions])
    unique_decisions = sorted(set(dec_labels))
    color_map = {d: c for d, c in DECISION_COLORS.items()}

    oracle_h = X_test[:, :2]
    n_frames = min(len(checkpoints_spo), len(checkpoints_mse))
    n_rounds = len(regrets_spo)

    # Pre-compute smoothed per-instance regret
    reg_spo = np.array(regrets_spo)
    reg_mse = np.array(regrets_mse)
    smooth_spo = _rolling_mean(reg_spo, smooth_window)
    smooth_mse = _rolling_mean(reg_mse, smooth_window)
    rounds_all = np.arange(1, n_rounds + 1)

    # Fixed y-limit for regret (shared across all 3 panels)
    reg_ymax = max(smooth_spo.max(), smooth_mse.max()) * 1.1

    # Fixed axis limits from oracle
    pad = 0.3
    ox_min, ox_max = oracle_h[:, 0].min() - pad, oracle_h[:, 0].max() + pad
    oy_min, oy_max = oracle_h[:, 1].min() - pad, oracle_h[:, 1].max() + pad

    # Layout: 2 rows x 3 cols
    fig = plt.figure(figsize=(14, 7.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.4, 1], hspace=0.35, wspace=0.3,
                           top=0.90, bottom=0.08, left=0.06, right=0.96)
    ax_scatter = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_regret = [fig.add_subplot(gs[1, i]) for i in range(3)]

    def draw_frame(frame_idx):
        for ax in ax_scatter + ax_regret:
            ax.clear()

        rnd = checkpoints_spo[frame_idx][0]
        W1_spo, b1_spo = checkpoints_spo[frame_idx][1], checkpoints_spo[frame_idx][2]
        W1_mse, b1_mse = checkpoints_mse[frame_idx][1], checkpoints_mse[frame_idx][2]

        h_spo = features_at_checkpoint(W1_spo, b1_spo, X_test)
        h_mse = features_at_checkpoint(W1_mse, b1_mse, X_test)

        # --- Top row: scatter plots ---
        for ax, h, title in [
            (ax_scatter[0], oracle_h, 'Oracle (x₀, x₁)'),
            (ax_scatter[1], h_spo, 'BayesianSPO'),
            (ax_scatter[2], h_mse, 'BayesianMSE'),
        ]:
            for dec in unique_decisions:
                mask = dec_labels == dec
                c = color_map.get(dec, '#999999')
                pts = h[mask]
                ax.scatter(pts[:, 0], pts[:, 1], c=c, s=12, alpha=0.4, label=dec)

                if pts.shape[0] >= 3:
                    try:
                        hull = ConvexHull(pts)
                        verts = np.append(hull.vertices, hull.vertices[0])
                        ax.plot(pts[verts, 0], pts[verts, 1], c=c, lw=1.2, alpha=0.7)
                    except Exception:
                        pass

            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Feature dim 1', fontsize=9)
            ax.set_ylabel('Feature dim 2', fontsize=9)
            ax.tick_params(labelsize=8)

            if title.startswith('Oracle'):
                ax.set_xlim(ox_min, ox_max)
                ax.set_ylim(oy_min, oy_max)
                ax.axhline(0, ls='--', c='gray', lw=0.8)
                ax.axvline(0, ls='--', c='gray', lw=0.8)

        ax_scatter[2].legend(title='decision', fontsize=7, title_fontsize=8,
                             loc='upper right', markerscale=1.5)

        # --- Bottom row: per-instance regret (only past rounds shown) ---
        end = max(rnd, 1)
        for ax, smooth, color, title in [
            (ax_regret[0], np.zeros(n_rounds), COLORS['Oracle'], 'Oracle regret'),
            (ax_regret[1], smooth_spo, COLORS['BayesianSPO'], 'SPO regret'),
            (ax_regret[2], smooth_mse, COLORS['BayesianMSE'], 'MSE regret'),
        ]:
            ax.plot(rounds_all[:end], smooth[:end], color=color, lw=1.8)
            ax.set_xlim(0, n_rounds)
            ax.set_ylim(-0.02 * reg_ymax, reg_ymax)
            ax.set_xlabel('Round', fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=8)
        ax_regret[0].set_ylabel('Instant regret\n(smoothed)', fontsize=9)

        fig.suptitle(f'Feature Evolution — Round {rnd}', fontsize=13, fontweight='bold')

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=interval_ms, repeat=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_feature_scatter(
    features_spo: np.ndarray,
    features_mse: np.ndarray,
    decisions: np.ndarray,
    title_suffix: str = "",
) -> 'ggplot':
    """2-panel scatter: SPO features (left) vs MSE features (right), colored by decision."""
    rows = []
    for i in range(len(decisions)):
        label = decision_label(decisions[i])
        rows.append({
            'h1': float(features_spo[i, 0]),
            'h2': float(features_spo[i, 1]),
            'decision': label,
            'agent': 'BayesianSPO',
        })
        rows.append({
            'h1': float(features_mse[i, 0]),
            'h2': float(features_mse[i, 1]),
            'decision': label,
            'agent': 'BayesianMSE',
        })
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('h1', 'h2', color='decision'))
        + geom_point(size=2, alpha=0.6)
        + facet_wrap('agent')
        + scale_color_manual(values=DECISION_COLORS)
        + coord_fixed()
        + labs(
            x='Feature dim 1',
            y='Feature dim 2',
            color='Selection',
            title=f'Learned Features by Decision Region{title_suffix}',
        )
        + theme_minimal()
        + ggsize(800, 400)
    )
    return p


def plot_alignment_trajectory(
    rounds_spo: List[int],
    angles_spo: List[float],
    rounds_mse: List[int],
    angles_mse: List[float],
) -> 'ggplot':
    """Line plot of subspace alignment angle over training rounds."""
    rows = []
    for r, a in zip(rounds_spo, angles_spo):
        rows.append({'round': r, 'angle': a, 'agent': 'BayesianSPO'})
    for r, a in zip(rounds_mse, angles_mse):
        rows.append({'round': r, 'angle': a, 'agent': 'BayesianMSE'})
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('round', 'angle', color='agent'))
        + geom_line(size=1.5)
        + geom_point(size=3)
        + scale_color_manual(values={'BayesianSPO': COLORS['BayesianSPO'], 'BayesianMSE': COLORS['BayesianMSE']})
        + labs(
            x='Training Round',
            y='Subspace Angle to Oracle (degrees)',
            color='Agent',
            title='Feature Subspace Alignment During Training',
        )
        + geom_hline(yintercept=0, linetype='dashed', color='gray')
        + theme_minimal()
        + ggsize(600, 400)
    )
    return p


def plot_feature_directions(
    W1_spo: np.ndarray,
    W1_mse: np.ndarray,
    dim_names: Optional[List[str]] = None,
) -> 'ggplot':
    """Stacked horizontal bars: one normalized bar per agent.

    Each segment is the row-norm share for one input dimension:
        ||W1[i, :]|| / sum_j ||W1[j, :]||

    Blue shades = decision dimensions, red shades = nuisance dimensions.
    This isolates allocation across dimensions instead of mixing allocation with
    the overall scale of W1.
    """
    from lets_plot import coord_flip, scale_x_discrete

    if dim_names is None:
        dim_names = ['x₀', 'x₁', 'x₂', 'x₃', 'x₄']

    # Dim type: first 2 = decision, rest = nuisance
    dim_colors = {
        'x₀ (decision)': '#2166ac',
        'x₁ (decision)': '#4393c3',
        'x₂ (nuisance)': '#d6604d',
        'x₃ (nuisance)': '#f4a582',
        'x₄ (nuisance)': '#fddbc7',
    }
    dim_labels = [
        'x₀ (decision)', 'x₁ (decision)',
        'x₂ (nuisance)', 'x₃ (nuisance)', 'x₄ (nuisance)',
    ]

    rows = []
    for agent_name, W1 in [('BayesianSPO', W1_spo), ('BayesianMSE', W1_mse)]:
        row_norms = np.linalg.norm(W1, axis=1)
        total_norm = float(np.sum(row_norms))
        if total_norm <= 1e-12:
            shares = np.zeros_like(row_norms)
        else:
            shares = row_norms / total_norm
        for i, dl in enumerate(dim_labels):
            rows.append({'agent': agent_name, 'dimension': dl, 'weight_share': float(shares[i])})
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('agent', 'weight_share', fill='dimension'))
        + geom_bar(stat='identity', position='stack')
        + coord_flip()
        + scale_fill_manual(values=dim_colors)
        + scale_x_discrete(limits=['BayesianMSE', 'BayesianSPO'])
        + labs(
            x='',
            y='Normalized Feature Weight Share',
            fill='Context Dimension',
            title='Where does each agent allocate feature capacity?',
        )
        + theme_minimal()
        + ggsize(600, 250)
    )
    return p


# ── Task 5: Realistic Problem (knapsack_32) Utilities ─────────────────────


def compute_effective_dimensionality(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """PCA on features, return explained variance ratios and cumulative ratios."""
    pca = PCA()
    pca.fit(features)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    return explained, cumulative


def project_features_2d(features: np.ndarray) -> Tuple[np.ndarray, 'PCA']:
    """Project features to 2D via PCA."""
    pca = PCA(n_components=2)
    projected = pca.fit_transform(features)
    return projected, pca


def compute_contested_labels(
    values: np.ndarray,
    solver,
    eps: float = 0.5,
) -> np.ndarray:
    """Label items as contested for each test context.

    Item i is contested if solving knapsack(values + eps * e_i) gives a
    different solution than solving knapsack(values).
    """
    n_samples, n_arms = values.shape
    contested = np.zeros((n_samples, n_arms), dtype=bool)

    for i in range(n_samples):
        base_sol = solver(values[i])
        for j in range(n_arms):
            perturbed = values[i].copy()
            perturbed[j] += eps
            new_sol = solver(perturbed)
            if not np.array_equal(base_sol.astype(int), new_sol.astype(int)):
                contested[i, j] = True
    return contested


def contested_probe_accuracy(
    features: np.ndarray,
    contested: np.ndarray,
) -> float:
    """Train linear probe to predict contestedness from features.

    Pools all (context, item) pairs. Returns mean cross-validated accuracy.
    """
    n_samples, n_arms = contested.shape
    X_probe = np.tile(features, (n_arms, 1))
    arm_ids = np.repeat(np.eye(n_arms), n_samples, axis=0)
    X_probe = np.hstack([X_probe, arm_ids])
    y_probe = contested.T.flatten().astype(int)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    scores = cross_val_score(clf, X_probe, y_probe, cv=5, scoring='accuracy')
    return float(np.mean(scores))


def plot_pca_variance(
    cumulative_spo: np.ndarray,
    cumulative_mse: np.ndarray,
) -> 'ggplot':
    """Plot cumulative explained variance for SPO vs MSE features."""
    rows = []
    for i, val in enumerate(cumulative_spo):
        rows.append({'component': i + 1, 'cumulative_variance': float(val), 'agent': 'BayesianSPO'})
    for i, val in enumerate(cumulative_mse):
        rows.append({'component': i + 1, 'cumulative_variance': float(val), 'agent': 'BayesianMSE'})
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('component', 'cumulative_variance', color='agent'))
        + geom_line(size=1.5)
        + geom_point(size=3)
        + geom_hline(yintercept=0.95, linetype='dashed', color='gray')
        + scale_color_manual(values={'BayesianSPO': COLORS['BayesianSPO'], 'BayesianMSE': COLORS['BayesianMSE']})
        + labs(
            x='Principal Component',
            y='Cumulative Explained Variance',
            color='Agent',
            title='Feature Effective Dimensionality (knapsack_32)',
        )
        + theme_minimal()
        + ggsize(600, 400)
    )
    return p


def plot_decision_clusters(
    projected_spo: np.ndarray,
    projected_mse: np.ndarray,
    decision_labels: List[str],
) -> 'ggplot':
    """2-panel PCA scatter colored by decision identity for knapsack_32."""
    rows = []
    counts = Counter(decision_labels)
    top_decisions = {d for d, _ in counts.most_common(6)}

    for i, label in enumerate(decision_labels):
        display_label = label if label in top_decisions else 'other'
        rows.append({
            'pc1': float(projected_spo[i, 0]),
            'pc2': float(projected_spo[i, 1]),
            'decision': display_label,
            'agent': 'BayesianSPO',
        })
        rows.append({
            'pc1': float(projected_mse[i, 0]),
            'pc2': float(projected_mse[i, 1]),
            'decision': display_label,
            'agent': 'BayesianMSE',
        })
    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes('pc1', 'pc2', color=as_discrete('decision')))
        + geom_point(size=2, alpha=0.5)
        + facet_wrap('agent')
        + labs(
            x='PC 1',
            y='PC 2',
            color='Decision',
            title='Feature Clustering by Decision (knapsack_32)',
        )
        + theme_minimal()
        + ggsize(800, 400)
    )
    return p

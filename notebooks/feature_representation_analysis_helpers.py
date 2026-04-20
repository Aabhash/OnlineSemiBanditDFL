import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from lets_plot import (
    aes,
    geom_line,
    geom_path,
    geom_point,
    geom_ribbon,
    geom_segment,
    geom_tile,
    gggrid,
    ggplot,
    ggsize,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
    geom_abline,
)
from lets_plot.export import ggsave

from src.agents.base import run_agent_on_episode
from src.agents.contextual import BayesianMSEAgent, BayesianSPOAgent


DEFAULT_LR = 0.001
DEFAULT_BETA = 0.0
DEFAULT_LAMBDA_REG = 1.0
DEFAULT_HIDDEN_DIM = 1
DEFAULT_CONTEXT_DIM = 8
DEFAULT_N_ARMS = 4
DEFAULT_WEIGHTS = np.ones(DEFAULT_N_ARMS, dtype=float)
DEFAULT_CAPACITY = 1.0

DECISION_COLORS = {
    '0,2': '#1f77b4',
    '0,3': '#ff7f0e',
    '1,2': '#2ca02c',
    '1,3': '#d62728',
}


def default_representation_configs() -> List[Tuple[str, Dict]]:
    return [
        (
            'BayesianSPO (partial, masked)',
            dict(agent_name='BayesianSPO', full_feedback=False, mask_grad=True, phased=False),
        ),
        (
            'BayesianMSE (partial)',
            dict(agent_name='BayesianMSE', full_feedback=False, mask_grad=False, phased=False),
        ),
    ]


def top1_solver(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).flatten()
    action = np.zeros_like(values)
    action[int(np.argmax(values))] = 1.0
    return action


def _template_values(z_dec: np.ndarray) -> np.ndarray:
    z_dec = np.asarray(z_dec).reshape(-1, 1)
    centers = np.array([-1.8, -0.6, 0.6, 1.8], dtype=float)[None, :]
    heights = np.array([0.10, 0.40, 0.40, 0.10], dtype=float)[None, :]
    return heights - 0.95 * (z_dec - centers) ** 2


def _offset_and_scale(z_nui: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z_nui = np.asarray(z_nui)
    shared_offset = 2.6 * z_nui[:, 0]
    shared_scale = 1.4 + 0.45 * np.tanh(1.3 * z_nui[:, 1])
    return shared_offset, shared_scale


def _observed_context(z_dec: np.ndarray, z_nui: np.ndarray, noise_scale: float, rng) -> np.ndarray:
    z_dec = np.asarray(z_dec)
    z_nui = np.asarray(z_nui)
    x = np.column_stack(
        [
            z_dec + 0.10 * z_nui[:, 0],
            z_nui[:, 0] + 0.10 * z_dec,
            z_nui[:, 1] - 0.10 * z_dec,
            z_dec * z_nui[:, 0],
            np.sin(z_dec),
            np.sin(z_nui[:, 0]),
            np.cos(z_nui[:, 1]),
            np.tanh(z_dec - 0.5 * z_nui[:, 1]),
        ]
    )
    if noise_scale > 0.0:
        x = x + noise_scale * rng.normal(size=x.shape)
    return x


def make_latent_dataset(
    n_samples: int,
    seed: int,
    noise_scale: float = 0.05,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z_dec = rng.uniform(-2.4, 2.4, size=n_samples)
    z_nui = rng.normal(size=(n_samples, 2))
    template = _template_values(z_dec)
    shared_offset, shared_scale = _offset_and_scale(z_nui)
    values = shared_offset[:, None] + shared_scale[:, None] * template
    context = _observed_context(z_dec, z_nui, noise_scale, rng)
    optimal_arm = np.argmax(values, axis=1)
    sorted_vals = np.sort(values, axis=1)
    decision_margin = sorted_vals[:, -1] - sorted_vals[:, -2]
    return {
        'X': context,
        'Y': values,
        'z_dec': z_dec,
        'z_nui': z_nui,
        'nuisance_score': shared_offset,
        'shared_offset': shared_offset,
        'shared_scale': shared_scale,
        'optimal_arm': optimal_arm,
        'decision_margin': decision_margin,
    }


def make_episode_from_dataset(dataset: Dict[str, np.ndarray]) -> Dict:
    rounds_data = []
    for i in range(len(dataset['X'])):
        true_values = np.asarray(dataset['Y'][i], dtype=float)
        optimal_action = top1_solver(true_values)
        rounds_data.append(
            {
                'context': np.asarray(dataset['X'][i], dtype=float),
                'true_values': true_values,
                'optimal_action': optimal_action,
                'optimal_value': float(np.max(true_values)),
            }
        )
    return {'rounds_data': rounds_data, 'minimize': False}


def _make_agent(
    cfg: Dict,
    hidden_dim: int,
    nonlinear: bool,
    lr: float,
    beta: float,
    lambda_reg: float,
    seed: int,
):
    agent_name = cfg['agent_name']
    common = dict(
        context_dim=DEFAULT_CONTEXT_DIM,
        n_arms=DEFAULT_N_ARMS,
        weights=DEFAULT_WEIGHTS,
        capacity=DEFAULT_CAPACITY,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        minimize=False,
        solver=top1_solver,
        seed=seed,
        nonlinear=nonlinear,
        phased=cfg.get('phased', False),
        phase_length=cfg.get('phase_length', 50),
        phase_epochs=cfg.get('phase_epochs', 5),
    )
    if agent_name == 'BayesianSPO':
        return BayesianSPOAgent(**common, mask_grad=cfg.get('mask_grad', True))
    if agent_name == 'BayesianMSE':
        return BayesianMSEAgent(**common, mask_grad=cfg.get('mask_grad', False))
    raise ValueError(f'Unknown agent: {agent_name}')


def _extract_features(agent, X: np.ndarray) -> np.ndarray:
    features = np.asarray(agent._features(X))
    if features.ndim == 1:
        features = features[:, None]
    return features


def _extract_predictions(agent, X: np.ndarray) -> np.ndarray:
    return np.asarray([np.asarray(agent.predict_mean(x), dtype=float) for x in X])


def _with_intercept(features: np.ndarray) -> np.ndarray:
    return np.column_stack([features, np.ones(len(features))])


def _regression_probe_r2(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
) -> float:
    design = _with_intercept(train_features)
    weights, _, _, _ = np.linalg.lstsq(design, train_targets, rcond=None)
    preds = _with_intercept(test_features) @ weights
    numer = np.mean((preds - test_targets) ** 2)
    denom = np.var(test_targets)
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - numer / denom))


def _fit_linear_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> np.ndarray:
    weights, _, _, _ = np.linalg.lstsq(_with_intercept(train_features), train_targets, rcond=None)
    return weights


def _predict_linear_probe(
    test_features: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return _with_intercept(test_features) @ weights


def _multioutput_r2(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
) -> float:
    values = []
    for i in range(train_targets.shape[1]):
        values.append(
            _regression_probe_r2(train_features, train_targets[:, i], test_features, test_targets[:, i])
        )
    return float(np.mean(values))


def _classification_probe_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
) -> float:
    targets = np.eye(n_classes)[train_labels]
    weights, _, _, _ = np.linalg.lstsq(_with_intercept(train_features), targets, rcond=None)
    logits = _with_intercept(test_features) @ weights
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == test_labels))


def _action_separation_score(features: np.ndarray, labels: np.ndarray) -> float:
    centroids = []
    within = []
    for label in np.unique(labels):
        pts = features[labels == label]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0)
        centroids.append(center)
        within.append(np.mean(np.linalg.norm(pts - center, axis=1)))
    if len(centroids) < 2:
        return 0.0
    centroid_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            centroid_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    return float(np.mean(centroid_dists) / max(np.mean(within), 1e-8))


def _make_probe_summary(
    agent,
    probe_train: Dict[str, np.ndarray],
    probe_test: Dict[str, np.ndarray],
) -> Dict[str, float]:
    train_features = _extract_features(agent, probe_train['X'])
    test_features = _extract_features(agent, probe_test['X'])
    preds = _extract_predictions(agent, probe_test['X'])
    decision_probe_acc = _classification_probe_accuracy(
        train_features,
        probe_train['optimal_arm'],
        test_features,
        probe_test['optimal_arm'],
        DEFAULT_N_ARMS,
    )
    decision_r2 = _regression_probe_r2(
        train_features, probe_train['z_dec'], test_features, probe_test['z_dec']
    )
    nuisance_r2 = _multioutput_r2(
        train_features, probe_train['z_nui'], test_features, probe_test['z_nui']
    )
    nuisance_score_r2 = _regression_probe_r2(
        train_features, probe_train['nuisance_score'], test_features, probe_test['nuisance_score']
    )
    offset_r2 = _regression_probe_r2(
        train_features, probe_train['shared_offset'], test_features, probe_test['shared_offset']
    )
    scale_r2 = _regression_probe_r2(
        train_features, probe_train['shared_scale'], test_features, probe_test['shared_scale']
    )
    margin_r2 = _regression_probe_r2(
        train_features, probe_train['decision_margin'], test_features, probe_test['decision_margin']
    )
    value_probe_r2 = _multioutput_r2(
        train_features, probe_train['Y'], test_features, probe_test['Y']
    )
    value_mse = float(np.mean((preds - probe_test['Y']) ** 2))
    greedy_actions = np.argmax(preds, axis=1)
    decision_acc = float(np.mean(greedy_actions == probe_test['optimal_arm']))
    action_separation = _action_separation_score(test_features, probe_test['optimal_arm'])
    return {
        'probe_action_acc': decision_probe_acc,
        'probe_z_dec_r2': decision_r2,
        'probe_nuisance_r2': nuisance_r2,
        'probe_nuisance_score_r2': nuisance_score_r2,
        'probe_offset_r2': offset_r2,
        'probe_scale_r2': scale_r2,
        'probe_margin_r2': margin_r2,
        'probe_value_r2': value_probe_r2,
        'probe_value_mse': value_mse,
        'probe_greedy_acc': decision_acc,
        'probe_action_separation': action_separation,
    }


def _trajectory_dataset(
    varying: str,
    n_traces: int = 4,
    n_points: int = 61,
) -> Tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(123 if varying == 'decision' else 456)
    rows = []
    contexts = []
    if varying == 'decision':
        dec_values = np.linspace(-2.4, 2.4, n_points)
        nuisances = rng.normal(scale=0.9, size=(n_traces, 3))
        for trace_id in range(n_traces):
            for step, z_dec in enumerate(dec_values):
                z_nui = nuisances[trace_id]
                x = _observed_context(
                    np.array([z_dec]),
                    z_nui[None, :],
                    noise_scale=0.0,
                    rng=np.random.default_rng(trace_id + step),
                )[0]
                contexts.append(x)
                rows.append(
                    {
                        'trace_id': trace_id,
                        'step': step,
                        'varying': 'decision',
                        'z_dec': z_dec,
                        'nuisance_score': _offset_and_scale(z_nui[None, :])[0][0],
                    }
                )
    else:
        nuisance_values = np.linspace(-2.4, 2.4, n_points)
        decision_vals = np.linspace(-2.1, 2.1, n_traces)
        for trace_id, z_dec in enumerate(decision_vals):
            for step, s in enumerate(nuisance_values):
                z_nui = np.array([s, 0.7 * np.sin(s), -0.5 * np.cos(s)], dtype=float)
                x = _observed_context(
                    np.array([z_dec]),
                    z_nui[None, :],
                    noise_scale=0.0,
                    rng=np.random.default_rng(trace_id + step),
                )[0]
                contexts.append(x)
                rows.append(
                    {
                        'trace_id': trace_id,
                        'step': step,
                        'varying': 'nuisance',
                        'z_dec': z_dec,
                        'nuisance_score': _offset_and_scale(z_nui[None, :])[0][0],
                    }
                )
    return np.asarray(contexts, dtype=float), pd.DataFrame(rows)


def _trajectory_summary(agent) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outputs = []
    sensitivity_rows = []
    for varying in ('decision', 'nuisance'):
        contexts, meta = _trajectory_dataset(varying)
        features = _extract_features(agent, contexts)
        meta = meta.copy()
        meta['feature_1'] = features[:, 0]
        meta['feature_2'] = features[:, 1] if features.shape[1] > 1 else 0.0
        diffs = features[1:] - features[:-1]
        same_trace = meta['trace_id'].to_numpy()[1:] == meta['trace_id'].to_numpy()[:-1]
        path_length = float(np.mean(np.linalg.norm(diffs[same_trace], axis=1)))
        sensitivity_rows.append({'varying': varying, 'mean_step_feature_change': path_length})
        outputs.append(meta)
    traces = pd.concat(outputs, ignore_index=True)
    sensitivity = pd.DataFrame(sensitivity_rows)
    decision_change = float(
        sensitivity.loc[sensitivity['varying'] == 'decision', 'mean_step_feature_change'].iloc[0]
    )
    nuisance_change = float(
        sensitivity.loc[sensitivity['varying'] == 'nuisance', 'mean_step_feature_change'].iloc[0]
    )
    sensitivity['decision_to_nuisance_ratio'] = (
        decision_change / max(nuisance_change, 1e-8)
    )
    return traces, sensitivity


def world_overview(
    n_grid: int = 121,
    n_nuisance_samples: int = 256,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    z_grid = np.linspace(-2.4, 2.4, n_grid)
    rng = np.random.default_rng(99)
    nuisance = rng.normal(size=(n_nuisance_samples, 2))
    rows = []
    stability_rows = []
    for z_dec in z_grid:
        z_dec_rep = np.full(n_nuisance_samples, z_dec)
        shared_offset, shared_scale = _offset_and_scale(nuisance)
        values = shared_offset[:, None] + shared_scale[:, None] * _template_values(z_dec_rep)
        optimal = np.argmax(values, axis=1)
        for arm in range(DEFAULT_N_ARMS):
            rows.append(
                {
                    'z_dec': z_dec,
                    'arm': f'Arm {arm + 1}',
                    'mean_value': float(values[:, arm].mean()),
                    'optimal_prob': float(np.mean(optimal == arm)),
                }
            )
        counts = np.bincount(optimal, minlength=DEFAULT_N_ARMS)
        stability_rows.append(
            {
                'z_dec': z_dec,
                'dominant_arm_prob': float(counts.max() / counts.sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(stability_rows)


def run_representation_suite(
    configs: Iterable[Tuple[str, Dict]],
    n_rounds: int = 1500,
    n_seeds: int = 5,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    nonlinear: bool = True,
    lr: float = DEFAULT_LR,
    beta: float = DEFAULT_BETA,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    diagnostics_interval: int = 25,
    probe_train_size: int = 1000,
    probe_test_size: int = 1500,
) -> Dict[str, pd.DataFrame]:
    rounds_records = []
    seed_summary = []
    probe_records = []
    feature_views = []
    trace_records = []
    sensitivity_records = []

    plot_seed = 0

    for seed in range(n_seeds):
        train_data = make_latent_dataset(n_rounds, seed=1000 + seed)
        episode = make_episode_from_dataset(train_data)
        probe_train = make_latent_dataset(probe_train_size, seed=2000 + seed)
        probe_test = make_latent_dataset(probe_test_size, seed=3000 + seed)

        for label, cfg in configs:
            agent = _make_agent(
                cfg,
                hidden_dim=hidden_dim,
                nonlinear=nonlinear,
                lr=lr,
                beta=beta,
                lambda_reg=lambda_reg,
                seed=seed,
            )
            out = run_agent_on_episode(
                agent,
                episode,
                n_rounds=n_rounds,
                full_feedback=cfg.get('full_feedback', False),
                diagnostics=True,
                diagnostics_interval=diagnostics_interval,
            )
            diag_rounds = out['diagnostic_rounds']
            for idx, round_idx in enumerate(diag_rounds):
                rounds_records.append(
                    {
                        'agent': label,
                        'seed': seed,
                        'round': round_idx,
                        'cumulative_regret': float(out['cumulative_regret'][round_idx - 1]),
                        'instant_regret': float(out['regrets'][round_idx - 1]),
                        'decision_correct': float(out['decision_correct'][idx]),
                        'overall_mae': float(out['overall_mae'][idx]),
                        'contested_mae': float(out['contested_mae'][idx]),
                        'rank_corr': float(out['rank_corr'][idx]),
                    }
                )

            probe = _make_probe_summary(agent, probe_train, probe_test)
            probe_records.append({'agent': label, 'seed': seed, **probe})
            seed_summary.append(
                {
                    'agent': label,
                    'seed': seed,
                    'final_regret': float(out['final_regret']),
                    'final_probe_value_mse': probe['probe_value_mse'],
                    'final_probe_greedy_acc': probe['probe_greedy_acc'],
                }
            )

            traces, sensitivity = _trajectory_summary(agent)
            traces['agent'] = label
            traces['seed'] = seed
            trace_records.append(traces)
            sensitivity['agent'] = label
            sensitivity['seed'] = seed
            sensitivity_records.append(sensitivity)

            if seed == plot_seed:
                features = _extract_features(agent, probe_test['X'])
                train_features = _extract_features(agent, probe_train['X'])
                z_dec_probe = _fit_linear_probe(train_features, probe_train['z_dec'])
                offset_probe = _fit_linear_probe(train_features, probe_train['shared_offset'])
                decoded_z_dec = _predict_linear_probe(features, z_dec_probe)
                decoded_offset = _predict_linear_probe(features, offset_probe)
                for i in range(len(features)):
                    feature_views.append(
                        {
                            'agent': label,
                            'seed': seed,
                            'feature_1': float(features[i, 0]),
                            'feature_2': float(features[i, 1] if features.shape[1] > 1 else 0.0),
                            'optimal_arm': f"Arm {int(probe_test['optimal_arm'][i]) + 1}",
                            'z_dec': float(probe_test['z_dec'][i]),
                            'nuisance_score': float(probe_test['nuisance_score'][i]),
                            'shared_offset': float(probe_test['shared_offset'][i]),
                            'decision_margin': float(probe_test['decision_margin'][i]),
                            'decoded_z_dec': float(decoded_z_dec[i]),
                            'decoded_offset': float(decoded_offset[i]),
                        }
                    )

    rounds_df = pd.DataFrame(rounds_records)
    seed_summary_df = pd.DataFrame(seed_summary)
    probe_df = pd.DataFrame(probe_records)
    sensitivity_df = pd.concat(sensitivity_records, ignore_index=True)
    traces_df = pd.concat(trace_records, ignore_index=True)
    feature_df = pd.DataFrame(feature_views)

    train_summary = (
        seed_summary_df.groupby('agent', as_index=False)
        .agg(
            mean_final_regret=('final_regret', 'mean'),
            std_final_regret=('final_regret', 'std'),
            mean_probe_value_mse=('final_probe_value_mse', 'mean'),
            mean_probe_greedy_acc=('final_probe_greedy_acc', 'mean'),
        )
    )

    probe_summary = (
        probe_df.groupby('agent', as_index=False)
        .agg(
            mean_probe_action_acc=('probe_action_acc', 'mean'),
            mean_probe_z_dec_r2=('probe_z_dec_r2', 'mean'),
            mean_probe_nuisance_r2=('probe_nuisance_r2', 'mean'),
            mean_probe_nuisance_score_r2=('probe_nuisance_score_r2', 'mean'),
            mean_probe_offset_r2=('probe_offset_r2', 'mean'),
            mean_probe_scale_r2=('probe_scale_r2', 'mean'),
            mean_probe_margin_r2=('probe_margin_r2', 'mean'),
            mean_probe_value_r2=('probe_value_r2', 'mean'),
            mean_probe_value_mse=('probe_value_mse', 'mean'),
            mean_probe_greedy_acc=('probe_greedy_acc', 'mean'),
            mean_probe_action_separation=('probe_action_separation', 'mean'),
        )
    )

    sensitivity_summary = (
        sensitivity_df.groupby(['agent', 'varying'], as_index=False)
        .agg(
            mean_step_feature_change=('mean_step_feature_change', 'mean'),
            mean_ratio=('decision_to_nuisance_ratio', 'mean'),
        )
    )

    return {
        'rounds': rounds_df,
        'train_summary': train_summary,
        'probe_seed': probe_df,
        'probe_summary': probe_summary,
        'feature_views': feature_df,
        'traces': traces_df,
        'sensitivity_seed': sensitivity_df,
        'sensitivity_summary': sensitivity_summary,
    }


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    series = pd.Series(values)
    return series.rolling(window, min_periods=1, center=True).mean().to_numpy()


def plot_training_regret(rounds: pd.DataFrame, title: str, smooth_window: int = 11):
    if rounds.empty:
        return ggplot() + labs(title=title)
    summary = (
        rounds.groupby(['agent', 'round'], as_index=False)
        .agg(
            mean_regret=('cumulative_regret', 'mean'),
            std_regret=('cumulative_regret', 'std'),
        )
    )
    summary['std_regret'] = summary['std_regret'].fillna(0.0)
    summary['lower'] = summary['mean_regret'] - summary['std_regret']
    summary['upper'] = summary['mean_regret'] + summary['std_regret']
    parts = []
    for agent, df_agent in summary.groupby('agent'):
        df_agent = df_agent.sort_values('round').copy()
        df_agent['mean_regret'] = _smooth(df_agent['mean_regret'], smooth_window)
        df_agent['lower'] = _smooth(df_agent['lower'], smooth_window)
        df_agent['upper'] = _smooth(df_agent['upper'], smooth_window)
        parts.append(df_agent)
    plot_df = pd.concat(parts, ignore_index=True)
    return (
        ggplot(plot_df, aes('round', 'mean_regret', color='agent', fill='agent'))
        + geom_ribbon(aes(ymin='lower', ymax='upper'), alpha=0.16, color='transparent', size=0)
        + geom_line(size=1.1)
        + ggsize(700, 340)
        + labs(title=title, x='Round', y='Cumulative Regret')
        + theme(legend_position='top')
    )


def plot_world_value_curves(curves: pd.DataFrame):
    return (
        ggplot(curves, aes('z_dec', 'mean_value', color='arm'))
        + geom_line(size=1.1)
        + ggsize(650, 320)
        + labs(
            title='True Latent World: Mean Arm Value vs Decision Latent',
            x='Decision latent z_dec',
            y='Mean arm value',
            color='Arm',
        )
        + theme(legend_position='top')
    )


def plot_world_optimal_regions(curves: pd.DataFrame, stability: pd.DataFrame):
    left = (
        ggplot(curves, aes('z_dec', 'optimal_prob', color='arm'))
        + geom_line(size=1.1)
        + ggsize(650, 320)
        + labs(
            title='Optimal-Arm Probability Under Nuisance Variation',
            x='Decision latent z_dec',
            y='P(arm is optimal)',
            color='Arm',
        )
        + theme(legend_position='top')
    )
    right = (
        ggplot(stability, aes('z_dec', 'dominant_arm_prob'))
        + geom_line(size=1.1)
        + ggsize(650, 320)
        + labs(
            title='Decision Stability Under Nuisance Variation',
            x='Decision latent z_dec',
            y='Dominant arm probability',
        )
    )
    return gggrid([left, right], ncol=2)


def plot_feature_scatter(feature_views: pd.DataFrame, color_by: str, title: str):
    plots = []
    for agent, df_agent in feature_views.groupby('agent'):
        plots.append(
            ggplot(df_agent, aes('feature_1', 'feature_2', color=color_by))
            + geom_point(alpha=0.55, size=2.0)
            + ggsize(340, 320)
            + labs(title=agent, x='Feature 1', y='Feature 2', color=color_by.replace('_', ' ').title())
        )
    return gggrid(plots, ncol=len(plots)) + labs(title=title)


def plot_representation_story(feature_views: pd.DataFrame):
    plots = []
    for color_by, subtitle in (
        ('optimal_arm', 'Optimal arm'),
        ('nuisance_score', 'Shared offset'),
    ):
        for agent, df_agent in feature_views.groupby('agent'):
            plots.append(
                ggplot(df_agent, aes('feature_1', 'feature_2', color=color_by))
                + geom_point(alpha=0.60, size=1.8)
                + ggsize(320, 300)
                + labs(
                    title=f'{agent}: {subtitle}',
                    x='Feature 1',
                    y='Feature 2',
                    color=subtitle,
                )
            )
    return gggrid(plots, ncol=2) + labs(
        title='Learned bottleneck: decision regions vs nuisance structure'
    )


def plot_decoded_latent_story(feature_views: pd.DataFrame):
    plots = []
    specs = [
        ('z_dec', 'decoded_z_dec', 'Decision latent', 'Decoded decision latent'),
        ('shared_offset', 'decoded_offset', 'Shared offset', 'Decoded shared offset'),
    ]
    for true_col, pred_col, axis_label, title_suffix in specs:
        for agent, df_agent in feature_views.groupby('agent'):
            low = float(min(df_agent[true_col].min(), df_agent[pred_col].min()))
            high = float(max(df_agent[true_col].max(), df_agent[pred_col].max()))
            line_df = pd.DataFrame({'x': [low, high], 'y': [low, high]})
            plots.append(
                ggplot()
                + geom_line(data=line_df, mapping=aes('x', 'y'), color='#999999', alpha=0.6, size=1.0)
                + geom_point(
                    data=df_agent,
                    mapping=aes(true_col, pred_col, color='optimal_arm'),
                    alpha=0.55,
                    size=1.8,
                )
                + ggsize(320, 300)
                + labs(
                    title=f'{agent}: {title_suffix}',
                    x=f'True {axis_label}',
                    y=f'Decoded {axis_label}',
                    color='Optimal arm',
                )
            )
    return gggrid(plots, ncol=2) + labs(
        title='What the bottleneck linearly encodes'
    )


def plot_feature_trajectories(traces: pd.DataFrame, varying: str, title: str):
    plots = []
    subset = traces[traces['varying'] == varying]
    for agent, df_agent in subset.groupby('agent'):
        df_agent = df_agent.copy()
        df_agent['trace_label'] = df_agent['trace_id'].astype(str)
        plots.append(
            ggplot(df_agent, aes('feature_1', 'feature_2', color='trace_label', group='trace_id'))
            + geom_path(size=1.0, alpha=0.85)
            + geom_point(size=1.6, alpha=0.75)
            + ggsize(340, 320)
            + labs(title=agent, x='Feature 1', y='Feature 2', color='Trace')
        )
    return gggrid(plots, ncol=len(plots)) + labs(title=title)


def _metric_panel_df(
    probe_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    group: str,
) -> pd.DataFrame:
    sensitivity_wide = (
        sensitivity_summary.pivot(index='agent', columns='varying', values='mean_step_feature_change')
        .reset_index()
    )
    ratio_df = (
        sensitivity_summary[sensitivity_summary['varying'] == 'decision'][['agent', 'mean_ratio']]
        .rename(columns={'mean_ratio': 'decision_to_nuisance_ratio'})
    )
    summary = probe_summary.merge(sensitivity_wide, on='agent', how='left').merge(
        ratio_df, on='agent', how='left'
    )

    if group == 'decision':
        metric_map = [
            ('Action probe accuracy', 'mean_probe_action_acc'),
            ('Decision latent R²', 'mean_probe_z_dec_r2'),
            ('Action separation', 'mean_probe_action_separation'),
            ('Decision / nuisance sensitivity', 'decision_to_nuisance_ratio'),
        ]
    else:
        metric_map = [
            ('Full-value probe R²', 'mean_probe_value_r2'),
            ('Offset probe R²', 'mean_probe_offset_r2'),
            ('Value MSE', 'mean_probe_value_mse'),
            ('Nuisance latent R²', 'mean_probe_nuisance_r2'),
        ]

    rows = []
    for metric_label, col in metric_map:
        for _, row in summary.iterrows():
            rows.append(
                {
                    'metric': metric_label,
                    'agent': row['agent'],
                    'value': float(row[col]),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_panels(
    probe_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    group: str,
    title: str,
):
    metric_df = _metric_panel_df(probe_summary, sensitivity_summary, group=group)
    plots = []
    for metric, df_metric in metric_df.groupby('metric'):
        xmin = float(df_metric['value'].min())
        xmax = float(df_metric['value'].max())
        pad = max((xmax - xmin) * 0.15, 0.02)
        segment_df = pd.DataFrame(
            {'x': [xmin], 'xend': [xmax], 'y': [0.0], 'yend': [0.0]}
        )
        plot_df = df_metric.copy()
        plot_df['y'] = 0.0
        plots.append(
            ggplot()
            + geom_segment(
                data=segment_df,
                mapping=aes(x='x', xend='xend', y='y', yend='yend'),
                size=1.2,
                alpha=0.35,
            )
            + geom_point(
                data=plot_df,
                mapping=aes('value', 'y', color='agent'),
                size=4.0,
                alpha=0.9,
            )
            + ggsize(280, 135)
            + labs(title=metric, x='', y='')
            + theme(
                legend_position='none',
                axis_ticks_y='blank',
                axis_text_y='blank',
                axis_title_y='blank',
            )
        )
    return gggrid(plots, ncol=2) + labs(title=title)


def plot_metric_story(
    probe_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    group: str,
    title: str,
):
    metric_df = _metric_panel_df(probe_summary, sensitivity_summary, group=group)
    if metric_df.empty:
        return ggplot() + labs(title=title)
    agents = list(metric_df['agent'].unique())
    if len(agents) != 2:
        return ggplot(metric_df, aes('value', 'metric', color='agent')) + geom_point(size=4) + labs(title=title)

    wide = metric_df.pivot(index='metric', columns='agent', values='value').reset_index()
    wide = wide.rename(columns={agents[0]: 'left_value', agents[1]: 'right_value'})
    long = metric_df.copy()
    return (
        ggplot()
        + geom_segment(
            data=wide,
            mapping=aes(x='left_value', xend='right_value', y='metric', yend='metric'),
            size=1.3,
            alpha=0.35,
        )
        + geom_point(
            data=long,
            mapping=aes('value', 'metric', color='agent'),
            size=4.5,
            alpha=0.9,
        )
        + ggsize(640, 240)
        + labs(title=title, x='', y='', color='Agent')
        + theme(legend_position='top')
    )


def plot_sensitivity_summary(sensitivity_summary: pd.DataFrame):
    df = sensitivity_summary.copy()
    plots = []
    for metric_label, value_col in (
        ('Decision-latent traversal', 'decision'),
        ('Nuisance traversal', 'nuisance'),
    ):
        pass
    pivot = (
        df.pivot(index='agent', columns='varying', values='mean_step_feature_change')
        .reset_index()
    )
    long = pivot.melt(id_vars='agent', value_vars=['decision', 'nuisance'], var_name='varying', value_name='value')
    long['varying'] = long['varying'].map(
        {'decision': 'Decision traversal', 'nuisance': 'Nuisance traversal'}
    )
    left = (
        ggplot(long, aes('varying', 'value', color='agent', group='agent'))
        + geom_line(size=1.0, alpha=0.65)
        + geom_point(size=3.5)
        + ggsize(360, 260)
        + labs(
            title='Mean feature movement by traversal type',
            x='',
            y='Mean step feature change',
            color='Agent',
        )
        + theme(legend_position='top')
    )
    ratio_df = (
        sensitivity_summary[sensitivity_summary['varying'] == 'decision'][['agent', 'mean_ratio']]
        .rename(columns={'mean_ratio': 'ratio'})
    )
    right = (
        ggplot(ratio_df, aes('agent', 'ratio', color='agent'))
        + geom_point(size=4.5)
        + ggsize(260, 260)
        + labs(
            title='Decision / nuisance sensitivity ratio',
            x='',
            y='Ratio',
            color='Agent',
        )
        + theme(legend_position='none')
    )
    return gggrid([left, right], ncol=2)


def print_regret_summary(summary: pd.DataFrame):
    for _, row in summary.sort_values('mean_final_regret').iterrows():
        std = row['std_final_regret']
        std_str = '0.0' if pd.isna(std) else f'{std:.1f}'
        print(f"{row['agent']:<24}: regret = {row['mean_final_regret']:.1f} +/- {std_str}")


LATENT_CONTEXT_DIM = 8
LATENT_N_ARMS = 6
LATENT_K = 2
LATENT_HIDDEN_DIM = 2
LATENT_BETA = 1.0
LATENT_WEIGHTS = np.ones(LATENT_N_ARMS, dtype=float)
LATENT_CAPACITY = float(LATENT_K)
DECISION_CENTERS = np.array([-1.8, 0.0, 1.8], dtype=float)
REGIME_NAMES = np.array(['Regime A', 'Regime B', 'Regime C'])
SET_NAMES = np.array(['{1,2}', '{3,4}', '{5,6}'])
PAIR_REGIME_ADJUST = 0.95 * np.array(
    [
        [1.0, -1.0],
        [-1.0, 1.0],
        [0.0, 0.0],
    ],
    dtype=float,
)
PAIR_BIAS = np.array([0.15, -0.15, 0.15, -0.15, 0.15, -0.15], dtype=float)
PAIR_INDEX = np.array([0, 0, 1, 1, 2, 2], dtype=int)


def latent_representation_configs() -> List[Tuple[str, Dict]]:
    return [
        (
            'BayesianSPO (partial, masked)',
            dict(agent_name='BayesianSPO', full_feedback=False, mask_grad=True, phased=False),
        ),
        (
            'BayesianMSE (partial)',
            dict(agent_name='BayesianMSE', full_feedback=False, mask_grad=False, phased=False),
        ),
    ]


def topk_solver(values: np.ndarray, k: int = LATENT_K) -> np.ndarray:
    values = np.asarray(values).flatten()
    idx = np.argsort(values)[-k:]
    action = np.zeros_like(values)
    action[idx] = 1.0
    return action


def _decision_group_scores(z_dec: np.ndarray) -> np.ndarray:
    z_dec = np.asarray(z_dec).reshape(-1, 1)
    return 4.6 - 1.55 * (z_dec - DECISION_CENTERS[None, :]) ** 2


def _context_from_latents(z_dec: np.ndarray, regime_idx: np.ndarray, noise_scale: float, rng) -> np.ndarray:
    z_dec = np.asarray(z_dec)
    regime_idx = np.asarray(regime_idx)
    regime_embed = np.array(
        [
            [1.0, -0.3, -0.8],
            [-0.6, 1.0, -0.2],
            [-0.4, -0.7, 1.0],
        ],
        dtype=float,
    )[regime_idx]
    x = np.column_stack(
        [
            z_dec + 0.20 * regime_embed[:, 0],
            np.sin(z_dec) + 0.25 * regime_embed[:, 1],
            np.cos(z_dec) + 0.25 * regime_embed[:, 2],
            regime_embed[:, 0] + 0.15 * z_dec,
            regime_embed[:, 1] - 0.10 * z_dec,
            regime_embed[:, 2] + 0.10 * z_dec ** 2,
            z_dec * regime_embed[:, 0],
            z_dec * regime_embed[:, 1],
        ]
    )
    if noise_scale > 0.0:
        x = x + noise_scale * rng.normal(size=x.shape)
    return x


def make_decision_equivalence_dataset(
    n_samples: int,
    seed: int,
    noise_scale: float = 0.04,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    decision_group = rng.integers(0, 3, size=n_samples)
    regime_idx = rng.integers(0, 3, size=n_samples)
    z_dec = rng.normal(loc=DECISION_CENTERS[decision_group], scale=0.22, size=n_samples)
    group_scores = _decision_group_scores(z_dec)
    base_values = group_scores[:, PAIR_INDEX] + PAIR_BIAS[None, :]
    values = base_values.copy()
    pair_cols = np.array([[0, 1], [2, 3], [4, 5]], dtype=int)
    for i in range(n_samples):
        cols = pair_cols[decision_group[i]]
        values[i, cols] += PAIR_REGIME_ADJUST[regime_idx[i]]
    context = _context_from_latents(z_dec, regime_idx, noise_scale, rng)
    selected_set = np.argmax(group_scores, axis=1)
    return {
        'X': context,
        'Y': values,
        'z_dec': z_dec,
        'decision_group': decision_group,
        'regime_idx': regime_idx,
        'selected_set': selected_set,
        'selected_set_label': SET_NAMES[selected_set],
        'regime_label': REGIME_NAMES[regime_idx],
    }


def make_topk_episode(dataset: Dict[str, np.ndarray]) -> Dict:
    rounds_data = []
    for i in range(len(dataset['X'])):
        true_values = np.asarray(dataset['Y'][i], dtype=float)
        optimal_action = topk_solver(true_values)
        rounds_data.append(
            {
                'context': np.asarray(dataset['X'][i], dtype=float),
                'true_values': true_values,
                'optimal_action': optimal_action,
                'optimal_value': float(np.sum(true_values[optimal_action > 0.5])),
            }
        )
    return {'rounds_data': rounds_data, 'minimize': False}


def _make_latent_agent(
    cfg: Dict,
    hidden_dim: int,
    nonlinear: bool,
    lr: float,
    beta: float,
    lambda_reg: float,
    seed: int,
):
    common = dict(
        context_dim=LATENT_CONTEXT_DIM,
        n_arms=LATENT_N_ARMS,
        weights=LATENT_WEIGHTS,
        capacity=LATENT_CAPACITY,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        minimize=False,
        solver=topk_solver,
        seed=seed,
        nonlinear=nonlinear,
        phased=cfg.get('phased', False),
        phase_length=cfg.get('phase_length', 50),
        phase_epochs=cfg.get('phase_epochs', 5),
    )
    if cfg['agent_name'] == 'BayesianSPO':
        return BayesianSPOAgent(**common, mask_grad=cfg.get('mask_grad', True))
    if cfg['agent_name'] == 'BayesianMSE':
        return BayesianMSEAgent(**common, mask_grad=cfg.get('mask_grad', False))
    raise ValueError(f"Unknown agent: {cfg['agent_name']}")


def run_latent_representation_suite(
    configs: Iterable[Tuple[str, Dict]],
    n_rounds: int = 2500,
    n_seeds: int = 3,
    hidden_dim: int = LATENT_HIDDEN_DIM,
    nonlinear: bool = True,
    lr: float = DEFAULT_LR,
    beta: float = LATENT_BETA,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    probe_size: int = 1800,
) -> Dict[str, pd.DataFrame]:
    regret_rows = []
    feature_rows = []

    for seed in range(n_seeds):
        train_data = make_decision_equivalence_dataset(n_rounds, seed=1000 + seed)
        episode = make_topk_episode(train_data)
        probe_data = make_decision_equivalence_dataset(probe_size, seed=2000 + seed, noise_scale=0.0)

        if seed == 0:
            for i in range(len(probe_data['X'])):
                feature_rows.append(
                    {
                        'view': 'original',
                        'seed': seed,
                        'selected_set': probe_data['selected_set_label'][i],
                        'regime': probe_data['regime_label'][i],
                        'regime_order': int(probe_data['regime_idx'][i]),
                        'z_dec': float(probe_data['z_dec'][i]),
                    }
                )

        for label, cfg in configs:
            agent = _make_latent_agent(
                cfg,
                hidden_dim=hidden_dim,
                nonlinear=nonlinear,
                lr=lr,
                beta=beta,
                lambda_reg=lambda_reg,
                seed=seed,
            )
            result = run_agent_on_episode(
                agent,
                episode,
                n_rounds=n_rounds,
                full_feedback=cfg.get('full_feedback', False),
                diagnostics=False,
            )
            regret_rows.append(
                {
                    'agent': label,
                    'seed': seed,
                    'final_regret': float(result['final_regret']),
                }
            )

            if seed == 0:
                pass

            features = _extract_features(agent, probe_data['X'])
            centroid_df = pd.DataFrame(
                {
                    'selected_set': probe_data['selected_set_label'],
                    'regime': probe_data['regime_label'],
                    'regime_order': probe_data['regime_idx'],
                    'feature_1': features[:, 0],
                    'feature_2': features[:, 1] if features.shape[1] > 1 else 0.0,
                }
            )
            centroids = (
                centroid_df.groupby(['selected_set', 'regime', 'regime_order'], as_index=False)
                .agg(
                    feature_1=('feature_1', 'mean'),
                    feature_2=('feature_2', 'mean'),
                )
            )
            for _, row in centroids.iterrows():
                feature_rows.append(
                    {
                        'view': 'learned_centroid',
                        'agent': label,
                        'seed': seed,
                        'selected_set': row['selected_set'],
                        'regime': row['regime'],
                        'regime_order': int(row['regime_order']),
                        'feature_1': float(row['feature_1']),
                        'feature_2': float(row['feature_2']),
                    }
                )

    regret_df = pd.DataFrame(regret_rows)
    regret_summary = (
        regret_df.groupby('agent', as_index=False)
        .agg(
            mean_final_regret=('final_regret', 'mean'),
            std_final_regret=('final_regret', 'std'),
        )
    )
    return {
        'feature_views': pd.DataFrame(feature_rows),
        'regret_seed': regret_df,
        'regret_summary': regret_summary,
    }


def plot_original_latent(feature_views: pd.DataFrame):
    original_df = feature_views[feature_views['view'] == 'original'].copy()
    centroid_df = (
        original_df.groupby(['selected_set', 'regime', 'regime_order'], as_index=False)
        .agg(z_dec=('z_dec', 'mean'))
    )
    return (
        ggplot()
        + geom_point(
            data=original_df,
            mapping=aes('z_dec', 'regime', color='selected_set'),
            alpha=0.28,
            size=1.8,
        )
        + geom_point(
            data=centroid_df,
            mapping=aes('z_dec', 'regime', color='selected_set', shape='regime'),
            alpha=0.95,
            size=4.2,
        )
        + ggsize(620, 300)
        + labs(
            title='Original latent world',
            x='True decision latent z_dec',
            y='Nuisance regime',
            color='Selected set',
            shape='Regime',
        )
        + theme(legend_position='right')
    )


def plot_learned_latent(feature_views: pd.DataFrame, regret_summary: pd.DataFrame | None = None):
    centroid_df = feature_views[feature_views['view'] == 'learned_centroid'].copy()
    centroid_df['path_group'] = centroid_df['selected_set'] + ' / seed ' + centroid_df['seed'].astype(str)
    mean_df = (
        centroid_df.groupby(['agent', 'selected_set', 'regime', 'regime_order'], as_index=False)
        .agg(
            feature_1=('feature_1', 'mean'),
            feature_2=('feature_2', 'mean'),
        )
    )
    mean_df['path_group'] = mean_df['selected_set']
    plots = []
    regret_map = {}
    if regret_summary is not None and not regret_summary.empty:
        for _, row in regret_summary.iterrows():
            std = row['std_final_regret']
            std_str = '0.0' if pd.isna(std) else f'{std:.0f}'
            regret_map[row['agent']] = f"{row['mean_final_regret']:.0f} +/- {std_str}"
    for agent, df_mean in mean_df.groupby('agent'):
        df_mean = df_mean.sort_values(['selected_set', 'regime_order']).copy()
        df_seed = centroid_df[centroid_df['agent'] == agent].sort_values(
            ['selected_set', 'seed', 'regime_order']
        )
        title = agent
        if agent in regret_map:
            title = f'{agent}\nfinal regret = {regret_map[agent]}'
        plots.append(
            ggplot()
            + geom_path(
                data=df_seed,
                mapping=aes('feature_1', 'feature_2', color='selected_set', group='path_group'),
                alpha=0.18,
                size=1.0,
            )
            + geom_path(
                data=df_mean,
                mapping=aes('feature_1', 'feature_2', color='selected_set', group='path_group'),
                alpha=0.95,
                size=1.7,
            )
            + geom_point(
                data=df_seed,
                mapping=aes('feature_1', 'feature_2', color='selected_set', shape='regime'),
                alpha=0.28,
                size=2.2,
            )
            + geom_point(
                data=df_mean,
                mapping=aes('feature_1', 'feature_2', color='selected_set', shape='regime'),
                size=5.2,
                alpha=0.98,
            )
            + ggsize(420, 360)
            + labs(
                title=title,
                x='Feature 1',
                y='Feature 2',
                color='Selected set',
                shape='Regime',
            )
            + theme(legend_position='right')
        )
    return gggrid(plots, ncol=2) + labs(
        title='Learned latent: mean regime trajectories within each selected-set cluster'
    )


# ── Oracle-Aligned Mechanism Experiment ───────────────────────────────────


from notebooks.feature_representation_analysis_claude_helpers import (
    DecisionFocusedEnv,
    compute_oracle_decision,
    create_agent_pair,
    train_with_checkpoints,
    extract_features,
    generate_test_contexts,
    decision_label,
    alignment_over_training,
    plot_alignment_trajectory,
    plot_feature_directions,
)


def oracle_agreement_summary(
    n_verify: int = 200,
    seed: int = 42,
    context_seed: int = 123,
    noise_width: float = 0.0,
) -> pd.DataFrame:
    env = DecisionFocusedEnv(seed=seed, noise_width=noise_width)
    rng = np.random.default_rng(context_seed)
    matches = 0
    for _ in range(n_verify):
        x = rng.standard_normal(env.context_dim)
        v = env.data._generate_values(x)
        oracle_sel = compute_oracle_decision(x)
        solver_sel = env.solve(v)
        if np.array_equal(oracle_sel.astype(int), solver_sel.astype(int)):
            matches += 1
    return pd.DataFrame(
        [
            {
                'noise_width': noise_width,
                'oracle_matches': matches,
                'n_verify': n_verify,
                'agreement_rate': matches / n_verify,
            }
        ]
    )


def plot_decision_dgp_overview(noise_width: float = 0.0):
    from lets_plot import scale_color_manual

    env = DecisionFocusedEnv(seed=42, noise_width=noise_width)
    grid = np.linspace(-2.5, 2.5, 121)
    item_colors = {
        'Item 0': '#1f77b4',
        'Item 1': '#4e79a7',
        'Item 2': '#d62728',
        'Item 3': '#e15759',
    }

    def sweep_df(vary_dim: int, fixed_x: np.ndarray, x_label: str) -> pd.DataFrame:
        rows = []
        for value in grid:
            x = fixed_x.copy()
            x[vary_dim] = value
            vals = env.data._generate_values(x)
            for item_idx, item_value in enumerate(vals):
                rows.append(
                    {
                        'x_value': float(value),
                        'item': f'Item {item_idx}',
                        'value': float(item_value),
                    }
                )
        return pd.DataFrame(rows), x_label

    df_x0, x0_label = sweep_df(0, np.zeros(5), 'Vary x0: pair (0,1) flips')
    df_x1, x1_label = sweep_df(1, np.zeros(5), 'Vary x1: pair (2,3) flips')
    fixed = np.array([0.9, -0.9, 0.0, 0.0, 0.0], dtype=float)
    df_x2, x2_label = sweep_df(
        2,
        fixed,
        'Vary x2 / x3 / x4: all item values shift together',
    )

    plots = []
    for title, df in (
        (x0_label, df_x0),
        (x1_label, df_x1),
        (x2_label, df_x2),
    ):
        plots.append(
            ggplot(df, aes('x_value', 'value', color='item'))
            + geom_line(size=1.3)
            + scale_color_manual(values=item_colors)
            + ggsize(320, 260)
            + labs(
                title=title,
                x='Input value',
                y='Item value',
                color='Item',
            )
            + theme(legend_position='right')
        )

    return gggrid(plots, ncol=3) + labs(
        title='How the synthetic world works'
    )


def plot_decision_pair_schematic():
    from lets_plot import (
        coord_cartesian,
        element_blank,
        element_text,
        geom_rect,
        geom_text,
        scale_fill_manual,
        theme_minimal,
    )

    pair_boxes = pd.DataFrame(
        [
            {'xmin': 0.4, 'xmax': 2.6, 'ymin': 1.20, 'ymax': 2.65},
            {'xmin': 3.4, 'xmax': 5.6, 'ymin': 1.20, 'ymax': 2.65},
        ]
    )
    context_boxes = pd.DataFrame(
        [
            {'xmin': 1.00, 'xmax': 1.45, 'ymin': 3.35, 'ymax': 3.85, 'fill': 'decision_ctx'},
            {'xmin': 1.55, 'xmax': 2.00, 'ymin': 3.35, 'ymax': 3.85, 'fill': 'decision_ctx'},
            {'xmin': 2.35, 'xmax': 2.80, 'ymin': 3.35, 'ymax': 3.85, 'fill': 'nuisance_ctx'},
            {'xmin': 2.90, 'xmax': 3.35, 'ymin': 3.35, 'ymax': 3.85, 'fill': 'nuisance_ctx'},
            {'xmin': 3.45, 'xmax': 3.90, 'ymin': 3.35, 'ymax': 3.85, 'fill': 'nuisance_ctx'},
        ]
    )
    item_boxes = pd.DataFrame(
        [
            {'xmin': 0.75, 'xmax': 1.5, 'ymin': 1.65, 'ymax': 2.25, 'fill': 'item'},
            {'xmin': 1.5, 'xmax': 2.25, 'ymin': 1.65, 'ymax': 2.25, 'fill': 'item'},
            {'xmin': 3.75, 'xmax': 4.5, 'ymin': 1.65, 'ymax': 2.25, 'fill': 'item'},
            {'xmin': 4.5, 'xmax': 5.25, 'ymin': 1.65, 'ymax': 2.25, 'fill': 'item'},
            {'xmin': 0.7, 'xmax': 5.3, 'ymin': -0.96, 'ymax': -0.28, 'fill': 'nuisance'},
        ]
    )
    text = pd.DataFrame(
        [
            {'x': 0.55, 'y': 3.60, 'label': 'Context x =', 'kind': 'context_label'},
            {'x': 1.5, 'y': 2.84, 'label': 'Pair 1', 'kind': 'title'},
            {'x': 4.5, 'y': 2.84, 'label': 'Pair 2', 'kind': 'title'},
            {'x': 1.5, 'y': 0.90, 'label': 'Pick one winner', 'kind': 'body'},
            {'x': 1.5, 'y': 0.66, 'label': 'from pair (0,1)', 'kind': 'body'},
            {'x': 1.5, 'y': 0.40, 'label': 'x0 decides which item wins', 'kind': 'body'},
            {'x': 1.5, 'y': 0.18, 'label': 'inside pair (0,1)', 'kind': 'body'},
            {'x': 4.5, 'y': 0.90, 'label': 'Pick one winner', 'kind': 'body'},
            {'x': 4.5, 'y': 0.66, 'label': 'from pair (2,3)', 'kind': 'body'},
            {'x': 4.5, 'y': 0.40, 'label': 'x1 decides which item wins', 'kind': 'body'},
            {'x': 4.5, 'y': 0.18, 'label': 'inside pair (2,3)', 'kind': 'body'},
            {'x': 3.0, 'y': -0.50, 'label': 'x2, x3, x4 shift overall item values', 'kind': 'body'},
            {'x': 3.0, 'y': -0.72, 'label': 'but do not change the winner inside either pair', 'kind': 'body'},
        ]
    )
    context_labels = pd.DataFrame(
        [
            {'x': 1.225, 'y': 3.60, 'label': 'x0'},
            {'x': 1.775, 'y': 3.60, 'label': 'x1'},
            {'x': 2.575, 'y': 3.60, 'label': 'x2'},
            {'x': 3.125, 'y': 3.60, 'label': 'x3'},
            {'x': 3.675, 'y': 3.60, 'label': 'x4'},
        ]
    )
    item_labels = pd.DataFrame(
        [
            {'x': 1.1, 'y': 1.86, 'label': '0'},
            {'x': 1.9, 'y': 1.86, 'label': '1'},
            {'x': 4.1, 'y': 1.86, 'label': '2'},
            {'x': 4.9, 'y': 1.86, 'label': '3'},
        ]
    )

    return (
        ggplot()
        + geom_rect(
            data=pair_boxes,
            mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax'),
            fill=None,
            color='#6b7280',
            size=1.2,
        )
        + geom_rect(
            data=context_boxes,
            mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'),
            color='#374151',
            size=0.9,
            alpha=0.45,
        )
        + geom_rect(
            data=item_boxes,
            mapping=aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='fill'),
            color='#374151',
            size=1.0,
            alpha=0.35,
        )
        + geom_text(
            data=context_labels,
            mapping=aes('x', 'y', label='label'),
            size=10,
            fontface='bold',
        )
        + geom_text(data=item_labels, mapping=aes('x', 'y', label='label'), size=18, fontface='bold')
        + geom_text(
            data=text[text['kind'] == 'title'],
            mapping=aes('x', 'y', label='label'),
            size=16,
            fontface='bold',
        )
        + geom_text(
            data=text[text['kind'] == 'body'],
            mapping=aes('x', 'y', label='label'),
            size=10,
        )
        + geom_text(
            data=text[text['kind'] == 'context_label'],
            mapping=aes('x', 'y', label='label'),
            size=10,
            fontface='bold',
        )
        + scale_fill_manual(
            values={
                'item': '#93c5fd',
                'nuisance': '#fde68a',
                'decision_ctx': '#93c5fd',
                'nuisance_ctx': '#fde68a',
            }
        )
        + coord_cartesian(xlim=(0.0, 6.0), ylim=(-1.08, 4.15))
        + ggsize(920, 520)
        + labs(title='How the decision problem is structured')
        + theme_minimal()
        + theme(
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            axis_title=element_blank(),
            panel_grid=element_blank(),
            legend_position='none',
            plot_title=element_text(size=18, face='bold'),
        )
    )


def _cost_correlation_metric(features: np.ndarray, total_cost: np.ndarray) -> float:
    max_corr = 0.0
    for j in range(features.shape[1]):
        r = np.corrcoef(features[:, j], total_cost)[0, 1]
        if np.isfinite(r):
            max_corr = max(max_corr, abs(r))
    return float(max_corr)


def _predict_mean_matrix(agent, X_test: np.ndarray) -> np.ndarray:
    rows = []
    for x in X_test:
        rows.append(np.asarray(agent.predict_mean(x), dtype=float))
    return np.vstack(rows)


def _safe_decision_probe_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    _, counts = np.unique(labels, return_counts=True)
    min_count = int(counts.min())
    if min_count < 2:
        return float('nan')
    n_splits = min(5, min_count)
    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(clf, features, labels, cv=cv)
    return float(scores.mean())


def _pair_decision_metrics(
    pred_values: np.ndarray,
    true_values: np.ndarray,
    solver,
    contested_quantile: float = 0.25,
) -> Dict[str, float]:
    true_sel = np.vstack([np.asarray(solver(v), dtype=float) for v in true_values])
    pred_sel = np.vstack([np.asarray(solver(v), dtype=float) for v in pred_values])
    greedy_decision_acc = float(np.mean(np.all(pred_sel == true_sel, axis=1)))

    pair_indices = [('pair_01', (0, 1)), ('pair_23', (2, 3))]
    pair_correct = []
    pair_margins = []
    per_pair_correct = {name: [] for name, _ in pair_indices}
    for pred_row, true_row in zip(pred_values, true_values):
        for name, (i, j) in pair_indices:
            pred_winner = i if pred_row[i] >= pred_row[j] else j
            true_winner = i if true_row[i] >= true_row[j] else j
            correct = float(pred_winner == true_winner)
            pair_correct.append(correct)
            pair_margins.append(float(abs(true_row[i] - true_row[j])))
            per_pair_correct[name].append(correct)
    pair_correct = np.asarray(pair_correct, dtype=float)
    pair_margins = np.asarray(pair_margins, dtype=float)
    pair_acc = float(pair_correct.mean())
    threshold = float(np.quantile(pair_margins, contested_quantile))
    contested_mask = pair_margins <= threshold
    contested_pair_acc = float(pair_correct[contested_mask].mean()) if np.any(contested_mask) else float('nan')
    return {
        'greedy_decision_acc': greedy_decision_acc,
        'pair_decision_acc': pair_acc,
        'contested_pair_acc': contested_pair_acc,
        'pair_01_acc': float(np.mean(per_pair_correct['pair_01'])),
        'pair_23_acc': float(np.mean(per_pair_correct['pair_23'])),
    }


def _mechanism_summary_table(
    regime: str,
    res_spo: Dict[str, list],
    res_mse: Dict[str, list],
    spo_agent,
    mse_agent,
    solver,
    h_spo: np.ndarray,
    h_mse: np.ndarray,
    X_test: np.ndarray,
    decisions_test: np.ndarray,
    values_test: np.ndarray,
) -> pd.DataFrame:
    oracle_h = X_test[:, :2]
    decision_labels = np.array([decision_label(d) for d in decisions_test])
    label_to_int = {label: i for i, label in enumerate(np.unique(decision_labels))}
    decision_ids = np.array([label_to_int[label] for label in decision_labels])
    total_cost = values_test.sum(axis=1)
    spo_pred = _predict_mean_matrix(spo_agent, X_test)
    mse_pred = _predict_mean_matrix(mse_agent, X_test)
    _, angles_spo = alignment_over_training(res_spo['W1_checkpoints'])
    _, angles_mse = alignment_over_training(res_mse['W1_checkpoints'])

    rows = [
        {
            'regime': regime,
            'agent': 'Oracle',
            'final_regret': np.nan,
            'decision_probe_acc': _safe_decision_probe_accuracy(oracle_h, decision_ids),
            'greedy_decision_acc': 1.0,
            'pair_decision_acc': 1.0,
            'contested_pair_acc': 1.0,
            'pair_01_acc': 1.0,
            'pair_23_acc': 1.0,
            'cost_correlation': _cost_correlation_metric(oracle_h, total_cost),
            'alignment_angle_deg': 0.0,
            'decision_loading': np.nan,
            'nuisance_loading': np.nan,
        }
    ]
    for label, result, features, checkpoints in (
        ('BayesianSPO', res_spo, h_spo, res_spo['W1_checkpoints']),
        ('BayesianMSE', res_mse, h_mse, res_mse['W1_checkpoints']),
    ):
        W1 = checkpoints[-1][1]
        row_norms = np.linalg.norm(W1, axis=1)
        pred_values = spo_pred if label == 'BayesianSPO' else mse_pred
        pair_metrics = _pair_decision_metrics(pred_values, values_test, solver)
        rows.append(
            {
                'regime': regime,
                'agent': label,
                'final_regret': float(result['cumulative_regret'][-1]),
                'decision_probe_acc': _safe_decision_probe_accuracy(features, decision_ids),
                'greedy_decision_acc': pair_metrics['greedy_decision_acc'],
                'pair_decision_acc': pair_metrics['pair_decision_acc'],
                'contested_pair_acc': pair_metrics['contested_pair_acc'],
                'pair_01_acc': pair_metrics['pair_01_acc'],
                'pair_23_acc': pair_metrics['pair_23_acc'],
                'cost_correlation': _cost_correlation_metric(features, total_cost),
                'alignment_angle_deg': angles_spo[-1] if label == 'BayesianSPO' else angles_mse[-1],
                'decision_loading': float(row_norms[:2].sum()),
                'nuisance_loading': float(row_norms[2:].sum()),
            }
        )
    return pd.DataFrame(rows)


def decode_latents(features: np.ndarray, values: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit linear probes to predict decision margins and shared offset from features."""
    from sklearn.linear_model import LinearRegression

    values = np.asarray(values, dtype=float)
    targets = {
        'margin (0,1)': values[:, 0] - values[:, 1],
        'margin (2,3)': values[:, 2] - values[:, 3],
        'shared offset': values.mean(axis=1),
    }
    predictions = {}
    for name, target in targets.items():
        reg = LinearRegression().fit(features, target)
        predictions[name] = reg.predict(features)
    return predictions


def plot_decoded_latents(
    features_spo: np.ndarray,
    features_mse: np.ndarray,
    values: np.ndarray,
    decisions: np.ndarray,
) -> 'ggplot':
    """Scatter: true task-relevant quantity vs decoded (predicted) quantity from features.

    3x2 grid: rows = pairwise decision margins and shared offset,
    cols = SPO / MSE. Diagonal = perfect recovery.
    """
    from lets_plot import geom_abline, gggrid

    dec_labels = [decision_label(d) for d in decisions]
    decoded_spo = decode_latents(features_spo, values)
    decoded_mse = decode_latents(features_mse, values)

    values = np.asarray(values, dtype=float)
    true_targets = {
        'margin (0,1)': values[:, 0] - values[:, 1],
        'margin (2,3)': values[:, 2] - values[:, 3],
        'shared offset': values.mean(axis=1),
    }

    panels = []
    for target_name, short, dim_type in [
        ('margin (0,1)', 'margin (0,1)', 'DECISION'),
        ('margin (2,3)', 'margin (2,3)', 'DECISION'),
        ('shared offset', 'shared offset', 'NUISANCE'),
    ]:
        for agent_name, decoded in [('BayesianSPO', decoded_spo), ('BayesianMSE', decoded_mse)]:
            true_vals = true_targets[target_name]
            pred_vals = decoded[target_name]
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
                    title=f'[{dim_type}] {agent_name} → {target_name} (R²={r2:.2f})',
                )
                + theme_minimal()
                + ggsize(460, 420)
            )
            panels.append(p)

    return gggrid(panels, ncol=2)


def plot_feedback_feature_grid(
    features_spo: np.ndarray,
    features_mse: np.ndarray,
    X_test: np.ndarray,
    decisions: np.ndarray,
    values: np.ndarray,
    title_prefix: str,
):
    from lets_plot import (
        geom_hline,
        geom_path,
        geom_text,
        geom_vline,
        gggrid,
        scale_color_gradient,
        scale_color_manual,
        theme_minimal,
    )
    from scipy.spatial import ConvexHull

    oracle_h = X_test[:, :2]
    total_cost = values.sum(axis=1)
    decision_labels = np.array([decision_label(d) for d in decisions])
    label_to_int = {label: i for i, label in enumerate(np.unique(decision_labels))}
    decision_ids = np.array([label_to_int[label] for label in decision_labels])
    decision_colors = {
        '0,2': '#1f77b4',
        '0,3': '#ff7f0e',
        '1,2': '#2ca02c',
        '1,3': '#d62728',
    }

    def convex_hull_df(h1, h2, group_label):
        points = np.column_stack([h1, h2])
        if len(points) < 3:
            return []
        try:
            hull = ConvexHull(points)
        except Exception:
            return []
        verts = hull.vertices.tolist()
        verts.append(verts[0])
        return [
            {
                'h1': float(points[v, 0]),
                'h2': float(points[v, 1]),
                'decision': group_label,
            }
            for v in verts
        ]

    panels_row1 = []
    for label, h in (
        ('Oracle (x0, x1)', oracle_h),
        ('BayesianSPO', features_spo),
        ('BayesianMSE', features_mse),
    ):
        acc = _safe_decision_probe_accuracy(h, decision_ids)
        df = pd.DataFrame(
            {
                'h1': h[:, 0],
                'h2': h[:, 1],
                'decision': decision_labels,
            }
        )
        hull_rows = []
        for decision in decision_colors:
            mask = df['decision'] == decision
            if int(mask.sum()) >= 3:
                hull_rows.extend(
                    convex_hull_df(
                        df.loc[mask, 'h1'].to_numpy(),
                        df.loc[mask, 'h2'].to_numpy(),
                        decision,
                    )
                )
        hull_df = pd.DataFrame(hull_rows)
        x_ann = float(df['h1'].min() + 0.02 * (df['h1'].max() - df['h1'].min()))
        y_ann = float(df['h2'].max() - 0.02 * (df['h2'].max() - df['h2'].min()))
        ann_df = pd.DataFrame([{'h1': x_ann, 'h2': y_ann, 'text': f'accuracy = {acc:.0%}'}])

        plot = (
            ggplot(df, aes('h1', 'h2', color='decision'))
            + geom_point(size=2.0, alpha=0.4)
            + scale_color_manual(values=decision_colors)
            + geom_text(
                data=ann_df,
                mapping=aes('h1', 'h2', label='text'),
                color='black',
                size=9,
                hjust=0,
                fontface='bold',
            )
            + labs(title=label, x='Feature dim 1', y='Feature dim 2')
            + theme_minimal()
            + ggsize(460, 460)
        )
        if not hull_df.empty:
            plot = plot + geom_path(
                data=hull_df,
                mapping=aes('h1', 'h2', color='decision', group='decision'),
                size=1.0,
                alpha=0.7,
            )
        if label.startswith('Oracle'):
            plot = (
                plot
                + geom_vline(xintercept=0, linetype='dashed', color='gray', size=0.8)
                + geom_hline(yintercept=0, linetype='dashed', color='gray', size=0.8)
            )
        panels_row1.append(plot)

    panels_row2 = []
    for label, h in (
        ('Oracle (x0, x1)', oracle_h),
        ('BayesianSPO', features_spo),
        ('BayesianMSE', features_mse),
    ):
        corr = _cost_correlation_metric(h, total_cost)
        df = pd.DataFrame({'h1': h[:, 0], 'h2': h[:, 1], 'cost': total_cost})
        x_ann = float(df['h1'].min() + 0.02 * (df['h1'].max() - df['h1'].min()))
        y_ann = float(df['h2'].max() - 0.02 * (df['h2'].max() - df['h2'].min()))
        ann_df = pd.DataFrame([{'h1': x_ann, 'h2': y_ann, 'text': f'|r| = {corr:.2f}'}])
        panels_row2.append(
            ggplot(df, aes('h1', 'h2', color='cost'))
            + geom_point(size=2.0, alpha=0.6)
            + scale_color_gradient(low='#2166ac', high='#b2182b')
            + geom_text(
                data=ann_df,
                mapping=aes('h1', 'h2', label='text'),
                color='black',
                size=9,
                hjust=0,
                fontface='bold',
            )
            + labs(title=label, x='Feature dim 1', y='Feature dim 2')
            + theme_minimal()
            + ggsize(460, 460)
        )
    return gggrid(panels_row1 + panels_row2, ncol=3) + labs(title=title_prefix)


def plot_action_timeline(
    oracle_actions,
    spo_actions,
    mse_actions,
    title_prefix: str,
):
    from itertools import combinations
    from lets_plot import theme_minimal

    oracle_actions = [np.asarray(a).reshape(-1) for a in oracle_actions]
    n_arms = len(oracle_actions[0])
    action_size = int(round(float(np.sum(oracle_actions[0]))))
    action_space = []
    for combo in combinations(range(n_arms), action_size):
        action = np.zeros(n_arms, dtype=float)
        action[list(combo)] = 1.0
        action_space.append(action)

    def _label(action: np.ndarray) -> str:
        return ','.join(str(i) for i in np.where(np.asarray(action) > 0.5)[0])

    action_labels = [_label(action) for action in action_space]

    def _panel(actions, agent_label):
        rows = []
        for t, (oracle_action, action) in enumerate(zip(oracle_actions, actions), start=1):
            oracle_label = _label(oracle_action)
            action_label = _label(np.asarray(action).reshape(-1))
            for label in action_labels:
                if label == action_label == oracle_label:
                    status = 'Correct action'
                elif label == action_label and action_label != oracle_label:
                    status = 'Chosen wrong'
                elif label == oracle_label and action_label != oracle_label:
                    status = 'Oracle missed'
                else:
                    status = 'Other'
                rows.append({'round': t, 'action': label, 'status': status})
        df = pd.DataFrame(rows, columns=['round', 'action', 'status'])
        return (
            ggplot(df, aes('round', 'action', fill='status'))
            + geom_tile()
            + scale_fill_manual(
                values={
                    'Correct action': '#2ca02c',
                    'Chosen wrong': '#d62728',
                    'Oracle missed': '#ffbf00',
                    'Other': '#f5f5f5',
                }
            )
            + labs(title=agent_label, x='Round', y='Action', fill='')
            + theme_minimal()
            + theme(
                panel_grid_minor='blank',
                panel_grid='blank',
            )
            + ggsize(420, 220)
        )

    plots = [
        _panel(oracle_actions, 'Oracle'),
        _panel(spo_actions, 'BayesianSPO'),
        _panel(mse_actions, 'BayesianMSE'),
    ]
    return gggrid(plots, ncol=3) + labs(title=title_prefix)


def plot_regret_per_round(
    res_spo: Dict[str, list],
    res_mse: Dict[str, list],
    title_prefix: str,
    smooth_window: int = 25,
):
    from lets_plot import scale_color_manual, theme_minimal

    rows = []
    for label, regrets in (('BayesianSPO', res_spo['regrets']), ('BayesianMSE', res_mse['regrets'])):
        series = pd.Series(np.asarray(regrets, dtype=float)).rolling(
            window=smooth_window, min_periods=1
        ).mean()
        for round_idx, value in enumerate(series, start=1):
            rows.append({'round': round_idx, 'regret': float(value), 'agent': label})
    df = pd.DataFrame(rows)
    return (
        ggplot(df, aes('round', 'regret', color='agent'))
        + geom_line(size=1.1)
        + scale_color_manual(values={'BayesianSPO': '#1f77b4', 'BayesianMSE': '#d62728'})
        + labs(title=title_prefix, x='Round', y='Per-round regret')
        + theme_minimal()
        + theme(panel_grid_minor='blank')
        + ggsize(900, 200)
    )


def plot_cumulative_regret(
    res_spo: Dict[str, list],
    res_mse: Dict[str, list],
    title_prefix: str,
):
    from lets_plot import scale_color_manual, theme_minimal

    rows = []
    for label, regrets in (
        ('BayesianSPO', res_spo['cumulative_regret']),
        ('BayesianMSE', res_mse['cumulative_regret']),
    ):
        for round_idx, value in enumerate(np.asarray(regrets, dtype=float), start=1):
            rows.append({'round': round_idx, 'cumulative_regret': float(value), 'agent': label})
    df = pd.DataFrame(rows)
    return (
        ggplot(df, aes('round', 'cumulative_regret', color='agent'))
        + geom_line(size=1.1)
        + scale_color_manual(values={'BayesianSPO': '#1f77b4', 'BayesianMSE': '#d62728'})
        + labs(title=title_prefix, x='Round', y='Cumulative regret')
        + theme_minimal()
        + theme(panel_grid_minor='blank')
        + ggsize(900, 200)
    )


def plot_prediction_error_trajectory(
    res_spo: Dict[str, list],
    res_mse: Dict[str, list],
    title_prefix: str,
):
    from lets_plot import scale_color_manual, theme_minimal

    rows = []
    for label, checkpoints in (
        ('BayesianSPO', res_spo.get('prediction_error_checkpoints', [])),
        ('BayesianMSE', res_mse.get('prediction_error_checkpoints', [])),
    ):
        for round_idx, value in checkpoints:
            rows.append({'round': int(round_idx), 'prediction_error': float(value), 'agent': label})
    df = pd.DataFrame(rows)
    return (
        ggplot(df, aes('round', 'prediction_error', color='agent'))
        + geom_line(size=1.1)
        + geom_point(size=2.0)
        + scale_color_manual(values={'BayesianSPO': '#1f77b4', 'BayesianMSE': '#d62728'})
        + labs(title=title_prefix, x='Round', y='Prediction MSE')
        + theme_minimal()
        + theme(panel_grid_minor='blank')
        + ggsize(900, 200)
    )


def build_regime_dashboard(
    feature_plot,
    decoded_latents_plot,
    regret_plot,
    cumulative_regret_plot,
    prediction_error_plot,
    loading_plot,
    action_plot,
    title_prefix: str,
):
    regret_row = gggrid(
        [regret_plot, cumulative_regret_plot],
        ncol=2,
    )
    diagnostics_row = gggrid(
        [prediction_error_plot, loading_plot],
        ncol=2,
    )
    return (
        gggrid(
            [
                feature_plot,
                decoded_latents_plot,
                regret_row,
                diagnostics_row,
                action_plot,
            ],
            ncol=1,
        )
        + ggsize(1700, 3600)
    )


def _slugify_regime(regime: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', regime.lower()).strip('_')
    return slug or 'regime'


def save_dashboard_plot(dashboard_plot, regime: str) -> str:
    output_dir = Path('results') / 'feature_representation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'{_slugify_regime(regime)}_dashboard.svg'
    return ggsave(dashboard_plot, filename, path=str(output_dir), iframe=False)


def run_feedback_regime_experiment(
    regime: str,
    full_feedback: bool,
    n_rounds: int = 5000,
    checkpoint_interval: int = 200,
    noise_width: float = 0.0,
    hidden_dim: int = 2,
    beta: float = 1.0,
    lambda_reg: float = 0.1,
    lr: float = 0.03,
    seed: int = 0,
    n_test: int = 500,
    test_seed: int = 99,
    mask_spo: bool = False,
    mask_mse: bool = False,
    pair_swap_augment_spo: bool = False,
    pair_swap_augment_mse: bool = False,
    pair_balanced_spo: bool = False,
    pair_balanced_mse: bool = False,
    disagreement_coverage_spo: bool = False,
    disagreement_coverage_mse: bool = False,
    disagreement_samples: int = 16,
    disagreement_lambda: float = 1.0,
) -> Dict[str, object]:
    from src.agents.base import pregenerate_episode

    env = DecisionFocusedEnv(seed=42, noise_width=noise_width)
    episode = pregenerate_episode(env, n_rounds=n_rounds)
    env_test = DecisionFocusedEnv(seed=42, noise_width=noise_width)
    X_test, decisions_test, values_test = generate_test_contexts(env_test, n_samples=n_test, seed=test_seed)
    spo_agent, mse_agent = create_agent_pair(
        env,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        seed=seed,
        mask_spo=mask_spo,
        mask_mse=mask_mse,
        pair_balanced_spo=pair_balanced_spo,
        pair_balanced_mse=pair_balanced_mse,
        disagreement_coverage_spo=disagreement_coverage_spo,
        disagreement_coverage_mse=disagreement_coverage_mse,
        disagreement_samples=disagreement_samples,
        disagreement_lambda=disagreement_lambda,
    )
    res_spo = train_with_checkpoints(
        spo_agent,
        episode,
        n_rounds,
        checkpoint_interval=checkpoint_interval,
        full_feedback=full_feedback,
        pair_swap_augment=pair_swap_augment_spo,
        eval_contexts=X_test,
        eval_values=values_test,
    )
    _, mse_agent = create_agent_pair(
        env,
        hidden_dim=hidden_dim,
        beta=beta,
        lambda_reg=lambda_reg,
        lr=lr,
        seed=seed,
        mask_spo=mask_spo,
        mask_mse=mask_mse,
        pair_balanced_spo=pair_balanced_spo,
        pair_balanced_mse=pair_balanced_mse,
        disagreement_coverage_spo=disagreement_coverage_spo,
        disagreement_coverage_mse=disagreement_coverage_mse,
        disagreement_samples=disagreement_samples,
        disagreement_lambda=disagreement_lambda,
    )
    res_mse = train_with_checkpoints(
        mse_agent,
        episode,
        n_rounds,
        checkpoint_interval=checkpoint_interval,
        full_feedback=full_feedback,
        pair_swap_augment=pair_swap_augment_mse,
        eval_contexts=X_test,
        eval_values=values_test,
    )
    h_spo = extract_features(spo_agent, X_test)
    h_mse = extract_features(mse_agent, X_test)
    summary = _mechanism_summary_table(
        regime,
        res_spo,
        res_mse,
        spo_agent,
        mse_agent,
        env_test.solve,
        h_spo,
        h_mse,
        X_test,
        decisions_test,
        values_test,
    )
    feature_plot = plot_feedback_feature_grid(
        h_spo,
        h_mse,
        X_test,
        decisions_test,
        values_test,
        title_prefix=f'{regime}: decision structure vs nuisance prediction',
    )
    regret_plot = plot_regret_per_round(
        res_spo,
        res_mse,
        title_prefix=f'{regime}: regret per round',
    )
    cumulative_regret_plot = plot_cumulative_regret(
        res_spo,
        res_mse,
        title_prefix=f'{regime}: cumulative regret',
    )
    prediction_error_plot = plot_prediction_error_trajectory(
        res_spo,
        res_mse,
        title_prefix=f'{regime}: prediction error over training',
    )
    rounds_spo, angles_spo = alignment_over_training(res_spo['W1_checkpoints'])
    rounds_mse, angles_mse = alignment_over_training(res_mse['W1_checkpoints'])
    alignment_plot = plot_alignment_trajectory(rounds_spo, angles_spo, rounds_mse, angles_mse)
    loading_plot = plot_feature_directions(
        res_spo['W1_checkpoints'][-1][1],
        res_mse['W1_checkpoints'][-1][1],
    )
    oracle_actions = [rd['optimal_action'] for rd in episode['rounds_data']]
    action_plot = plot_action_timeline(
        oracle_actions,
        res_spo['actions'],
        res_mse['actions'],
        title_prefix=f'{regime}: action-space match vs oracle',
    )
    decoded_latents_plot = plot_decoded_latents(
        h_spo, h_mse, values_test, decisions_test,
    )
    dashboard_plot = build_regime_dashboard(
        feature_plot,
        decoded_latents_plot,
        regret_plot,
        cumulative_regret_plot,
        prediction_error_plot,
        loading_plot,
        action_plot,
        title_prefix=f'{regime}: dashboard',
    )
    dashboard_path = save_dashboard_plot(dashboard_plot, regime)
    return {
        'regime': regime,
        'summary': summary,
        'feature_plot': feature_plot,
        'regret_plot': regret_plot,
        'cumulative_regret_plot': cumulative_regret_plot,
        'prediction_error_plot': prediction_error_plot,
        'alignment_plot': alignment_plot,
        'loading_plot': loading_plot,
        'action_plot': action_plot,
        'decoded_latents_plot': decoded_latents_plot,
        'dashboard_plot': dashboard_plot,
        'dashboard_path': dashboard_path,
        'res_spo': res_spo,
        'res_mse': res_mse,
        'h_spo': h_spo,
        'h_mse': h_mse,
        'X_test': X_test,
        'decisions_test': decisions_test,
        'values_test': values_test,
    }

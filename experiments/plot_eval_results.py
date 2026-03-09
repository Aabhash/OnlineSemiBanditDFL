import argparse
import os
import pickle

import numpy as np
import pandas as pd

from lets_plot import (
    ggplot, aes, geom_line, geom_ribbon, geom_bar, geom_errorbar,
    ggsave, ggsize,
    scale_color_manual, scale_fill_manual, scale_linetype_manual,
    scale_x_continuous, scale_y_continuous,
    guides, guide_legend,
    labs, theme_minimal, theme, element_text, element_blank,
    coord_flip,
)
from lets_plot import LetsPlot
LetsPlot.setup_html()


AGENT_DISPLAY = {
    'BayesianMSE':        'BayesianMSE',
    'BayesianSPO':        'BayesianSPO',
    'BayesianSPO-DFE':    'BSPO-DFE',
    'BayesianSPO-DI':     'BSPO-DI',
    'BayesianSPO-DR':     'BSPO-DR',
    'BayesianSPO-DFE+DR': 'BSPO-DFE+DR',
    'NeuralUCB':          'NeuralUCB',
    'LinUCB':             'LinUCB',
    'UCB-Bayesian':       'UCB-Bayesian',
    'ThompsonSampling':   'TS',
    'EpsilonGreedy':      'EpsGreedy',
    'Greedy':             'Greedy',
    'Random':             'Random',
}

DATASET_LABELS = {
    'knapsack':            'Knapsack',
    'knapsack_unit_weight':'Knapsack (Unit Wt)',
    'shortest_path_8x8':  'Shortest Path 8×8',
    'shortest_path_20x20':'Shortest Path 20×20',
}

# Line plot colors — every agent gets a distinct color
COLORS = {
    'BayesianMSE':       '#d62728',   # red
    'BayesianSPO':       '#1f77b4',   # blue
    'BSPO-DFE':     '#9467bd',   # purple
    'BSPO-DFE+DR':  '#bcbd22',   # olive
    'BSPO-DI':      '#2ca02c',   # green
    'BSPO-DR':      '#ff7f0e',   # orange
    'NeuralUCB':    '#17becf',   # cyan
    'LinUCB':       '#8c564b',   # brown
    'UCB-Bayesian':    '#e377c2',   # pink
    'TS':           '#7f7f7f',   # grey
    'EpsGreedy':    '#aec7e8',   # light blue
    'Greedy':       '#ffbb78',   # light orange
    'Random':       '#c7c7c7',   # light grey
}

LINESTYLES = {
    'BayesianMSE':       'dotted',
    'BayesianSPO':       'dashed',
    'BSPO-DFE':     'solid',
    'BSPO-DFE+DR':  'solid',
    'BSPO-DI':      'solid',
    'BSPO-DR':      'solid',
    'NeuralUCB':    'dotdash',
    'LinUCB':       'dotdash',
    'UCB-Bayesian':    'dashed',
    'TS':           'dotted',
    'EpsGreedy':    'twodash',
    'Greedy':       'longdash',
    'Random':       'dotted',
}

# Bar chart colors: BayesianSPO=blue, DFE variants=orange, DI=green, baselines=grey
BAR_COLORS = {
    'BayesianSPO':       '#1f77b4',
    'BSPO-DFE':     '#ff7f0e',
    'BSPO-DI':      '#2ca02c',
    'BSPO-DR':      '#ff7f0e',
    'BSPO-DFE+DR':  '#ff7f0e',
    'BayesianMSE':       '#aaaaaa',
    'NeuralUCB':    '#aaaaaa',
    'LinUCB':       '#aaaaaa',
    'UCB-Bayesian':    '#aaaaaa',
    'TS':           '#aaaaaa',
    'EpsGreedy':    '#aaaaaa',
    'Greedy':       '#aaaaaa',
    'Random':       '#aaaaaa',
}

WINDOW = 50  # smoothing window for per-round regret

# Agents to include in LINE plots (None = all agents present)
LINE_AGENTS = None

LINE_LEGEND_THEME = theme(
    legend_position='bottom', legend_title=element_blank(),
    legend_text=element_text(size=8),
)


def load_trajectories(path):
    """Load trajectories.pkl and return organized dict."""
    with open(path, 'rb') as f:
        raw = pickle.load(f)

    organized = {}
    for (agent, dataset, feedback, seed), data in raw.items():
        organized.setdefault(dataset, {}).setdefault(feedback, {}).setdefault(agent, {})[seed] = data

    return organized


def build_dataframes(organized):
    rows_cum, rows_pr, rows_bar = [], [], []

    for dataset, fb_dict in organized.items():
        for feedback, agent_dict in fb_dict.items():
            for agent_raw, seed_dict in agent_dict.items():
                agent = AGENT_DISPLAY.get(agent_raw, agent_raw)

                for seed, data in seed_dict.items():
                    cum_reg = data['cumulative_regret']
                    regrets = data['regrets']
                    n = len(cum_reg)

                    # Subsample cumulative regret (~500 points per line)
                    step = max(1, n // 500)
                    indices = list(range(0, n, step))
                    if (n - 1) not in indices:
                        indices.append(n - 1)
                    for t in indices:
                        rows_cum.append(dict(
                            dataset=dataset, feedback=feedback, agent=agent,
                            seed=seed, round=t + 1, cumulative_regret=cum_reg[t],
                        ))

                    # Per-round regret (smoothed)
                    kernel = np.ones(WINDOW) / WINDOW
                    smoothed = np.convolve(regrets, kernel, mode='valid')
                    m = len(smoothed)
                    step_pr = max(1, m // 500)
                    indices_pr = list(range(0, m, step_pr))
                    if (m - 1) not in indices_pr:
                        indices_pr.append(m - 1)
                    for t in indices_pr:
                        rows_pr.append(dict(
                            dataset=dataset, feedback=feedback, agent=agent,
                            seed=seed, round=t + WINDOW, per_round_regret=smoothed[t],
                        ))

                    # Bar chart data
                    rows_bar.append(dict(
                        dataset=dataset, feedback=feedback, agent=agent,
                        seed=seed, final_cumulative_regret=cum_reg[-1],
                    ))

    df_cum = pd.DataFrame(rows_cum)
    df_pr = pd.DataFrame(rows_pr)
    df_bar = pd.DataFrame(rows_bar)

    # Aggregate across seeds
    df_cum_agg = df_cum.groupby(['dataset', 'feedback', 'agent', 'round']).agg(
        mean=('cumulative_regret', 'mean'), std=('cumulative_regret', 'std'),
    ).reset_index()
    df_cum_agg['ymin'] = df_cum_agg['mean'] - df_cum_agg['std']
    df_cum_agg['ymax'] = df_cum_agg['mean'] + df_cum_agg['std']

    df_pr_agg = df_pr.groupby(['dataset', 'feedback', 'agent', 'round']).agg(
        mean=('per_round_regret', 'mean'), std=('per_round_regret', 'std'),
    ).reset_index()
    df_pr_agg['ymin'] = df_pr_agg['mean'] - df_pr_agg['std']
    df_pr_agg['ymax'] = df_pr_agg['mean'] + df_pr_agg['std']

    return df_cum_agg, df_pr_agg, df_bar


def make_line_plots(df_cum_agg, df_pr_agg, datasets, feedback='semi_bandit',
                    output_dir='results/figures', fmt='svg', line_agents=None):

    available = set(df_cum_agg['agent'].unique())
    requested = line_agents or LINE_AGENTS
    if requested is not None:
        agents = [a for a in requested if a in available]
    else:
        agents = sorted(available)

    # Build color/linetype dicts for agents in data
    colors = {a: COLORS.get(a, '#333333') for a in agents}
    linetypes = {a: LINESTYLES.get(a, 'solid') for a in agents}

    plot_datasets = [ds for ds in datasets
                     if not df_cum_agg[(df_cum_agg['dataset'] == ds) &
                                       (df_cum_agg['feedback'] == feedback)].empty]
    if not plot_datasets:
        print('No line plot data available.')
        return

    from lets_plot import gggrid

    for ds in plot_datasets:
        label = DATASET_LABELS.get(ds, ds)

        d_cum = df_cum_agg[(df_cum_agg['dataset'] == ds) &
                           (df_cum_agg['feedback'] == feedback) &
                           (df_cum_agg['agent'].isin(agents))]
        d_pr = df_pr_agg[(df_pr_agg['dataset'] == ds) &
                         (df_pr_agg['feedback'] == feedback) &
                         (df_pr_agg['agent'].isin(agents))]

        if d_cum.empty:
            continue

        p_cum = (
            ggplot(d_cum, aes(x='round', y='mean', color='agent', fill='agent'))
            + geom_ribbon(aes(ymin='ymin', ymax='ymax'), alpha=0.15, color='rgba(0,0,0,0)')
            + geom_line(aes(linetype='agent'), size=1.0)
            + scale_color_manual(values=colors)
            + scale_fill_manual(values=colors)
            + scale_linetype_manual(values=linetypes)
            + guides(color=guide_legend(nrow=2), fill='none', linetype=guide_legend(nrow=2))
            + labs(title=f'{label}: Cumulative Regret', x='Round', y='Cumulative Regret')
            + theme_minimal() + LINE_LEGEND_THEME
        )

        p_pr = (
            ggplot(d_pr, aes(x='round', y='mean', color='agent', fill='agent'))
            + geom_ribbon(aes(ymin='ymin', ymax='ymax'), alpha=0.15, color='rgba(0,0,0,0)')
            + geom_line(aes(linetype='agent'), size=1.0)
            + scale_color_manual(values=colors)
            + scale_fill_manual(values=colors)
            + scale_linetype_manual(values=linetypes)
            + guides(color=guide_legend(nrow=2), fill='none', linetype=guide_legend(nrow=2))
            + labs(title=f'{label}: Per-Round ({WINDOW}-avg)', x='Round', y='Per-Round Regret')
            + theme_minimal() + LINE_LEGEND_THEME
        )

        fig = gggrid([p_cum, p_pr], ncol=2, align=True, hspace=30) + ggsize(1400, 450)
        outpath = os.path.abspath(os.path.join(output_dir, f'regret_trajectories_{ds}.{fmt}'))
        ggsave(fig, outpath)
        print(f'Saved line plots: {outpath}')


def make_bar_charts(df_bar, datasets, feedback='semi_bandit',
                    output_dir='results/figures', fmt='svg'):

    df = df_bar[df_bar['feedback'] == feedback].copy()
    ds_list = [ds for ds in datasets if ds in df['dataset'].unique()]
    if not ds_list:
        print('No bar chart data available.')
        return

    for ds in ds_list:
        d = df[df['dataset'] == ds].copy()

        # Aggregate
        agg = d.groupby('agent')['final_cumulative_regret'].agg(['mean', 'std']).reset_index()

        # Sort: worst on top (highest mean first → reversed for coord_flip)
        agg = agg.sort_values('mean', ascending=True)
        agent_order = agg['agent'].tolist()  # best-first → bottom after coord_flip

        scale = 1e3
        agg['mean_k'] = agg['mean'] / scale
        agg['std_k'] = agg['std'] / scale
        agg['ymin_k'] = (agg['mean'] - agg['std']) / scale
        agg['ymax_k'] = (agg['mean'] + agg['std']) / scale

        # Map colors
        bar_colors = {a: BAR_COLORS.get(a, '#aaaaaa') for a in agent_order}

        n_agents = len(agent_order)
        plot_height = max(300, 20 * n_agents + 80)

        from lets_plot import scale_x_discrete

        p = (
            ggplot(agg, aes(x='agent', y='mean_k', fill='agent'))
            + geom_bar(stat='identity', width=0.7, color='#333333', size=0.3)
            + geom_errorbar(aes(ymin='ymin_k', ymax='ymax_k'), width=0.25, size=0.5)
            + scale_fill_manual(values=bar_colors)
            + scale_x_discrete(limits=agent_order)
            + coord_flip()
            + labs(title=DATASET_LABELS.get(ds, ds), x='', y='Cumul. Regret (×1k)')
            + theme_minimal()
            + theme(legend_position='none',
                    axis_text_y=element_text(size=9),
                    plot_title=element_text(size=12, face='bold'))
            + ggsize(700, plot_height)
        )

        outpath = os.path.abspath(os.path.join(output_dir, f'regret_barplot_{ds}.{fmt}'))
        ggsave(p, outpath)
        print(f'Saved bar chart: {outpath}')



def load_final_regrets(path):
    import json
    with open(path) as f:
        data = json.load(f)

    rows = []
    for dataset, agent_dict in data.items():
        for agent_raw, regrets in agent_dict.items():
            agent = AGENT_DISPLAY.get(agent_raw, agent_raw)
            for i, r in enumerate(regrets):
                rows.append(dict(
                    dataset=dataset, feedback='semi_bandit',
                    agent=agent, seed=i, final_cumulative_regret=r,
                ))
    return pd.DataFrame(rows)


# ─────────────────────────── Main ───────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot evaluation results')
    parser.add_argument('--input', type=str,
                        default='results/eval/metrics/trajectories.pkl',
                        help='Path to trajectories.pkl')
    parser.add_argument('--regrets_json', type=str, default=None,
                        help='Path to final_eval_regrets.json (for bar charts with more datasets)')
    parser.add_argument('--output', type=str, default='results/figures',
                        help='Output directory for plots')
    parser.add_argument('--format', type=str, default='svg', choices=['svg', 'png', 'pdf'],
                        help='Output format')
    parser.add_argument('--feedback', type=str, default='semi_bandit',
                        help='Feedback type to plot')
    parser.add_argument('--bar_agents', type=str, nargs='*', default=None,
                        help='Agents to include in bar chart (default: all)')
    parser.add_argument('--line_agents', type=str, nargs='*', default=None,
                        help='Agents to include in line plots (default: all)')
    parser.add_argument('--exclude_nl', action='store_true', default=True,
                        help='Exclude NL (nonlinear) agent variants from plots (default: True)')
    parser.add_argument('--include_nl', action='store_true',
                        help='Include NL (nonlinear) agent variants in plots')
    args = parser.parse_args()

    exclude_nl = args.exclude_nl and not args.include_nl

    os.makedirs(args.output, exist_ok=True)

    # Load trajectories for line plots
    # Try per-dataset files first (trajectories_<dataset>.pkl), then combined
    organized = None
    input_dir = os.path.dirname(args.input)
    import glob as glob_mod
    per_ds_files = sorted(glob_mod.glob(os.path.join(input_dir, 'trajectories_*.pkl')))
    if per_ds_files:
        organized = {}
        for f in per_ds_files:
            print(f'Loading {os.path.basename(f)}...')
            part = load_trajectories(f)
            for ds, fb_dict in part.items():
                organized.setdefault(ds, {}).update(fb_dict)
    elif os.path.exists(args.input):
        print(f'Loading trajectories from {args.input}...')
        organized = load_trajectories(args.input)

    # Load regrets JSON for bar charts (if provided or auto-detect)
    df_bar_extra = None
    regrets_path = args.regrets_json
    if regrets_path is None:
        default_path = 'results/final_eval_regrets.json'
        if os.path.exists(default_path):
            regrets_path = default_path
    if regrets_path and os.path.exists(regrets_path):
        print(f'Loading final regrets from {regrets_path}...')
        df_bar_extra = load_final_regrets(regrets_path)

    # Build dataframes from trajectories
    df_cum_agg, df_pr_agg, df_bar = None, None, pd.DataFrame()
    if organized:
        datasets_traj = sorted(organized.keys())
        agents_all = set()
        for ds in organized.values():
            for fb in ds.values():
                agents_all.update(AGENT_DISPLAY.get(a, a) for a in fb.keys())
        print(f'Trajectory datasets: {datasets_traj}')
        print(f'Trajectory agents: {sorted(agents_all)}')

        df_cum_agg, df_pr_agg, df_bar = build_dataframes(organized)

    # Combine bar data: use trajectories (newer) for datasets it covers,
    # fall back to regrets JSON for remaining datasets
    if df_bar_extra is not None and not df_bar.empty:
        traj_datasets = set(df_bar['dataset'].unique())
        df_json_only = df_bar_extra[~df_bar_extra['dataset'].isin(traj_datasets)]
        df_bar_combined = pd.concat([df_bar, df_json_only], ignore_index=True)
        bar_datasets = sorted(df_bar_combined['dataset'].unique())
        print(f'Bar chart datasets (trajectories: {sorted(traj_datasets)}, '
              f'JSON fallback: {sorted(df_json_only["dataset"].unique())})')
    elif df_bar_extra is not None:
        df_bar_combined = df_bar_extra
        bar_datasets = sorted(df_bar_extra['dataset'].unique())
        print(f'Bar chart datasets (from JSON): {bar_datasets}')
    else:
        df_bar_combined = df_bar
        bar_datasets = sorted(df_bar['dataset'].unique()) if not df_bar.empty else []

    # Exclude NL variants
    if exclude_nl and not df_bar_combined.empty:
        nl_mask = df_bar_combined['agent'].str.contains('-NL$', regex=True)
        n_excluded = nl_mask.sum()
        if n_excluded > 0:
            df_bar_combined = df_bar_combined[~nl_mask]
            print(f'Excluded {n_excluded} NL variant entries from bar charts')

    # Filter agents if specified
    if args.bar_agents and not df_bar_combined.empty:
        df_bar_combined = df_bar_combined[df_bar_combined['agent'].isin(args.bar_agents)]

    # Generate line plots (only from trajectories)
    if df_cum_agg is not None and df_pr_agg is not None:
        traj_datasets = sorted(set(df_cum_agg['dataset'].unique()))
        make_line_plots(df_cum_agg, df_pr_agg, traj_datasets, args.feedback,
                        args.output, args.format, args.line_agents)

    # Generate bar charts
    if not df_bar_combined.empty:
        make_bar_charts(df_bar_combined, bar_datasets, args.feedback,
                        args.output, args.format)

    print('Done.')


if __name__ == '__main__':
    main()

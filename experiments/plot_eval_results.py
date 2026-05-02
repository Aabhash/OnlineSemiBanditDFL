import argparse
import glob as glob_mod
import json
import os
import pickle
import re

for _cairo_lib in ("/opt/homebrew/opt/cairo/lib", "/usr/local/opt/cairo/lib"):
    if os.path.exists(os.path.join(_cairo_lib, "libcairo.2.dylib")):
        os.environ.setdefault("DYLD_LIBRARY_PATH", _cairo_lib)
        break

import cairosvg
import numpy as np
import pandas as pd

from lets_plot import (
    ggplot, aes, geom_line, geom_ribbon, geom_bar, geom_errorbar,
    gggrid, ggsave, ggsize,
    scale_color_manual, scale_fill_manual, scale_linetype_manual,
    scale_x_continuous, scale_x_discrete, scale_y_continuous,
    guides, guide_legend,
    labs, theme_minimal, theme, element_text, element_blank,
    coord_flip, element_rect,
)
from lets_plot import LetsPlot
LetsPlot.setup_html()

PAPER_FONT = "Avenir Next"
PAPER_BOLD_WEIGHT = "600"
PAPER_EMPHASIS_FONT = "Avenir Next Demi Bold"


def force_svg_font(svg_path):
    with open(svg_path) as f:
        text = f.read()
    text = text.replace("font-family: sans-serif", f"font-family: {PAPER_FONT}")
    text = text.replace("font-family: sans", f"font-family: {PAPER_FONT}")
    text = re.sub(
        rf"font-weight:\s*bold;\s*font-style:\s*normal;\s*font-family:\s*{re.escape(PAPER_FONT)};",
        (
            "font-weight: normal;\n"
            "font-style: normal;\n"
            f"font-family: {PAPER_EMPHASIS_FONT};"
        ),
        text,
    )
    text = text.replace("font-weight: bold", f"font-weight: {PAPER_BOLD_WEIGHT}")
    text = text.replace("font-weight:bold", f"font-weight:{PAPER_BOLD_WEIGHT}")
    text = re.sub(
        rf"font-weight:\s*{re.escape(PAPER_BOLD_WEIGHT)};([^\"<>]*?)font-family:\s*{re.escape(PAPER_FONT)};",
        rf"font-weight:normal;\1font-family:{PAPER_EMPHASIS_FONT};",
        text,
    )
    with open(svg_path, "w") as f:
        f.write(text)


def save_lets_plot(plot, outpath, fmt):
    if fmt == "pdf":
        svg_path = os.path.splitext(outpath)[0] + ".svg"
        ggsave(plot, svg_path)
        force_svg_font(svg_path)
        cairosvg.svg2pdf(url=svg_path, write_to=outpath)
    else:
        ggsave(plot, outpath)
        if fmt == "svg":
            force_svg_font(outpath)


AGENT_DISPLAY = {
    'BayesianMSE':              'BayesianMSE',
    'BayesianSPO':              'BayesianSPO',
    'BayesianPFY':              'BayesianPFY',
    'NeuralUCB':                'NeuralUCB',
    'LinUCB':                   'LinUCB',
    'TS':                       'TS',
    'EpsilonGreedy':            'EpsGreedy',
    'Greedy':                   'Greedy',
    'Random':                   'Random',
    'WeberBLL-Online':          'BLL-Online',
    'BLL-Online':               'BLL-Online',
}

DATASET_LABELS = {
    'knapsack':            'Knapsack',
    'knapsack_32':         'Knapsack (n=32)',
    'shortest_path_8x8':  'Shortest Path 8×8',
    'shortest_path_20x20':'Shortest Path 20×20',
    'topk_8arm':           'TopK (8 arms)',
    'topk_8arm_linear':    'TopK Linear (8 arms)',
}


COLORS = {
    'BayesianMSE':              '#d62728',
    'BayesianSPO':              '#1f77b4',
    'BayesianPFY':              '#bcbd22',
    'NeuralUCB':                '#7f7f7f',
    'LinUCB':                   '#8c564b',
    'TS':                       '#7f7f7f',
    'EpsGreedy':                '#aec7e8',
    'Greedy':                   '#ffbb78',
    'Random':                   '#c7c7c7',
    'WeberBLL-Online':          '#8c564b',
    'BLL-Online':               '#8c564b',
}

LINESTYLES = {
    'BayesianMSE':              'solid',
    'BayesianSPO':              'solid',
    'BayesianPFY':              'dashed',
    'NeuralUCB':                'dotdash',
    'LinUCB':                   'dotdash',
    'TS':                       'dotted',
    'EpsGreedy':                'twodash',
    'Greedy':                   'longdash',
    'Random':                   'dotted',
    'WeberBLL-Online':          'dashed',
    'BLL-Online':               'dashed',
}


BAR_COLORS = {
    'BayesianSPO':              '#1f77b4',
    'BayesianPFY':              '#bcbd22',
    'BayesianMSE':              '#aaaaaa',
    'NeuralUCB':                '#aaaaaa',
    'LinUCB':                   '#aaaaaa',
    'TS':                       '#aaaaaa',
    'EpsGreedy':                '#aaaaaa',
    'Greedy':                   '#aaaaaa',
    'Random':                   '#aaaaaa',
    'WeberBLL-Online':          '#8c564b',
    'BLL-Online':               '#8c564b',
}

WINDOW = 100


LINE_AGENTS = [
    'LinUCB', 'NeuralUCB',
    'BayesianMSE',
    'BayesianSPO',
    'BLL-Online', 'WeberBLL-Online',
]
BAR_AGENTS = [
    'Random', 'EpsGreedy', 'TS', 'LinUCB',
    'NeuralUCB',
    'BayesianMSE',
    'BayesianSPO',
    'BLL-Online', 'WeberBLL-Online',
]

LINE_LEGEND_THEME_TL = theme(
    text=element_text(family=PAPER_FONT),
    legend_position=[0.02, 0.98], legend_justification=[0, 1],
    legend_title=element_blank(),
    legend_text=element_text(size=14, face='bold'),
    legend_background=element_rect(fill='white', color='#cccccc', size=0.5),
)

LINE_LEGEND_THEME_TR = theme(
    text=element_text(family=PAPER_FONT),
    legend_position=[0.98, 0.98], legend_justification=[1, 1],
    legend_title=element_blank(),
    legend_text=element_text(size=14, face='bold'),
    legend_background=element_rect(fill='white', color='#cccccc', size=0.5),
)

THEME_LINE = (
    theme_minimal()
    + theme(
        text=element_text(family=PAPER_FONT),
        legend_title=element_blank(),
        legend_text=element_text(size=10, face='bold'),
        plot_title=element_text(size=12, face='bold'),
        plot_subtitle=element_text(size=11, face='bold'),
    )
)


def load_trajectories(path):
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


                    step = max(1, n // 500)
                    indices = list(range(0, n, step))
                    if (n - 1) not in indices:
                        indices.append(n - 1)
                    for t in indices:
                        rows_cum.append(dict(
                            dataset=dataset, feedback=feedback, agent=agent,
                            seed=seed, round=t + 1, cumulative_regret=cum_reg[t],
                        ))


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


                    rows_bar.append(dict(
                        dataset=dataset, feedback=feedback, agent=agent,
                        seed=seed, final_cumulative_regret=cum_reg[-1],
                    ))

    df_cum = pd.DataFrame(rows_cum)
    df_pr = pd.DataFrame(rows_pr)
    df_bar = pd.DataFrame(rows_bar)


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


    colors = {a: COLORS.get(a, '#333333') for a in agents}
    linetypes = {a: LINESTYLES.get(a, 'solid') for a in agents}

    plot_datasets = [ds for ds in datasets
                     if not df_cum_agg[(df_cum_agg['dataset'] == ds) &
                                       (df_cum_agg['feedback'] == feedback)].empty]
    if not plot_datasets:
        print('No line plot data available.')
        return

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


        d_cum = d_cum.copy()
        d_cum['mean_k'] = d_cum['mean'] / 1e3
        d_cum['ymin_k'] = d_cum['ymin'] / 1e3
        d_cum['ymax_k'] = d_cum['ymax'] / 1e3

        p_cum = (
            ggplot(d_cum, aes(x='round', y='mean_k', color='agent', fill='agent'))
            + geom_ribbon(aes(ymin='ymin_k', ymax='ymax_k'), alpha=0.15, color='rgba(0,0,0,0)')
            + geom_line(aes(linetype='agent'), size=1.0)
            + scale_color_manual(values=colors)
            + scale_fill_manual(values=colors)
            + scale_linetype_manual(values=linetypes)
            + guides(color=guide_legend(nrow=1), fill='none', linetype=guide_legend(nrow=1))
            + labs(title=f'{label}: Cumulative Regret', x='Round', y='Cumulative Regret (×1k)')
            + theme_minimal() + LINE_LEGEND_THEME_TL
        )

        p_pr = (
            ggplot(d_pr, aes(x='round', y='mean', color='agent', fill='agent'))
            + geom_ribbon(aes(ymin='ymin', ymax='ymax'), alpha=0.15, color='rgba(0,0,0,0)')
            + geom_line(aes(linetype='agent'), size=1.0)
            + scale_color_manual(values=colors)
            + scale_fill_manual(values=colors)
            + scale_linetype_manual(values=linetypes)
            + guides(color=guide_legend(nrow=1), fill='none', linetype=guide_legend(nrow=1))
            + labs(title=f'{label}: Per-Round Regret', x='Round', y='Per-Round Regret')
            + theme_minimal() + LINE_LEGEND_THEME_TR
        )

        fig = gggrid([p_cum, p_pr], ncol=2, align=True, hspace=30) + ggsize(1400, 450)
        outpath = os.path.abspath(os.path.join(output_dir, f'regret_trajectories_{ds}.{fmt}'))
        save_lets_plot(fig, outpath, fmt)
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


        agg = d.groupby('agent')['final_cumulative_regret'].agg(['mean', 'std']).reset_index()


        agg = agg.sort_values('mean', ascending=True)
        agent_order = agg['agent'].tolist()

        scale = 1e3
        agg['mean_k'] = agg['mean'] / scale
        agg['std_k'] = agg['std'] / scale
        agg['ymin_k'] = (agg['mean'] - agg['std']) / scale
        agg['ymax_k'] = (agg['mean'] + agg['std']) / scale


        bar_colors = {a: BAR_COLORS.get(a, '#aaaaaa') for a in agent_order}

        n_agents = len(agent_order)
        plot_height = max(300, 22 * n_agents + 80)
        plot_width = 1200

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
                    text=element_text(family=PAPER_FONT),
                    axis_text_y=element_text(size=9, face='bold'),
                    plot_title=element_text(size=12, face='bold'))
            + ggsize(plot_width, plot_height)
        )

        outpath = os.path.abspath(os.path.join(output_dir, f'regret_barplot_{ds}.{fmt}'))
        save_lets_plot(p, outpath, fmt)
        print(f'Saved bar chart: {outpath}')


def build_diagnostic_dataframes(organized):
    diag_metrics = {
        'flip_set_size': 'Flip Set Size',
        'contested_mae': 'Contested MAE',
        'decision_correct': 'Decision Accuracy',
        'rank_corr': 'Rank Correlation',
    }

    all_rows = {metric: [] for metric in diag_metrics}

    for dataset, fb_dict in organized.items():
        for feedback, agent_dict in fb_dict.items():
            for agent_raw, seed_dict in agent_dict.items():
                agent = AGENT_DISPLAY.get(agent_raw, agent_raw)
                for seed, data in seed_dict.items():
                    for metric_key in diag_metrics:
                        values = data.get(metric_key, [])
                        if not values or not isinstance(values, list):
                            continue
                        diagnostic_rounds = data.get('diagnostic_rounds', [])

                        arr = np.array(values, dtype=float)
                        arr = arr[~np.isnan(arr)]
                        if len(arr) < WINDOW:
                            continue
                        if (
                            isinstance(diagnostic_rounds, list)
                            and len(diagnostic_rounds) == len(values)
                        ):
                            rounds_arr = np.array(diagnostic_rounds, dtype=float)
                            rounds_arr = rounds_arr[~np.isnan(np.array(values, dtype=float))]
                        else:
                            rounds_arr = np.arange(1, len(arr) + 1, dtype=float)
                        kernel = np.ones(WINDOW) / WINDOW
                        smoothed = np.convolve(arr, kernel, mode='valid')
                        step = max(1, len(smoothed) // 300)
                        for t in range(0, len(smoothed), step):
                            round_idx = min(t + WINDOW - 1, len(rounds_arr) - 1)
                            all_rows[metric_key].append(dict(
                                dataset=dataset, feedback=feedback, agent=agent,
                                seed=seed, round=float(rounds_arr[round_idx]), value=smoothed[t],
                            ))

    result = {}
    for metric_key, label in diag_metrics.items():
        if all_rows[metric_key]:
            df = pd.DataFrame(all_rows[metric_key])
            agg = df.groupby(['dataset', 'feedback', 'agent', 'round']).agg(
                mean=('value', 'mean'), std=('value', 'std'),
            ).reset_index()
            result[metric_key] = (agg, label)

    return result


def make_diagnostic_plots(organized, datasets, feedback='semi_bandit',
                          output_dir='results/figures', fmt='svg',
                          line_agents=None):
    diag_dfs = build_diagnostic_dataframes(organized)
    if not diag_dfs:
        print('No diagnostic data available.')
        return

    requested = line_agents or LINE_AGENTS

    for ds in datasets:
        label = DATASET_LABELS.get(ds, ds)
        plots = []

        for metric_key, (df_agg, metric_label) in diag_dfs.items():
            d = df_agg[(df_agg['dataset'] == ds) & (df_agg['feedback'] == feedback)]
            if d.empty:
                continue

            available_agents = set(d['agent'].unique())
            agents = [a for a in requested if a in available_agents] if requested else sorted(available_agents)
            if not agents:
                agents = sorted(available_agents)

            d = d[d['agent'].isin(agents)]
            colors = {a: COLORS.get(a, '#333333') for a in agents}

            p = (
                ggplot(d, aes(x='round', y='mean', color='agent'))
                + geom_line(size=1)
                + scale_color_manual(values=colors)
                + labs(x='Round', y=metric_label, title=f'{metric_label}')
                + THEME_LINE
            )
            plots.append(p)

        if not plots:
            continue

        ncol = min(2, len(plots))
        fig = gggrid(plots, ncol=ncol, align=True) + ggsize(700 * ncol, 400 * ((len(plots) + ncol - 1) // ncol))
        outpath = os.path.abspath(os.path.join(output_dir, f'diagnostics_{ds}.{fmt}'))
        save_lets_plot(fig, outpath, fmt)
        print(f'Saved diagnostics: {outpath}')


def load_final_regrets(path):
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


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation results')
    parser.add_argument('--input', type=str, nargs='+',
                        default=[
                            'results/eval/metrics',
                            'results/eval_online/metrics',
                        ],
                        help='One or more directories containing trajectories_*.pkl files')
    parser.add_argument('--regrets_json', type=str, default=None,
                        help='Path to final_eval_regrets.json (for bar charts with more datasets)')
    parser.add_argument('--output', type=str, default='results/figures',
                        help='Output directory for plots')
    parser.add_argument('--format', type=str, default='pdf', choices=['pdf', 'svg', 'png'],
                        help='Output format for plots (default: vector PDF)')
    parser.add_argument('--feedback', type=str, default='semi_bandit',
                        help='Feedback type to plot')
    parser.add_argument('--line_agents', type=str, nargs='*', default=None,
                        help='Agents to include in line plots (default: all)')
    parser.add_argument('--exclude_nl', action='store_true', default=True,
                        help='Exclude NL (nonlinear) agent variants from plots (default: True)')
    parser.add_argument('--include_nl', action='store_true',
                        help='Include NL (nonlinear) agent variants in plots')
    args = parser.parse_args()

    exclude_nl = args.exclude_nl and not args.include_nl

    os.makedirs(args.output, exist_ok=True)


    organized = {}
    input_dirs = args.input
    for input_dir in input_dirs:

        if os.path.isfile(input_dir):
            input_dir = os.path.dirname(input_dir)
        per_ds_files = sorted(glob_mod.glob(os.path.join(input_dir, 'trajectories_*.pkl')))
        for f in per_ds_files:
            print(f'Loading {os.path.basename(f)} from {input_dir}...')
            part = load_trajectories(f)
            for ds, fb_dict in part.items():
                if ds not in organized:
                    organized[ds] = {}
                for fb_mode, agent_dict in fb_dict.items():
                    if fb_mode not in organized[ds]:
                        organized[ds][fb_mode] = {}
                    organized[ds][fb_mode].update(agent_dict)
    if not organized:
        print('No trajectory files found. Run evaluate.py or evaluate_online.py first.')
        organized = None


    df_bar_extra = None
    regrets_path = args.regrets_json
    if regrets_path is None:
        candidate_paths = []
        for input_path in input_dirs:
            normalized = os.path.dirname(input_path) if os.path.isfile(input_path) else input_path
            candidate_path = os.path.join(normalized, 'final_eval_regrets.json')
            if candidate_path not in candidate_paths:
                candidate_paths.append(candidate_path)
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                regrets_path = candidate_path
                break
    if regrets_path and os.path.exists(regrets_path):
        print(f'Loading final regrets from {regrets_path}...')
        df_bar_extra = load_final_regrets(regrets_path)


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


    if exclude_nl and not df_bar_combined.empty:
        nl_mask = df_bar_combined['agent'].str.contains('-NL$', regex=True)
        n_excluded = nl_mask.sum()
        if n_excluded > 0:
            df_bar_combined = df_bar_combined[~nl_mask]
            print(f'Excluded {n_excluded} NL variant entries from bar charts')


    if BAR_AGENTS and not df_bar_combined.empty:
        df_bar_combined = df_bar_combined[df_bar_combined['agent'].isin(BAR_AGENTS)]


    if df_cum_agg is not None and df_pr_agg is not None:
        traj_datasets = sorted(set(df_cum_agg['dataset'].unique()))
        make_line_plots(df_cum_agg, df_pr_agg, traj_datasets, args.feedback,
                        args.output, args.format, args.line_agents)


    if not df_bar_combined.empty:
        make_bar_charts(df_bar_combined, bar_datasets, args.feedback,
                        args.output, args.format)


    if organized:
        make_diagnostic_plots(organized, sorted(organized.keys()), args.feedback,
                              args.output, args.format, args.line_agents)

    print('Done.')


if __name__ == '__main__':
    main()

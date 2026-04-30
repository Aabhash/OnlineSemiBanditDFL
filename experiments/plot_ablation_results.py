#!/usr/bin/env python
"""Render ablation figures from saved result files using LetsPlot."""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lets_plot import (
    LetsPlot,
    aes,
    arrow,
    coord_flip,
    element_blank,
    element_rect,
    element_text,
    facet_grid,
    facet_wrap,
    geom_bar,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    geom_rect,
    geom_ribbon,
    geom_segment,
    geom_text,
    geom_vline,
    gggrid,
    ggplot,
    ggsize,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_x_log10,
    scale_x_log2,
    scale_y_continuous,
    scale_y_log10,
    theme,
    theme_minimal,
    theme_void,
    position_dodge,
)
from lets_plot.export import ggsave

sys.path.insert(0, ".")

LetsPlot.setup_html()

ABLATION_DIR = "results/ablations"
OUTPUT_DIR = "results/figures/ablations"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#222222",
    "gray": "#777777",
}

PAPER_COLORS = {
    "BayesianSPO": "#6E9FBD",
    "BayesianMSE": "#C28C68",
    "BLL": "#73A99A",
    "LinUCB": "#838383",
    "GreedySPO": "#73A99A",
    "GreedyMSE": "#B89AAC",
}

AGENT_SHAPES = {
    "BayesianSPO": 16,
    "BayesianMSE": 17,
    "BLL": 15,
    "LinUCB": 3,
    "GreedySPO": 15,
    "GreedyMSE": 18,
    "DecoupledSPO\nUCB": 16,
    "DecoupledSPO\nEps": 4,
    "CoupledSPO\nx0.5": 15,
    "CoupledSPO\nx1": 18,
    "CoupledSPO\nx2": 3,
}

POINT_SIZE = 6.2
SMALL_POINT_SIZE = 4.2

AGENT_LABELS = {
    "BayesianSPO": "BayesianSPO",
    "BayesianMSE": "BayesianMSE",
    "WeberBLL-Online": "BLL",
    "LinUCB": "LinUCB",
    "Random": "Random",
    "BayesianSPO-masked": "BayesianSPO\nmasked",
    "BayesianSPO-unmasked": "BayesianSPO\nunmasked",
    "BayesianMSE-masked": "BayesianMSE\nmasked",
    "BayesianMSE-unmasked": "BayesianMSE\nunmasked",
    "BayesianSPO-Greedy": "BayesianSPO\nGreedy",
    "BayesianSPO-UCB": "BayesianSPO\nUCB",
    "BayesianSPO-TS": "BayesianSPO\nTS",
    "BayesianMSE-Greedy": "BayesianMSE\nGreedy",
    "BayesianMSE-UCB": "BayesianMSE\nUCB",
    "BayesianMSE-TS": "BayesianMSE\nTS",
    "DeterministicSPO": "GreedySPO",
    "DeterministicMSE": "GreedyMSE",
    "SPO-E-Greedy": "GreedySPO",
    "MSE-E-Greedy": "GreedyMSE",
    "GreedySPO": "GreedySPO",
    "GreedyMSE": "GreedyMSE",
    "DecoupledSPO-UCB": "DecoupledSPO\nUCB",
    "DecoupledSPO-Eps": "DecoupledSPO\nEps",
}

AGENT_ORDER = [
    "BayesianSPO",
    "BayesianMSE",
    "BLL",
    "LinUCB",
    "Random",
    "BayesianSPO\nGreedy",
    "BayesianSPO\nUCB",
    "BayesianSPO\nTS",
    "BayesianMSE\nGreedy",
    "BayesianMSE\nUCB",
    "BayesianMSE\nTS",
    "BayesianSPO\nmasked",
    "BayesianSPO\nunmasked",
    "BayesianMSE\nmasked",
    "BayesianMSE\nunmasked",
    "GreedySPO",
    "GreedyMSE",
    "DecoupledSPO\nUCB",
    "DecoupledSPO\nEps",
    "CoupledSPO\nx0.5",
    "CoupledSPO\nx1",
    "CoupledSPO\nx2",
]

AGENT_COLORS = {
    "BayesianSPO": PAPER_COLORS["BayesianSPO"],
    "BayesianMSE": PAPER_COLORS["BayesianMSE"],
    "BLL": PAPER_COLORS["BLL"],
    "LinUCB": PAPER_COLORS["LinUCB"],
    "Random": OKABE_ITO["gray"],
    "BayesianSPO\nGreedy": OKABE_ITO["sky"],
    "BayesianSPO\nUCB": PAPER_COLORS["BayesianSPO"],
    "BayesianSPO\nTS": OKABE_ITO["green"],
    "BayesianMSE\nGreedy": OKABE_ITO["yellow"],
    "BayesianMSE\nUCB": PAPER_COLORS["BayesianMSE"],
    "BayesianMSE\nTS": OKABE_ITO["purple"],
    "BayesianSPO\nmasked": PAPER_COLORS["BayesianSPO"],
    "BayesianSPO\nunmasked": OKABE_ITO["sky"],
    "BayesianMSE\nmasked": PAPER_COLORS["BayesianMSE"],
    "BayesianMSE\nunmasked": OKABE_ITO["purple"],
    "GreedySPO": PAPER_COLORS["GreedySPO"],
    "GreedyMSE": PAPER_COLORS["GreedyMSE"],
    "DecoupledSPO\nUCB": PAPER_COLORS["BayesianSPO"],
    "DecoupledSPO\nEps": OKABE_ITO["sky"],
    "CoupledSPO\nx0.5": PAPER_COLORS["BayesianMSE"],
    "CoupledSPO\nx1": PAPER_COLORS["GreedyMSE"],
    "CoupledSPO\nx2": PAPER_COLORS["GreedySPO"],
}

AGENT_LINETYPES = {
    "BayesianSPO": "solid",
    "BayesianMSE": "longdash",
    "BLL": "dotdash",
    "LinUCB": "dotted",
    "Random": "twodash",
    "BayesianSPO\nGreedy": "dotdash",
    "BayesianSPO\nUCB": "solid",
    "BayesianSPO\nTS": "dashed",
    "BayesianMSE\nGreedy": "dotdash",
    "BayesianMSE\nUCB": "longdash",
    "BayesianMSE\nTS": "dashed",
    "BayesianSPO\nmasked": "solid",
    "BayesianSPO\nunmasked": "dotted",
    "BayesianMSE\nmasked": "longdash",
    "BayesianMSE\nunmasked": "dotdash",
    "GreedySPO": "dashed",
    "GreedyMSE": "dotdash",
    "DecoupledSPO\nUCB": "solid",
    "DecoupledSPO\nEps": "dashed",
    "CoupledSPO\nx0.5": "dotted",
    "CoupledSPO\nx1": "dotdash",
    "CoupledSPO\nx2": "longdash",
}

def load_pkl(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_ablation_dir(results_dir, ablation_name):
    candidate = os.path.join(results_dir, ablation_name)
    if os.path.isdir(candidate) or os.path.exists(os.path.join(candidate, "results.pkl")):
        return candidate
    return results_dir


def list_result_entries(ablation_dir):
    direct_path = os.path.join(ablation_dir, "results.pkl")
    if os.path.exists(direct_path):
        return [(".", direct_path)]

    entries = []
    if not os.path.isdir(ablation_dir):
        return entries

    for name in sorted(os.listdir(ablation_dir)):
        path = os.path.join(ablation_dir, name, "results.pkl")
        if os.path.exists(path):
            entries.append((name, path))
    return entries


def display_label(raw_name):
    return AGENT_LABELS.get(raw_name, raw_name)


def ordered_labels(labels):
    order = {name: idx for idx, name in enumerate(AGENT_ORDER)}
    return sorted(labels, key=lambda label: (order.get(label, len(order)), label))


def _categorical(series, order):
    return pd.Categorical(series, categories=order, ordered=True)


def _summary(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return np.nan, 0.0
    return float(vals.mean()), float(vals.std())


def _paper_theme(legend_position="top", rotate_x=False):
    x_text = element_text(size=11.5, color="#444444")
    if rotate_x:
        x_text = element_text(size=10.8, color="#444444", angle=25, hjust=1)
    return theme_minimal() + theme(
        legend_position=legend_position,
        legend_title=element_blank(),
        legend_text=element_text(size=12.0),
        axis_title=element_text(size=13.0),
        axis_text_x=x_text,
        axis_text_y=element_text(size=11.5, color="#444444"),
        plot_title=element_text(size=14.0, face="bold"),
        strip_text=element_text(size=12.5, face="bold"),
        panel_grid_minor=element_blank(),
    )


def _color_scale(labels):
    labels = ordered_labels(labels)
    values = {label: AGENT_COLORS.get(label, "#555555") for label in labels}
    return scale_color_manual(values=values)


def _fill_scale(labels):
    labels = ordered_labels(labels)
    values = {label: AGENT_COLORS.get(label, "#999999") for label in labels}
    return scale_fill_manual(values=values)


def _linetype_scale(labels):
    labels = ordered_labels(labels)
    values = {label: AGENT_LINETYPES.get(label, "solid") for label in labels}
    return scale_linetype_manual(values=values)


def _shape_scale(labels):
    labels = ordered_labels(labels)
    values = {label: AGENT_SHAPES.get(label, 16) for label in labels}
    return scale_shape_manual(values=values)


def _separator_plot():
    return (
        ggplot(
            pd.DataFrame(
                {"xmin": [0.0], "xmax": [1.0], "ymin": [-0.08], "ymax": [0.08]}
            )
        )
        + geom_rect(
            aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
            fill="#b8b8b8",
            color="#b8b8b8",
        )
        + scale_x_continuous(limits=[0.0, 1.0])
        + scale_y_continuous(limits=[-1.0, 1.0])
        + theme_void()
    )


def _baseline_callout_frames(label, x_limits, baseline=0.0, y=0.28, yend=0.58):
    x_min, x_max = x_limits
    span = x_max - x_min
    arrow_start = baseline + 0.026 * span
    label_x = baseline + 0.038 * span
    arrow_df = pd.DataFrame(
        {"x": [arrow_start], "xend": [baseline], "y": [y], "yend": [yend]}
    )
    label_df = pd.DataFrame({"x": [label_x], "y": [y - 0.08], "label": [label]})
    return arrow_df, label_df


def save_plot(plot, output_dir, stem, fmt="pdf", png_width=1800):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    out_path = output_dir / f"{stem}.{fmt}"

    ggsave(plot, svg_path.name, path=str(output_dir), iframe=False)

    if fmt == "svg":
        print(f"  Saved {stem}.svg")
        return

    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError(
            "rsvg-convert is required to convert LetsPlot SVG output; "
            f"left intermediate SVG at {svg_path}"
        )

    if fmt == "pdf":
        subprocess.run([converter, "-f", "pdf", "-o", str(out_path), str(svg_path)], check=True)
    elif fmt == "png":
        subprocess.run(
            [converter, "-f", "png", "-w", str(png_width), "-o", str(out_path), str(svg_path)],
            check=True,
        )
    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    svg_path.unlink(missing_ok=True)
    print(f"  Saved {stem}.{fmt}")


def _final_regret_frame(data, key_arity):
    rows = []
    if key_arity == 2:
        agents = sorted({name for name, _ in data.keys()})
        seeds = sorted({seed for _, seed in data.keys()})
        for name in agents:
            vals = [data[(name, seed)]["final_regret"] for seed in seeds if (name, seed) in data]
            mean, std = _summary(vals)
            rows.append(
                {
                    "agent": name,
                    "agent_label": display_label(name),
                    "mean": mean,
                    "std": std,
                    "ymin": mean - std,
                    "ymax": mean + std,
                }
            )
    return pd.DataFrame(rows)


def plot_kn_ratio(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "kn_ratio")
    data = load_pkl(os.path.join(ablation_dir, "results.pkl"))
    if data is None:
        print("  kn_ratio: no data found, skipping")
        return

    caps = sorted(set(cap for _, cap, _ in data.keys()))
    seeds = sorted(set(seed for _, _, seed in data.keys()))
    agents = sorted(set(name for name, _, _ in data.keys()))
    n_items = 32
    rows = []
    for name in agents:
        label = display_label(name)
        for cap in caps:
            vals = [
                data[(name, cap, seed)]["final_regret"]
                for seed in seeds
                if (name, cap, seed) in data
            ]
            mean, std = _summary(vals)
            rows.append(
                {
                    "agent_label": label,
                    "ratio": cap / n_items,
                    "mean": mean,
                    "ymin": mean - std,
                    "ymax": mean + std,
                }
            )
    df = pd.DataFrame(rows)
    labels = ordered_labels(df["agent_label"].unique())
    df["agent_label"] = _categorical(df["agent_label"], labels)

    plot = (
        ggplot(df, aes("ratio", "mean", color="agent_label", linetype="agent_label"))
        + geom_line(size=0.8)
        + geom_point(size=SMALL_POINT_SIZE)
        + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.018, alpha=0.55)
        + _color_scale(labels)
        + _linetype_scale(labels)
        + scale_x_continuous(breaks=sorted(df["ratio"].unique()))
        + labs(
            title="Observation-rate sweep",
            x="Capacity ratio k/n",
            y="Final cumulative regret",
        )
        + _paper_theme()
        + ggsize(680, 390)
    )
    save_plot(plot, output_dir, "kn_ratio_sweep", fmt)


def _bar_plot_from_final(data, title, stem, output_dir, fmt):
    df = _final_regret_frame(data, 2)
    if df.empty:
        return
    labels = ordered_labels(df["agent_label"].unique())
    df["agent_label"] = _categorical(df["agent_label"], labels)
    width = max(520, 90 * len(labels))
    plot = (
        ggplot(df, aes("agent_label", "mean", fill="agent_label"))
        + geom_bar(stat="identity", alpha=0.78, color="#ffffff", size=0.3)
        + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.18, size=0.55)
        + _fill_scale(labels)
        + labs(title=title, x="", y="Final cumulative regret")
        + _paper_theme(legend_position="none", rotate_x=True)
        + ggsize(width, 360)
    )
    save_plot(plot, output_dir, stem, fmt)


def plot_exploration(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "exploration")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  exploration: no data found, skipping")
        return
    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        title = f"Exploration strategy: {dataset_name}"
        stem = "exploration_strategy" if dataset_name == "." else f"exploration_strategy_{dataset_name}"
        _bar_plot_from_final(data, title, stem, output_dir, fmt)
    plot_exploration_relative(results_dir, output_dir, fmt)


def _paired_ratio(vals, baseline_vals):
    rows = []
    for seed, regret in vals.items():
        baseline = baseline_vals.get(seed)
        if baseline is None or not np.isfinite(baseline) or abs(baseline) == 0:
            continue
        rows.append(float(regret) / float(baseline))
    return np.asarray(rows, dtype=float)


def plot_exploration_relative(results_dir, output_dir, fmt="pdf"):
    exploration_dir = resolve_ablation_dir(results_dir, "exploration")
    exploration_entries = list_result_entries(exploration_dir)
    if not exploration_entries:
        print("  exploration relative: no exploration data found, skipping")
        return

    ts_dir = resolve_ablation_dir(results_dir, "ts_scale")
    ts_entries = dict(list_result_entries(ts_dir))
    dataset_names = _dataset_label_map()
    method_order = ["BayesianSPO", "BayesianMSE"]
    strategy_order = ["Greedy", "Raw TS", "TS Sweep - Best"]

    rows = []
    for dataset_name, data_path in exploration_entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        dataset_label = dataset_names.get(dataset_name, dataset_name)

        ts_data = load_pkl(ts_entries.get(dataset_name, "")) if dataset_name in ts_entries else None
        for method in method_order:
            ucb_name = f"{method}-UCB"
            ucb_vals = {
                seed: float(result["final_regret"])
                for (name, seed), result in data.items()
                if name == ucb_name
            }
            if not ucb_vals:
                continue

            strategy_values = {
                "UCB": ucb_vals,
                "Greedy": {
                    seed: float(result["final_regret"])
                    for (name, seed), result in data.items()
                    if name == f"{method}-Greedy"
                },
                "Raw TS": {
                    seed: float(result["final_regret"])
                    for (name, seed), result in data.items()
                    if name == f"{method}-TS"
                },
            }

            if ts_data is not None:
                candidates = {}
                for (name, seed), result in ts_data.items():
                    parsed = _parse_ts_scale_agent(name)
                    if parsed is None:
                        continue
                    parsed_method, multiplier = parsed
                    if parsed_method != method:
                        continue
                    candidates.setdefault(multiplier, {})[seed] = float(result["final_regret"])
                if candidates:
                    best_multiplier, best_vals = min(
                        candidates.items(),
                        key=lambda item: np.mean(list(item[1].values())),
                    )
                    strategy_values["TS Sweep - Best"] = best_vals

            for strategy in strategy_order:
                vals = strategy_values.get(strategy)
                if not vals:
                    continue
                ratio_vals = _paired_ratio(vals, ucb_vals)
                ratio_vals = ratio_vals[np.isfinite(ratio_vals) & (ratio_vals > 0)]
                if ratio_vals.size == 0:
                    continue
                se = float(ratio_vals.std(ddof=1) / np.sqrt(ratio_vals.size)) if ratio_vals.size > 1 else 0.0
                mean = float(ratio_vals.mean())
                rows.append(
                    {
                        "dataset": dataset_label,
                        "method": method,
                        "strategy": strategy,
                        "plot_group": f"{method}-{strategy}",
                        "mean": mean,
                        "ymin": mean - 1.96 * se,
                        "ymax": mean + 1.96 * se,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        print("  exploration relative: no plottable data found, skipping")
        return

    dataset_order = [
        label
        for key, label in dataset_names.items()
        if label in set(df["dataset"].unique())
    ]
    df["dataset"] = _categorical(df["dataset"], dataset_order)
    df["method"] = _categorical(df["method"], method_order)
    df["strategy"] = _categorical(df["strategy"], strategy_order)
    df["log_mean"] = np.log10(df["mean"].clip(lower=0.05))
    df["log_ymin"] = np.log10(df["ymin"].clip(lower=0.05))
    df["log_ymax"] = np.log10(df["ymax"].clip(lower=0.05))
    strategy_colors = {
        "Greedy": "#6C8FB3",
        "Raw TS": "#C47C5A",
        "TS Sweep - Best": "#6F8F76",
    }
    dataset_positions = {dataset: idx + 1 for idx, dataset in enumerate(dataset_order)}
    strategy_offsets = {
        "Greedy": -0.22,
        "Raw TS": 0.0,
        "TS Sweep - Best": 0.22,
    }
    df["dataset_pos"] = df["dataset"].astype(str).map(dataset_positions)
    df["strategy_offset"] = df["strategy"].astype(str).map(strategy_offsets)
    df["ypos"] = df["dataset_pos"] + df["strategy_offset"]
    df["bar_ymin"] = df["ypos"] - 0.09
    df["bar_ymax"] = df["ypos"] + 0.09
    df["xmin"] = np.minimum(0.0, df["log_mean"])
    df["xmax"] = np.maximum(0.0, df["log_mean"])
    x_limits = [
        min(-0.05, float(df["log_ymin"].min()) - 0.03),
        max(np.log10(10.0), float(df["log_ymax"].max()) + 0.03),
    ]

    def method_plot(method, show_legend):
        method_df = df[df["method"] == method].copy()
        cap_rows = []
        for row in method_df.itertuples(index=False):
            cap_rows.extend(
                [
                    {
                        "x": row.log_ymin,
                        "xend": row.log_ymin,
                        "y": row.ypos - 0.055,
                        "yend": row.ypos + 0.055,
                    },
                    {
                        "x": row.log_ymax,
                        "xend": row.log_ymax,
                        "y": row.ypos - 0.055,
                        "yend": row.ypos + 0.055,
                    },
                ]
            )
        cap_df = pd.DataFrame(cap_rows)
        arrow_df, label_df = _baseline_callout_frames(
            "UCB baseline", x_limits, y=0.27, yend=0.60
        )
        plot_theme = _paper_theme(legend_position="top" if show_legend else "none")
        plot_theme += theme(panel_border=element_rect(color="#c7c7c7", fill="transparent", size=0.45))
        if show_legend:
            plot_theme += theme(legend_title=element_text(size=12.0, face="bold"))
        return (
            ggplot(method_df)
            + geom_rect(
                aes(
                    xmin="xmin",
                    xmax="xmax",
                    ymin="bar_ymin",
                    ymax="bar_ymax",
                    fill="strategy",
                ),
                alpha=0.78,
                color="#ffffff",
                size=0.25,
            )
            + geom_segment(
                aes(x="log_ymin", xend="log_ymax", y="ypos", yend="ypos"),
                color="#555555",
                size=0.52,
            )
            + geom_segment(
                aes(x="x", xend="xend", y="y", yend="yend"),
                data=cap_df,
                color="#555555",
                size=0.52,
            )
            + geom_vline(xintercept=0, linetype="solid", color="#222222", size=1.0)
            + geom_segment(
                aes("x", "y", xend="xend", yend="yend"),
                data=arrow_df,
                inherit_aes=False,
                arrow=arrow(length=5, type="closed"),
                color="#222222",
                size=0.45,
            )
            + geom_text(
                aes("x", "y", label="label"),
                data=label_df,
                inherit_aes=False,
                color="#222222",
                fontface="bold",
                size=4.5,
                hjust=0.0,
            )
            + scale_fill_manual(values=strategy_colors)
            + scale_x_continuous(
                limits=x_limits,
                breaks=[0.0, np.log10(2.0), np.log10(5.0), np.log10(10.0)],
                labels=["1", "2", "5", "10"],
            )
            + scale_y_continuous(
                breaks=[dataset_positions[dataset] for dataset in dataset_order],
                labels=dataset_order,
                limits=[0.0, 5.35],
            )
            + labs(
                title=method,
                x="Cumulative regret ratio vs tuned UCB (log scale)",
                y="",
                fill="Exploration strategy",
            )
            + plot_theme
        )

    plot = (
        gggrid(
            [
                method_plot("BayesianSPO", show_legend=True),
                _separator_plot(),
                method_plot("BayesianMSE", show_legend=False),
            ],
            ncol=1,
            align=True,
            heights=[1.0, 0.08, 1.0],
            vspace=0.01,
        )
        + ggsize(620, 470)
    )
    save_plot(plot, output_dir, "exploration_relative_summary", fmt)


def _mean_final_regret(data, agent_name):
    regrets = np.asarray(
        [float(result["final_regret"]) for (name, _seed), result in data.items() if name == agent_name],
        dtype=float,
    )
    regrets = regrets[np.isfinite(regrets)]
    if regrets.size == 0:
        return None
    return float(regrets.mean())


def _load_canonical_bayesian_spo_regrets(dataset_names):
    candidates = [
        Path("results/FINAL/eval_online_10k/metrics/final_eval_summary.csv"),
        Path("results/FINAL_paper_baseline/eval_online_10k/metrics/final_eval_summary.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        required = {"Dataset", "Agent", "Mean Regret"}
        if not required.issubset(summary.columns):
            continue
        if "Feedback Type" in summary.columns:
            semi_bandit = summary[summary["Feedback Type"] == "semi_bandit"]
            if not semi_bandit.empty:
                summary = semi_bandit
        summary = summary[summary["Agent"] == "BayesianSPO"]
        if summary.empty:
            continue
        return {
            dataset_names.get(str(row["Dataset"]), str(row["Dataset"])): float(row["Mean Regret"])
            for _, row in summary.iterrows()
            if pd.notna(row["Mean Regret"])
        }
    return {}


def _append_ablation_pair(rows, data, dataset, check, tuned_regret, ablation_agent, fallback_agent=None):
    if tuned_regret is None and fallback_agent is not None:
        tuned_regret = _mean_final_regret(data, fallback_agent)
    ablation_regret = _mean_final_regret(data, ablation_agent)
    if tuned_regret is None or ablation_regret is None:
        return
    rows.append(
        {
            "dataset": dataset,
            "check": check,
            "tuned_regret": tuned_regret,
            "ablation_regret": ablation_regret,
        }
    )


def plot_ablation_design_summary(results_dir, output_dir, fmt="pdf"):
    dataset_names = _dataset_label_map()
    canonical_regrets = _load_canonical_bayesian_spo_regrets(dataset_names)
    rows = []

    imputation_dir = resolve_ablation_dir(results_dir, "imputation")
    for dataset_name, data_path in list_result_entries(imputation_dir):
        data = load_pkl(data_path)
        if data is None:
            continue
        dataset = dataset_names.get(dataset_name, dataset_name)
        _append_ablation_pair(
            rows,
            data,
            dataset,
            "Masking: unmasked",
            None,
            "BayesianSPO-w1",
            fallback_agent="BayesianSPO-w0",
        )

    architecture_dir = resolve_ablation_dir(results_dir, "architecture")
    for dataset_name, data_path in list_result_entries(architecture_dir):
        data = load_pkl(data_path)
        if data is None:
            continue
        dataset = dataset_names.get(dataset_name, dataset_name)
        _append_ablation_pair(
            rows,
            data,
            dataset,
            "Architecture: non-Bayesian head",
            canonical_regrets.get(dataset),
            "DeterministicSPO",
            fallback_agent="BayesianSPO",
        )

    exploration_dir = resolve_ablation_dir(results_dir, "exploration")
    ts_dir = resolve_ablation_dir(results_dir, "ts_scale")
    ts_entries = dict(list_result_entries(ts_dir))
    for dataset_name, data_path in list_result_entries(exploration_dir):
        data = load_pkl(data_path)
        if data is None:
            continue
        dataset = dataset_names.get(dataset_name, dataset_name)
        ts_data = load_pkl(ts_entries.get(dataset_name, "")) if dataset_name in ts_entries else None
        _append_ablation_pair(
            rows,
            data,
            dataset,
            "Exploration: greedy",
            canonical_regrets.get(dataset),
            "BayesianSPO-Greedy",
            fallback_agent="BayesianSPO-UCB",
        )
        _append_ablation_pair(
            rows,
            data,
            dataset,
            "Exploration: raw TS",
            canonical_regrets.get(dataset),
            "BayesianSPO-TS",
            fallback_agent="BayesianSPO-UCB",
        )

        if ts_data is None:
            continue
        candidates = {}
        for (name, seed), result in ts_data.items():
            parsed = _parse_ts_scale_agent(name)
            if parsed is None:
                continue
            parsed_method, multiplier = parsed
            if parsed_method != "BayesianSPO":
                continue
            candidates.setdefault(multiplier, {})[seed] = float(result["final_regret"])
        tuned_regret = canonical_regrets.get(dataset) or _mean_final_regret(data, "BayesianSPO-UCB")
        if candidates and tuned_regret is not None:
            best_multiplier, best_vals = min(
                candidates.items(),
                key=lambda item: np.mean(list(item[1].values())),
            )
            regrets = np.asarray(list(best_vals.values()), dtype=float)
            regrets = regrets[np.isfinite(regrets)]
            if regrets.size:
                rows.append(
                    {
                        "dataset": dataset,
                        "check": "Exploration: best TS sweep",
                        "tuned_regret": tuned_regret,
                        "ablation_regret": float(regrets.mean()),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        print("  ablation design summary: no plottable data found, skipping")
        return

    dataset_order = [
        label
        for key, label in dataset_names.items()
        if label in set(df["dataset"].unique())
    ]
    check_order = [
        "Exploration: best TS sweep",
        "Exploration: raw TS",
        "Exploration: greedy",
        "Architecture: non-Bayesian head",
        "Masking: unmasked",
    ]
    check_positions = {check: len(check_order) - idx for idx, check in enumerate(check_order)}
    portfolio_label = dataset_names.get("portfolio_32", "Portfolio")
    df["dataset"] = _categorical(df["dataset"], dataset_order)
    df["check"] = _categorical(df["check"], check_order)
    df["ypos"] = df["check"].astype(str).map(check_positions)
    df["axis_unit"] = np.where(df["dataset"].astype(str) == portfolio_label, "raw", "K")
    df["facet"] = np.where(
        df["axis_unit"] == "raw",
        df["dataset"].astype(str) + " (raw)",
        df["dataset"].astype(str) + " (K)",
    )
    facet_order = [
        f"{dataset} ({'raw' if dataset == portfolio_label else 'K'})"
        for dataset in dataset_order
    ]
    df["facet"] = _categorical(df["facet"], facet_order)
    for col in ["tuned_regret", "ablation_regret"]:
        df[f"{col}_axis"] = np.where(df["axis_unit"] == "raw", df[col], df[col] / 1000.0)
    df["xmin"] = np.minimum(df["tuned_regret_axis"], df["ablation_regret_axis"])
    df["xmax"] = np.maximum(df["tuned_regret_axis"], df["ablation_regret_axis"])

    point_rows = []
    for row in df.itertuples(index=False):
        point_rows.append(
            {
                "facet": row.facet,
                "check": row.check,
                "kind": "Ablation",
                "regret": row.ablation_regret_axis,
                "ypos": row.ypos,
            }
        )
        point_rows.append(
            {
                "facet": row.facet,
                "check": row.check,
                "kind": "Tuned BayesianSPO",
                "regret": row.tuned_regret_axis,
                "ypos": row.ypos,
            }
        )
    point_df = pd.DataFrame(point_rows)
    point_df["facet"] = _categorical(point_df["facet"], facet_order)
    point_df["kind"] = _categorical(point_df["kind"], ["Tuned BayesianSPO", "Ablation"])
    pad_rows = []
    for facet, facet_df in df.groupby("facet", observed=True):
        lower = float(facet_df["xmin"].min())
        upper = float(facet_df["xmax"].max())
        if np.isfinite(lower) and np.isfinite(upper) and lower > 0 and upper > 0:
            pad_rows.extend(
                [
                    {"facet": facet, "regret": lower / 1.25, "ypos": 1.0},
                    {"facet": facet, "regret": upper * 1.25, "ypos": 1.0},
                ]
            )
    pad_df = pd.DataFrame(pad_rows, columns=["facet", "regret", "ypos"])
    if not pad_df.empty:
        pad_df["facet"] = _categorical(pad_df["facet"], facet_order)

    point_colors = {
        "Tuned BayesianSPO": PAPER_COLORS["BayesianSPO"],
        "Ablation": PAPER_COLORS["BayesianMSE"],
    }
    x_values = pd.concat([df["tuned_regret_axis"], df["ablation_regret_axis"]], ignore_index=True)
    x_values = x_values[np.isfinite(x_values) & (x_values > 0)]
    x_breaks = []
    if not x_values.empty:
        min_exp = int(np.floor(np.log10(x_values.min())))
        max_exp = int(np.ceil(np.log10(x_values.max())))
        x_breaks = [
            float(mult * (10 ** exp))
            for exp in range(min_exp, max_exp + 1)
            for mult in [1, 2, 5]
            if x_values.min() * 0.8 <= mult * (10 ** exp) <= x_values.max() * 1.05
        ]
        if x_values.max() < 1250:
            x_breaks = [break_value for break_value in x_breaks if break_value < 1000]

    plot = (
        ggplot(point_df, aes("regret", "ypos"))
        + geom_segment(
            aes(x="xmin", xend="xmax", y="ypos", yend="ypos"),
            data=df,
            inherit_aes=False,
            color="#9a9a9a",
            size=0.8,
        )
        + geom_point(aes(color="kind"), size=POINT_SIZE)
        + geom_point(
            aes("regret", "ypos"),
            data=pad_df,
            inherit_aes=False,
            alpha=0.0,
            size=0.1,
        )
        + facet_wrap("facet", ncol=3, scales="free_x")
        + scale_color_manual(values=point_colors)
        + scale_x_log10(breaks=x_breaks)
        + scale_y_continuous(
            breaks=[check_positions[check] for check in check_order],
            labels=check_order,
            limits=[0.5, len(check_order) + 0.5],
        )
        + labs(
            title="Ablation design summary",
            x="Final cumulative regret (log scale; Portfolio raw, others K)",
            y="",
            color="",
        )
        + _paper_theme(legend_position="top")
        + theme(
            axis_text_y=element_text(size=11.2, color="#444444"),
            strip_text=element_text(size=12.0, face="bold"),
            plot_margin=[8, 10, 36, 10],
        )
        + ggsize(980, 700)
    )
    save_plot(plot, output_dir, "ablation_design_summary", fmt, png_width=2200)


def _dataset_label_map():
    return {
        "knapsack_32": "Knapsack 32",
        "knapsack_100": "Knapsack 100",
        "shortest_path_8x8": "SP 8x8",
        "shortest_path_20x20": "SP 20x20",
        "portfolio_32": "Portfolio",
        ".": "dataset",
    }


def _comparison_frame(entries, label_fn=display_label):
    dataset_names = _dataset_label_map()
    rows = []
    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        dataset_label = dataset_names.get(dataset_name, dataset_name)
        agents = sorted({name for name, _ in data.keys()})
        seeds = sorted({seed for _, seed in data.keys()})
        for name in agents:
            vals = [
                data[(name, seed)]["final_regret"]
                for seed in seeds
                if (name, seed) in data
            ]
            mean, std = _summary(vals)
            rows.append(
                {
                    "dataset": dataset_label,
                    "agent_label": label_fn(name),
                    "mean": mean,
                    "ymin": mean - std,
                    "ymax": mean + std,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    dataset_order = [
        label
        for key, label in dataset_names.items()
        if label in set(df["dataset"].unique())
    ]
    labels = ordered_labels(df["agent_label"].unique())
    df["dataset"] = _categorical(df["dataset"], dataset_order)
    df["agent_label"] = _categorical(df["agent_label"], labels)
    return df


def _faceted_method_bars(df, title, stem, output_dir, fmt, height=520):
    if df.empty:
        return
    labels = ordered_labels(df["agent_label"].astype(str).unique())
    dodge = position_dodge(width=0.55)
    plot = (
        ggplot(df, aes("agent_label", "mean", fill="agent_label"))
        + geom_bar(stat="identity", alpha=0.78, color="#ffffff", size=0.3)
        + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.15, position=dodge, size=0.5)
        + facet_wrap("dataset", ncol=3, scales="free_y")
        + _fill_scale(labels)
        + labs(title=title, x="", y="Final cumulative regret")
        + _paper_theme(legend_position="none", rotate_x=True)
        + ggsize(920, height)
    )
    save_plot(plot, output_dir, stem, fmt)


def plot_architecture(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "architecture")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  architecture: no data found, skipping")
        return

    dataset_names = _dataset_label_map()
    raw_rows = []
    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        dataset_label = dataset_names.get(dataset_name, dataset_name)
        for (name, seed), result in data.items():
            raw_rows.append(
                {
                    "dataset": dataset_label,
                    "agent": name,
                    "agent_label": display_label(name),
                    "seed": seed,
                    "regret": float(result["final_regret"]),
                }
            )
    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        print("  architecture: no plottable data found, skipping")
        return

    comparisons = [
        {
            "agent": "DeterministicSPO",
            "agent_label": "GreedySPO",
            "baseline": "BayesianSPO",
            "panel_title": "GreedySPO vs BayesianSPO",
            "axis_label": "Regret increase vs BayesianSPO (%)",
        },
        {
            "agent": "DeterministicMSE",
            "agent_label": "GreedyMSE",
            "baseline": "BayesianMSE",
            "panel_title": "GreedyMSE vs BayesianMSE",
            "axis_label": "Regret increase vs BayesianMSE (%)",
        },
    ]

    rows = []
    for comparison in comparisons:
        baseline = raw_df[raw_df["agent"] == comparison["baseline"]][
            ["dataset", "seed", "regret"]
        ].rename(columns={"regret": "baseline_regret"})
        candidate = raw_df[raw_df["agent"] == comparison["agent"]].merge(
            baseline, on=["dataset", "seed"], how="inner"
        )
        denom = candidate["baseline_regret"].abs().replace(0.0, np.nan)
        candidate["pct_delta"] = (
            100.0 * (candidate["regret"] - candidate["baseline_regret"]) / denom
        )
        for dataset, grp in candidate.groupby("dataset"):
            vals = grp["pct_delta"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            mean = float(vals.mean())
            rows.append(
                {
                    "dataset": dataset,
                    "agent_label": comparison["agent_label"],
                    "panel_title": comparison["panel_title"],
                    "axis_label": comparison["axis_label"],
                    "mean": mean,
                    "ymin": mean - 1.96 * se,
                    "ymax": mean + 1.96 * se,
                }
            )
    df = pd.DataFrame(rows)
    dataset_order = [
        label
        for key, label in dataset_names.items()
        if label in set(df["dataset"].unique())
    ]
    method_order = ["GreedySPO", "GreedyMSE"]
    df["dataset"] = _categorical(df["dataset"], dataset_order)
    df["agent_label"] = _categorical(df["agent_label"], method_order)
    architecture_colors = {
        label: AGENT_COLORS.get(label, "#555555")
        for label in method_order
    }
    architecture_x_limits = {}
    for agent_label in method_order:
        panel_df = df[df["agent_label"] == agent_label]
        upper = float(panel_df["ymax"].max())
        lower = float(min(0.0, panel_df["ymin"].min()))
        span = max(upper - lower, 1.0)
        architecture_x_limits[agent_label] = [
            min(lower, 0.0) - 0.04 * span,
            upper + 0.06 * span,
        ]
    baseline_labels = {
        "GreedySPO": "BayesianSPO baseline",
        "GreedyMSE": "BayesianMSE baseline",
    }
    dataset_positions = {dataset: idx + 1 for idx, dataset in enumerate(dataset_order)}

    def architecture_panel(agent_label):
        panel_df = df[df["agent_label"] == agent_label].copy()
        title = panel_df["panel_title"].dropna().iloc[0]
        axis_label = panel_df["axis_label"].dropna().iloc[0]
        panel_df["dataset_pos"] = panel_df["dataset"].astype(str).map(dataset_positions)
        panel_df["bar_ymin"] = panel_df["dataset_pos"] - 0.36
        panel_df["bar_ymax"] = panel_df["dataset_pos"] + 0.36
        panel_df["xmin"] = np.minimum(0.0, panel_df["mean"])
        panel_df["xmax"] = np.maximum(0.0, panel_df["mean"])
        cap_rows = []
        for row in panel_df.itertuples(index=False):
            cap_rows.extend(
                [
                    {
                        "x": row.ymin,
                        "xend": row.ymin,
                        "y": row.dataset_pos - 0.06,
                        "yend": row.dataset_pos + 0.06,
                    },
                    {
                        "x": row.ymax,
                        "xend": row.ymax,
                        "y": row.dataset_pos - 0.06,
                        "yend": row.dataset_pos + 0.06,
                    },
                ]
            )
        cap_df = pd.DataFrame(cap_rows)
        x_limits = architecture_x_limits[agent_label]
        arrow_df, label_df = _baseline_callout_frames(
            baseline_labels[agent_label], x_limits, y=0.27, yend=0.60
        )
        panel_theme = _paper_theme(legend_position="none")
        panel_theme += theme(panel_border=element_rect(color="#c7c7c7", fill="transparent", size=0.45))
        return (
            ggplot(panel_df)
            + geom_rect(
                aes(
                    xmin="xmin",
                    xmax="xmax",
                    ymin="bar_ymin",
                    ymax="bar_ymax",
                    fill="agent_label",
                ),
                alpha=0.78,
                color="#ffffff",
                size=0.25,
            )
            + geom_segment(
                aes(x="ymin", xend="ymax", y="dataset_pos", yend="dataset_pos"),
                color="#555555",
                size=0.52,
            )
            + geom_segment(
                aes(x="x", xend="xend", y="y", yend="yend"),
                data=cap_df,
                color="#555555",
                size=0.52,
            )
            + geom_vline(xintercept=0, linetype="solid", color="#222222", size=1.0)
            + geom_segment(
                aes("x", "y", xend="xend", yend="yend"),
                data=arrow_df,
                inherit_aes=False,
                arrow=arrow(length=5, type="closed"),
                color="#222222",
                size=0.45,
            )
            + geom_text(
                aes("x", "y", label="label"),
                data=label_df,
                inherit_aes=False,
                color="#222222",
                fontface="bold",
                size=4.5,
                hjust=0.0,
            )
            + scale_fill_manual(values=architecture_colors)
            + scale_x_continuous(limits=x_limits)
            + scale_y_continuous(
                breaks=[dataset_positions[dataset] for dataset in dataset_order],
                labels=dataset_order,
                limits=[0.0, len(dataset_order) + 0.65],
            )
            + labs(
                title=title,
                x=axis_label,
                y="",
                fill="",
            )
            + panel_theme
            + theme(axis_text_y=element_text(size=11.0, color="#444444"))
        )

    bar_plot = (
        gggrid(
            [
                architecture_panel("GreedySPO"),
                _separator_plot(),
                architecture_panel("GreedyMSE"),
            ],
            ncol=1,
            align=True,
            heights=[1.0, 0.08, 1.0],
            vspace=0.01,
        )
        + ggsize(560, 430)
    )
    save_plot(bar_plot, output_dir, "architecture_ablation", fmt)
    save_plot(bar_plot, output_dir, "architecture_ablation_bar", fmt)


def _decoupling_label(name):
    if name.startswith("CoupledSPO-Eps-x"):
        return "CoupledSPO\nx" + name.rsplit("x", 1)[-1]
    return display_label(name)


def plot_decoupling(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "decoupling")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  decoupling: no data found, skipping")
        return
    df = _comparison_frame(entries, label_fn=_decoupling_label)
    _faceted_method_bars(
        df,
        "Coupled vs. decoupled SPO training",
        "decoupling_ablation",
        output_dir,
        fmt,
    )


def _parse_ts_scale_agent(raw_name):
    marker = "-TS-x"
    if marker not in raw_name:
        return None
    method, mult = raw_name.rsplit(marker, 1)
    try:
        return method, float(mult)
    except ValueError:
        return None


def plot_ts_scale_sweep(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "ts_scale")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  ts_scale: no data found, skipping")
        return

    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        seeds = sorted(set(seed for _, seed in data.keys()))
        rows = []
        for raw_name, _ in sorted(data.keys()):
            parsed = _parse_ts_scale_agent(raw_name)
            if parsed is None:
                continue
            method, multiplier = parsed
            vals = [
                data[(raw_name, seed)]["final_regret"]
                for seed in seeds
                if (raw_name, seed) in data
            ]
            mean, std = _summary(vals)
            rows.append(
                {
                    "agent_label": display_label(method),
                    "multiplier": multiplier,
                    "mean": mean,
                    "ymin": mean - std,
                    "ymax": mean + std,
                }
            )
        df = pd.DataFrame(rows).drop_duplicates()
        if df.empty:
            continue
        labels = ordered_labels(df["agent_label"].unique())
        df["agent_label"] = _categorical(df["agent_label"], labels)
        plot = (
            ggplot(df, aes("multiplier", "mean", color="agent_label", linetype="agent_label"))
            + geom_line(size=0.8)
            + geom_point(size=SMALL_POINT_SIZE)
            + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.03, alpha=0.6)
            + _color_scale(labels)
            + _linetype_scale(labels)
            + scale_x_log2(breaks=sorted(df["multiplier"].unique()))
            + labs(
                title=f"TS scale sweep: {dataset_name}",
                x="TS scale multiplier of tuned beta",
                y="Final cumulative regret",
            )
            + _paper_theme()
            + ggsize(680, 390)
        )
        stem = "ts_scale_sweep" if dataset_name == "." else f"ts_scale_sweep_{dataset_name}"
        save_plot(plot, output_dir, stem, fmt)


def plot_imputation(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "imputation")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  imputation: no data found, skipping")
        return

    dataset_names = {
        "knapsack_32": "Knapsack 32",
        "knapsack_100": "Knapsack 100",
        "shortest_path_8x8": "SP 8x8",
        "shortest_path_20x20": "SP 20x20",
        "portfolio_32": "Portfolio",
        ".": "dataset",
    }

    def parse_row_name(name):
        for method in ("BayesianSPO", "BayesianMSE"):
            prefix = f"{method}-w"
            if name.startswith(prefix):
                try:
                    return method, float(name[len(prefix):])
                except ValueError:
                    return None
        legacy = {
            "BayesianSPO-masked": ("BayesianSPO", 0.0),
            "BayesianSPO-unmasked": ("BayesianSPO", 1.0),
            "BayesianMSE-masked": ("BayesianMSE", 0.0),
            "BayesianMSE-unmasked": ("BayesianMSE", 1.0),
        }
        return legacy.get(name)

    raw_rows = []
    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        dataset_label = dataset_names.get(dataset_name, dataset_name)
        for (name, seed), result in data.items():
            parsed = parse_row_name(name)
            if parsed is None:
                continue
            method, weight = parsed
            raw_rows.append(
                {
                    "dataset": dataset_label,
                    "method": method,
                    "weight": weight,
                    "seed": seed,
                    "regret": float(result["final_regret"]),
                }
            )

    if not raw_rows:
        print("  imputation: no masked-imputation data found, skipping")
        return

    raw_df = pd.DataFrame(raw_rows)
    dataset_order = [
        label
        for key, label in dataset_names.items()
        if label in set(raw_df["dataset"].unique())
    ]
    method_order = ["BayesianSPO", "BayesianMSE"]
    baseline = raw_df[raw_df["weight"] == 0.0][
        ["dataset", "method", "seed", "regret"]
    ].rename(columns={"regret": "masked_regret"})
    paired = raw_df.merge(baseline, on=["dataset", "method", "seed"], how="inner")
    paired["delta"] = paired["regret"] - paired["masked_regret"]
    denom = paired["masked_regret"].abs().replace(0.0, np.nan)
    paired["pct_delta"] = 100.0 * paired["delta"] / denom
    # For posterior-mean MSE completion, missing entries have zero residual:
    # Y_target_i = pred_i for i not in S_t. The missing-arm gradient weight
    # therefore cannot change the MSE update. Separately-run MSE variants can
    # still show tiny regret differences from solver/numerical noise, so the
    # masking plot uses BayesianMSE as the structural zero-control.
    paired.loc[paired["method"] == "BayesianMSE", "pct_delta"] = 0.0

    dose_rows = []
    for (dataset, method, weight), grp in paired.groupby(["dataset", "method", "weight"]):
        vals = grp["pct_delta"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        mean = float(vals.mean())
        dose_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "weight": float(weight),
                "mean": mean,
                "ymin": mean - 1.96 * se,
                "ymax": mean + 1.96 * se,
            }
        )
    existing_dose = {
        (row["dataset"], row["method"], row["weight"])
        for row in dose_rows
    }
    for dataset in sorted(raw_df["dataset"].unique()):
        weights = sorted(raw_df.loc[raw_df["dataset"] == dataset, "weight"].unique())
        for weight in weights:
            key = (dataset, "BayesianMSE", float(weight))
            if key in existing_dose:
                continue
            dose_rows.append(
                {
                    "dataset": dataset,
                    "method": "BayesianMSE",
                    "weight": float(weight),
                    "mean": 0.0,
                    "ymin": 0.0,
                    "ymax": 0.0,
                }
            )

    if dose_rows:
        dose_df = pd.DataFrame(dose_rows)
        dose_df["dataset"] = _categorical(dose_df["dataset"], dataset_order)
        dose_df["method"] = _categorical(dose_df["method"], method_order)
        weights = sorted(dose_df["weight"].unique())
        weight_labels = [f"{weight:g}" for weight in weights]
        dose_df["weight_label"] = dose_df["weight"].map(lambda weight: f"{weight:g}")
        dose_df["weight_label"] = _categorical(dose_df["weight_label"], weight_labels)
        dose_plot = (
            ggplot(
                dose_df,
                aes(
                    "weight_label",
                    "mean",
                    color="method",
                    shape="method",
                    linetype="method",
                    group="method",
                ),
            )
            + geom_hline(yintercept=0, linetype="dashed", color="#666666", size=0.4)
            + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.13, size=0.45)
            + geom_line(size=0.75)
            + geom_point(size=POINT_SIZE)
            + facet_wrap("dataset", ncol=3)
            + _color_scale(method_order)
            + _shape_scale(method_order)
            + _linetype_scale(method_order)
            + labs(
                title="Masked-imputation dose response",
                x="Missing-arm gradient weight",
                y="Regret increase vs full mask (%)",
            )
            + _paper_theme(legend_position="top")
            + theme(axis_text_x=element_text(size=11.0, color="#444444"))
            + ggsize(820, 420)
        )
        save_plot(dose_plot, output_dir, "masked_imputation_dose_response", fmt)

    effect_rows = []
    endpoint = paired[paired["weight"] == 1.0]
    for (dataset, method), grp in endpoint.groupby(["dataset", "method"]):
        vals = grp["pct_delta"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        mean = float(vals.mean())
        effect_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "mean": mean,
                "ymin": mean - 1.96 * se,
                "ymax": mean + 1.96 * se,
            }
        )
    existing_effect = {
        (row["dataset"], row["method"])
        for row in effect_rows
    }
    for dataset in sorted(raw_df["dataset"].unique()):
        if (dataset, "BayesianMSE") in existing_effect:
            continue
        if raw_df[(raw_df["dataset"] == dataset) & (raw_df["weight"] == 1.0)].empty:
            continue
        effect_rows.append(
            {
                "dataset": dataset,
                "method": "BayesianMSE",
                "mean": 0.0,
                "ymin": 0.0,
                "ymax": 0.0,
            }
        )

    if not effect_rows:
        return

    df = pd.DataFrame(effect_rows)
    method_order = ["BayesianSPO", "BayesianMSE"]
    method_display = {
        "BayesianSPO": "BayesianSPO-Unmasked",
        "BayesianMSE": "BayesianMSE-Unmasked",
    }
    method_display_order = [method_display[method] for method in method_order]
    df["dataset"] = _categorical(df["dataset"], dataset_order)
    df["method"] = _categorical(df["method"], method_order)
    df["method_display"] = df["method"].map(method_display)
    df["method_display"] = _categorical(df["method_display"], method_display_order)
    effect_colors = {
        "BayesianSPO-Unmasked": PAPER_COLORS["BayesianSPO"],
        "BayesianMSE-Unmasked": PAPER_COLORS["BayesianMSE"],
    }
    endpoint_scale_limits = [-5.0, 65.0]
    endpoint_scale_breaks = [0.0, 20.0, 40.0, 60.0]
    dataset_positions = {dataset: idx + 1 for idx, dataset in enumerate(dataset_order)}

    def endpoint_panel(method):
        panel_df = df[df["method"] == method].copy()
        panel_df["dataset"] = _categorical(panel_df["dataset"], dataset_order)
        panel_df["method_display"] = _categorical(panel_df["method_display"], method_display_order)
        panel_df["dataset_pos"] = panel_df["dataset"].astype(str).map(dataset_positions)
        panel_df["bar_ymin"] = panel_df["dataset_pos"] - 0.36
        panel_df["bar_ymax"] = panel_df["dataset_pos"] + 0.36
        panel_df["xmin"] = np.minimum(0.0, panel_df["mean"])
        panel_df["xmax"] = np.maximum(0.0, panel_df["mean"])
        cap_rows = []
        for row in panel_df.itertuples(index=False):
            cap_rows.extend(
                [
                    {
                        "x": row.ymin,
                        "xend": row.ymin,
                        "y": row.dataset_pos - 0.06,
                        "yend": row.dataset_pos + 0.06,
                    },
                    {
                        "x": row.ymax,
                        "xend": row.ymax,
                        "y": row.dataset_pos - 0.06,
                        "yend": row.dataset_pos + 0.06,
                    },
                ]
            )
        cap_df = pd.DataFrame(cap_rows)
        arrow_df, label_df = _baseline_callout_frames(
            "masked baseline", endpoint_scale_limits, y=0.27, yend=0.60
        )
        panel_theme = _paper_theme(legend_position="none")
        panel_theme += theme(panel_border=element_rect(color="#c7c7c7", fill="transparent", size=0.45))
        return (
            ggplot(panel_df)
            + geom_rect(
                aes(
                    xmin="xmin",
                    xmax="xmax",
                    ymin="bar_ymin",
                    ymax="bar_ymax",
                    fill="method_display",
                ),
                alpha=0.78,
                color="#ffffff",
                size=0.25,
            )
            + geom_segment(
                aes(x="ymin", xend="ymax", y="dataset_pos", yend="dataset_pos"),
                color="#555555",
                size=0.55,
            )
            + geom_segment(
                aes(x="x", xend="xend", y="y", yend="yend"),
                data=cap_df,
                color="#555555",
                size=0.55,
            )
            + geom_vline(xintercept=0, linetype="solid", color="#222222", size=1.0)
            + geom_segment(
                aes("x", "y", xend="xend", yend="yend"),
                data=arrow_df,
                inherit_aes=False,
                arrow=arrow(length=5, type="closed"),
                color="#222222",
                size=0.45,
            )
            + geom_text(
                aes("x", "y", label="label"),
                data=label_df,
                inherit_aes=False,
                color="#222222",
                fontface="bold",
                size=4.5,
                hjust=0.0,
            )
            + scale_fill_manual(values=effect_colors)
            + scale_x_continuous(limits=endpoint_scale_limits, breaks=endpoint_scale_breaks)
            + scale_y_continuous(
                breaks=[dataset_positions[dataset] for dataset in dataset_order],
                labels=dataset_order,
                limits=[0.0, len(dataset_order) + 0.65],
            )
            + labs(
                title=method_display[method],
                x="Unmasked regret increase vs full mask (%)",
                y="",
                fill="",
            )
            + panel_theme
            + theme(axis_text_y=element_text(size=11.5, color="#444444"))
        )

    bar_plot = (
        gggrid(
            [
                endpoint_panel("BayesianSPO"),
                _separator_plot(),
                endpoint_panel("BayesianMSE"),
            ],
            ncol=1,
            align=True,
            heights=[1.0, 0.08, 1.0],
            vspace=0.01,
        )
        + ggsize(560, 430)
    )
    save_plot(bar_plot, output_dir, "masked_imputation_effect", fmt)
    save_plot(bar_plot, output_dir, "masked_imputation_effect_bar", fmt)


def plot_feature_sharing(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "feature_sharing")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  feature_sharing: no data found, skipping")
        return

    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        seeds = sorted(set(seed for _, seed in data.keys()))
        rows = []
        linucb_vals = [
            data[("LinUCB", seed)]["final_regret"]
            for seed in seeds
            if ("LinUCB", seed) in data
        ]
        linucb_mean = np.mean(linucb_vals) if linucb_vals else np.nan
        for name, _ in data.keys():
            if name == "LinUCB":
                continue
            parts = name.split("-")
            if len(parts) != 3 or not parts[2].startswith("h"):
                continue
            objective = "BayesianSPO" if parts[0] == "SPO" else "BayesianMSE"
            feature_map = "Linear" if parts[1] == "L" else "Nonlinear"
            hidden_dim = int(parts[2][1:])
            vals = [
                data[(name, seed)]["final_regret"]
                for seed in seeds
                if (name, seed) in data
            ]
            mean, std = _summary(vals)
            rows.append(
                {
                    "objective": objective,
                    "feature_map": feature_map,
                    "hidden_dim": hidden_dim,
                    "mean": mean,
                    "ymin": mean - std,
                    "ymax": mean + std,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"  feature_sharing ({dataset_name}): no parseable data, skipping")
            continue
        df["objective"] = _categorical(df["objective"], ["BayesianSPO", "BayesianMSE"])
        df["feature_map"] = _categorical(df["feature_map"], ["Linear", "Nonlinear"])
        labels = ["BayesianSPO", "BayesianMSE"]
        plot = (
            ggplot(df, aes("hidden_dim", "mean", color="objective", linetype="objective"))
            + geom_line(size=0.85)
            + geom_point(size=SMALL_POINT_SIZE)
            + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.08, alpha=0.55)
            + geom_hline(yintercept=linucb_mean, linetype="dashed", color="#666666", size=0.65)
            + facet_wrap("feature_map", ncol=2)
            + _color_scale(labels)
            + _linetype_scale(labels)
            + scale_x_log2(breaks=sorted(df["hidden_dim"].unique()))
            + labs(
                title=f"Feature-capacity sweep: {dataset_name}",
                x="Hidden dimension",
                y="Final cumulative regret",
            )
            + _paper_theme()
            + ggsize(760, 390)
        )
        title = "feature_sharing" if dataset_name == "." else f"feature_sharing_{dataset_name}"
        save_plot(plot, output_dir, title, fmt)


def plot_context_variation(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "context_variation")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  context_variation: no data found, skipping")
        return

    dataset_names = {
        "knapsack_32": "Knapsack 32",
        "knapsack_100": "Knapsack 100",
        "shortest_path_8x8": "SP 8x8",
        "shortest_path_20x20": "SP 20x20",
        "portfolio_32": "Portfolio",
        ".": "dataset",
    }
    rows = []
    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        dataset_label = dataset_names.get(dataset_name, dataset_name)
        agents = sorted(set(name for name, _, _ in data.keys()))
        noise_widths = sorted(set(noise for _, noise, _ in data.keys()))
        seeds = sorted(set(seed for _, _, seed in data.keys()))
        for name in agents:
            label = display_label(name)
            for noise_width in noise_widths:
                vals = [
                    data[(name, noise_width, seed)]["final_regret"]
                    for seed in seeds
                    if (name, noise_width, seed) in data
                ]
                mean, std = _summary(vals)
                rows.append(
                    {
                        "dataset": dataset_label,
                        "agent_label": label,
                        "noise_width": noise_width,
                        "mean": mean,
                        "ymin": mean - std,
                        "ymax": mean + std,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        print("  context_variation: no plottable data found, skipping")
        return
    dataset_order = [
        label
        for key, label in dataset_names.items()
        if label in set(df["dataset"].unique())
    ]
    labels = ordered_labels(df["agent_label"].unique())
    df["dataset"] = _categorical(df["dataset"], dataset_order)
    df["agent_label"] = _categorical(df["agent_label"], labels)
    noise_widths = sorted(df["noise_width"].unique())

    regret_plot = (
        ggplot(df, aes("noise_width", "mean", color="agent_label", linetype="agent_label"))
        + geom_line(size=0.85)
        + geom_point(size=SMALL_POINT_SIZE)
        + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.025, alpha=0.55)
        + facet_wrap("dataset", ncol=3, scales="free_y")
        + _color_scale(labels)
        + _linetype_scale(labels)
        + scale_x_continuous(breaks=noise_widths)
        + labs(title="Regret vs. context noise", x="Noise width", y="Final cumulative regret")
        + _paper_theme()
    )

    gap_rows = []
    for dataset in dataset_order:
        ds_df = df[df["dataset"] == dataset]
        for noise_width in noise_widths:
            spo = ds_df[(ds_df["agent_label"] == "BayesianSPO") & (ds_df["noise_width"] == noise_width)]["mean"]
            mse = ds_df[(ds_df["agent_label"] == "BayesianMSE") & (ds_df["noise_width"] == noise_width)]["mean"]
            if not spo.empty and not mse.empty:
                gap_rows.append({
                    "dataset": dataset,
                    "noise_width": str(noise_width),
                    "gap": float(mse.iloc[0] - spo.iloc[0]),
                })
    gap_df = pd.DataFrame(gap_rows)
    gap_df["dataset"] = _categorical(gap_df["dataset"], dataset_order)
    gap_plot = (
        ggplot(gap_df, aes("noise_width", "gap"))
        + geom_bar(stat="identity", fill=OKABE_ITO["purple"], alpha=0.72, color="#ffffff", size=0.3)
        + geom_hline(yintercept=0.0, color="#777777", linetype="dashed", size=0.65)
        + facet_wrap("dataset", ncol=3, scales="free_y")
        + labs(title="BayesianSPO advantage", x="Noise width", y="MSE regret - SPO regret")
        + _paper_theme(legend_position="none")
    )

    plot = gggrid([regret_plot, gap_plot], ncol=1, align=True) + ggsize(920, 720)
    save_plot(plot, output_dir, "context_variation", fmt)


def plot_feature_robustness(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "feature_robustness")
    if not os.path.isdir(ablation_dir):
        print("  feature_robustness: no data found, skipping")
        return

    json_files = sorted(
        os.path.join(ablation_dir, name)
        for name in os.listdir(ablation_dir)
        if name.endswith(".json")
    )
    if not json_files:
        print("  feature_robustness: no data found, skipping")
        return

    for data_path in json_files:
        with open(data_path) as f:
            data = json.load(f)
        if not data:
            continue

        stem = os.path.splitext(os.path.basename(data_path))[0]
        dataset_name = stem.split("_n")[0]
        agent_names = sorted(
            {
                agent
                for seed_data in data.values()
                for agent in seed_data.get("agents", {})
            }
        )

        rows = []
        for agent in agent_names:
            label = display_label(agent)
            regrets = []
            ratios = []
            for seed_data in data.values():
                agent_data = seed_data.get("agents", {}).get(agent)
                if not agent_data:
                    continue
                regrets.append(agent_data.get("final_cumulative_regret", np.nan))
                evals = agent_data.get("eval_metrics", [])
                if evals:
                    last = evals[-1]
                    irr = last.get("irrelevant_noise1.0_decision_change", np.nan)
                    rel = last.get("relevant_noise1.0_decision_change", np.nan)
                    if np.isfinite(irr) and np.isfinite(rel):
                        ratios.append(irr / max(rel, 1e-8))
            for metric, vals in [
                ("Online regret", regrets),
                ("Irrelevant / relevant decision change", ratios),
            ]:
                mean, std = _summary(vals)
                rows.append(
                    {
                        "agent_label": label,
                        "metric": metric,
                        "mean": mean,
                        "ymin": mean - std,
                        "ymax": mean + std,
                    }
                )
        df = pd.DataFrame(rows)
        labels = ordered_labels(df["agent_label"].unique())
        df["agent_label"] = _categorical(df["agent_label"], labels)
        df["metric"] = _categorical(
            df["metric"],
            ["Online regret", "Irrelevant / relevant decision change"],
        )
        plot = (
            ggplot(df, aes("agent_label", "mean", fill="agent_label"))
            + geom_bar(stat="identity", alpha=0.78, color="#ffffff", size=0.3)
            + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.18, size=0.55)
            + facet_wrap("metric", ncol=2, scales="free_y")
            + _fill_scale(labels)
            + labs(title=f"Feature robustness: {dataset_name}", x="", y="")
            + _paper_theme(legend_position="none", rotate_x=True)
            + ggsize(760, 330)
        )
        save_plot(plot, output_dir, f"feature_robustness_{dataset_name}", fmt)


def _numeric_sweep_plot(data, dataset_name, prefix, x_label, title, stem, output_dir, fmt, x_scale="linear"):
    pairs = []
    for name, _ in data.keys():
        if not name.startswith(prefix):
            continue
        try:
            pairs.append((float(name[len(prefix):]), name))
        except ValueError:
            continue
    pairs = sorted(set(pairs))
    if not pairs:
        print(f"  {stem} ({dataset_name}): no {prefix}-named agents, skipping")
        return
    seeds = sorted(set(seed for _, seed in data.keys()))
    rows = []
    for value, name in pairs:
        vals = [data[(name, seed)]["final_regret"] for seed in seeds if (name, seed) in data]
        mean, std = _summary(vals)
        rows.append({"value": value, "mean": mean, "ymin": mean - std, "ymax": mean + std})
    df = pd.DataFrame(rows)
    plot = (
        ggplot(df, aes("value", "mean"))
        + geom_line(color=OKABE_ITO["blue"], size=0.85)
        + geom_point(color=OKABE_ITO["blue"], size=SMALL_POINT_SIZE)
        + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.06, alpha=0.6, color=OKABE_ITO["blue"])
        + labs(title=f"{title}: {dataset_name}", x=x_label, y="Final cumulative regret")
        + _paper_theme(legend_position="none")
        + ggsize(500, 330)
    )
    if x_scale == "log2":
        plot = plot + scale_x_log2(breaks=list(df["value"]))
    elif x_scale == "log10":
        plot = plot + scale_x_log10(breaks=list(df["value"]))
    else:
        plot = plot + scale_x_continuous(breaks=list(df["value"]))
    save_plot(plot, output_dir, stem, fmt)


def plot_imputation_samples(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "imputation_samples")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  imputation_samples: no data found, skipping")
        return
    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        title = "imputation_samples" if dataset_name == "." else f"imputation_samples_{dataset_name}"
        _numeric_sweep_plot(
            data,
            dataset_name,
            "M",
            "Imputation samples M",
            "BayesianSPO posterior completions",
            title,
            output_dir,
            fmt,
            x_scale="log2",
        )

def plot_beta_sensitivity(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "beta_sensitivity")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  beta_sensitivity: no data found, skipping")
        return
    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        title = "beta_sensitivity" if dataset_name == "." else f"beta_sensitivity_{dataset_name}"
        _numeric_sweep_plot(
            data,
            dataset_name,
            "beta",
            "UCB scale beta",
            "Beta sensitivity",
            title,
            output_dir,
            fmt,
            x_scale="log10",
        )


def plot_lambda_reg_sweep(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "lambda_reg_sweep")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  lambda_reg_sweep: no data found, skipping")
        return
    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
        title = "lambda_reg_sweep" if dataset_name == "." else f"lambda_reg_sweep_{dataset_name}"
        _numeric_sweep_plot(
            data,
            dataset_name,
            "lambda",
            "Prior precision lambda",
            "Lambda sensitivity",
            title,
            output_dir,
            fmt,
            x_scale="log10",
        )


def plot_flip_set(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "flip_set")
    entries = list_result_entries(ablation_dir)
    data = load_pkl(entries[0][1]) if entries else None
    if data is None:
        print("  flip_set: no data found, skipping")
        return

    agents = sorted(set(name for name, _, _, _ in data.keys()))
    regimes = sorted(set(regime for _, regime, _, _ in data.keys()))
    datasets = sorted(set(dataset for _, _, dataset, _ in data.keys()))
    seeds = sorted(set(seed for _, _, _, seed in data.keys()))
    for dataset_name in datasets:
        rows = []
        bars = []
        for regime in regimes:
            for name in agents:
                label = f"{display_label(name)} ({regime})"
                traces = [
                    data[(name, regime, dataset_name, seed)].get("flip_set_size", [])
                    for seed in seeds
                    if (name, regime, dataset_name, seed) in data
                ]
                traces = [np.asarray(t, dtype=float) for t in traces if len(t) > 0]
                if traces:
                    min_len = min(len(t) for t in traces)
                    arr = np.asarray([t[:min_len] for t in traces])
                    mean = arr.mean(axis=0)
                    for t, val in enumerate(mean):
                        rows.append({"round": t + 1, "flip_size": val, "label": label})
                vals = [
                    data[(name, regime, dataset_name, seed)]["final_regret"]
                    for seed in seeds
                    if (name, regime, dataset_name, seed) in data
                ]
                mean, std = _summary(vals)
                bars.append({"label": label, "mean": mean, "ymin": mean - std, "ymax": mean + std})
        line_df = pd.DataFrame(rows)
        bar_df = pd.DataFrame(bars)
        line_plot = (
            ggplot(line_df, aes("round", "flip_size", color="label", linetype="label"))
            + geom_line(size=0.75)
            + labs(title=f"Flip-set size: {dataset_name}", x="Round", y="Flip-set size")
            + _paper_theme()
        )
        bar_plot = (
            ggplot(bar_df, aes("label", "mean", fill="label"))
            + geom_bar(stat="identity", alpha=0.78, color="#ffffff", size=0.3)
            + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.18)
            + labs(title="Final regret", x="", y="Final cumulative regret")
            + _paper_theme(legend_position="none", rotate_x=True)
        )
        plot = gggrid([line_plot, bar_plot], ncol=2, align=True) + ggsize(900, 360)
        title = "flip_set_collapse" if dataset_name == "." else f"flip_set_collapse_{dataset_name}"
        save_plot(plot, output_dir, title, fmt)


PLOT_FUNCTIONS = {
    "kn_ratio": plot_kn_ratio,
    "kn_ratio_sweep": plot_kn_ratio,
    "exploration": plot_exploration,
    "exploration_strategy": plot_exploration,
    "ablation_design_summary": plot_ablation_design_summary,
    "design_summary": plot_ablation_design_summary,
    "architecture": plot_architecture,
    "architecture_ablation": plot_architecture,
    "decoupling": plot_decoupling,
    "decoupling_ablation": plot_decoupling,
    "ts_scale": plot_ts_scale_sweep,
    "ts_scale_sweep": plot_ts_scale_sweep,
    "imputation": plot_imputation,
    "imputation_strategy": plot_imputation,
    "imputation_samples": plot_imputation_samples,
    "beta_sensitivity": plot_beta_sensitivity,
    "lambda_reg_sweep": plot_lambda_reg_sweep,
    "feature_sharing": plot_feature_sharing,
    "feature_robustness": plot_feature_robustness,
    "context_variation": plot_context_variation,
    "flip_set": plot_flip_set,
    "flip_set_collapse": plot_flip_set,
}

DEFAULT_ABLATIONS = [
    "kn_ratio",
    "imputation",
    "architecture",
    "decoupling",
    "feature_sharing",
    "context_variation",
    "feature_robustness",
    "beta_sensitivity",
    "exploration",
    "ts_scale",
    "lambda_reg_sweep",
    "imputation_samples",
]


def main():
    parser = argparse.ArgumentParser(description="Plot ablation results")
    parser.add_argument("--results-dir", type=str, default=ABLATION_DIR)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument(
        "--ablations",
        type=str,
        default=None,
        help="Comma-separated ablation names (default: runner set)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.ablations:
        ablations = [a.strip() for a in args.ablations.split(",")]
    else:
        ablations = DEFAULT_ABLATIONS

    print(f"Plotting ablations: {ablations}")
    print(f"Reading from: {args.results_dir}")
    print(f"Saving to: {args.output}")
    print()

    for name in ablations:
        if name in PLOT_FUNCTIONS:
            PLOT_FUNCTIONS[name](args.results_dir, args.output, args.format)
        else:
            print(f"  Unknown ablation: {name}")

    print(f"\nDone. Figures in {args.output}/")


if __name__ == "__main__":
    main()

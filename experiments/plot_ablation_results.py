#!/usr/bin/env python
"""Render ablation figures from saved result files using LetsPlot."""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

for _cairo_lib in ("/opt/homebrew/opt/cairo/lib", "/usr/local/opt/cairo/lib"):
    if (Path(_cairo_lib) / "libcairo.2.dylib").exists():
        os.environ.setdefault("DYLD_LIBRARY_PATH", _cairo_lib)
        break

import cairosvg
import numpy as np
import pandas as pd
from lets_plot import (
    LetsPlot,
    aes,
    arrow,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_blank,
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
    theme,
    theme_minimal,
    theme_void,
)
from lets_plot.export import ggsave

sys.path.insert(0, ".")

LetsPlot.setup_html()

ABLATION_DIR = "results/ablations"
OUTPUT_DIR = "results/figures/ablations"
CORE_DESIGN_DATASET_KEYS = ["knapsack_100", "shortest_path_20x20", "portfolio_32"]
CORE_DESIGN_DATASET_SET = set(CORE_DESIGN_DATASET_KEYS)

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
PAPER_FONT = "Avenir Next"
PAPER_BOLD_WEIGHT = "600"
PAPER_EMPHASIS_FONT = "Avenir Next Demi Bold"

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


def core_design_entries(ablation_dir):
    return [
        (name, path)
        for name, path in list_result_entries(ablation_dir)
        if name == "." or name in CORE_DESIGN_DATASET_SET
    ]


def core_design_dataset_order(dataset_names, present_labels):
    ordered = [
        dataset_names[key]
        for key in CORE_DESIGN_DATASET_KEYS
        if dataset_names.get(key) in present_labels
    ]
    ordered.extend(sorted(set(present_labels) - set(ordered)))
    return ordered


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


def _force_svg_font(svg_path: Path) -> None:
    text = svg_path.read_text()
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
    svg_path.write_text(text)


def _paper_theme(legend_position="top", rotate_x=False):
    x_text = element_text(family=PAPER_FONT, size=13.0, color="#444444")
    if rotate_x:
        x_text = element_text(family=PAPER_FONT, size=12.2, color="#444444", angle=25, hjust=1)
    return theme_minimal() + theme(
        legend_position=legend_position,
        legend_title=element_blank(),
        legend_text=element_text(family=PAPER_FONT, size=14.0, face="bold"),
        axis_title=element_text(family=PAPER_FONT, size=15.0),
        axis_text_x=x_text,
        axis_text_y=element_text(family=PAPER_FONT, size=13.0, color="#444444"),
        plot_title=element_text(family=PAPER_FONT, size=16.0, face="bold"),
        plot_subtitle=element_text(family=PAPER_FONT, size=14.0, face="bold"),
        strip_text=element_text(family=PAPER_FONT, size=14.5, face="bold"),
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

    if fmt not in {"pdf", "png", "svg"}:
        raise ValueError(f"Unsupported output format: {fmt}")

    if fmt == "pdf":
        ggsave(plot, svg_path.name, path=str(output_dir), iframe=False)
        _force_svg_font(svg_path)
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(out_path))
        print(f"  Saved {stem}.{fmt}")
        return

    kwargs = {}
    if fmt == "png" and png_width:
        kwargs = {"w": png_width, "unit": "px", "dpi": 96}

    ggsave(plot, out_path.name, path=str(output_dir), iframe=False, **kwargs)
    if fmt == "svg":
        _force_svg_font(out_path)

    print(f"  Saved {stem}.{fmt}")

def plot_kn_ratio(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "kn_ratio")
    data = load_pkl(os.path.join(ablation_dir, "results.pkl"))
    if data is None:
        print("  kn_ratio: no data found, skipping")
        return

    dataset_names = {
        "knapsack_32": "Knapsack 32",
        "knapsack_100": "Knapsack 100",
        "portfolio_32": "Portfolio",
    }
    records = []
    for key, result in data.items():
        if not isinstance(key, tuple) or not isinstance(result, dict):
            continue
        if len(key) == 4:
            dataset, name, cap, seed = key
        elif len(key) == 3:
            name, cap, seed = key
            dataset = result.get("dataset", "knapsack_32")
        else:
            continue
        if name not in {"BayesianSPO", "BayesianMSE"}:
            continue
        default_n_items = 100 if dataset == "knapsack_100" else 32
        n_items = int(result.get("n_items", default_n_items))
        selected_count = result.get("selected_count_mean")
        if selected_count is None or not np.isfinite(float(selected_count)):
            selected_count = float(cap) if float(cap) <= n_items else np.nan
        records.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_names.get(dataset, dataset),
                "agent": name,
                "agent_label": display_label(name),
                "capacity": float(cap),
                "seed": int(seed),
                "final_regret": float(result["final_regret"]),
                "regret_scale": 1.0 if dataset == "portfolio_32" else 1000.0,
                "selected_count_mean": float(selected_count),
                "selected_fraction_mean": float(selected_count) / float(n_items),
            }
        )
    rec_df = pd.DataFrame(records)
    if rec_df.empty:
        print("  kn_ratio: no plottable data, skipping")
        return

    agents = sorted(rec_df["agent"].unique())
    raw_rows = []
    for (dataset, label, name, cap), grp in rec_df.groupby(
        ["dataset_label", "agent_label", "agent", "capacity"]
    ):
        vals = grp["final_regret"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        raw_rows.append(
            {
                "dataset_label": dataset,
                "agent_label": label,
                "agent": name,
                "capacity": cap,
                "selected_count_mean": float(grp["selected_count_mean"].mean()),
                "selected_fraction_mean": float(grp["selected_fraction_mean"].mean()),
                "mean": mean,
                "mean_display": mean / float(grp["regret_scale"].iloc[0]),
                "ymin_display": max((mean - 1.96 * se) / float(grp["regret_scale"].iloc[0]), 0.0),
                "ymax_display": (mean + 1.96 * se) / float(grp["regret_scale"].iloc[0]),
            }
        )
    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        print("  kn_ratio: no plottable data, skipping")
        return

    raw_labels = ordered_labels(raw_df["agent_label"].unique())
    raw_df["agent_label"] = _categorical(raw_df["agent_label"], raw_labels)
    dataset_order = ["Knapsack 32", "Knapsack 100", "Portfolio"]
    present_dataset_order = [label for label in dataset_order if label in set(raw_df["dataset_label"])]
    raw_df["dataset_label"] = _categorical(raw_df["dataset_label"], present_dataset_order)
    raw_df = raw_df.sort_values(["dataset_label", "agent_label", "selected_count_mean"]).reset_index(drop=True)

    gain_rows = []
    if {"BayesianSPO", "BayesianMSE"}.issubset(set(agents)):
        paired_rows = []
        for (dataset, cap, seed), grp in rec_df.groupby(["dataset_label", "capacity", "seed"]):
            by_agent = {row["agent"]: row for _, row in grp.iterrows()}
            if "BayesianSPO" not in by_agent or "BayesianMSE" not in by_agent:
                continue
            mse_regret = float(by_agent["BayesianMSE"]["final_regret"])
            spo_regret = float(by_agent["BayesianSPO"]["final_regret"])
            if np.isfinite(mse_regret) and np.isfinite(spo_regret) and mse_regret > 0:
                paired_rows.append(
                    {
                        "dataset_label": dataset,
                        "capacity": cap,
                        "seed": seed,
                        "gain": 100.0 * (mse_regret - spo_regret) / mse_regret,
                        "selected_count_mean": 0.5 * (
                            float(by_agent["BayesianSPO"]["selected_count_mean"])
                            + float(by_agent["BayesianMSE"]["selected_count_mean"])
                        ),
                        "selected_fraction_mean": 0.5 * (
                            float(by_agent["BayesianSPO"]["selected_fraction_mean"])
                            + float(by_agent["BayesianMSE"]["selected_fraction_mean"])
                        ),
                    }
                )
        paired_df = pd.DataFrame(paired_rows)
        for (dataset, cap), grp in paired_df.groupby(["dataset_label", "capacity"]):
            gains = grp["gain"].to_numpy(dtype=float)
            if gains.size:
                mean = float(gains.mean())
                se = float(gains.std(ddof=1) / np.sqrt(gains.size)) if gains.size > 1 else 0.0
                gain_rows.append(
                    {
                        "dataset_label": dataset,
                        "capacity": cap,
                        "selected_count_mean": float(grp["selected_count_mean"].mean()),
                        "selected_fraction_mean": float(grp["selected_fraction_mean"].mean()),
                        "mean": mean,
                        "ymin": mean - 1.96 * se,
                        "ymax": mean + 1.96 * se,
                    }
                )
    gain_df = pd.DataFrame(gain_rows)
    if not gain_df.empty:
        gain_df["dataset_label"] = _categorical(gain_df["dataset_label"], present_dataset_order)
        gain_df = gain_df.sort_values(["dataset_label", "selected_count_mean"]).reset_index(drop=True)

    raw_plot = (
        ggplot(raw_df, aes("selected_count_mean", "mean_display", color="agent_label", linetype="agent_label"))
        + geom_line(size=0.95)
        + geom_point(size=3.2)
        + geom_errorbar(aes(ymin="ymin_display", ymax="ymax_display"), width=0.0, alpha=0.58, size=0.45)
        + facet_wrap("dataset_label", ncol=3, scales="free", order=0)
        + _color_scale(raw_labels)
        + _linetype_scale(raw_labels)
        + labs(
            x="",
            y="Final Cumulative Regret",
            color="",
            linetype="",
        )
        + _paper_theme()
        + theme(
            legend_position="top",
            legend_text=element_text(family=PAPER_FONT, size=15.0, face="bold"),
            axis_title=element_text(size=16.0),
            axis_text_x=element_text(size=14.0, color="#444444"),
            axis_text_y=element_text(size=14.0, color="#444444"),
            axis_line_x=element_line(color="#7a7a7a", size=0.35),
            axis_ticks_x=element_line(color="#7a7a7a", size=0.35),
            axis_ticks_length_x=4,
            panel_grid_minor=element_blank(),
            plot_margin=[4, 8, 0, 54],
        )
    )
    if not gain_df.empty:
        gain_plot = (
            ggplot(gain_df, aes("selected_count_mean", "mean"))
            + geom_hline(yintercept=0.0, linetype="dashed", color="#555555", size=0.55)
            + geom_ribbon(aes(ymin="ymin", ymax="ymax"), fill=PAPER_COLORS["BayesianSPO"], alpha=0.16)
            + geom_line(color=PAPER_COLORS["BayesianSPO"], size=1.05)
            + geom_point(color="#222222", size=3.7)
            + geom_point(color=PAPER_COLORS["BayesianSPO"], size=2.9)
            + facet_wrap("dataset_label", ncol=3, scales="free_x", order=0)
            + labs(
                x="Capacity/cardinality sweep (empirical selected components per round)",
                y="Regret gap vs\nBayesianSPO (%)",
            )
            + _paper_theme(legend_position="none")
            + theme(
                axis_title=element_text(size=16.0),
                axis_text_x=element_text(size=14.0, color="#444444"),
                axis_text_y=element_text(size=14.0, color="#444444"),
                axis_line_x=element_line(color="#7a7a7a", size=0.35),
                axis_ticks_x=element_line(color="#7a7a7a", size=0.35),
                axis_ticks_length_x=4,
                panel_grid_minor=element_blank(),
                plot_margin=[0, 8, 8, 54],
            )
        )
        plot = gggrid([raw_plot, gain_plot], ncol=1, align=True, heights=[0.52, 0.48]) + ggsize(1080, 560)
    else:
        plot = raw_plot + ggsize(1080, 340)
    save_plot(plot, output_dir, "kn_ratio_sweep", fmt, png_width=1800)


def plot_exploration(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "exploration")
    entries = core_design_entries(ablation_dir)
    if not entries:
        print("  exploration: no data found, skipping")
        return
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
    exploration_entries = core_design_entries(exploration_dir)
    if not exploration_entries:
        print("  exploration relative: no exploration data found, skipping")
        return

    ts_dir = resolve_ablation_dir(results_dir, "ts_scale")
    ts_entries = dict(core_design_entries(ts_dir))
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
                    complete_candidates = {
                        multiplier: vals
                        for multiplier, vals in candidates.items()
                        if set(ucb_vals).issubset(set(vals))
                    }
                    if complete_candidates:
                        best_multiplier, best_vals = min(
                            complete_candidates.items(),
                            key=lambda item: np.mean(
                                [item[1][seed] for seed in ucb_vals]
                            ),
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

    dataset_order = core_design_dataset_order(dataset_names, set(df["dataset"].unique()))
    strategy_order = [strategy for strategy in strategy_order if strategy in set(df["strategy"].unique())]
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
    strategy_colors = {strategy: strategy_colors[strategy] for strategy in strategy_order}
    dataset_positions = {dataset: idx + 1 for idx, dataset in enumerate(dataset_order)}
    offset_values = np.linspace(-0.18, 0.18, len(strategy_order)) if len(strategy_order) > 1 else [0.0]
    strategy_offsets = dict(zip(strategy_order, offset_values))
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
            f"{method} Baseline - UCB", x_limits, y=0.27, yend=0.60
        )
        plot_theme = _paper_theme(legend_position="top" if show_legend else "none")
        plot_theme += theme(panel_border=element_rect(color="#c7c7c7", fill="transparent", size=0.45))
        if show_legend:
            plot_theme += theme(legend_title=element_text(family=PAPER_FONT, size=12.0, face="bold"))
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
                family=PAPER_FONT,
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
                limits=[0.0, len(dataset_order) + 0.35],
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
        + ggsize(620, 330)
    )
    save_plot(plot, output_dir, "exploration_relative_summary", fmt)

def plot_ablation_design_summary(results_dir, output_dir, fmt="pdf"):
    dataset_names = _dataset_label_map()
    dataset_order = core_design_dataset_order(dataset_names, set(dataset_names.values()))
    rows = []

    def append_final_method_rows(data, dataset, check, ablation_agent, baseline_agent):
        paired_baseline = []
        paired_ablation = []
        seeds = sorted(
            {
                seed
                for name, seed in data.keys()
                if name in {baseline_agent, ablation_agent}
            }
        )
        for seed in seeds:
            baseline_key = (baseline_agent, seed)
            ablation_key = (ablation_agent, seed)
            if baseline_key not in data or ablation_key not in data:
                continue
            baseline_regret = float(data[baseline_key]["final_regret"])
            ablation_regret = float(data[ablation_key]["final_regret"])
            if not np.isfinite(baseline_regret) or not np.isfinite(ablation_regret):
                continue
            if baseline_regret <= 0 or ablation_regret <= 0:
                continue
            paired_baseline.append(baseline_regret)
            paired_ablation.append(ablation_regret)
        if not paired_baseline:
            return
        rows.append(
            {
                "dataset": dataset,
                "check": check,
                "baseline_regret": float(np.mean(paired_baseline)),
                "ablation_regret": float(np.mean(paired_ablation)),
            }
        )

    architecture_dir = resolve_ablation_dir(results_dir, "architecture")
    for dataset_name, data_path in core_design_entries(architecture_dir):
        data = load_pkl(data_path)
        if data is None:
            continue
        dataset = dataset_names.get(dataset_name, dataset_name)
        append_final_method_rows(
            data,
            dataset,
            "Non-Bayesian head SPO",
            "DeterministicSPO",
            "BayesianSPO",
        )

    decoupling_dir = resolve_ablation_dir(results_dir, "decoupling")
    for dataset_name, data_path in core_design_entries(decoupling_dir):
        data = load_pkl(data_path)
        if data is None:
            continue
        dataset = dataset_names.get(dataset_name, dataset_name)
        append_final_method_rows(
            data,
            dataset,
            "Coupled head SPO (0.5x lr)",
            "CoupledSPO-Eps-x0.5",
            "DecoupledSPO-UCB",
        )
        append_final_method_rows(
            data,
            dataset,
            "Coupled head SPO (1x lr)",
            "CoupledSPO-Eps-x1",
            "DecoupledSPO-UCB",
        )
        append_final_method_rows(
            data,
            dataset,
            "Coupled head SPO (2x lr)",
            "CoupledSPO-Eps-x2",
            "DecoupledSPO-UCB",
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("  ablation design summary: no plottable data found, skipping")
        return

    check_order = [
        "Non-Bayesian head SPO",
        "Coupled head SPO (0.5x lr)",
        "Coupled head SPO (1x lr)",
        "Coupled head SPO (2x lr)",
    ]
    check_positions = {check: len(check_order) - idx for idx, check in enumerate(check_order)}
    present_datasets = [
        dataset
        for dataset in dataset_order
        if dataset in set(df["dataset"].unique())
    ]
    portfolio_label = dataset_names.get("portfolio_32", "Portfolio")
    df["dataset"] = _categorical(df["dataset"], present_datasets)
    df["check"] = _categorical(df["check"], check_order)
    df["ypos"] = df["check"].astype(str).map(check_positions)
    df["axis_unit"] = np.where(df["dataset"].astype(str) == portfolio_label, "raw", "K")
    df["facet"] = df["dataset"].astype(str)
    facet_order = present_datasets
    df["facet"] = _categorical(df["facet"], facet_order)
    for col in ["baseline_regret", "ablation_regret"]:
        df[f"{col}_axis"] = np.where(df["axis_unit"] == "raw", df[col], df[col] / 1000.0)
    df["xmin"] = np.minimum(df["baseline_regret_axis"], df["ablation_regret_axis"])
    df["xmax"] = np.maximum(df["baseline_regret_axis"], df["ablation_regret_axis"])

    point_rows = []
    for row in df.itertuples(index=False):
        point_rows.append(
            {
                "facet": row.facet,
                "check": row.check,
                "kind": "Tuned BayesianSPO",
                "regret": row.baseline_regret_axis,
                "ypos": row.ypos,
            }
        )
        point_rows.append(
            {
                "facet": row.facet,
                "check": row.check,
                "kind": "Ablation",
                "regret": row.ablation_regret_axis,
                "ypos": row.ypos,
            }
        )
    point_df = pd.DataFrame(point_rows)
    point_df["facet"] = _categorical(point_df["facet"], facet_order)
    point_df["kind"] = _categorical(point_df["kind"], ["Tuned BayesianSPO", "Ablation"])

    pad_rows = []
    linear_pad_rows = []
    for facet, facet_df in df.groupby("facet", observed=True):
        lower = float(facet_df["xmin"].min())
        upper = float(facet_df["xmax"].max())
        if np.isfinite(lower) and np.isfinite(upper) and lower > 0 and upper > 0:
            pad_rows.extend(
                [
                    {"facet": facet, "regret": lower / 1.25, "ypos": 1.0},
                    {"facet": facet, "regret": upper * 1.03, "ypos": 1.0},
                ]
            )
            linear_pad_rows.extend(
                [
                    {"facet": facet, "regret": -upper * 0.04, "ypos": 1.0},
                    {"facet": facet, "regret": upper * 1.03, "ypos": 1.0},
                ]
            )
    pad_df = pd.DataFrame(pad_rows, columns=["facet", "regret", "ypos"])
    if not pad_df.empty:
        pad_df["facet"] = _categorical(pad_df["facet"], facet_order)
    linear_pad_df = pd.DataFrame(linear_pad_rows, columns=["facet", "regret", "ypos"])
    if not linear_pad_df.empty:
        linear_pad_df["facet"] = _categorical(linear_pad_df["facet"], facet_order)

    linear_minor_grid_rows = []
    for facet, facet_df in df.groupby("facet", observed=True):
        upper = float(facet_df["xmax"].max())
        if not np.isfinite(upper) or upper <= 0:
            continue
        axis_upper = upper * 1.03
        facet_name = str(facet)
        if facet_name == portfolio_label:
            major_step = 250.0
            minor_step = 125.0
        elif facet_name.startswith("SP"):
            major_step = 50.0
            minor_step = 25.0
        else:
            major_step = 200.0
            minor_step = 100.0
        grid_max = np.ceil(axis_upper / minor_step) * minor_step
        for x in np.arange(0.0, grid_max + 0.5 * minor_step, minor_step):
            if x <= 0.0:
                continue
            if abs((x / major_step) - round(x / major_step)) < 1e-9:
                continue
            linear_minor_grid_rows.append({"facet": facet, "xintercept": float(x)})
    linear_minor_grid_df = pd.DataFrame(linear_minor_grid_rows, columns=["facet", "xintercept"])
    if not linear_minor_grid_df.empty:
        linear_minor_grid_df["facet"] = _categorical(linear_minor_grid_df["facet"], facet_order)

    point_colors = {
        "Tuned BayesianSPO": PAPER_COLORS["BayesianSPO"],
        "Ablation": PAPER_COLORS["BayesianMSE"],
    }
    x_values = pd.concat([df["baseline_regret_axis"], df["ablation_regret_axis"]], ignore_index=True)
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

    def make_design_plot(log_scale=True):
        scale_pad_df = pad_df if log_scale else linear_pad_df
        x_scale = (
            scale_x_log10(breaks=x_breaks, expand=[0.04, 0.0])
            if log_scale
            else scale_x_continuous(expand=[0.02, 0.0])
        )
        x_label = (
            "Final cumulative regret (log scale)"
            if log_scale
            else "Final cumulative regret"
        )
        plot = ggplot(point_df, aes("regret", "ypos"))
        if not log_scale and not linear_minor_grid_df.empty:
            plot = plot + geom_vline(
                aes(xintercept="xintercept"),
                data=linear_minor_grid_df,
                inherit_aes=False,
                color="#f0f0f0",
                size=0.30,
            )
        return (
            plot
            + geom_segment(
                aes(x="xmin", xend="xmax", y="ypos", yend="ypos"),
                data=df,
                inherit_aes=False,
                color="#9a9a9a",
                size=0.8,
                lineend="butt",
            )
            + geom_point(color="#111111", shape=16, size=5.2)
            + geom_point(aes(color="kind"), shape=16, size=4.0)
            + geom_blank(
                aes("regret", "ypos"),
                data=scale_pad_df,
                inherit_aes=False,
            )
            + facet_wrap("facet", ncol=3, scales="free_x")
            + scale_color_manual(values=point_colors)
            + x_scale
            + scale_y_continuous(
                breaks=[check_positions[check] for check in check_order],
                labels=check_order,
                limits=[0.5, len(check_order) + 0.5],
            )
            + labs(
                x=x_label,
                y="",
                color="",
            )
            + _paper_theme(legend_position="top")
            + theme(
                legend_background=element_rect(fill="#f7f7f7", color="#d0d0d0", size=0.6),
                legend_justification="center",
                legend_box_just="center",
                legend_text=element_text(family=PAPER_FONT, size=18.5, face="bold"),
                axis_title_x=element_text(size=17.5),
                axis_text_x=element_text(family=PAPER_FONT, size=15.0, color="#3d3d3d"),
                axis_text_spacing_x=2,
                axis_text_y=element_text(family=PAPER_FONT, size=14.2, color="#444444"),
                axis_ticks_x=element_line(color="#8f8f8f", size=0.45),
                axis_minor_ticks_x=element_line(color="#cfcfcf", size=0.30),
                axis_ticks_length_x=5,
                axis_minor_ticks_length_x=3,
                strip_text=element_text(family=PAPER_FONT, size=15.5, face="bold"),
                panel_grid_major_x=element_line(color="#e5e5e5", size=0.45),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_x=element_line(color="#f0f0f0", size=0.30),
                panel_grid_minor_y=element_blank(),
                panel_spacing_x=2.15,
                panel_spacing_y=0.80,
                plot_title=element_blank(),
                plot_margin=[4, 10, 28, 10],
            )
            + ggsize(980, 350)
        )

    linear_plot = make_design_plot(log_scale=False)
    save_plot(linear_plot, output_dir, "ablation_design_summary_linear", fmt, png_width=2200)


def _dataset_label_map():
    return {
        "knapsack_32": "Knapsack 32",
        "knapsack_100": "Knapsack 100",
        "shortest_path_8x8": "SP 8x8",
        "shortest_path_20x20": "SP 20x20",
        "portfolio_32": "Portfolio",
        ".": "dataset",
    }


def _parse_ts_scale_agent(raw_name):
    marker = "-TS-x"
    if marker not in raw_name:
        return None
    method, mult = raw_name.rsplit(marker, 1)
    try:
        return method, float(mult)
    except ValueError:
        return None


def plot_imputation(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "imputation")
    entries = core_design_entries(ablation_dir)
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
    dataset_order = core_design_dataset_order(dataset_names, set(raw_df["dataset"].unique()))
    method_order = ["BayesianSPO", "BayesianMSE"]
    baseline = raw_df[raw_df["weight"] == 0.0][
        ["dataset", "method", "seed", "regret"]
    ].rename(columns={"regret": "masked_regret"})
    paired = raw_df.merge(baseline, on=["dataset", "method", "seed"], how="inner")
    paired["delta"] = paired["regret"] - paired["masked_regret"]
    denom = paired["masked_regret"].abs().replace(0.0, np.nan)
    paired["pct_delta"] = 100.0 * paired["delta"] / denom
    # For posterior-mean MSE completion, missing entries have zero residual:
    # Y_target_i = pred_i for i not in S_t. The unobserved-arm gradient weight
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
    if dose_rows:
        dose_df = pd.DataFrame(dose_rows)
        dose_df = dose_df[dose_df["method"] == "BayesianSPO"].copy()
        dose_df["dataset"] = _categorical(dose_df["dataset"], dataset_order)
        dose_df["method"] = _categorical(dose_df["method"], ["BayesianSPO"])
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
            + _color_scale(["BayesianSPO"])
            + _shape_scale(["BayesianSPO"])
            + _linetype_scale(["BayesianSPO"])
            + labs(
                title="Unobserved-arm weight response",
                x="Unobserved arm gradient weight",
                y="Regret increase vs full mask (%)",
            )
            + _paper_theme(legend_position="none")
            + theme(axis_text_x=element_text(size=11.0, color="#444444"))
            + ggsize(820, 340)
        )
        save_plot(dose_plot, output_dir, "masked_imputation_dose_response", fmt)

    spo_raw = raw_df[raw_df["method"] == "BayesianSPO"].copy()
    baseline_regret = spo_raw[spo_raw["weight"] == 0.0][
        ["dataset", "seed", "regret"]
    ].rename(columns={"regret": "masked_regret"})
    dose_pairs = spo_raw[spo_raw["weight"] > 0.0].merge(
        baseline_regret,
        on=["dataset", "seed"],
        how="inner",
    )
    matched_rows = []
    for (dataset, weight), grp in dose_pairs.groupby(["dataset", "weight"]):
        masked_vals = grp["masked_regret"].to_numpy(dtype=float)
        dose_vals = grp["regret"].to_numpy(dtype=float)
        masked_vals = masked_vals[np.isfinite(masked_vals)]
        dose_vals = dose_vals[np.isfinite(dose_vals)]
        if masked_vals.size == 0 or dose_vals.size == 0:
            continue
        matched_rows.append(
            {
                "dataset": dataset,
                "weight": float(weight),
                "masked_regret": float(masked_vals.mean()),
                "dose_regret": float(dose_vals.mean()),
            }
        )

    if matched_rows:
        matched_df = pd.DataFrame(matched_rows)
        matched_df = matched_df[~np.isclose(matched_df["weight"], 0.1)].copy()
        portfolio_label = dataset_names.get("portfolio_32", "Portfolio")
        matched_df["axis_unit"] = np.where(matched_df["dataset"] == portfolio_label, "raw", "K")
        matched_df["facet"] = matched_df["dataset"]
        facet_order = dataset_order
        matched_df["dataset"] = _categorical(matched_df["dataset"], dataset_order)
        matched_df["facet"] = _categorical(matched_df["facet"], facet_order)
        for col in ["masked_regret", "dose_regret"]:
            matched_df[f"{col}_axis"] = np.where(
                matched_df["axis_unit"] == "raw",
                matched_df[col],
                matched_df[col] / 1000.0,
            )
        baseline_rows = []
        for dataset, grp in baseline_regret.groupby("dataset"):
            vals = grp["masked_regret"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            raw_regret = float(vals.mean())
            axis_unit = "raw" if dataset == portfolio_label else "K"
            baseline_rows.append(
                {
                    "dataset": dataset,
                    "weight": 0.0,
                    "facet": dataset,
                    "axis_unit": axis_unit,
                    "regret": raw_regret,
                    "regret_axis": raw_regret if axis_unit == "raw" else raw_regret / 1000.0,
                }
            )
        baseline_df = pd.DataFrame(
            baseline_rows,
            columns=["dataset", "weight", "facet", "axis_unit", "regret", "regret_axis"],
        )
        if not baseline_df.empty:
            baseline_df["dataset"] = _categorical(baseline_df["dataset"], dataset_order)
            baseline_df["facet"] = _categorical(baseline_df["facet"], facet_order)

        weight_order = sorted(set(matched_df["weight"].unique()).union({0.0}))
        weight_labels = {weight: f"w={weight:g}" for weight in weight_order}
        weight_positions = {weight: len(weight_order) - idx for idx, weight in enumerate(weight_order)}
        matched_df["weight_label"] = matched_df["weight"].map(weight_labels)
        matched_df["ypos"] = matched_df["weight"].map(weight_positions)
        if not baseline_df.empty:
            baseline_df["weight_label"] = baseline_df["weight"].map(weight_labels)
            baseline_df["ypos"] = baseline_df["weight"].map(weight_positions)
        matched_df["xmin"] = np.minimum(
            matched_df["masked_regret_axis"],
            matched_df["dose_regret_axis"],
        )
        matched_df["xmax"] = np.maximum(
            matched_df["masked_regret_axis"],
            matched_df["dose_regret_axis"],
        )

        point_rows = []
        for row in matched_df.itertuples(index=False):
            point_rows.append(
                {
                    "facet": row.facet,
                    "kind": "Imputation Weight",
                    "regret": row.dose_regret_axis,
                    "ypos": row.ypos,
                }
            )
        for row in baseline_df.itertuples(index=False):
            point_rows.append(
                {
                    "facet": row.facet,
                    "kind": "BayesianSPO (Full Mask)",
                    "regret": row.regret_axis,
                    "ypos": row.ypos,
                }
            )
        point_df = pd.DataFrame(point_rows)
        point_df["facet"] = _categorical(point_df["facet"], facet_order)
        legend_order = ["BayesianSPO (Full Mask)", "Imputation Weight"]
        point_df["kind"] = _categorical(point_df["kind"], legend_order)

        path_rows = [
            {
                "facet": row.facet,
                "kind": "Imputation Weight",
                "regret": row.dose_regret_axis,
                "ypos": row.ypos,
            }
            for row in matched_df.itertuples(index=False)
        ]
        path_rows.extend(
            {
                "facet": row.facet,
                "kind": "Imputation Weight",
                "regret": row.regret_axis,
                "ypos": row.ypos,
            }
            for row in baseline_df.itertuples(index=False)
        )
        path_df = pd.DataFrame(path_rows, columns=["facet", "kind", "regret", "ypos"])
        if not path_df.empty:
            path_df["facet"] = _categorical(path_df["facet"], facet_order)
            path_df["kind"] = _categorical(path_df["kind"], legend_order)

        trajectory_rows = []
        for facet, facet_df in path_df.groupby("facet", observed=True):
            ordered = facet_df.sort_values("ypos")
            for start, end in zip(
                ordered.iloc[:-1].itertuples(index=False),
                ordered.iloc[1:].itertuples(index=False),
            ):
                trajectory_rows.append(
                    {
                        "facet": facet,
                        "kind": "Imputation Weight",
                        "x": start.regret,
                        "xend": end.regret,
                        "y": start.ypos,
                        "yend": end.ypos,
                    }
                )
        trajectory_df = pd.DataFrame(
            trajectory_rows, columns=["facet", "kind", "x", "xend", "y", "yend"]
        )
        if not trajectory_df.empty:
            trajectory_df["facet"] = _categorical(trajectory_df["facet"], facet_order)
            trajectory_df["kind"] = _categorical(trajectory_df["kind"], legend_order)

        baseline_line_rows = []
        for row in baseline_df.itertuples(index=False):
            if np.isfinite(row.regret_axis) and row.regret_axis > 0:
                baseline_line_rows.append(
                    {
                        "facet": row.facet,
                        "kind": "BayesianSPO (Full Mask)",
                        "x": row.regret_axis,
                        "xend": row.regret_axis,
                        "y": 0.5,
                        "yend": len(weight_order) + 0.5,
                    }
                )
        baseline_line_df = pd.DataFrame(
            baseline_line_rows, columns=["facet", "kind", "x", "xend", "y", "yend"]
        )
        if not baseline_line_df.empty:
            baseline_line_df["facet"] = _categorical(baseline_line_df["facet"], facet_order)
            baseline_line_df["kind"] = _categorical(baseline_line_df["kind"], legend_order)

        pad_rows = []
        for facet, facet_df in matched_df.groupby("facet", observed=True):
            lower = float(facet_df["xmin"].min())
            upper = float(facet_df["xmax"].max())
            if np.isfinite(lower) and np.isfinite(upper) and lower > 0 and upper > 0:
                pad_rows.extend(
                    [
                        {"facet": facet, "regret": lower / 1.2, "ypos": 1.0},
                        {"facet": facet, "regret": upper * 1.2, "ypos": 1.0},
                    ]
                )
        pad_df = pd.DataFrame(pad_rows, columns=["facet", "regret", "ypos"])
        if not pad_df.empty:
            pad_df["facet"] = _categorical(pad_df["facet"], facet_order)

        x_values = pd.concat(
            [matched_df["masked_regret_axis"], matched_df["dose_regret_axis"]],
            ignore_index=True,
        )
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

        matched_colors = {
            "BayesianSPO (Full Mask)": PAPER_COLORS["BayesianSPO"],
            "Imputation Weight": PAPER_COLORS["BayesianMSE"],
        }
        matched_point_size = 3.6
        matched_point_border_size = matched_point_size + 0.4
        legend_bg_df = pd.DataFrame(
            [{"xmin": 0.14, "xmax": 0.91, "ymin": 0.13, "ymax": 0.87}]
        )
        legend_key_df = pd.DataFrame(
            [
                {
                    "kind": "BayesianSPO (Full Mask)",
                    "x": 0.22,
                    "xend": 0.28,
                    "xmid": 0.25,
                    "y": 0.5,
                    "label_x": 0.30,
                    "label": "BayesianSPO (Full Mask)",
                },
                {
                    "kind": "Imputation Weight",
                    "x": 0.60,
                    "xend": 0.66,
                    "xmid": 0.63,
                    "y": 0.5,
                    "label_x": 0.68,
                    "label": "Imputation Weight",
                },
            ]
        )
        legend_key_df["kind"] = _categorical(legend_key_df["kind"], legend_order)
        legend_plot = (
            ggplot()
            + geom_rect(
                aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                data=legend_bg_df,
                inherit_aes=False,
                fill="#f7f7f7",
                color="#d0d0d0",
                size=0.6,
            )
            + geom_segment(
                aes(x="x", xend="xend", y="y", yend="y", color="kind"),
                data=legend_key_df,
                inherit_aes=False,
                size=0.8,
            )
            + geom_point(
                aes("xmid", "y"),
                data=legend_key_df,
                inherit_aes=False,
                color="#222222",
                size=matched_point_border_size,
            )
            + geom_point(
                aes("xmid", "y", color="kind"),
                data=legend_key_df,
                inherit_aes=False,
                size=matched_point_size,
            )
            + geom_text(
                aes("label_x", "y", label="label"),
                data=legend_key_df,
                inherit_aes=False,
                color="#222222",
                family=PAPER_FONT,
                fontface="bold",
                hjust=0.0,
                size=7.8,
            )
            + scale_color_manual(values=matched_colors)
            + scale_x_continuous(limits=[0.0, 1.0])
            + scale_y_continuous(limits=[0.0, 1.0])
            + theme_void()
            + theme(legend_position="none", plot_margin=[0, 0, 0, 0])
        )
        matched_body = (
            ggplot(point_df, aes("regret", "ypos"))
            + geom_segment(
                aes(x="xmin", xend="xmax", y="ypos", yend="ypos"),
                data=matched_df,
                inherit_aes=False,
                color="#9a9a9a",
                size=0.8,
            )
            + geom_segment(
                aes(x="x", xend="xend", y="y", yend="yend", color="kind"),
                data=baseline_line_df,
                inherit_aes=False,
                size=0.95,
                alpha=0.9,
            )
            + geom_segment(
                aes(x="x", xend="xend", y="y", yend="yend", color="kind"),
                data=trajectory_df,
                inherit_aes=False,
                size=0.75,
                alpha=0.85,
            )
            + geom_point(color="#222222", size=matched_point_border_size)
            + geom_point(aes(color="kind"), size=matched_point_size)
            + geom_point(
                aes("regret", "ypos"),
                data=pad_df,
                inherit_aes=False,
                alpha=0.0,
                size=0.1,
            )
            + facet_wrap("facet", ncol=3, scales="free_x")
            + scale_color_manual(values=matched_colors)
            + scale_x_continuous()
            + scale_y_continuous(
                breaks=[weight_positions[weight] for weight in weight_order],
                labels=[weight_labels[weight] for weight in weight_order],
                limits=[0.5, len(weight_order) + 0.5],
            )
            + labs(
                x="Final cumulative regret",
                y="Unobserved arm gradient weight",
                color="",
            )
            + _paper_theme(legend_position="none")
            + theme(
                axis_text_x=element_text(size=16.5, color="#444444"),
                axis_text_y=element_text(size=16.5, color="#444444"),
                axis_title_x=element_text(size=19.0),
                axis_title_y=element_text(size=19.0),
                strip_text=element_text(family=PAPER_FONT, size=18.5, face="bold"),
                plot_margin=[2, 10, 20, 10],
            )
        )
        matched_plot = (
            gggrid(
                [legend_plot, matched_body],
                ncol=1,
                align=True,
                heights=[0.075, 0.925],
                vspace=0.02,
            )
            + ggsize(980, 360)
        )

        max_weight = float(max(weight_order))
        swapped_pair_rows = []
        for row in matched_df.itertuples(index=False):
            swapped_pair_rows.append(
                {
                    "facet": row.facet,
                    "x": row.weight,
                    "xend": row.weight,
                    "y": min(row.masked_regret_axis, row.dose_regret_axis),
                    "yend": max(row.masked_regret_axis, row.dose_regret_axis),
                }
            )
        swapped_pair_df = pd.DataFrame(swapped_pair_rows, columns=["facet", "x", "xend", "y", "yend"])
        if not swapped_pair_df.empty:
            swapped_pair_df["facet"] = _categorical(swapped_pair_df["facet"], facet_order)

        swapped_curve_rows = []
        for row in matched_df.itertuples(index=False):
            swapped_curve_rows.append(
                {
                    "facet": row.facet,
                    "kind": "Imputation Weight",
                    "weight": row.weight,
                    "regret": row.dose_regret_axis,
                }
            )
        for row in baseline_df.itertuples(index=False):
            swapped_curve_rows.append(
                {
                    "facet": row.facet,
                    "kind": "Imputation Weight",
                    "weight": 0.0,
                    "regret": row.regret_axis,
                }
            )
        swapped_curve_df = pd.DataFrame(
            swapped_curve_rows, columns=["facet", "kind", "weight", "regret"]
        )
        if not swapped_curve_df.empty:
            swapped_curve_df["facet"] = _categorical(swapped_curve_df["facet"], facet_order)
            swapped_curve_df["kind"] = _categorical(swapped_curve_df["kind"], legend_order)

        swapped_point_rows = [
            {
                "facet": row.facet,
                "kind": "Imputation Weight",
                "weight": row.weight,
                "regret": row.dose_regret_axis,
            }
            for row in matched_df.itertuples(index=False)
        ]
        swapped_point_rows.extend(
            {
                "facet": row.facet,
                "kind": "BayesianSPO (Full Mask)",
                "weight": 0.0,
                "regret": row.regret_axis,
            }
            for row in baseline_df.itertuples(index=False)
        )
        swapped_point_df = pd.DataFrame(
            swapped_point_rows, columns=["facet", "kind", "weight", "regret"]
        )
        if not swapped_point_df.empty:
            swapped_point_df["facet"] = _categorical(swapped_point_df["facet"], facet_order)
            swapped_point_df["kind"] = _categorical(swapped_point_df["kind"], legend_order)

        swapped_baseline_rows = [
            {
                "facet": row.facet,
                "kind": "BayesianSPO (Full Mask)",
                "x": 0.0,
                "xend": max_weight,
                "y": row.regret_axis,
                "yend": row.regret_axis,
            }
            for row in baseline_df.itertuples(index=False)
            if np.isfinite(row.regret_axis)
        ]
        swapped_baseline_df = pd.DataFrame(
            swapped_baseline_rows, columns=["facet", "kind", "x", "xend", "y", "yend"]
        )
        if not swapped_baseline_df.empty:
            swapped_baseline_df["facet"] = _categorical(swapped_baseline_df["facet"], facet_order)
            swapped_baseline_df["kind"] = _categorical(swapped_baseline_df["kind"], legend_order)

        matched_swapped_body = (
            ggplot(swapped_point_df, aes("weight", "regret"))
            + geom_segment(
                aes(x="x", xend="xend", y="y", yend="yend"),
                data=swapped_pair_df,
                inherit_aes=False,
                color="#9a9a9a",
                size=0.8,
            )
            + geom_segment(
                aes(x="x", xend="xend", y="y", yend="yend", color="kind"),
                data=swapped_baseline_df,
                inherit_aes=False,
                size=0.95,
                alpha=0.9,
            )
            + geom_line(
                aes("weight", "regret", color="kind", group="kind"),
                data=swapped_curve_df,
                inherit_aes=False,
                size=0.75,
                alpha=0.85,
            )
            + geom_point(color="#222222", size=matched_point_border_size)
            + geom_point(aes(color="kind"), size=matched_point_size)
            + facet_wrap("facet", ncol=3, scales="free_y")
            + scale_color_manual(values=matched_colors)
            + scale_x_continuous(
                breaks=weight_order,
                labels=[f"{weight:g}" for weight in weight_order],
                limits=[-0.03, max_weight + 0.03],
            )
            + labs(
                x="Unobserved arm gradient weight",
                y="Final cumulative regret",
                color="",
            )
            + _paper_theme(legend_position="none")
            + theme(
                axis_text_x=element_text(size=16.5, color="#444444", angle=0, hjust=0.5),
                axis_text_y=element_text(size=16.5, color="#444444"),
                axis_title_x=element_text(size=19.0),
                axis_title_y=element_text(size=19.0),
                panel_grid_major=element_line(color="#e4e4e4", size=0.55),
                panel_grid_minor=element_line(color="#f0f0f0", size=0.35),
                strip_text=element_text(family=PAPER_FONT, size=18.5, face="bold"),
                plot_margin=[2, 10, 20, 10],
            )
        )
        matched_swapped_plot = (
            gggrid(
                [legend_plot, matched_swapped_body],
                ncol=1,
                align=True,
                heights=[0.075, 0.925],
                vspace=0.02,
            )
            + ggsize(980, 360)
        )
        save_plot(
            matched_swapped_plot,
            output_dir,
            "masked_imputation_dose_matched",
            fmt,
            png_width=2200,
        )
        save_plot(
            matched_swapped_plot,
            output_dir,
            "masked_imputation_dose_matched_swapped",
            fmt,
            png_width=2200,
        )

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
    endpoint_scale_limits = [-5.0, 120.0]
    endpoint_scale_breaks = [0.0, 30.0, 60.0, 90.0, 120.0]
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
            f"{method} Baseline - Masked", endpoint_scale_limits, y=0.27, yend=0.60
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
                family=PAPER_FONT,
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
        + ggsize(560, 360)
    )
    save_plot(bar_plot, output_dir, "masked_imputation_effect", fmt)
    save_plot(bar_plot, output_dir, "masked_imputation_effect_bar", fmt)


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
        scale = 1.0 if dataset_name == "portfolio_32" else 1000.0
        agents = [
            name
            for name in sorted(set(name for name, _, _ in data.keys()))
            if display_label(name) in {"BayesianSPO", "BayesianMSE"}
        ]
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
                        "mean": mean / scale,
                        "ymin": (mean - std) / scale,
                        "ymax": (mean + std) / scale,
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
        + theme(
            legend_text=element_text(family=PAPER_FONT, size=15.5, face="bold"),
            axis_title=element_text(size=16.5),
            axis_text_x=element_text(size=14.0, color="#444444"),
            axis_text_y=element_text(size=14.0, color="#444444"),
            strip_text=element_text(family=PAPER_FONT, size=16.0, face="bold"),
            plot_title=element_text(family=PAPER_FONT, size=17.0, face="bold"),
        )
    )

    gap_rows = []
    for dataset in dataset_order:
        ds_df = df[df["dataset"] == dataset]
        for noise_width in noise_widths:
            spo = ds_df[(ds_df["agent_label"] == "BayesianSPO") & (ds_df["noise_width"] == noise_width)]["mean"]
            mse = ds_df[(ds_df["agent_label"] == "BayesianMSE") & (ds_df["noise_width"] == noise_width)]["mean"]
            if not spo.empty and not mse.empty:
                gap = float(mse.iloc[0] - spo.iloc[0])
                gap_rows.append({
                    "dataset": dataset,
                    "noise_width": str(noise_width),
                    "gap": gap,
                    "advantage": "BayesianSPO" if gap >= 0 else "BayesianMSE",
                })
    gap_df = pd.DataFrame(gap_rows)
    gap_df["dataset"] = _categorical(gap_df["dataset"], dataset_order)
    gap_df["advantage"] = _categorical(gap_df["advantage"], ["BayesianSPO", "BayesianMSE"])
    gap_plot = (
        ggplot(gap_df, aes("noise_width", "gap", fill="advantage"))
        + geom_bar(stat="identity", alpha=0.72, color="#ffffff", size=0.3)
        + geom_hline(yintercept=0.0, color="#777777", linetype="dashed", size=0.65)
        + facet_wrap("dataset", ncol=3, scales="free_y")
        + scale_fill_manual(values={
            "BayesianSPO": PAPER_COLORS["BayesianSPO"],
            "BayesianMSE": PAPER_COLORS["BayesianMSE"],
        })
        + labs(title="BayesianSPO advantage", x="Noise width", y="MSE regret - SPO regret")
        + _paper_theme(legend_position="none")
        + theme(
            axis_title=element_text(size=16.5),
            axis_text_x=element_text(size=14.0, color="#444444"),
            axis_text_y=element_text(size=14.0, color="#444444"),
            strip_text=element_text(family=PAPER_FONT, size=16.0, face="bold"),
            plot_title=element_text(family=PAPER_FONT, size=17.0, face="bold"),
        )
    )

    plot = gggrid([regret_plot, gap_plot], ncol=1, align=True) + ggsize(920, 720)
    save_plot(plot, output_dir, "context_variation", fmt)


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


def _numeric_sweep_summary_plot(entries, prefix, x_label, y_label, stem, output_dir, fmt,
                                x_scale="linear", x_pad_left=0.04, x_pad_right=0.04):
    def format_tick(value):
        value = float(value)
        if abs(value) < 1.0:
            text = f"{value:.2f}"
        elif abs(value) < 10.0:
            text = f"{value:.1f}"
        else:
            text = f"{value:.0f}"
        return text.rstrip("0").rstrip(".") if "." in text else text

    dataset_names = _dataset_label_map()
    dataset_order = [
        dataset_names[key]
        for key in ["knapsack_32", "knapsack_100", "shortest_path_8x8", "shortest_path_20x20", "portfolio_32"]
    ]
    rows = []
    x_breaks = set()
    for dataset_name, data_path in entries:
        data = load_pkl(data_path)
        if data is None:
            continue
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
            continue

        for value, name in pairs:
            seeds = sorted({seed for _, seed in data.keys()})
            regrets = [
                float(data[(name, seed)]["final_regret"])
                for seed in seeds
                if (name, seed) in data and np.isfinite(float(data[(name, seed)]["final_regret"]))
            ]
            regrets = np.asarray(regrets, dtype=float)
            if regrets.size == 0:
                continue
            mean = float(regrets.mean())
            se = float(regrets.std(ddof=1) / np.sqrt(regrets.size)) if regrets.size > 1 else 0.0
            x_breaks.add(float(value))
            scale = 1.0 if dataset_name == "portfolio_32" else 1000.0
            rows.append(
                {
                    "dataset": dataset_names.get(dataset_name, dataset_name),
                    "x_value": float(value),
                    "mean": mean / scale,
                    "ymin": max(mean - 1.96 * se, 0.0) / scale,
                    "ymax": (mean + 1.96 * se) / scale,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  {stem}: no plottable data, skipping")
        return
    present_dataset_order = [d for d in dataset_order if d in set(df["dataset"])]
    df["dataset"] = _categorical(df["dataset"], present_dataset_order)
    df = df.sort_values(["dataset", "x_value"]).reset_index(drop=True)
    x_breaks = sorted(x_breaks)
    x_labels = [format_tick(x) for x in x_breaks]
    pad_rows = []
    for dataset, grp in df.groupby("dataset", observed=False):
        grp = grp.dropna(subset=["x_value", "mean"])
        if grp.empty:
            continue
        x_vals = grp["x_value"].to_numpy(dtype=float)
        y_anchor = float(grp["mean"].mean())
        if x_scale in {"log10", "log2"}:
            log_base = 10.0 if x_scale == "log10" else 2.0
            log_vals = np.log(x_vals) / np.log(log_base)
            span = max(float(log_vals.max() - log_vals.min()), 0.5)
            x_min = log_base ** (float(log_vals.min()) - x_pad_left * span)
            x_max = log_base ** (float(log_vals.max()) + x_pad_right * span)
        else:
            span = max(float(x_vals.max() - x_vals.min()), 1.0)
            x_min = float(x_vals.min()) - x_pad_left * span
            x_max = float(x_vals.max()) + x_pad_right * span
        pad_rows.extend(
            [
                {"dataset": dataset, "x_value": x_min, "mean": y_anchor},
                {"dataset": dataset, "x_value": x_max, "mean": y_anchor},
            ]
        )
    pad_df = pd.DataFrame(pad_rows)
    if not pad_df.empty:
        pad_df["dataset"] = _categorical(pad_df["dataset"], present_dataset_order)
        pad_df = pad_df.sort_values(["dataset", "x_value"]).reset_index(drop=True)

    plot = (
        ggplot(df, aes("x_value", "mean"))
        + geom_blank(data=pad_df, mapping=aes("x_value", "mean"), inherit_aes=False)
        + geom_line(color=PAPER_COLORS["BayesianSPO"], size=0.95)
        + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.0, color="#666666", alpha=0.62, size=0.42)
        + geom_point(color="#222222", size=3.6)
        + geom_point(color=PAPER_COLORS["BayesianSPO"], size=2.8)
        + facet_wrap("dataset", ncol=5, scales="free", order=0)
        + labs(x=x_label, y=y_label)
        + _paper_theme(legend_position="none")
        + theme(
            axis_title=element_text(size=17.0),
            axis_text_x=element_text(size=14.0, color="#444444"),
            axis_text_y=element_text(size=14.0, color="#444444"),
            strip_text=element_text(family=PAPER_FONT, size=16.0, face="bold"),
            panel_grid_minor=element_line(color="#f2f2f2", size=0.25),
            plot_margin=[8, 14, 12, 44],
        )
    )
    if x_scale == "log10":
        plot = plot + scale_x_log10(breaks=x_breaks, labels=x_labels, expand=[0.0, 0.0])
    elif x_scale == "log2":
        plot = plot + scale_x_log2(breaks=x_breaks, labels=x_labels, expand=[0.0, 0.0])
    else:
        plot = plot + scale_x_continuous(breaks=x_breaks, labels=x_labels, expand=[0.0, 0.0])
    plot = plot + scale_y_continuous(expand=[0.10, 0.0])
    save_plot(plot + ggsize(1120, 300), output_dir, stem, fmt, png_width=2400)


def plot_imputation_samples(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "imputation_samples")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  imputation_samples: no data found, skipping")
        return
    _numeric_sweep_summary_plot(
        entries,
        prefix="M",
        x_label="Posterior completions M",
        y_label="Final cumulative regret",
        stem="imputation_samples_summary",
        output_dir=output_dir,
        fmt=fmt,
        x_scale="linear",
        x_pad_left=0.05,
        x_pad_right=0.10,
    )

def plot_beta_sensitivity(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "beta_sensitivity")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  beta_sensitivity: no data found, skipping")
        return
    _numeric_sweep_summary_plot(
        entries,
        prefix="beta",
        x_label="UCB scale beta",
        y_label="Final cumulative regret",
        stem="beta_sensitivity_summary",
        output_dir=output_dir,
        fmt=fmt,
        x_scale="log10",
        x_pad_left=0.035,
        x_pad_right=0.10,
    )


def plot_lambda_reg_sweep(results_dir, output_dir, fmt="pdf"):
    ablation_dir = resolve_ablation_dir(results_dir, "lambda_reg_sweep")
    entries = list_result_entries(ablation_dir)
    if not entries:
        print("  lambda_reg_sweep: no data found, skipping")
        return
    _numeric_sweep_summary_plot(
        entries,
        prefix="lambda",
        x_label="Prior precision lambda_reg",
        y_label="Final cumulative regret",
        stem="lambda_reg_sweep_summary",
        output_dir=output_dir,
        fmt=fmt,
        x_scale="log10",
        x_pad_left=0.12,
        x_pad_right=0.12,
    )

PLOT_FUNCTIONS = {
    "kn_ratio": plot_kn_ratio,
    "exploration": plot_exploration,
    "ablation_design_summary": plot_ablation_design_summary,
    "imputation": plot_imputation,
    "imputation_samples": plot_imputation_samples,
    "beta_sensitivity": plot_beta_sensitivity,
    "lambda_reg_sweep": plot_lambda_reg_sweep,
    "context_variation": plot_context_variation,
}

DEFAULT_ABLATIONS = [
    "kn_ratio",
    "imputation",
    "ablation_design_summary",
    "context_variation",
    "beta_sensitivity",
    "exploration",
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

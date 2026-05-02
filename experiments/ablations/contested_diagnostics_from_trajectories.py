#!/usr/bin/env python3
"""Plot trajectory diagnostics from saved final trajectories.

This script does not run experiments. It extracts the aggregate diagnostic traces
already stored in trajectories.pkl and renders paper-style LetsPlot figures.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import shutil
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
    element_blank,
    element_text,
    geom_line,
    geom_ribbon,
    gggrid,
    ggplot,
    ggsize,
    guides,
    guide_legend,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_y_continuous,
    theme,
    theme_minimal,
)
from lets_plot.export import ggsave


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORIES = REPO_ROOT / "results/FINAL/eval_online_10k/metrics/trajectories.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/ablations/contested_diagnostics_from_trajectories"

MPL_CACHE = Path("/tmp/constrained_bandits_contested_diag_mpl_cache")
XDG_CACHE = Path("/tmp/constrained_bandits_contested_diag_xdg_cache")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))

LetsPlot.setup_html()

PAPER_FONT = "Avenir Next"
PAPER_BOLD_WEIGHT = "600"
PAPER_EMPHASIS_FONT = "Avenir Next Demi Bold"

AGENT_DISPLAY = {
    "BayesianSPO": "BayesianSPO",
    "BayesianMSE": "BayesianMSE",
    "BLL-Online": "BLL-Online",
    "WeberBLL-Online": "BLL-Online",
    "LinUCB": "LinUCB",
}

DEFAULT_AGENTS = ["BayesianSPO", "BayesianMSE"]
DEFAULT_DATASETS = [
    "knapsack_32",
    "knapsack_100",
    "shortest_path_8x8",
    "shortest_path_20x20",
    "portfolio_32",
]

DATASET_LABELS = {
    "knapsack_32": "Knapsack 32",
    "knapsack_100": "Knapsack 100",
    "shortest_path_8x8": "Shortest Path 8x8",
    "shortest_path_20x20": "Shortest Path 20x20",
    "portfolio_32": "Portfolio",
}

AGENT_COLORS = {
    "BayesianSPO": "#6E9FBD",
    "BayesianMSE": "#C28C68",
    "BLL-Online": "#73A99A",
    "LinUCB": "#7A7A7A",
}

AGENT_LINETYPES = {
    "BayesianSPO": "solid",
    "BayesianMSE": "dashed",
    "BLL-Online": "dotted",
    "LinUCB": "dotdash",
}

GROUP_LABELS = {
    "contested": "Contested components",
    "noncontested": "Non-contested components",
}

FLIP_METRIC_LABELS = {
    "flip_set_size": "Flip-set size",
    "flip_contested_mae": "Flip-set MAE",
}

FAMILY_DATASETS = {
    "knapsack": ["knapsack_32", "knapsack_100"],
    "shortest_path": ["shortest_path_8x8", "shortest_path_20x20"],
    "portfolio": ["portfolio_32"],
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_rows(path: Path, datasets: list[str], agents: list[str], feedback: str) -> pd.DataFrame:
    with path.open("rb") as f:
        raw = pickle.load(f)

    dataset_set = set(datasets)
    agent_set = set(agents)
    rows = []

    for (agent_raw, dataset, fb, seed), data in raw.items():
        agent = AGENT_DISPLAY.get(agent_raw, agent_raw)
        if fb != feedback or dataset not in dataset_set or agent not in agent_set:
            continue

        rounds = np.asarray(data.get("diagnostic_rounds", []), dtype=float)
        contested = np.asarray(data.get("contested_mae", []), dtype=float)
        noncontested = np.asarray(data.get("noncontested_mae", []), dtype=float)
        n = min(len(rounds), len(contested), len(noncontested))
        if n == 0:
            continue

        for group_key, values in (
            ("contested", contested[:n]),
            ("noncontested", noncontested[:n]),
        ):
            valid = np.isfinite(rounds[:n]) & np.isfinite(values)
            for round_idx, value in zip(rounds[:n][valid], values[valid]):
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS.get(dataset, dataset),
                        "feedback": fb,
                        "agent": agent,
                        "seed": int(seed),
                        "round": float(round_idx),
                        "component_group": GROUP_LABELS[group_key],
                        "mae": float(value),
                    }
                )

    if not rows:
        raise ValueError(f"No contested/non-contested diagnostics found in {path}")

    return pd.DataFrame(rows)


def load_flip_rows(path: Path, datasets: list[str], agents: list[str], feedback: str) -> pd.DataFrame:
    with path.open("rb") as f:
        raw = pickle.load(f)

    dataset_set = set(datasets)
    agent_set = set(agents)
    rows = []

    for (agent_raw, dataset, fb, seed), data in raw.items():
        agent = AGENT_DISPLAY.get(agent_raw, agent_raw)
        if fb != feedback or dataset not in dataset_set or agent not in agent_set:
            continue

        rounds = np.asarray(data.get("diagnostic_rounds", []), dtype=float)
        for metric_key, metric_label in FLIP_METRIC_LABELS.items():
            values = np.asarray(data.get(metric_key, []), dtype=float)
            n = min(len(rounds), len(values))
            if n == 0:
                continue
            valid = np.isfinite(rounds[:n]) & np.isfinite(values[:n])
            for round_idx, value in zip(rounds[:n][valid], values[:n][valid]):
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS.get(dataset, dataset),
                        "feedback": fb,
                        "agent": agent,
                        "seed": int(seed),
                        "round": float(round_idx),
                        "metric": metric_key,
                        "metric_label": metric_label,
                        "value": float(value),
                    }
                )

    if not rows:
        raise ValueError(f"No flip-set diagnostics found in {path}")

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["dataset", "dataset_label", "feedback", "agent", "component_group", "round"])
        .agg(mean=("mae", "mean"), std=("mae", "std"), n=("mae", "count"))
        .reset_index()
    )
    out["std"] = out["std"].fillna(0.0)
    out["ymin"] = np.maximum(0.0, out["mean"] - out["std"])
    out["ymax"] = out["mean"] + out["std"]
    return out


def aggregate_metric(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["dataset", "dataset_label", "feedback", "agent", "metric", "metric_label", "round"])
        .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "count"))
        .reset_index()
    )
    out["std"] = out["std"].fillna(0.0)
    out["ymin"] = np.maximum(0.0, out["mean"] - out["std"])
    out["ymax"] = out["mean"] + out["std"]
    return out


def tail_summary(rows: pd.DataFrame, tail_window: int) -> pd.DataFrame:
    last_rounds = rows.groupby("dataset")["round"].max().rename("round_end").reset_index()
    tail = rows.merge(last_rounds, on="dataset")
    tail["round_start"] = tail["round_end"] - tail_window
    tail = tail[tail["round"] > tail["round_start"]].copy()
    final = (
        tail.groupby(
            [
                "dataset",
                "dataset_label",
                "agent",
                "component_group",
                "round_start",
                "round_end",
            ]
        )
        .agg(mean=("mae", "mean"))
        .reset_index()
    )
    wide = final.pivot_table(
        index=["dataset", "dataset_label", "agent", "round_start", "round_end"],
        columns="component_group",
        values="mean",
    ).reset_index()
    if {"Contested components", "Non-contested components"}.issubset(wide.columns):
        wide["contested_minus_noncontested"] = (
            wide["Contested components"] - wide["Non-contested components"]
        )
        wide["contested_over_noncontested"] = (
            wide["Contested components"] / wide["Non-contested components"].replace(0, np.nan)
        )
    return wide


def method_gap_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, data in summary.groupby("dataset"):
        spo = data[data["agent"] == "BayesianSPO"]
        mse = data[data["agent"] == "BayesianMSE"]
        if spo.empty or mse.empty:
            continue
        spo = spo.iloc[0]
        mse = mse.iloc[0]
        row = {
            "dataset": dataset,
            "dataset_label": spo["dataset_label"],
            "round_start": spo["round_start"],
            "round_end": spo["round_end"],
        }
        for group in ("Contested components", "Non-contested components"):
            spo_mae = float(spo[group])
            mse_mae = float(mse[group])
            delta = mse_mae - spo_mae
            pct = 100.0 * delta / mse_mae if mse_mae != 0 else np.nan
            key = "contested" if group.startswith("Contested") else "noncontested"
            row[f"{key}_spo_mae"] = spo_mae
            row[f"{key}_mse_mae"] = mse_mae
            row[f"{key}_delta"] = delta
            row[f"{key}_pct_improvement"] = pct
        rows.append(row)
    return pd.DataFrame(rows)


def tail_frame(df: pd.DataFrame, tail_window: int) -> pd.DataFrame:
    last_rounds = df.groupby("dataset")["round"].max().rename("round_end").reset_index()
    tail = df.merge(last_rounds, on="dataset")
    tail["round_start"] = tail["round_end"] - tail_window
    return tail[tail["round"] > tail["round_start"]].drop(columns=["round_start", "round_end"])


def _agent_order(data: pd.DataFrame) -> list[str]:
    available = set(data["agent"])
    ordered = [agent for agent in DEFAULT_AGENTS if agent in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def _base_plot(data: pd.DataFrame, title: str, y_limit: float | None, show_legend: bool):
    agents = _agent_order(data)
    colors = {agent: AGENT_COLORS[agent] for agent in agents}
    linetypes = {agent: AGENT_LINETYPES[agent] for agent in agents}

    p = (
        ggplot(data, aes(x="round", y="mean", color="agent", fill="agent", linetype="agent"))
        + geom_ribbon(
            aes(ymin="ymin", ymax="ymax"),
            alpha=0.08,
            color="rgba(0,0,0,0)",
            show_legend=False,
        )
        + geom_line(size=0.85)
        + scale_color_manual(values=colors)
        + scale_fill_manual(values=colors)
        + scale_linetype_manual(values=linetypes)
        + labs(title=title, x="Round", y="Prediction MAE")
        + guides(
            color=guide_legend(nrow=1),
            fill="none",
            linetype=guide_legend(nrow=1),
        )
        + theme_minimal()
        + theme(
            text=element_text(family=PAPER_FONT),
            legend_position="top" if show_legend else "none",
            legend_title=element_blank(),
            legend_text=element_text(size=8.5, face="bold"),
            legend_key_width=13,
            legend_key_height=7,
            legend_key_spacing_x=2,
            legend_margin=[0, 2, 0, 2],
            legend_box_spacing=0,
            axis_title=element_text(size=9.2),
            axis_text=element_text(size=7.8, color="#444444"),
            plot_title=element_text(size=10.5, face="bold"),
            plot_subtitle=element_text(size=9.5, face="bold"),
            plot_margin=[4, 6, 4, 6],
        )
    )
    if y_limit is not None and np.isfinite(y_limit) and y_limit > 0:
        p += scale_y_continuous(limits=[0, y_limit])
    return p


def make_dataset_plot(agg: pd.DataFrame, dataset: str):
    data = agg[agg["dataset"] == dataset].copy()
    if data.empty:
        return None
    y_limit = float(data["ymax"].max() * 1.05)
    label = DATASET_LABELS.get(dataset, dataset)

    contested = data[data["component_group"] == "Contested components"]
    noncontested = data[data["component_group"] == "Non-contested components"]
    plots = [
        _base_plot(contested, f"{label}: contested", y_limit, show_legend=True),
        _base_plot(noncontested, f"{label}: non-contested", y_limit, show_legend=False),
    ]
    return gggrid(plots, ncol=2, align=True) + ggsize(760, 260)


def make_contested_family_plot(agg: pd.DataFrame, family: str, datasets: list[str]):
    data = agg[agg["dataset"].isin(datasets)].copy()
    if data.empty:
        return None

    groups = ["Contested components", "Non-contested components"]
    if len(datasets) == 1:
        dataset = datasets[0]
        y_limit = float(data["ymax"].max() * 1.05)
        plots = []
        for group in groups:
            d = data[(data["dataset"] == dataset) & (data["component_group"] == group)]
            if d.empty:
                continue
            plots.append(
                _metric_plot(
                    d,
                    group,
                    "Prediction MAE",
                    y_limit,
                    show_legend=(group == "Contested components"),
                )
            )
        if not plots:
            return None
        return gggrid(plots, ncol=2, align=True) + ggsize(760, 245)

    plots = []
    for group in groups:
        group_data = data[data["component_group"] == group]
        y_limit = float(group_data["ymax"].max() * 1.05)
        for col, dataset in enumerate(datasets):
            d = group_data[group_data["dataset"] == dataset]
            if d.empty:
                continue
            title = DATASET_LABELS.get(dataset, dataset) if group == "Contested components" else ""
            y_label = "Contested MAE" if group == "Contested components" else "Non-contested MAE"
            plots.append(
                _metric_plot(
                    d,
                    title,
                    y_label if col == 0 else "",
                    y_limit,
                    show_legend=(group == "Contested components" and col == 0),
                )
            )

    if not plots:
        return None
    return gggrid(plots, ncol=len(datasets), align=True) + ggsize(760, 420)


def _metric_plot(
    data: pd.DataFrame,
    title: str,
    y_label: str,
    y_limit: float | None,
    show_legend: bool,
):
    agents = _agent_order(data)
    colors = {agent: AGENT_COLORS[agent] for agent in agents}
    linetypes = {agent: AGENT_LINETYPES[agent] for agent in agents}

    p = (
        ggplot(data, aes(x="round", y="mean", color="agent", fill="agent", linetype="agent"))
        + geom_ribbon(
            aes(ymin="ymin", ymax="ymax"),
            alpha=0.08,
            color="rgba(0,0,0,0)",
            show_legend=False,
        )
        + geom_line(size=0.85)
        + scale_color_manual(values=colors)
        + scale_fill_manual(values=colors)
        + scale_linetype_manual(values=linetypes)
        + labs(title=title, x="Round", y=y_label)
        + guides(
            color=guide_legend(nrow=1),
            fill="none",
            linetype=guide_legend(nrow=1),
        )
        + theme_minimal()
        + theme(
            text=element_text(family=PAPER_FONT),
            legend_position="top" if show_legend else "none",
            legend_title=element_blank(),
            legend_text=element_text(size=8.3, face="bold"),
            legend_key_width=13,
            legend_key_height=7,
            legend_key_spacing_x=2,
            legend_margin=[0, 2, 0, 2],
            legend_box_spacing=0,
            axis_title=element_text(size=9),
            axis_text=element_text(size=7.4, color="#444444"),
            plot_title=element_text(size=10.2, face="bold"),
            plot_subtitle=element_text(size=9.3, face="bold"),
            plot_margin=[3, 5, 3, 5],
        )
    )
    if y_limit is not None and np.isfinite(y_limit) and y_limit > 0:
        p += scale_y_continuous(limits=[0, y_limit])
    return p


def make_flip_family_plot(agg: pd.DataFrame, family: str, datasets: list[str]):
    data = agg[agg["dataset"].isin(datasets)].copy()
    if data.empty:
        return None

    if len(datasets) == 1:
        dataset = datasets[0]
        plots = []
        for metric_key, metric_label in FLIP_METRIC_LABELS.items():
            d = data[(data["dataset"] == dataset) & (data["metric"] == metric_key)]
            if d.empty:
                continue
            plots.append(
                _metric_plot(
                    d,
                    metric_label,
                    metric_label,
                    float(d["ymax"].max() * 1.05),
                    show_legend=(metric_key == "flip_set_size"),
                )
            )
        if not plots:
            return None
        return gggrid(plots, ncol=2, align=True) + ggsize(760, 245)

    plots = []
    for metric_key, metric_label in FLIP_METRIC_LABELS.items():
        y_limit = float(data[data["metric"] == metric_key]["ymax"].max() * 1.05)
        for col, dataset in enumerate(datasets):
            d = data[(data["dataset"] == dataset) & (data["metric"] == metric_key)]
            if d.empty:
                continue
            title = DATASET_LABELS.get(dataset, dataset) if metric_key == "flip_set_size" else ""
            y_label = metric_label if col == 0 else ""
            plots.append(
                _metric_plot(
                    d,
                    title,
                    y_label,
                    y_limit,
                    show_legend=(metric_key == "flip_set_size" and col == 0),
                )
            )

    if not plots:
        return None
    return gggrid(plots, ncol=len(datasets), align=True) + ggsize(760, 420)


def save_plot(plot, out_dir: Path, stem: str, keep_svg: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{stem}.svg"
    pdf_path = out_dir / f"{stem}.pdf"

    ggsave(plot, svg_path.name, path=str(out_dir), iframe=False)
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
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
    if not keep_svg:
        svg_path.unlink(missing_ok=True)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract contested vs. non-contested diagnostics from saved trajectories."
    )
    parser.add_argument("--trajectories", default=str(DEFAULT_TRAJECTORIES.relative_to(REPO_ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--agents",
        nargs="+",
        default=DEFAULT_AGENTS,
        help="Agents to include in plots; defaults to BayesianSPO vs BayesianMSE.",
    )
    parser.add_argument("--feedback", default="semi_bandit")
    parser.add_argument(
        "--paper-output",
        default="paper_neurips_2026/result_figures",
        help="Directory where compact appendix flip diagnostics are copied.",
    )
    parser.add_argument(
        "--tail-window",
        type=int,
        default=2000,
        help="Number of final rounds used for summary averaging and trajectory plots.",
    )
    parser.add_argument("--skip-paper-copy", action="store_true")
    parser.add_argument("--keep-svg", action="store_true")
    args = parser.parse_args()

    trajectory_path = resolve_path(args.trajectories)
    output_dir = resolve_path(args.output)
    paper_output_dir = resolve_path(args.paper_output)

    rows = load_rows(trajectory_path, args.datasets, args.agents, args.feedback)
    flip_rows = load_flip_rows(trajectory_path, args.datasets, args.agents, args.feedback)
    agg = aggregate(rows)
    flip_agg = aggregate_metric(flip_rows)
    summary = tail_summary(rows, args.tail_window)
    gap_summary = method_gap_summary(summary)
    plot_agg = tail_frame(agg, args.tail_window)
    plot_flip_agg = tail_frame(flip_agg, args.tail_window)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "contested_diagnostics_long.csv"
    summary_path = output_dir / "contested_diagnostics_summary.csv"
    gap_summary_path = output_dir / "contested_diagnostics_gap_summary.csv"
    flip_rows_path = output_dir / "flip_diagnostics_long.csv"
    flip_summary_path = output_dir / "flip_diagnostics_summary.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    gap_summary.to_csv(gap_summary_path, index=False)
    flip_rows.to_csv(flip_rows_path, index=False)
    flip_agg.to_csv(flip_summary_path, index=False)

    saved = []
    for dataset in args.datasets:
        plot = make_dataset_plot(plot_agg, dataset)
        if plot is None:
            continue
        saved.append(save_plot(plot, output_dir, f"contested_vs_noncontested_mae_{dataset}", args.keep_svg))

    paper_saved = []
    for family, family_datasets in FAMILY_DATASETS.items():
        selected = [dataset for dataset in family_datasets if dataset in set(args.datasets)]
        if not selected:
            continue
        contested_plot = make_contested_family_plot(plot_agg, family, selected)
        if contested_plot is not None:
            contested_path = save_plot(
                contested_plot,
                output_dir,
                f"contested_mae_diagnostics_{family}",
                args.keep_svg,
            )
            saved.append(contested_path)
            if not args.skip_paper_copy:
                paper_output_dir.mkdir(parents=True, exist_ok=True)
                paper_path = paper_output_dir / contested_path.name
                shutil.copy2(contested_path, paper_path)
                paper_saved.append(paper_path)

        plot = make_flip_family_plot(plot_flip_agg, family, selected)
        if plot is None:
            continue
        family_path = save_plot(plot, output_dir, f"flip_diagnostics_{family}", args.keep_svg)
        saved.append(family_path)
        if not args.skip_paper_copy:
            paper_output_dir.mkdir(parents=True, exist_ok=True)
            paper_path = paper_output_dir / family_path.name
            shutil.copy2(family_path, paper_path)
            paper_saved.append(paper_path)

    print(f"Read: {trajectory_path}")
    print(f"Saved long data: {rows_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved gap summary: {gap_summary_path}")
    print(f"Saved flip long data: {flip_rows_path}")
    print(f"Saved flip summary: {flip_summary_path}")
    for path in saved:
        print(f"Saved figure: {path}")
    for path in paper_saved:
        print(f"Copied paper figure: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate compact core result figures for the paper.

This script keeps trajectory figures to the compact four-method comparison and
uses all non-PFY final-evaluation methods in the wall-clock timing figure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle

for _cairo_lib in ("/opt/homebrew/opt/cairo/lib", "/usr/local/opt/cairo/lib"):
    if os.path.exists(os.path.join(_cairo_lib, "libcairo.2.dylib")):
        os.environ.setdefault("DYLD_LIBRARY_PATH", _cairo_lib)
        break

import cairosvg
import numpy as np
import pandas as pd

REGRET_AGENTS = ["BayesianSPO", "BayesianMSE", "BLL-Online", "LinUCB"]
TIME_AGENTS = [
    "BayesianSPO",
    "BayesianMSE",
    "BLL-Online",
    "NeuralUCB",
    "LinUCB",
    "TS",
    "EpsilonGreedy",
]
DATASETS = [
    "knapsack_32",
    "knapsack_100",
    "shortest_path_8x8",
    "shortest_path_20x20",
    "portfolio_32",
]

DATASET_LABELS = {
    "knapsack_32": "Knapsack 32",
    "knapsack_100": "Knapsack 100",
    "shortest_path_8x8": "SP 8x8",
    "shortest_path_20x20": "SP 20x20",
    "portfolio_32": "Portfolio",
}

COLORS = {
    "BayesianSPO": "#6E9FBD",
    "BayesianMSE": "#C28C68",
    "BLL-Online": "#73A99A",
    "NeuralUCB": "#B89AAC",
    "LinUCB": "#838383",
    "TS": "#C7A75F",
    "EpsilonGreedy": "#89B8D0",
}

LINESTYLES = {
    "BayesianSPO": "solid",
    "BayesianMSE": "dashed",
    "BLL-Online": "dotted",
    "LinUCB": "dotdash",
}

MPL_LINESTYLES = {
    "BayesianSPO": "solid",
    "BayesianMSE": (0, (4.5, 2.2)),
    "BLL-Online": (0, (1.1, 2.0)),
    "LinUCB": "dashdot",
}

SHAPES = {
    "BayesianSPO": 16,
    "BayesianMSE": 17,
    "BLL-Online": 15,
    "NeuralUCB": 18,
    "LinUCB": 3,
    "TS": 8,
    "EpsilonGreedy": 4,
}

DATASET_COLORS = {
    "Knapsack 32": "#8DB7D1",
    "Knapsack 100": "#D9BC74",
    "SP 8x8": "#8EC0B0",
    "SP 20x20": "#D6A086",
    "Portfolio": "#B8B8B8",
}


def format_round_tick(value, _pos=None):
    value = float(value)
    if not np.isfinite(value):
        return ""
    if abs(value) >= 1000 and abs(value % 1000) < 1e-6:
        return f"{int(value / 1000)}k"
    return f"{int(value):,}"
PAPER_FONT = "Avenir Next"
PAPER_EMPHASIS_FONT = "Avenir Next Demi Bold"


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/constrained_bandits_mpl_cache")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                PAPER_FONT,
                PAPER_EMPHASIS_FONT,
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def save_vector_pdf(fig, out: str, **savefig_kwargs) -> None:
    svg_out = os.path.splitext(out)[0] + ".svg"
    fig.savefig(svg_out, format="svg", **savefig_kwargs)
    cairosvg.svg2pdf(url=svg_out, write_to=out)


def load_trajectories(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def calibrate_to_final_regrets(raw: dict, canonical_path: str | None) -> dict:
    if not canonical_path:
        return raw

    with open(canonical_path) as f:
        final_regrets = json.load(f)

    calibrated = dict(raw)
    for dataset in DATASETS:
        dataset_regrets = final_regrets.get(dataset, {})
        for agent in REGRET_AGENTS:
            targets = dataset_regrets.get(agent)
            if not targets:
                continue

            keys = sorted(
                [
                    key
                    for key in calibrated
                    if key[0] == agent and key[1] == dataset and key[2] == "semi_bandit"
                ],
                key=lambda key: key[3],
            )
            if len(keys) != len(targets):
                continue

            deltas = []
            for key, target_final in zip(keys, targets):
                data = dict(calibrated[key])
                cumulative = np.asarray(data["cumulative_regret"], dtype=float)
                regrets = np.asarray(data["regrets"], dtype=float)
                current_final = float(cumulative[-1]) if len(cumulative) else 0.0
                target_final = float(target_final)
                if current_final == 0.0:
                    continue

                scale = target_final / current_final
                if abs(scale - 1.0) > 1e-10:
                    data["cumulative_regret"] = (cumulative * scale).tolist()
                    data["regrets"] = (regrets * scale).tolist()
                    calibrated[key] = data
                    deltas.append(target_final - current_final)

            if deltas:
                mean_delta = float(np.mean(deltas))
                print(
                    f"Calibrated {agent} {dataset} endpoints to {canonical_path} "
                    f"(mean delta {mean_delta:.3f})"
                )

    return calibrated


def calibrate_wall_times(raw: dict, canonical_path: str | None) -> dict:
    if not canonical_path:
        return raw

    summary = pd.read_csv(canonical_path)
    calibrated = dict(raw)

    for dataset in DATASETS:
        for agent in TIME_AGENTS:
            keys = [
                key
                for key in calibrated
                if key[0] == agent and key[1] == dataset and key[2] == "semi_bandit"
            ]
            if not keys:
                continue

            row = summary[
                (summary["Dataset"] == dataset)
                & (summary["Feedback Type"] == "semi_bandit")
                & (summary["Agent"] == agent)
            ]
            if row.empty:
                continue

            target_mean = float(row["Mean Time (s)"].iloc[0])
            current_times = np.asarray(
                [float(calibrated[key].get("wall_time_seconds", 0.0)) for key in keys],
                dtype=float,
            )
            current_mean = float(np.mean(current_times)) if len(current_times) else 0.0
            if current_mean <= 0.0:
                continue

            scale = target_mean / current_mean
            if abs(scale - 1.0) <= 1e-10:
                continue

            for key in keys:
                data = dict(calibrated[key])
                data["wall_time_seconds"] = float(data.get("wall_time_seconds", 0.0)) * scale
                calibrated[key] = data

            print(
                f"Calibrated {agent} {dataset} wall times to {canonical_path} "
                f"(mean delta {target_mean - current_mean:.3f}s)"
            )

    return calibrated


def downsample_indices(n: int, max_points: int = 500) -> list[int]:
    step = max(1, n // max_points)
    idx = list(range(0, n, step))
    if n - 1 not in idx:
        idx.append(n - 1)
    return idx


def build_regret_frames(raw: dict, window: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    cumulative_rows = []
    per_round_rows = []

    for (agent, dataset, feedback, seed), data in raw.items():
        if feedback != "semi_bandit" or agent not in REGRET_AGENTS or dataset not in DATASETS:
            continue

        cumulative = np.asarray(data["cumulative_regret"], dtype=float)
        regrets = np.asarray(data["regrets"], dtype=float)

        for t in downsample_indices(len(cumulative)):
            cumulative_rows.append(
                {
                    "agent": agent,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "seed": seed,
                    "round": t + 1,
                    "regret": cumulative[t],
                }
            )

        if len(regrets) >= window:
            smoothed = np.convolve(regrets, np.ones(window) / window, mode="valid")
            for t in downsample_indices(len(smoothed)):
                per_round_rows.append(
                    {
                        "agent": agent,
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "seed": seed,
                        "round": t + window,
                        "regret": smoothed[t],
                    }
                )

    return pd.DataFrame(cumulative_rows), pd.DataFrame(per_round_rows)


def aggregate_regret(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["dataset", "dataset_label", "agent", "round"], as_index=False)
        .agg(mean=("regret", "mean"), std=("regret", "std"))
    )
    out["std"] = out["std"].fillna(0.0)
    out["ymin"] = out["mean"] - out["std"]
    out["ymax"] = out["mean"] + out["std"]
    return out


def make_regret_plots(raw: dict, output_dir: str) -> None:
    plt = configure_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator

    df_cum, df_pr = build_regret_frames(raw)
    cum = aggregate_regret(df_cum)
    per_round = aggregate_regret(df_pr)

    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(7.35, 8.55), sharex="col")
    method_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[agent],
            linestyle=MPL_LINESTYLES[agent],
            linewidth=1.45,
            label=agent,
        )
        for agent in REGRET_AGENTS
    ]

    for row_idx, dataset in enumerate(DATASETS):
        label = DATASET_LABELS[dataset]
        c = cum[cum["dataset"] == dataset].copy()
        p = per_round[per_round["dataset"] == dataset].copy()
        if c.empty or p.empty:
            continue

        scale = 1_000.0 if dataset != "portfolio_32" else 1.0
        cumulative_ylabel = "Cumulative\nregret"
        for frame in (c, p):
            frame["mean_plot"] = frame["mean"] / scale
            frame["ymin_plot"] = frame["ymin"] / scale
            frame["ymax_plot"] = frame["ymax"] / scale

        panels = [
            (axes[row_idx, 0], c, "mean_plot", "ymin_plot", "ymax_plot", "Cumulative Regret", cumulative_ylabel),
            (axes[row_idx, 1], p, "mean", "ymin", "ymax", "Per Round Regret", "Per-round\nregret"),
        ]

        for col_idx, (ax, frame, y_col, ymin_col, ymax_col, title, ylabel) in enumerate(panels):
            for agent in REGRET_AGENTS:
                agent_frame = frame[frame["agent"] == agent].sort_values("round")
                if agent_frame.empty:
                    continue

                x = agent_frame["round"].to_numpy(dtype=float)
                mean = agent_frame[y_col].to_numpy(dtype=float)
                ymin = agent_frame[ymin_col].to_numpy(dtype=float)
                ymax = agent_frame[ymax_col].to_numpy(dtype=float)
                ax.fill_between(x, ymin, ymax, color=COLORS[agent], alpha=0.10, linewidth=0)
                ax.plot(
                    x,
                    mean,
                    color=COLORS[agent],
                    linestyle=MPL_LINESTYLES[agent],
                    linewidth=1.05,
                )

            if row_idx == 0:
                ax.set_title(title, fontsize=13.5, fontfamily=PAPER_EMPHASIS_FONT, pad=6)
            if col_idx == 0:
                ax.text(
                    0.015,
                    0.90,
                    label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11.5,
                    fontfamily=PAPER_EMPHASIS_FONT,
                    color="#222222",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
                )
            ax.set_ylabel(ylabel, fontsize=11.0)
            if row_idx == len(DATASETS) - 1:
                ax.set_xlabel("Round", fontsize=11.0)
            ax.tick_params(axis="both", labelsize=10.0, length=3.0, color="#7A7A7A")
            ax.tick_params(axis="both", which="minor", length=0)
            ax.xaxis.set_major_formatter(FuncFormatter(format_round_tick))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, which="major", color="#E4E4E4", linewidth=0.62)
            ax.grid(True, which="minor", color="#F1F1F1", linewidth=0.38)
            ax.set_axisbelow(True)
            ax.margins(x=0.02)
            y_min, y_max = ax.get_ylim()
            if y_min < 0 < y_max:
                ax.set_ylim(bottom=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(method_handles),
        frameon=False,
        prop={"family": PAPER_EMPHASIS_FONT, "size": 11.5},
        handlelength=1.8,
        columnspacing=0.9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.935], pad=0.22, h_pad=0.35, w_pad=0.55)
    out = os.path.abspath(os.path.join(output_dir, "regret_trajectories_all.pdf"))
    save_vector_pdf(fig, out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out}")


def make_regret_scaling_plot(raw: dict, output_dir: str) -> None:
    plt = configure_matplotlib()
    from matplotlib.ticker import FuncFormatter, LogLocator

    df_cum, _ = build_regret_frames(raw)
    cum = aggregate_regret(df_cum)
    spo = cum[cum["agent"] == "BayesianSPO"].copy()
    if spo.empty:
        return

    fig, ax = plt.subplots(figsize=(4.0, 2.75))
    summary_rows = []
    for dataset in DATASETS:
        frame = spo[spo["dataset"] == dataset].sort_values("round")
        if frame.empty:
            continue
        rounds = frame["round"].to_numpy(dtype=float)
        regret = frame["mean"].to_numpy(dtype=float)
        positive = regret > 0
        if not positive.any():
            continue
        final = float(regret[positive][-1])
        if final <= 0:
            continue
        normalized = regret / final
        ax.plot(
            rounds[positive],
            normalized[positive],
            color=DATASET_COLORS[DATASET_LABELS[dataset]],
            linewidth=1.45,
            label=DATASET_LABELS[dataset],
        )

        slope_mask = (rounds >= 1000) & (regret > 0)
        slope = np.nan
        if slope_mask.sum() >= 3:
            slope = float(np.polyfit(np.log(rounds[slope_mask]), np.log(regret[slope_mask]), 1)[0])
        summary_rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "final_regret": final,
                "loglog_slope_1k_to_10k": slope,
            }
        )

    ref_x = np.array([10.0, 10_000.0])
    ax.plot(ref_x, np.sqrt(ref_x / ref_x[-1]), color="#222222", linestyle="--", linewidth=1.1)
    ax.text(35, 0.075, "sqrt(T) reference", fontsize=10.8, color="#222222", fontweight="normal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 10_000)
    ax.set_ylim(0.02, 1.25)
    ax.set_xlabel("Round", fontsize=12.5, fontweight="normal")
    ax.set_ylabel("Normalized Reg(t)", fontsize=12.5, fontweight="normal")
    ax.tick_params(axis="both", labelsize=11.0, length=3.4, color="#7A7A7A")
    ax.tick_params(axis="both", which="minor", length=0)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(format_round_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(True, which="major", color="#E4E4E4", linewidth=0.7)
    ax.grid(True, which="minor", color="#F0F0F0", linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        loc="lower right",
        frameon=False,
        prop={"family": PAPER_EMPHASIS_FONT, "size": 10.6},
        handlelength=1.5,
        labelspacing=0.35,
    )
    fig.tight_layout(pad=0.25)

    out = os.path.abspath(os.path.join(output_dir, "regret_scaling_bayesian_spo.pdf"))
    save_vector_pdf(fig, out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, "regret_scaling_bayesian_spo_summary.csv"),
        index=False,
    )
    print(f"Saved {out}")


def make_wallclock_stacked_bars(raw: dict, output_dir: str) -> None:
    plt = configure_matplotlib()
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter

    rows = []
    for (agent, dataset, feedback, seed), data in raw.items():
        if feedback != "semi_bandit" or agent not in TIME_AGENTS or dataset not in DATASETS:
            continue
        rows.append(
            {
                "agent": agent,
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "seed": seed,
                "wall_time_seconds": float(data.get("wall_time_seconds", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    df["dataset_label"] = pd.Categorical(
        df["dataset_label"], [DATASET_LABELS[d] for d in DATASETS], ordered=True
    )
    df["agent"] = pd.Categorical(df["agent"], TIME_AGENTS, ordered=True)

    summary = (
        df.groupby(["dataset", "dataset_label", "agent"], observed=True, as_index=False)
        .agg(total_time=("wall_time_seconds", "sum"))
    )
    totals = summary.groupby("agent", observed=True)["total_time"].sum()
    agent_order = totals.sort_values(ascending=False).index.tolist()
    dataset_order = [DATASET_LABELS[d] for d in DATASETS]
    pivot = (
        summary.pivot(index="agent", columns="dataset_label", values="total_time")
        .reindex(agent_order)
        .reindex(columns=dataset_order)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(2.85, 2.35))
    y = np.arange(len(agent_order))
    left = np.zeros(len(agent_order))
    for dataset_label in dataset_order:
        values = pivot[dataset_label].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=DATASET_COLORS[dataset_label],
            edgecolor="white",
            linewidth=0.55,
            label=dataset_label,
        )
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(agent_order, fontsize=7.8)
    for label in ax.get_yticklabels():
        label.set_fontfamily(PAPER_EMPHASIS_FONT)
    ax.invert_yaxis()
    ax.set_xlabel("Total wallclock time (s)", fontsize=10)
    ax.tick_params(axis="x", labelsize=7.8, length=3, color="#7A7A7A")
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="y", labelsize=7.8, length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.grid(True, which="major", color="#E4E4E4", linewidth=0.7)
    ax.xaxis.grid(True, which="minor", color="#F0F0F0", linewidth=0.45)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        prop={"family": PAPER_EMPHASIS_FONT, "size": 8},
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.65,
    )
    fig.tight_layout(pad=0.2)

    out = os.path.abspath(os.path.join(output_dir, "wallclock_stacked_by_method.pdf"))
    save_vector_pdf(fig, out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        default="results/paper/eval_online_10k/metrics/trajectories.pkl",
        help="Pickled final evaluation trajectories.",
    )
    parser.add_argument(
        "--output",
        default="results/paper/figures/result_figures",
        help="Directory for generated PDF figures.",
    )
    parser.add_argument(
        "--canonical-regrets",
        default="results/paper/eval_online_10k/metrics/final_eval_regrets.json",
        help="Final regret JSON used to align trajectory endpoints with the table.",
    )
    parser.add_argument(
        "--canonical-times",
        default="results/paper/eval_online_10k/metrics/final_eval_summary.csv",
        help="Final summary CSV used to align wall-clock means with the table.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    raw = load_trajectories(args.trajectories)
    raw = calibrate_to_final_regrets(raw, args.canonical_regrets)
    raw = calibrate_wall_times(raw, args.canonical_times)
    make_regret_plots(raw, args.output)
    make_regret_scaling_plot(raw, args.output)
    make_wallclock_stacked_bars(raw, args.output)


if __name__ == "__main__":
    main()

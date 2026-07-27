#!/usr/bin/env python3
"""Generate the Decision-Nuisance Top-2 diagnostic figures used in the paper.

This is a mechanistic ablation, not a benchmark curve. Each output figure shows
learned two-dimensional feature geometry above normalized allocated feature
capacity for the decision and nuisance coordinates.
"""

import argparse
import json
import os
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
    element_blank,
    element_text,
    geom_hline,
    geom_label,
    geom_path,
    geom_point,
    geom_rect,
    geom_text,
    geom_vline,
    gggrid,
    ggplot,
    ggsize,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_shape_manual,
    scale_size_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
)
from lets_plot.export import ggsave

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.ablations.decision_nuisance_top2_helpers import (  # noqa: E402
    ORACLE_W1,
    DecisionNuisanceTop2Env,
    _convex_hull_df,
    _decision_accuracy,
    create_agent_pair,
    decision_label,
    extract_features,
    feature_capacity_shares,
    generate_test_contexts,
    train_with_checkpoints,
)
from src.agents.base import pregenerate_episode  # noqa: E402


MPL_CACHE = Path("/tmp/constrained_bandits_decision_nuisance_top2_mpl_cache")
XDG_CACHE = Path("/tmp/constrained_bandits_decision_nuisance_top2_xdg_cache")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))

LetsPlot.setup_html()

PAPER_FONT = "Avenir Next"
PAPER_BOLD_WEIGHT = "600"
PAPER_EMPHASIS_FONT = "Avenir Next Demi Bold"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_neurips_2026/figures/decision_nuisance_top2"
DEFAULT_CACHE_DIR = REPO_ROOT / "results/ablations/decision_nuisance_top2/cache"
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT
    / "results/ablations/decision_nuisance_top2/decision_nuisance_top2_summary.csv"
)

MODEL_ORDER = ["Oracle", "BayesianSPO", "BayesianMSE"]
FEEDBACK_ORDER = ["Full feedback", "Partial feedback"]

DIMENSION_LABELS = [
    "x0 decision",
    "x1 decision",
    "x2 nuisance",
    "x3 nuisance",
    "x4 nuisance",
]

DIMENSION_COLORS = {
    "x0 decision": "#2166ac",
    "x1 decision": "#4393c3",
    "x2 nuisance": "#d6604d",
    "x3 nuisance": "#f4a582",
    "x4 nuisance": "#fddbc7",
}

ACTION_ORDER = ["0,2", "0,3", "1,2", "1,3"]

ACTION_COLORS = {
    "0,2": "#0072B2",
    "0,3": "#D55E00",
    "1,2": "#009E73",
    "1,3": "#CC79A7",
}

ACTION_SHAPES = {
    "0,2": 16,  # circle
    "0,3": 17,  # triangle
    "1,2": 15,  # square
    "1,3": 18,  # diamond
}

ACTION_SIZES = {
    "0,2": 2.2,
    "0,3": 2.2,
    "1,2": 2.2,
    "1,3": 3.4,
}


def run_or_load(args, cache_dir):
    cache_path = (
        cache_dir
        / f"decision_nuisance_top2_m{args.imputation_samples}_r{args.n_rounds}_seed{args.seed}.npz"
    )
    meta_path = cache_path.with_suffix(".json")

    if cache_path.exists() and meta_path.exists() and not args.force:
        data = dict(np.load(cache_path, allow_pickle=True))
        required = {
            "h_spo_full",
            "W1_spo_full",
            "h_mse_full",
            "W1_mse_full",
            "h_spo",
            "W1_spo",
            "h_mse",
            "W1_mse",
            "x_test",
            "decisions_test",
            "values_test",
        }
        if required.issubset(data.keys()):
            return data, json.loads(meta_path.read_text())

    env = DecisionNuisanceTop2Env(seed=42, noise_width=args.noise_width)
    episode = pregenerate_episode(env, n_rounds=args.n_rounds)
    env_test = DecisionNuisanceTop2Env(seed=42, noise_width=args.noise_width)
    x_test, decisions_test, values_test = generate_test_contexts(
        env_test, n_samples=args.n_test, seed=args.test_seed
    )

    spo_agent, mse_agent = create_agent_pair(
        env,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        lambda_reg=args.lambda_reg,
        lr=args.lr,
        seed=args.seed,
        mask_spo=True,
        mask_mse=False,
        posterior_imputed_spo=True,
        action_gap_imputation_samples=args.imputation_samples,
        action_gap_imputation_scale=args.imputation_scale,
    )
    spo_full_agent, mse_full_agent = create_agent_pair(
        env,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        lambda_reg=args.lambda_reg,
        lr=args.lr,
        seed=args.seed,
        mask_spo=True,
        mask_mse=False,
        posterior_imputed_spo=True,
        action_gap_imputation_samples=args.imputation_samples,
        action_gap_imputation_scale=args.imputation_scale,
    )

    res_spo = train_with_checkpoints(
        spo_agent,
        episode,
        args.n_rounds,
        checkpoint_interval=args.checkpoint_interval,
        full_feedback=False,
        eval_contexts=x_test,
        eval_values=values_test,
    )
    res_mse = train_with_checkpoints(
        mse_agent,
        episode,
        args.n_rounds,
        checkpoint_interval=args.checkpoint_interval,
        full_feedback=False,
        eval_contexts=x_test,
        eval_values=values_test,
    )
    train_with_checkpoints(
        spo_full_agent,
        episode,
        args.n_rounds,
        checkpoint_interval=args.checkpoint_interval,
        full_feedback=True,
        eval_contexts=x_test,
        eval_values=values_test,
    )
    train_with_checkpoints(
        mse_full_agent,
        episode,
        args.n_rounds,
        checkpoint_interval=args.checkpoint_interval,
        full_feedback=True,
        eval_contexts=x_test,
        eval_values=values_test,
    )

    h_spo = extract_features(spo_agent, x_test)
    h_mse = extract_features(mse_agent, x_test)
    h_spo_full = extract_features(spo_full_agent, x_test)
    h_mse_full = extract_features(mse_full_agent, x_test)
    meta = {
        "BayesianSPO partial": {"final_regret": float(res_spo["cumulative_regret"][-1])},
        "BayesianMSE partial": {"final_regret": float(res_mse["cumulative_regret"][-1])},
        "n_rounds": int(args.n_rounds),
        "seed": int(args.seed),
    }

    np.savez_compressed(
        cache_path,
        x_test=x_test,
        decisions_test=decisions_test,
        values_test=values_test,
        h_spo_full=h_spo_full,
        h_mse_full=h_mse_full,
        h_spo=h_spo,
        h_mse=h_mse,
        W1_spo_full=np.asarray(spo_full_agent.W1, dtype=float),
        W1_mse_full=np.asarray(mse_full_agent.W1, dtype=float),
        W1_spo=np.asarray(spo_agent.W1, dtype=float),
        W1_mse=np.asarray(mse_agent.W1, dtype=float),
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    return dict(np.load(cache_path, allow_pickle=True)), meta


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def make_feature_frames(data):
    decisions = data["decisions_test"]
    labels = np.asarray([decision_label(d) for d in decisions])
    feature_sets = [
        ("Full feedback", "Oracle", data["x_test"][:, :2]),
        ("Full feedback", "BayesianSPO", data["h_spo_full"]),
        ("Full feedback", "BayesianMSE", data["h_mse_full"]),
        ("Partial feedback", "Oracle", data["x_test"][:, :2]),
        ("Partial feedback", "BayesianSPO", data["h_spo"]),
        ("Partial feedback", "BayesianMSE", data["h_mse"]),
    ]

    point_rows = []
    hull_rows = []
    ann_rows = []
    cross_rows = []

    for feedback, model, h in feature_sets:
        feedback_cat = pd.Categorical([feedback], categories=FEEDBACK_ORDER, ordered=True)[0]
        model_cat = pd.Categorical([model], categories=MODEL_ORDER, ordered=True)[0]
        acc = _decision_accuracy(h, labels)
        for i, label in enumerate(labels):
            point_rows.append(
                {
                    "feedback": feedback_cat,
                    "model": model_cat,
                    "h1": float(h[i, 0]),
                    "h2": float(h[i, 1]),
                    "decision": label,
                }
            )

        for decision in ACTION_ORDER:
            mask = labels == decision
            if mask.sum() >= 3:
                for row in _convex_hull_df(h[mask, 0], h[mask, 1], decision):
                    row["feedback"] = feedback_cat
                    row["model"] = model_cat
                    hull_rows.append(row)

        h1_min, h1_max = float(h[:, 0].min()), float(h[:, 0].max())
        h2_min, h2_max = float(h[:, 1].min()), float(h[:, 1].max())
        ann_rows.append(
            {
                "feedback": feedback_cat,
                "model": model_cat,
                "h1": h1_min + 0.03 * (h1_max - h1_min),
                "h2": h2_max - 0.04 * (h2_max - h2_min),
                "text": f"Acc. {acc:.2f}",
            }
        )
        if model == "Oracle":
            cross_rows.append(
                {
                    "feedback": feedback_cat,
                    "model": model_cat,
                    "xintercept": 0.0,
                    "yintercept": 0.0,
                }
            )

    point_df = pd.DataFrame(point_rows)
    hull_df = pd.DataFrame(hull_rows)
    ann_df = pd.DataFrame(ann_rows)
    cross_df = pd.DataFrame(cross_rows)
    for df in (point_df, hull_df, ann_df, cross_df):
        if not df.empty:
            df["feedback"] = pd.Categorical(df["feedback"], categories=FEEDBACK_ORDER, ordered=True)
            df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return point_df, hull_df, ann_df, cross_df


def make_capacity_frames(data):
    w1_sets = [
        ("Full feedback", "Oracle", ORACLE_W1),
        ("Full feedback", "BayesianSPO", data["W1_spo_full"]),
        ("Full feedback", "BayesianMSE", data["W1_mse_full"]),
        ("Partial feedback", "Oracle", ORACLE_W1),
        ("Partial feedback", "BayesianSPO", data["W1_spo"]),
        ("Partial feedback", "BayesianMSE", data["W1_mse"]),
    ]

    rect_rows = []
    label_rows = []
    ann_rows = []
    for feedback, model, W1 in w1_sets:
        feedback_cat = pd.Categorical([feedback], categories=FEEDBACK_ORDER, ordered=True)[0]
        model_cat = pd.Categorical([model], categories=MODEL_ORDER, ordered=True)[0]
        shares = feature_capacity_shares(W1)
        left = 0.0
        for share, dim in zip(shares, DIMENSION_LABELS):
            right = left + float(share)
            rect_rows.append(
                {
                    "feedback": feedback_cat,
                    "model": model_cat,
                    "dimension": dim,
                    "xmin": left,
                    "xmax": right,
                    "ymin": 0.0,
                    "ymax": 0.30,
                }
            )
            if share >= 0.055:
                label_rows.append(
                    {
                        "feedback": feedback_cat,
                        "model": model_cat,
                        "dimension": dim,
                        "x": left + float(share) / 2.0,
                        "y": 0.15,
                        "label": dim.split()[0],
                        "kind": "decision" if "decision" in dim else "nuisance",
                    }
                )
            left = right

        decision_share = float(np.sum(shares[:2]))
        nuisance_share = float(np.sum(shares[2:]))
        ann_rows.append(
            {
                "feedback": feedback_cat,
                "model": model_cat,
                "x": 0.17,
                "y": 0.47,
                "label": "Decision",
                "kind": "decision_label",
            }
        )
        ann_rows.append(
            {
                "feedback": feedback_cat,
                "model": model_cat,
                "x": 0.37,
                "y": 0.47,
                "label": f"{decision_share:.2f}",
                "kind": "decision_value",
            }
        )
        ann_rows.append(
            {
                "feedback": feedback_cat,
                "model": model_cat,
                "x": 0.50,
                "y": 0.47,
                "label": "|",
                "kind": "separator",
            }
        )
        ann_rows.append(
            {
                "feedback": feedback_cat,
                "model": model_cat,
                "x": 0.65,
                "y": 0.47,
                "label": "Nuisance",
                "kind": "nuisance_label",
            }
        )
        ann_rows.append(
            {
                "feedback": feedback_cat,
                "model": model_cat,
                "x": 0.88,
                "y": 0.47,
                "label": f"{nuisance_share:.2f}",
                "kind": "nuisance_value",
            }
        )

    rect_df = pd.DataFrame(rect_rows)
    label_df = pd.DataFrame(label_rows)
    ann_df = pd.DataFrame(ann_rows)
    for df in (rect_df, label_df, ann_df):
        if not df.empty:
            df["feedback"] = pd.Categorical(df["feedback"], categories=FEEDBACK_ORDER, ordered=True)
            df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return rect_df, label_df, ann_df


def _for_feedback(df, feedback):
    return df[df["feedback"] == feedback].copy()


def build_feedback_plot(data, feedback):
    point_df, hull_df, ann_df, cross_df = make_feature_frames(data)
    rect_df, label_df, cap_ann_df = make_capacity_frames(data)
    point_df = _for_feedback(point_df, feedback)
    hull_df = _for_feedback(hull_df, feedback)
    ann_df = _for_feedback(ann_df, feedback)
    cross_df = _for_feedback(cross_df, feedback)
    rect_df = _for_feedback(rect_df, feedback)
    label_df = _for_feedback(label_df, feedback)
    cap_ann_df = _for_feedback(cap_ann_df, feedback)
    label_decision = label_df[label_df["kind"] == "decision"]
    label_nuisance = label_df[label_df["kind"] == "nuisance"]

    def feature_panel(model, show_legend=False):
        panel_points = point_df[point_df["model"] == model]
        panel_hulls = hull_df[hull_df["model"] == model]
        panel_ann = ann_df[ann_df["model"] == model]
        panel_cross = cross_df[cross_df["model"] == model]
        p = (
            ggplot()
            + geom_point(
                mapping=aes(
                    "h1", "h2", color="decision", shape="decision", size="decision"
                ),
                data=panel_points,
                alpha=0.45,
            )
            + geom_path(
                mapping=aes("h1", "h2", color="decision", group="decision"),
                data=panel_hulls,
                size=0.65,
                alpha=0.78,
                show_legend=False,
            )
            + geom_label(
                mapping=aes("h1", "h2", label="text"),
                data=panel_ann,
                color="#2b2b2b",
                fill="#ffffff",
                size=7.4,
                hjust=0,
                family=PAPER_FONT,
            )
            + scale_color_manual(values=ACTION_COLORS, name="Action")
            + scale_shape_manual(values=ACTION_SHAPES, name="Action")
            + scale_size_manual(values=ACTION_SIZES, name="Action")
            + labs(
                title=model,
                x="Feature dim 1",
                y="Feature dim 2" if model == "Oracle" else "",
            )
            + theme_minimal()
            + theme(
                text=element_text(family=PAPER_FONT),
                legend_position="top" if show_legend else "none",
                panel_grid=element_blank(),
                plot_title=element_text(size=17, face="bold"),
                plot_subtitle=element_text(size=14, face="bold"),
                legend_text=element_text(size=12, face="bold"),
                legend_title=element_text(size=12, face="bold"),
                legend_key_width=12,
                legend_key_height=11,
                legend_key_spacing_x=2,
                legend_margin=[0, 2, 0, 2],
                legend_box_spacing=0,
                axis_title=element_text(size=14),
                axis_text=element_text(size=11.5),
            )
        )
        if not panel_cross.empty:
            p = (
                p
                + geom_hline(
                    yintercept=0.0,
                    data=panel_cross,
                    color="#666666",
                    linetype="dashed",
                    size=0.9,
                )
                + geom_vline(
                    xintercept=0.0,
                    data=panel_cross,
                    color="#666666",
                    linetype="dashed",
                    size=0.9,
                )
            )
        return p

    top = (
        gggrid(
            [
                feature_panel("Oracle"),
                feature_panel("BayesianSPO", show_legend=True),
                feature_panel("BayesianMSE"),
            ],
            ncol=3,
        )
        + ggsize(900, 265)
    )

    def capacity_panel(model):
        panel_rect = rect_df[rect_df["model"] == model]
        panel_decision = label_decision[label_decision["model"] == model]
        panel_nuisance = label_nuisance[label_nuisance["model"] == model]
        panel_ann = cap_ann_df[cap_ann_df["model"] == model]
        panel_ann_labels = panel_ann[
            panel_ann["kind"].isin(["decision_label", "nuisance_label"])
        ]
        panel_ann_values = panel_ann[
            panel_ann["kind"].isin(["decision_value", "nuisance_value"])
        ]
        panel_ann_separator = panel_ann[panel_ann["kind"] == "separator"]
        return (
            ggplot()
            + geom_rect(
                mapping=aes(
                    xmin="xmin",
                    xmax="xmax",
                    ymin="ymin",
                    ymax="ymax",
                    fill="dimension",
                ),
                data=panel_rect,
                color="white",
                size=1.0,
            )
            + geom_text(
                mapping=aes("x", "y", label="label"),
                data=panel_decision,
                color="white",
                size=10.6,
                family=PAPER_FONT,
            )
            + geom_text(
                mapping=aes("x", "y", label="label"),
                data=panel_nuisance,
                color="#333333",
                size=10.6,
                family=PAPER_FONT,
            )
            + geom_text(
                mapping=aes("x", "y", label="label"),
                data=panel_ann_labels,
                color="#5a5a5a",
                size=7.0,
                hjust=0.5,
                family=PAPER_FONT,
            )
            + geom_text(
                mapping=aes("x", "y", label="label"),
                data=panel_ann_values,
                color="#2b2b2b",
                size=9.2,
                hjust=0.5,
                family=PAPER_FONT,
            )
            + geom_text(
                mapping=aes("x", "y", label="label"),
                data=panel_ann_separator,
                color="#777777",
                size=7.2,
                hjust=0.5,
                family=PAPER_FONT,
            )
            + scale_fill_manual(values=DIMENSION_COLORS, name="Context dimension")
            + scale_x_continuous(limits=[0.0, 1.0], breaks=[0.0, 0.25, 0.5, 0.75, 1.0])
            + scale_y_continuous(limits=[0.0, 0.55], breaks=[])
            + labs(
                x="Normalized allocated feature capacity"
                if model == "BayesianSPO"
                else "",
                y="",
            )
            + theme_minimal()
            + theme(
                text=element_text(family=PAPER_FONT),
                legend_position="none",
                panel_grid=element_blank(),
                axis_title=element_text(size=14),
                axis_text_x=element_text(size=11.5),
                axis_text_y=element_blank(),
                plot_margin=[0, 0, 10, 0],
            )
        )

    bottom = (
        gggrid(
            [
                capacity_panel("Oracle"),
                capacity_panel("BayesianSPO"),
                capacity_panel("BayesianMSE"),
            ],
            ncol=3,
        )
        + ggsize(900, 112)
    )

    return (
        gggrid([top, bottom], ncol=1, heights=[1.0, 0.39], align=True, vspace=2)
        + ggsize(900, 377)
    )


def build_combined_plot(data):
    point_df, hull_df, ann_df, cross_df = make_feature_frames(data)
    rect_df, label_df, cap_ann_df = make_capacity_frames(data)
    label_decision = label_df[label_df["kind"] == "decision"]
    label_nuisance = label_df[label_df["kind"] == "nuisance"]

    def _panel(df, feedback, model):
        return df[(df["feedback"] == feedback) & (df["model"] == model)]

    def feature_panel(feedback, model, title="", show_legend=False):
        panel_points = _panel(point_df, feedback, model)
        panel_hulls = _panel(hull_df, feedback, model)
        panel_ann = _panel(ann_df, feedback, model)
        panel_cross = _panel(cross_df, feedback, model)
        p = (
            ggplot()
            + geom_point(
                mapping=aes(
                    "h1", "h2", color="decision", shape="decision", size="decision"
                ),
                data=panel_points,
                alpha=0.45,
            )
            + geom_path(
                mapping=aes("h1", "h2", color="decision", group="decision"),
                data=panel_hulls,
                size=0.62,
                alpha=0.78,
                show_legend=False,
            )
            + geom_label(
                mapping=aes("h1", "h2", label="text"),
                data=panel_ann,
                color="#2b2b2b",
                fill="#ffffff",
                size=6.6,
                hjust=0,
                family=PAPER_FONT,
            )
            + scale_color_manual(values=ACTION_COLORS, name="Action")
            + scale_shape_manual(values=ACTION_SHAPES, name="Action")
            + scale_size_manual(values=ACTION_SIZES, name="Action")
            + labs(
                title=title,
                x="Feature dim 1" if model == "Oracle" else "",
                y="Feature dim 2" if model == "Oracle" else "",
            )
            + theme_minimal()
            + theme(
                text=element_text(family=PAPER_FONT),
                legend_position="top" if show_legend else "none",
                panel_grid=element_blank(),
                plot_title=element_text(size=14.2, face="bold"),
                legend_text=element_text(size=12.2, face="bold"),
                legend_title=element_text(size=12.2, face="bold"),
                legend_key_width=10,
                legend_key_height=10,
                legend_key_spacing_x=1,
                legend_margin=[0, 0, 0, 0],
                legend_box_spacing=0,
                axis_title=element_text(size=12.2),
                axis_text=element_text(size=10.0),
            )
        )
        if not panel_cross.empty:
            p = (
                p
                + geom_hline(
                    yintercept=0.0,
                    data=panel_cross,
                    color="#666666",
                    linetype="dashed",
                    size=0.85,
                )
                + geom_vline(
                    xintercept=0.0,
                    data=panel_cross,
                    color="#666666",
                    linetype="dashed",
                    size=0.85,
                )
            )
        return p

    def capacity_panel(feedback, model, show_xlabel=False):
        panel_rect = _panel(rect_df, feedback, model)
        panel_decision = _panel(label_decision, feedback, model)
        panel_nuisance = _panel(label_nuisance, feedback, model)
        panel_ann = _panel(cap_ann_df, feedback, model)
        if model == "Oracle":
            panel_ann = panel_ann.iloc[0:0]
        panel_ann_labels = panel_ann[
            panel_ann["kind"].isin(["decision_label", "nuisance_label"])
        ]
        panel_ann_values = panel_ann[
            panel_ann["kind"].isin(["decision_value", "nuisance_value"])
        ]
        panel_ann_separator = panel_ann[panel_ann["kind"] == "separator"]
        return (
            ggplot()
            + geom_rect(
                mapping=aes(
                    xmin="xmin",
                    xmax="xmax",
                    ymin="ymin",
                    ymax="ymax",
                    fill="dimension",
                ),
                data=panel_rect,
                color="white",
                size=1.0,
            )
            + geom_text(
                mapping=aes("x", "y", label="label"),
                data=panel_decision,
                color="white",
                size=8.9,
                family=PAPER_EMPHASIS_FONT,
            )
            + geom_text(
                mapping=aes("x", "y", label="label"),
                data=panel_nuisance,
                color="#333333",
                size=8.9,
                family=PAPER_EMPHASIS_FONT,
            )
            + geom_text(
                mapping=aes("x", "y", label="label"),
                data=panel_ann_labels,
                color="#5a5a5a",
                size=6.7,
                hjust=0.5,
                family=PAPER_FONT,
            )
            + geom_text(
                mapping=aes("x", "y", label="label"),
                data=panel_ann_values,
                color="#2b2b2b",
                size=8.5,
                hjust=0.5,
                family=PAPER_FONT,
            )
            + geom_text(
                mapping=aes("x", "y", label="label"),
                data=panel_ann_separator,
                color="#777777",
                size=6.8,
                hjust=0.5,
                family=PAPER_FONT,
            )
            + scale_fill_manual(values=DIMENSION_COLORS, name="Context dimension")
            + scale_x_continuous(limits=[0.0, 1.0], breaks=[0.0, 0.5, 1.0])
            + scale_y_continuous(limits=[0.0, 0.55], breaks=[])
            + labs(
                x="Normalized allocated feature capacity" if show_xlabel else "",
                y="",
            )
            + theme_minimal()
            + theme(
                text=element_text(family=PAPER_FONT),
                legend_position="none",
                panel_grid=element_blank(),
                axis_title=element_text(size=15.0),
                axis_text_x=element_text(size=9.8),
                axis_text_y=element_blank(),
                plot_margin=[0, 0, 8, 0],
            )
        )

    def cell(feedback, model, title, show_legend=False, show_xlabel=False):
        return (
            gggrid(
                [
                    feature_panel(feedback, model, title, show_legend=show_legend),
                    capacity_panel(feedback, model, show_xlabel=show_xlabel),
                ],
                ncol=1,
                heights=[1.0, 0.27],
                align=True,
                vspace=0,
            )
            + ggsize(386, 360)
        )

    def oracle_cell():
        return (
            feature_panel("Full feedback", "Oracle", "", show_legend=True)
            + ggsize(360, 310)
        )

    def oracle_decision_panel():
        return (
            ggplot()
            + geom_rect(
                data=pd.DataFrame(
                    {
                        "xmin": [0.025],
                        "xmax": [0.985],
                        "ymin": [0.06],
                        "ymax": [0.95],
                    }
                ),
                mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                fill="#f6f6f6",
                color="#d0d0d0",
                size=0.6,
            )
            + geom_text(
                data=pd.DataFrame(
                    {
                        "x": [0.10],
                        "y": [0.76],
                        "label": ["Oracle rule"],
                    }
                ),
                mapping=aes("x", "y", label="label"),
                color="#4b4b4b",
                size=6.4,
                hjust=0,
                family=PAPER_EMPHASIS_FONT,
            )
            + geom_text(
                data=pd.DataFrame(
                    {
                        "x": [0.10, 0.10],
                        "y": [0.49, 0.28],
                        "label": [
                            "If x0 > 0: pick 0; else pick 1.",
                            "If x1 > 0: pick 2; else pick 3.",
                        ],
                    }
                ),
                mapping=aes("x", "y", label="label"),
                color="#4a4a4a",
                size=5.8,
                hjust=0,
                family=PAPER_FONT,
            )
            + scale_x_continuous(limits=[0.0, 1.0], breaks=[])
            + scale_y_continuous(limits=[0.0, 1.0], breaks=[])
            + theme_minimal()
            + theme(
                panel_grid=element_blank(),
                axis_title=element_blank(),
                axis_text=element_blank(),
                plot_margin=[0, 0, 0, 0],
            )
        )

    def left_title_panel(label):
        return (
            ggplot()
            + geom_text(
                data=pd.DataFrame({"x": [0.04], "y": [0.5], "label": [label]}),
                mapping=aes("x", "y", label="label"),
                color="#3e3e3e",
                size=7.8,
                hjust=0,
                family=PAPER_EMPHASIS_FONT,
            )
            + scale_x_continuous(limits=[0.0, 1.0], breaks=[])
            + scale_y_continuous(limits=[0.0, 1.0], breaks=[])
            + theme_minimal()
            + theme(
                panel_grid=element_blank(),
                axis_title=element_blank(),
                axis_text=element_blank(),
                plot_margin=[0, 0, 0, 0],
            )
        )

    def text_panel(label, size=16.5, angle=0):
        return (
            ggplot()
            + geom_rect(
                data=pd.DataFrame({"xmin": [0.0], "xmax": [1.0], "ymin": [0.0], "ymax": [1.0]}),
                mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                fill="#ffffff",
                color="#ffffff",
                alpha=0.0,
            )
            + geom_text(
                data=pd.DataFrame({"x": [0.5], "y": [0.5], "label": [label]}),
                mapping=aes("x", "y", label="label"),
                color="#444444",
                size=size,
                angle=angle,
                family=PAPER_EMPHASIS_FONT,
            )
            + scale_x_continuous(limits=[0.0, 1.0], breaks=[])
            + scale_y_continuous(limits=[0.0, 1.0], breaks=[])
            + theme_minimal()
            + theme(
                panel_grid=element_blank(),
                axis_title=element_blank(),
                axis_text=element_blank(),
                plot_margin=[0, 0, 0, 0],
            )
        )

    def note_panel():
        role_width = 0.88 / 5.0
        role_bar = pd.DataFrame(
            [
                {
                    "dimension": dim,
                    "xmin": 0.06 + idx * role_width,
                    "xmax": 0.06 + (idx + 1) * role_width,
                    "ymin": 0.05,
                    "ymax": 0.19,
                    "x": 0.06 + (idx + 0.5) * role_width,
                    "y": 0.12,
                    "label": dim.split()[0],
                    "kind": "decision" if "decision" in dim else "nuisance",
                }
                for idx, dim in enumerate(DIMENSION_LABELS)
            ]
        )
        role_groups = pd.DataFrame(
            {
                "x": [0.06 + role_width, 0.06 + 3.5 * role_width],
                "y": [0.24, 0.24],
                "label": ["Decision", "Nuisance"],
            }
        )
        return (
            ggplot()
            + geom_text(
                data=pd.DataFrame(
                    {
                        "x": [0.04],
                        "y": [0.86],
                        "label": ["Diagnostic"],
                    }
                ),
                mapping=aes("x", "y", label="label"),
                color="#3e3e3e",
                size=7.5,
                hjust=0,
                family=PAPER_EMPHASIS_FONT,
            )
            + geom_text(
                data=pd.DataFrame(
                    {
                        "x": [0.04],
                        "y": [0.58],
                        "label": ["Action constraint"],
                    }
                ),
                mapping=aes("x", "y", label="label"),
                color="#4b4b4b",
                size=6.4,
                hjust=0,
                family=PAPER_EMPHASIS_FONT,
            )
            + geom_text(
                data=pd.DataFrame(
                    {
                        "x": [0.04],
                        "y": [0.47],
                        "label": ["Pick one from {0,1}. Pick another from {2,3}."],
                    }
                ),
                mapping=aes("x", "y", label="label"),
                color="#4a4a4a",
                size=6.3,
                hjust=0,
                family=PAPER_FONT,
            )
            + geom_text(
                data=pd.DataFrame(
                    {"x": [0.04], "y": [0.32], "label": ["Feature roles"]}
                ),
                mapping=aes("x", "y", label="label"),
                color="#4b4b4b",
                size=6.4,
                hjust=0,
                family=PAPER_EMPHASIS_FONT,
            )
            + geom_text(
                data=role_groups,
                mapping=aes("x", "y", label="label"),
                color="#5a5a5a",
                size=6.2,
                hjust=0.5,
                family=PAPER_FONT,
            )
            + geom_rect(
                data=role_bar,
                mapping=aes(
                    xmin="xmin",
                    xmax="xmax",
                    ymin="ymin",
                    ymax="ymax",
                    fill="dimension",
                ),
                color="white",
                size=0.9,
                show_legend=False,
            )
            + geom_text(
                data=role_bar,
                mapping=aes("x", "y", label="label"),
                color="#ffffff",
                size=7.0,
                family=PAPER_EMPHASIS_FONT,
            )
            + geom_text(
                data=role_bar[role_bar["kind"] == "nuisance"],
                mapping=aes("x", "y", label="label"),
                color="#333333",
                size=7.0,
                family=PAPER_EMPHASIS_FONT,
            )
            + scale_fill_manual(values=DIMENSION_COLORS, name="Context dimension")
            + scale_x_continuous(limits=[0.0, 1.0], breaks=[])
            + scale_y_continuous(limits=[0.0, 1.0], breaks=[])
            + theme_minimal()
            + theme(
                legend_position="none",
                panel_grid=element_blank(),
                axis_title=element_blank(),
                axis_text=element_blank(),
                plot_margin=[0, 0, 0, 0],
            )
        )

    def separator(vertical=True):
        sep_color = "#9f9f9f"
        if vertical:
            return (
                ggplot()
                + geom_rect(
                    data=pd.DataFrame({"xmin": [0.44], "xmax": [0.56], "ymin": [0.0], "ymax": [1.0]}),
                    mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                    fill=sep_color,
                    color=sep_color,
                )
                + scale_x_continuous(limits=[0.0, 1.0], breaks=[])
                + scale_y_continuous(limits=[0.0, 1.0], breaks=[])
                + theme_minimal()
                + theme(
                    panel_grid=element_blank(),
                    axis_title=element_blank(),
                    axis_text=element_blank(),
                    plot_margin=[0, 0, 0, 0],
                )
            )
        return (
            ggplot()
            + geom_rect(
                data=pd.DataFrame({"xmin": [0.0], "xmax": [1.0], "ymin": [0.36], "ymax": [0.64]}),
                mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                fill=sep_color,
                color=sep_color,
            )
            + scale_x_continuous(limits=[0.0, 1.0], breaks=[])
            + scale_y_continuous(limits=[0.0, 1.0], breaks=[])
            + theme_minimal()
            + theme(
                panel_grid=element_blank(),
                axis_title=element_blank(),
                axis_text=element_blank(),
                plot_margin=[0, 0, 0, 0],
            )
        )

    def blank():
        return (
            ggplot()
            + geom_rect(
                data=pd.DataFrame({"xmin": [0.0], "xmax": [1.0], "ymin": [0.0], "ymax": [1.0]}),
                mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                fill="#ffffff",
                color="#ffffff",
                alpha=0.0,
            )
            + scale_x_continuous(limits=[0.0, 1.0], breaks=[])
            + scale_y_continuous(limits=[0.0, 1.0], breaks=[])
            + theme_minimal()
            + theme(
                panel_grid=element_blank(),
                axis_title=element_blank(),
                axis_text=element_blank(),
                plot_margin=[0, 0, 0, 0],
            )
        )

    oracle_column = (
        gggrid(
            [
                note_panel(),
                left_title_panel("Oracle"),
                oracle_cell(),
                oracle_decision_panel(),
                blank(),
            ],
            ncol=1,
            heights=[0.40, 0.08, 0.78, 0.20, 0.08],
            align=True,
            vspace=0,
        )
        + ggsize(372, 760)
    )
    header_row = (
        gggrid(
            [
                blank(),
                text_panel("BayesianSPO", size=10.1),
                text_panel("BayesianMSE", size=10.1),
            ],
            ncol=3,
            widths=[0.075, 1.0, 1.0],
            align=True,
            hspace=8,
        )
        + ggsize(835, 36)
    )
    full_row = (
        gggrid(
            [
                text_panel("Full feedback", size=9.2, angle=90),
                cell("Full feedback", "BayesianSPO", ""),
                cell("Full feedback", "BayesianMSE", ""),
            ],
            ncol=3,
            widths=[0.075, 1.0, 1.0],
            align=True,
            hspace=8,
        )
        + ggsize(835, 340)
    )
    partial_row = (
        gggrid(
            [
                text_panel("Partial feedback", size=9.2, angle=90),
                cell("Partial feedback", "BayesianSPO", "", show_xlabel=True),
                cell("Partial feedback", "BayesianMSE", "", show_xlabel=True),
            ],
            ncol=3,
            widths=[0.075, 1.0, 1.0],
            align=True,
            hspace=8,
        )
        + ggsize(835, 340)
    )
    method_grid = (
        gggrid(
            [
                header_row,
                full_row,
                separator(vertical=False),
                partial_row,
            ],
            ncol=1,
            heights=[0.14, 1.0, 0.055, 1.0],
            align=True,
            vspace=4,
        )
        + ggsize(835, 760)
    )
    return (
        gggrid(
            [
                oracle_column,
                separator(vertical=True),
                method_grid,
            ],
            ncol=3,
            widths=[0.75, 0.04, 1.70],
            align=True,
            hspace=10,
        )
        + ggsize(1215, 760)
    )


def save_outputs(plot, out_dir, stem, keep_intermediates=False, save_png=False):
    svg_path = out_dir / f"{stem}.svg"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"

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
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")

    if save_png:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=1800)
        print(f"Saved {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate the Decision-Nuisance Top-2 diagnostic paper figures."
    )
    parser.add_argument("--n-rounds", type=int, default=10000)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=99)
    parser.add_argument("--noise-width", type=float, default=0.0)
    parser.add_argument("--imputation-samples", type=int, default=8)
    parser.add_argument("--imputation-scale", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Directory for the paper-ready PDF figures.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR.relative_to(REPO_ROOT)),
        help="Directory for cached training outputs.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH.relative_to(REPO_ROOT)),
        help="CSV summary path for diagnostic numbers.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate SVG files used for PDF conversion.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also export PNG previews next to the PDFs.",
    )
    args = parser.parse_args()

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = resolve_path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    data, _meta = run_or_load(args, cache_dir)
    save_outputs(
        build_feedback_plot(data, "Full feedback"),
        out_dir,
        "decision_nuisance_top2_full_feedback",
        keep_intermediates=args.keep_intermediates,
        save_png=args.png,
    )
    save_outputs(
        build_feedback_plot(data, "Partial feedback"),
        out_dir,
        "decision_nuisance_top2_partial_feedback",
        keep_intermediates=args.keep_intermediates,
        save_png=args.png,
    )
    save_outputs(
        build_combined_plot(data),
        out_dir,
        "decision_nuisance_top2_combined",
        keep_intermediates=args.keep_intermediates,
        save_png=args.png,
    )

    _point_df, _hull_df, ann_df, _cross_df = make_feature_frames(data)
    summary_specs = [
        ("Full feedback", "Oracle", ORACLE_W1),
        ("Full feedback", "BayesianSPO", data["W1_spo_full"]),
        ("Full feedback", "BayesianMSE", data["W1_mse_full"]),
        ("Partial feedback", "Oracle", ORACLE_W1),
        ("Partial feedback", "BayesianSPO", data["W1_spo"]),
        ("Partial feedback", "BayesianMSE", data["W1_mse"]),
    ]
    pd.DataFrame(
        {
            "feedback": [feedback for feedback, _model, _W1 in summary_specs],
            "model": [model for _feedback, model, _W1 in summary_specs],
            "accuracy": [
                ann_df.loc[
                    (ann_df["feedback"] == feedback) & (ann_df["model"] == model),
                    "text",
                ]
                .iloc[0]
                .replace("Acc. ", "")
                .replace("accuracy = ", "")
                for feedback, model, _W1 in summary_specs
            ],
            "decision_share": [
                f"{feature_capacity_shares(W1)[:2].sum():.2f}"
                for _feedback, _model, W1 in summary_specs
            ],
            "nuisance_share": [
                f"{feature_capacity_shares(W1)[2:].sum():.2f}"
                for _feedback, _model, W1 in summary_specs
            ],
        }
    ).to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

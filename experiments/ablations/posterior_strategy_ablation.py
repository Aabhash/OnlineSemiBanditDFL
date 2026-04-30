#!/usr/bin/env python
"""Shared posterior-strategy ablation runner.

This file owns the overlapping agent construction for the masked-imputation and
exploration-strategy ablations. The presets differ only in which rows from the
same Bayesian last-layer design grid they evaluate.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

from experiments.ablations.common import (  # noqa: E402
    get_quick_params,
    filter_results_for_run,
    load_config,
    load_results,
    make_agent_solver,
    make_dataset_config,
    run_ablation_sweep,
    save_results,
    select_ablation_seeds,
)
from src.agents.contextual import BayesianMSEAgent, BayesianSPOAgent  # noqa: E402


STRATEGY_LABELS = {
    "greedy": "Greedy",
    "ucb": "UCB",
    "ts": "TS",
}

IMPUTATION_VARIANTS = [
    {
        "label": "BayesianSPO-masked",
        "method": "spo",
        "exploration_strategy": "ucb",
        "mask_grad": True,
    },
    {
        "label": "BayesianSPO-unmasked",
        "method": "spo",
        "exploration_strategy": "ucb",
        "mask_grad": False,
    },
    {
        "label": "BayesianMSE-masked",
        "method": "mse",
        "exploration_strategy": "ucb",
        "mask_grad": True,
    },
    {
        "label": "BayesianMSE-unmasked",
        "method": "mse",
        "exploration_strategy": "ucb",
        "mask_grad": False,
    },
]

PRESETS = {
    "imputation": {
        "config_key": "imputation_strategy",
        "title": "Masked-imputation ablation",
        "diagnostics": False,
        "plotter": "plot_imputation",
        "default_hidden_dim_key": "hidden_dim_small",
        "variants": IMPUTATION_VARIANTS,
        "dose_response": True,
    },
    "exploration": {
        "config_key": "exploration_strategy",
        "title": "Exploration-policy ablation",
        "diagnostics": False,
        "plotter": "plot_exploration",
        "default_hidden_dim_key": "hidden_dim",
        "variants": [
            {
                "label": "BayesianSPO-Greedy",
                "method": "spo",
                "exploration_strategy": "greedy",
                "mask_grad": True,
            },
            {
                "label": "BayesianMSE-Greedy",
                "method": "mse",
                "exploration_strategy": "greedy",
            },
            {
                "label": "BayesianSPO-UCB",
                "method": "spo",
                "exploration_strategy": "ucb",
                "mask_grad": True,
            },
            {
                "label": "BayesianMSE-UCB",
                "method": "mse",
                "exploration_strategy": "ucb",
            },
            {
                "label": "BayesianSPO-TS",
                "method": "spo",
                "exploration_strategy": "ts",
                "mask_grad": True,
            },
            {
                "label": "BayesianMSE-TS",
                "method": "mse",
                "exploration_strategy": "ts",
            },
        ],
    },
}


def weight_label(weight):
    return f"{float(weight):g}"


def build_dose_variants(weights, methods=None):
    variants = []
    method_specs = {
        "spo": ("BayesianSPO", "spo"),
        "mse": ("BayesianMSE", "mse"),
    }
    selected_methods = methods or ["spo", "mse"]
    for method_key in selected_methods:
        method_label, method = method_specs[method_key]
        for weight in weights:
            variants.append(
                {
                    "label": f"{method_label}-w{weight_label(weight)}",
                    "method": method,
                    "exploration_strategy": "ucb",
                    "mask_grad": True,
                    "missing_grad_weight": float(weight),
                }
            )
    return variants


def resolve_variants(preset_cfg, abl_cfg, quick=False, methods=None):
    if not preset_cfg.get("dose_response", False):
        variants = preset_cfg["variants"]
        if methods is not None:
            method_set = set(methods)
            variants = [variant for variant in variants if variant["method"] in method_set]
        return variants

    weights = [float(w) for w in abl_cfg.get("missing_grad_weights", [0.0, 1.0])]
    if quick:
        quick_weights = [0.0, 0.25, 1.0]
        weights = [w for w in quick_weights if w in weights] or quick_weights
    return build_dose_variants(weights, methods=methods)


def load_tuned_params(tuning_dir, dataset, agent_name, feedback_type="semi_bandit"):
    path = os.path.join(tuning_dir, f"best_hyperparams_{dataset}_{feedback_type}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    entry = data.get(agent_name)
    if entry is None:
        return None
    if isinstance(entry, dict) and "params" in entry:
        return dict(entry["params"])
    return dict(entry)


def resolve_base_hp(
    defaults,
    tuned_params,
    include_imputation=False,
    default_hidden_dim_key="hidden_dim",
):
    tuned_params = tuned_params or {}
    default_hidden_dim = defaults.get(default_hidden_dim_key, defaults["hidden_dim"])
    hp = {
        "lr": tuned_params.get("init_lr", tuned_params.get("lr", defaults["lr"])),
        "beta": tuned_params.get("beta", defaults["beta"]),
        "hidden_dim": tuned_params.get("hidden_dim", default_hidden_dim),
        "lambda_reg": tuned_params.get("lambda_reg", defaults["lambda_reg"]),
    }
    if include_imputation:
        hp.update(
            {
                "imputation_samples": tuned_params.get("imputation_samples", 1),
                "imputation_scale": tuned_params.get("imputation_scale", 1.0),
                "use_posterior_predictive": tuned_params.get(
                    "use_posterior_predictive", False
                ),
                "noise_var": tuned_params.get("noise_var", 1.0),
            }
        )
    return hp


def make_agents(env, seed, spo_hp, mse_hp, variants):
    minimize = getattr(env, "minimize", False)

    base_common = dict(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        seed=seed,
        minimize=minimize,
    )

    spo_common = dict(
        **base_common,
        hidden_dim=spo_hp["hidden_dim"],
        lambda_reg=spo_hp["lambda_reg"],
        lr=spo_hp["lr"],
        beta=spo_hp["beta"],
        imputation_samples=int(spo_hp.get("imputation_samples", 1)),
        imputation_scale=spo_hp.get("imputation_scale", 1.0),
        use_posterior_predictive=spo_hp.get("use_posterior_predictive", False),
        noise_var=spo_hp.get("noise_var", 1.0),
    )
    mse_common = dict(
        **base_common,
        hidden_dim=mse_hp["hidden_dim"],
        lambda_reg=mse_hp["lambda_reg"],
        lr=mse_hp["lr"],
        beta=mse_hp["beta"],
    )

    agents = {}
    for variant in variants:
        strategy = variant["exploration_strategy"]
        if variant["method"] == "spo":
            agents[variant["label"]] = BayesianSPOAgent(
                **spo_common,
                solver=make_agent_solver(env),
                exploration_strategy=strategy,
                mask_grad=variant.get("mask_grad", True),
                missing_grad_weight=variant.get("missing_grad_weight", 0.0),
            )
        elif variant["method"] == "mse":
            agents[variant["label"]] = BayesianMSEAgent(
                **mse_common,
                solver=make_agent_solver(env),
                exploration_strategy=strategy,
                mask_grad=variant.get("mask_grad", False),
                missing_grad_weight=variant.get("missing_grad_weight", 0.0),
            )
        else:
            raise ValueError(f"Unknown variant method: {variant['method']}")
    return agents


def _plot_results(preset, output_dir, plot_format):
    from experiments import plot_ablation_results

    plotter = getattr(plot_ablation_results, PRESETS[preset]["plotter"])
    plotter(output_dir, output_dir, plot_format)


def main(default_preset=None):
    parser = argparse.ArgumentParser(description="Posterior-strategy ablation")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=default_preset,
        help="Ablation preset to run.",
    )
    parser.add_argument("--config", default="experiments/configs/experiment.yaml")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run (default: preset config datasets)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Save data only, skip plotting")
    parser.add_argument(
        "--use-tuned-params",
        action="store_true",
        help="Use tuned BayesianSPO and BayesianMSE hyperparameters from <tuning_dir>.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing dataset results and recompute from scratch.",
    )
    parser.add_argument(
        "--tuning-dir",
        default=None,
        help="Override tuning dir (default: preset config tuning_dir).",
    )
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument(
        "--diagnostics",
        dest="diagnostics",
        action="store_true",
        default=None,
        help="Force diagnostic trajectory metrics on.",
    )
    diagnostics.add_argument(
        "--no-diagnostics",
        dest="diagnostics",
        action="store_false",
        help="Force diagnostic trajectory metrics off.",
    )
    parser.add_argument(
        "--plot-format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Format used when rendering LetsPlot figures.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["spo", "mse"],
        default=None,
        help="Optional method subset to run for dose-response presets.",
    )
    args = parser.parse_args()

    if args.preset is None:
        parser.error("--preset is required when no default preset is provided")
    preset_cfg = PRESETS[args.preset]

    config = load_config(args.config)
    abl_cfg = config["ablations"][preset_cfg["config_key"]]
    quick = args.mode == "quick"
    if args.mode == "quick":
        abl_cfg = get_quick_params(abl_cfg)

    datasets = args.datasets or abl_cfg.get("datasets", ["knapsack_32"])
    n_rounds = abl_cfg["n_rounds"]
    n_seeds = abl_cfg["n_seeds"]
    seeds = select_ablation_seeds(config, n_seeds, quick=quick)
    output_dir = args.output or abl_cfg["output_dir"]
    tuning_dir = args.tuning_dir or abl_cfg.get("tuning_dir", "results/FINAL/tuning")
    defaults = config["agent_defaults"]
    diagnostics_enabled = (
        preset_cfg["diagnostics"] if args.diagnostics is None else args.diagnostics
    )
    variants = resolve_variants(preset_cfg, abl_cfg, quick=quick, methods=args.methods)

    variant_labels = [variant["label"] for variant in variants]
    print(f"\n=== {preset_cfg['title']} ===")
    print(f"Preset: {args.preset}")
    print(f"Datasets: {datasets}")
    print(f"n_rounds={n_rounds}, n_seeds={n_seeds}")
    print(f"Seeds: {seeds}")
    print(f"Variants: {', '.join(variant_labels)}")
    print(f"Diagnostics: {diagnostics_enabled}")
    print(f"Use tuned params: {args.use_tuned_params} (tuning_dir={tuning_dir})")

    hidden_key = preset_cfg["default_hidden_dim_key"]
    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        dc = make_dataset_config(dataset, config)
        tuned_spo = (
            load_tuned_params(tuning_dir, dataset, "BayesianSPO")
            if args.use_tuned_params
            else None
        )
        tuned_mse = (
            load_tuned_params(tuning_dir, dataset, "BayesianMSE")
            if args.use_tuned_params
            else None
        )
        if args.use_tuned_params and tuned_spo is None:
            print(f"  warning: no tuned BayesianSPO params for {dataset}; using defaults")
        if args.use_tuned_params and tuned_mse is None:
            print(f"  warning: no tuned BayesianMSE params for {dataset}; using defaults")

        spo_hp = resolve_base_hp(
            defaults,
            tuned_spo,
            include_imputation=True,
            default_hidden_dim_key=hidden_key,
        )
        mse_hp = resolve_base_hp(
            defaults,
            tuned_mse,
            include_imputation=False,
            default_hidden_dim_key=hidden_key,
        )
        print(
            "  BayesianSPO hp: "
            f"hidden_dim={spo_hp['hidden_dim']}, lr={spo_hp['lr']}, "
            f"lambda={spo_hp['lambda_reg']}, beta={spo_hp['beta']}, "
            f"M={spo_hp.get('imputation_samples', 1)}"
        )
        print(
            "  BayesianMSE hp: "
            f"hidden_dim={mse_hp['hidden_dim']}, lr={mse_hp['lr']}, "
            f"lambda={mse_hp['lambda_reg']}, beta={mse_hp['beta']}"
        )

        ds_dir = os.path.join(output_dir, dataset)
        result_path = os.path.join(ds_dir, "results.pkl")
        existing_results = {}
        if os.path.exists(result_path) and not args.no_resume:
            loaded_results = load_results(result_path)
            existing_results = filter_results_for_run(
                loaded_results, variant_labels, seeds, n_rounds=n_rounds
            )
            print(
                f"  resuming from {result_path} "
                f"({len(existing_results)}/{len(loaded_results)} matching entries)"
            )

        results = run_ablation_sweep(
            make_agents_fn=lambda env, seed: make_agents(
                env, seed, spo_hp, mse_hp, variants
            ),
            dataset=dataset,
            dataset_config=dc,
            seeds=seeds,
            n_rounds=n_rounds,
            lr=defaults["lr"],
            diagnostics=diagnostics_enabled,
            output_dir=output_dir,
            compact=True,
            keep_trajectory=False,
            initial_results=existing_results,
            save_path=result_path,
        )

        save_results(results, result_path)

    if not args.no_plot:
        _plot_results(args.preset, output_dir, args.plot_format)


if __name__ == "__main__":
    main()

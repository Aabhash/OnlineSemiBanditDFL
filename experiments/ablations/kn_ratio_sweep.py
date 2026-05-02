#!/usr/bin/env python
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, '.')

from experiments.ablations.common import (
    load_config, make_dataset_config, recompute_optimals,
    save_results, get_quick_params, compact_result, load_results,
    select_ablation_seeds, result_matches_rounds,
)
from experiments.ablations.posterior_strategy_ablation import (
    load_tuned_params,
    resolve_base_hp,
)
from src.envs import create_env
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.solvers.knapsack import KnapsackSolver
from src.solvers.portfolio import PortfolioSolver
from src.agents.contextual import (
    BayesianSPOAgent, BayesianMSEAgent, RandomAgent, LinUCBAgent,
)


def make_solver(env, capacity):
    data = getattr(env, 'data', None)
    covariance = getattr(data, 'covariance', None)
    if covariance is not None and hasattr(env, 'gamma') and hasattr(env, 'max_assets'):
        return PortfolioSolver(
            covariance=covariance,
            gamma=env.gamma,
            max_assets=int(round(capacity)),
            factor_loadings=getattr(data, 'L', None),
            idiosyncratic_var=getattr(data, 'idiosyncratic_var', None),
            time_limit=getattr(env, 'solver_time_limit', None),
        )
    return KnapsackSolver(np.asarray(env.weights, dtype=float), float(capacity))


def make_agents(env, seed, config, capacity, spo_hp, mse_hp):
    defaults = config['agent_defaults']

    weights = np.asarray(env.weights, dtype=float)

    base_common = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=weights, capacity=capacity,
        seed=seed,
    )

    agents = {
        'BayesianSPO': BayesianSPOAgent(
            **base_common,
            solver=make_solver(env, capacity).solve,
            hidden_dim=spo_hp['hidden_dim'],
            lambda_reg=spo_hp['lambda_reg'],
            beta=spo_hp['beta'],
            lr=spo_hp['lr'],
            imputation_samples=int(spo_hp.get('imputation_samples', 1)),
            imputation_scale=spo_hp.get('imputation_scale', 1.0),
            use_posterior_predictive=spo_hp.get('use_posterior_predictive', False),
            noise_var=spo_hp.get('noise_var', 1.0),
            mask_grad=True,
            minimize=False,
        ),
        'BayesianMSE': BayesianMSEAgent(
            **base_common,
            solver=make_solver(env, capacity).solve,
            hidden_dim=mse_hp['hidden_dim'],
            lambda_reg=mse_hp['lambda_reg'],
            beta=mse_hp['beta'],
            lr=mse_hp['lr'],
            minimize=False,
        ),
        'LinUCB': LinUCBAgent(
            env.context_dim, env.n_arms, weights, capacity,
            alpha=defaults['beta'],
            lambda_reg=defaults['lambda_reg'],
            minimize=False,
            solver=make_solver(env, capacity).solve,
        ),
        'Random': RandomAgent(
            env.n_arms, weights, capacity, seed=seed,
        ),
    }
    return agents


def summarize_selection(result, env, capacity):
    actions = np.asarray(result.get('actions', []), dtype=float)
    if actions.ndim != 2 or actions.shape[0] == 0:
        counts = np.asarray([], dtype=float)
        selected_weights = np.asarray([], dtype=float)
    else:
        selected = actions > 0.5
        counts = selected.sum(axis=1).astype(float)
        selected_weights = selected @ np.asarray(env.weights, dtype=float)

    optimal_actions = np.asarray(result.get('optimal_actions', []), dtype=float)
    if optimal_actions.ndim == 2 and optimal_actions.shape[0] > 0:
        optimal_counts = (optimal_actions > 0.5).sum(axis=1).astype(float)
    else:
        optimal_counts = np.asarray([], dtype=float)

    def mean_or_nan(values):
        return float(np.mean(values)) if values.size else np.nan

    def std_or_nan(values):
        return float(np.std(values, ddof=1)) if values.size > 1 else np.nan

    return {
        'capacity': float(capacity),
        'n_items': int(env.n_arms),
        'weight_sum': float(np.sum(env.weights)),
        'capacity_weight_fraction': float(capacity / np.sum(env.weights)),
        'selected_count_mean': mean_or_nan(counts),
        'selected_count_std': std_or_nan(counts),
        'selected_count_min': float(np.min(counts)) if counts.size else np.nan,
        'selected_count_max': float(np.max(counts)) if counts.size else np.nan,
        'selected_fraction_mean': mean_or_nan(counts) / float(env.n_arms) if counts.size else np.nan,
        'selected_weight_mean': mean_or_nan(selected_weights),
        'selected_weight_fraction_mean': mean_or_nan(selected_weights) / float(capacity) if selected_weights.size and capacity > 0 else np.nan,
        'optimal_selected_count_mean': mean_or_nan(optimal_counts),
        'optimal_selected_fraction_mean': mean_or_nan(optimal_counts) / float(env.n_arms) if optimal_counts.size else np.nan,
    }


def plot_results(*args, **kwargs):
    print('Skipping inline matplotlib plot. Use experiments/plot_ablation_results.py for paper figures.')


def main():
    parser = argparse.ArgumentParser(description='Capacity-constrained feedback-density sweep')
    parser.add_argument('--config', default='experiments/configs/experiment.yaml')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--no-plot', action='store_true', help='Save data only, skip plotting')
    parser.add_argument(
        '--use-tuned-params',
        action='store_true',
        help='Use tuned BayesianSPO and BayesianMSE hyperparameters.',
    )
    parser.add_argument('--tuning-dir', default=None)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config['ablations']['kn_ratio']
    quick = args.mode == 'quick'

    if quick:
        abl_cfg = get_quick_params(abl_cfg)

    datasets = args.datasets or abl_cfg.get('datasets') or [abl_cfg.get('dataset', 'knapsack_32')]
    if quick and args.datasets is None:
        datasets = datasets[:1]
    capacity_multipliers = [float(x) for x in abl_cfg.get(
        'capacity_multipliers', [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    )]
    portfolio_max_assets = [int(x) for x in abl_cfg.get(
        'portfolio_max_assets', [2, 3, 4, 5, 6, 8, 10]
    )]
    if quick:
        capacity_multipliers = [0.2, 1.0, 1.4]
        portfolio_max_assets = [2, 5, 10]
    n_rounds = abl_cfg['n_rounds']
    n_seeds = abl_cfg['n_seeds']
    seeds = select_ablation_seeds(config, n_seeds, quick=quick)
    output_dir = args.output or abl_cfg['output_dir']
    result_path = os.path.join(output_dir, 'results.pkl')
    tuning_dir = args.tuning_dir or abl_cfg.get('tuning_dir', 'results/FINAL/tuning')
    lr = config['agent_defaults']['lr']

    print(f'\n=== Capacity-constrained feedback-density sweep ===')
    print(f'datasets={datasets}, n_rounds={n_rounds}, seeds={seeds}')
    print(f'capacity_multipliers={capacity_multipliers}')
    print(f'portfolio_max_assets={portfolio_max_assets}')
    print(f'use_tuned_params={args.use_tuned_params} (tuning_dir={tuning_dir})')

    defaults = config['agent_defaults']
    agent_names = list(abl_cfg.get('agents', ['BayesianSPO', 'BayesianMSE']))
    all_results = {}
    if os.path.exists(result_path) and not args.no_resume:
        loaded = load_results(result_path)
        for key, value in loaded.items():
            if (
                isinstance(key, tuple)
                and len(key) == 4
                and key[0] in datasets
                and key[1] in agent_names
                and key[3] in seeds
                and result_matches_rounds(value, n_rounds)
            ):
                all_results[key] = value
        print(f'resuming from {result_path} ({len(all_results)}/{len(loaded)} matching entries)')

    for dataset in datasets:
        base_dc = make_dataset_config(dataset, config)
        if dataset == 'portfolio_32':
            base_capacity = float(base_dc['max_assets'])
            capacities = [float(k) for k in portfolio_max_assets]
            sweep_kind = 'max_assets'
        else:
            base_capacity = float(base_dc['capacity'])
            capacities = [float(base_capacity * multiplier) for multiplier in capacity_multipliers]
            sweep_kind = 'capacity'
        print(
            f'\n  Dataset {dataset}: base_{sweep_kind}={base_capacity}, '
            f'{"max_assets" if sweep_kind == "max_assets" else "capacities"}={capacities}'
        )

        tuned_spo = (
            load_tuned_params(tuning_dir, dataset, 'BayesianSPO')
            if args.use_tuned_params else None
        )
        tuned_mse = (
            load_tuned_params(tuning_dir, dataset, 'BayesianMSE')
            if args.use_tuned_params else None
        )
        spo_hp = resolve_base_hp(
            defaults, tuned_spo, include_imputation=True,
            default_hidden_dim_key='hidden_dim_small',
        )
        mse_hp = resolve_base_hp(
            defaults, tuned_mse, include_imputation=False,
            default_hidden_dim_key='hidden_dim_small',
        )

        for seed_idx, seed in enumerate(seeds):
            print(f'\n    Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):')

            env = create_env(dataset, seed=seed, **base_dc)
            base_episode = pregenerate_episode(env, n_rounds)
            weights = np.asarray(env.weights, dtype=float)

            for cap in capacities:
                cap_key = float(cap)
                if dataset == 'portfolio_32':
                    print(f'      max_assets={int(round(cap_key))}:')
                else:
                    print(
                        f'      capacity={cap_key:.3g} '
                        f'(capacity/sum(weights)={cap_key / np.sum(weights):.3f}):'
                    )

                solver = make_solver(env, cap_key)
                episode = recompute_optimals(base_episode, solver.solve)
                agents = make_agents(env, seed, config, cap_key, spo_hp, mse_hp)

                for agent_name, agent in agents.items():
                    if agent_name not in agent_names:
                        continue
                    key = (dataset, agent_name, cap_key, seed)
                    if key in all_results:
                        print(f'        {agent_name}... skipped', flush=True)
                        continue

                    print(f'        {agent_name}...', end=' ', flush=True)

                    result = run_agent_on_episode(
                        agent, episode, n_rounds,
                        lr=lr, full_feedback=False,
                        diagnostics=False, co_solver=solver,
                    )
                    compact = compact_result(result, keep_trajectory=False)
                    compact.update(summarize_selection(result, env, cap_key))
                    compact['dataset'] = dataset
                    all_results[key] = compact
                    save_results(all_results, result_path)
                    print(
                        f'regret={result["final_regret"]:.0f}, '
                        f'selected={compact["selected_count_mean"]:.1f}/{env.n_arms}',
                        flush=True,
                    )

    save_results(all_results, result_path)
    if not args.no_plot:
        plot_results()


if __name__ == '__main__':
    main()

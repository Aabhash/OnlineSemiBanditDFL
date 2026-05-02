#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, '.')

from experiments.ablations.common import (
    load_config, make_agent_solver, make_co_solver, make_dataset_config,
    save_results, get_quick_params, compact_result, select_ablation_seeds,
)
from experiments.ablations.posterior_strategy_ablation import (
    load_tuned_params,
    resolve_base_hp,
)
from src.agents.contextual import (
    BayesianSPOAgent, BayesianMSEAgent, LinUCBAgent,
)
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.envs import create_env


def resolve_linucb_hp(defaults, tuned_params):
    tuned_params = tuned_params or {}
    return {
        'alpha': tuned_params.get('alpha', defaults['alpha']),
        'lambda_reg': tuned_params.get('lambda_reg', defaults['lambda_reg']),
    }


def make_agents_for_config(env, seed, spo_hp, mse_hp, hidden_dim):
    minimize = getattr(env, 'minimize', False)

    spo_base = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=env.weights, capacity=env.capacity,
        hidden_dim=hidden_dim, lambda_reg=spo_hp['lambda_reg'],
        lr=spo_hp['lr'], solver=make_agent_solver(env),
        seed=seed, nonlinear=False,
    )
    mse_base = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=env.weights, capacity=env.capacity,
        hidden_dim=hidden_dim, lambda_reg=mse_hp['lambda_reg'],
        lr=mse_hp['lr'], solver=make_agent_solver(env),
        seed=seed, nonlinear=False,
    )

    agents = {
        f'BayesianSPO-h{hidden_dim}': BayesianSPOAgent(
            **spo_base, beta=spo_hp['beta'], mask_grad=True, minimize=minimize,
            imputation_samples=int(spo_hp.get('imputation_samples', 1)),
            imputation_scale=spo_hp.get('imputation_scale', 1.0),
            use_posterior_predictive=spo_hp.get('use_posterior_predictive', False),
            noise_var=spo_hp.get('noise_var', 1.0),
        ),
        f'BayesianMSE-h{hidden_dim}': BayesianMSEAgent(
            **mse_base, beta=mse_hp['beta'], minimize=minimize,
        ),
    }
    return agents


def plot_results(all_results, hidden_dims, seeds, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plots')
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for name_prefix, label in [('BayesianSPO', 'BayesianSPO'), ('BayesianMSE', 'BayesianMSE')]:
        means = []
        stds = []
        for hd in hidden_dims:
            name = f'{name_prefix}-h{hd}'
            regrets = [all_results[(name, s)]['final_regret']
                       for s in seeds if (name, s) in all_results]
            means.append(np.mean(regrets) if regrets else np.nan)
            stds.append(np.std(regrets) if regrets else 0)

        ax.errorbar(hidden_dims, means, yerr=stds, marker='o', capsize=3, label=label)

    ax.set_xlabel('Hidden dimension')
    ax.set_ylabel('Final cumulative regret')
    ax.set_xscale('log', base=2)

                                      
    linucb_regrets = [all_results[('LinUCB', s)]['final_regret']
                       for s in seeds if ('LinUCB', s) in all_results]
    if linucb_regrets:
        linucb_mean = np.mean(linucb_regrets)
        ax.axhline(linucb_mean, color='gray', linestyle='--', label='LinUCB')
    ax.legend()

    plt.title('Feature capacity sweep')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'feature_sharing.pdf'), bbox_inches='tight')
    print(f'Plots saved to {output_dir}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Feature Sharing Ablation')
    parser.add_argument('--config', default='experiments/configs/experiment.yaml')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--output', default=None)
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: config datasets)')
    hp_group = parser.add_mutually_exclusive_group()
    hp_group.add_argument('--use-tuned-params', dest='use_tuned_params',
                          action='store_true', default=True,
                          help='Use per-dataset tuned HPs from <tuning_dir> (default).')
    hp_group.add_argument('--use-default-params', dest='use_tuned_params',
                          action='store_false',
                          help='Use agent_defaults instead of tuned HPs.')
    parser.add_argument('--tuning-dir', default=None,
                        help='Override tuning dir (default: ablations.feature_sharing.tuning_dir).')
    parser.add_argument('--no-plot', action='store_true', help='Save data only, skip plotting')
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config['ablations']['feature_sharing']
    quick = args.mode == 'quick'

    if quick:
        abl_cfg = get_quick_params(abl_cfg)
        abl_cfg['hidden_dims'] = [4, 16, 64]

    datasets = args.datasets or abl_cfg.get('datasets', ['knapsack_32'])
    hidden_dims = abl_cfg['hidden_dims']
    n_rounds = abl_cfg['n_rounds']
    n_seeds = abl_cfg['n_seeds']
    seeds = select_ablation_seeds(config, n_seeds, quick=quick)
    output_dir = args.output or abl_cfg['output_dir']
    tuning_dir = args.tuning_dir or abl_cfg.get('tuning_dir', 'results/FINAL/tuning')
    lr = config['agent_defaults']['lr']
    defaults = config['agent_defaults']

    print(f'\n=== Feature Sharing ===')
    print(f'datasets={datasets}')
    print(f'hidden_dims={hidden_dims}')
    print(f'seeds={seeds}')
    print(f'use_tuned_params={args.use_tuned_params} (tuning_dir={tuning_dir})')

    for dataset in datasets:
        print(f'\n=== Feature Sharing: {dataset} ===')
        dc = make_dataset_config(dataset, config)
        tuned_spo = load_tuned_params(tuning_dir, dataset, 'BayesianSPO') if args.use_tuned_params else None
        tuned_mse = load_tuned_params(tuning_dir, dataset, 'BayesianMSE') if args.use_tuned_params else None
        tuned_linucb = load_tuned_params(tuning_dir, dataset, 'LinUCB') if args.use_tuned_params else None
        spo_hp = resolve_base_hp(defaults, tuned_spo, include_imputation=True)
        mse_hp = resolve_base_hp(defaults, tuned_mse, include_imputation=False)
        linucb_hp = resolve_linucb_hp(defaults, tuned_linucb)
        print(
            f'  BayesianSPO hp: lr={spo_hp["lr"]}, lambda={spo_hp["lambda_reg"]}, '
            f'beta={spo_hp["beta"]}, M={spo_hp.get("imputation_samples", 1)}'
        )
        print(
            f'  BayesianMSE hp: lr={mse_hp["lr"]}, lambda={mse_hp["lambda_reg"]}, '
            f'beta={mse_hp["beta"]}'
        )
        print(f'  LinUCB hp: alpha={linucb_hp["alpha"]}, lambda={linucb_hp["lambda_reg"]}')

        all_results = {}

        for seed_idx, seed in enumerate(seeds):
            print(f'  Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):')

            env = create_env(dataset, seed=seed, **dc)
            co_solver = make_co_solver(env)
            episode = pregenerate_episode(env, n_rounds)

                                                  
            linucb = LinUCBAgent(
                env.context_dim, env.n_arms, env.weights, env.capacity,
                alpha=linucb_hp['alpha'],
                lambda_reg=linucb_hp['lambda_reg'],
                minimize=getattr(env, 'minimize', False),
                solver=make_agent_solver(env),
            )
            print(f'    LinUCB...', end=' ', flush=True)
            result = run_agent_on_episode(
                linucb, episode, n_rounds,
                lr=lr, full_feedback=False,
                diagnostics=True, co_solver=co_solver,
            )
            all_results[('LinUCB', seed)] = compact_result(result, keep_trajectory=False)
            print(f'regret={result["final_regret"]:.0f}', flush=True)

            for hd in hidden_dims:
                agents = make_agents_for_config(env, seed, spo_hp, mse_hp, hd)

                for agent_name, agent in agents.items():
                    print(f'    {agent_name}...', end=' ', flush=True)
                    result = run_agent_on_episode(
                        agent, episode, n_rounds,
                        lr=lr, full_feedback=False,
                        diagnostics=True, co_solver=co_solver,
                    )
                    all_results[(agent_name, seed)] = compact_result(result, keep_trajectory=False)
                    print(f'regret={result["final_regret"]:.0f}', flush=True)

        ds_dir = os.path.join(output_dir, dataset)
        save_results(all_results, os.path.join(ds_dir, 'results.pkl'))
        if not args.no_plot:
            plot_results(all_results, hidden_dims, seeds, ds_dir)


if __name__ == '__main__':
    main()

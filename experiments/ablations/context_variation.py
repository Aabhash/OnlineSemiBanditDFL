#!/usr/bin/env python
import argparse
import sys
import os
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
from src.envs import create_env
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.agents.contextual import (
    BayesianSPOAgent, BayesianMSEAgent, LinUCBAgent,
)


def resolve_linucb_hp(defaults, tuned_params):
    tuned_params = tuned_params or {}
    return {
        'alpha': tuned_params.get('alpha', defaults['alpha']),
        'lambda_reg': tuned_params.get('lambda_reg', defaults['lambda_reg']),
    }


def make_agents(env, seed, spo_hp, mse_hp, linucb_hp):
    minimize = getattr(env, 'minimize', False)

    spo_base = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=env.weights, capacity=env.capacity,
        hidden_dim=spo_hp['hidden_dim'], lambda_reg=spo_hp['lambda_reg'],
        lr=spo_hp['lr'], solver=make_agent_solver(env), seed=seed,
    )
    mse_base = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=env.weights, capacity=env.capacity,
        hidden_dim=mse_hp['hidden_dim'], lambda_reg=mse_hp['lambda_reg'],
        lr=mse_hp['lr'], solver=make_agent_solver(env), seed=seed,
    )

    agents = {
        'BayesianSPO': BayesianSPOAgent(
            **spo_base, beta=spo_hp['beta'], mask_grad=True, minimize=minimize,
            imputation_samples=int(spo_hp.get('imputation_samples', 1)),
            imputation_scale=spo_hp.get('imputation_scale', 1.0),
            use_posterior_predictive=spo_hp.get('use_posterior_predictive', False),
            noise_var=spo_hp.get('noise_var', 1.0),
        ),
        'BayesianMSE': BayesianMSEAgent(
            **mse_base, beta=mse_hp['beta'], minimize=minimize,
        ),
        'LinUCB': LinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=linucb_hp['alpha'], lambda_reg=linucb_hp['lambda_reg'],
            minimize=minimize, solver=make_agent_solver(env),
        ),
    }
    return agents


def plot_results(all_results, noise_widths, seeds, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plots')
        return

    agent_names = ['BayesianSPO', 'BayesianMSE', 'LinUCB']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    ax = axes[0]
    for name in agent_names:
        means = []
        stds = []
        for nw in noise_widths:
            regrets = [all_results[(name, nw, s)]['final_regret']
                       for s in seeds if (name, nw, s) in all_results]
            means.append(np.mean(regrets) if regrets else np.nan)
            stds.append(np.std(regrets) if regrets else 0)
        ax.errorbar(noise_widths, means, yerr=stds, marker='o', label=name, capsize=3)

    ax.set_xlabel('Noise Width')
    ax.set_ylabel('Final Cumulative Regret')
    ax.set_title('Context Variation: Regret vs Noise')
    ax.legend()


    ax = axes[1]
    gaps = []
    for nw in noise_widths:
        spo = [all_results[('BayesianSPO', nw, s)]['final_regret']
               for s in seeds if ('BayesianSPO', nw, s) in all_results]
        mse = [all_results[('BayesianMSE', nw, s)]['final_regret']
               for s in seeds if ('BayesianMSE', nw, s) in all_results]
        if spo and mse:
            gaps.append(np.mean(mse) - np.mean(spo))
        else:
            gaps.append(np.nan)

    ax.bar(range(len(noise_widths)), gaps, tick_label=[str(nw) for nw in noise_widths],
           color='purple', alpha=0.7)
    ax.set_xlabel('Noise Width')
    ax.set_ylabel('MSE Regret - SPO Regret')
    ax.set_title('SPO+ Advantage (MSE - SPO)')
    ax.axhline(0, color='gray', linestyle='--')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'context_variation.pdf'), bbox_inches='tight')
    print(f'Plots saved to {output_dir}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Context Variation Ablation')
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
                        help='Override tuning dir (default: ablations.context_variation.tuning_dir).')
    parser.add_argument('--no-plot', action='store_true', help='Save data only, skip plotting')
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config['ablations']['context_variation']
    quick = args.mode == 'quick'

    if quick:
        abl_cfg = get_quick_params(abl_cfg)
        abl_cfg['noise_widths'] = [0.0, 0.25, 0.5, 1.0]

    datasets = args.datasets or abl_cfg.get('datasets', [abl_cfg.get('dataset', 'knapsack_32')])
    noise_widths = abl_cfg['noise_widths']
    n_rounds = abl_cfg['n_rounds']
    n_seeds = abl_cfg['n_seeds']
    seeds = select_ablation_seeds(config, n_seeds, quick=quick)
    output_dir = args.output or abl_cfg['output_dir']
    tuning_dir = args.tuning_dir or abl_cfg.get('tuning_dir', 'results/FINAL/tuning')
    lr = config['agent_defaults']['lr']

    print(f'\n=== Context Variation ===')
    print(f'datasets={datasets}')
    print(f'noise_widths={noise_widths}')
    print(f'seeds={seeds}')
    print(f'use_tuned_params={args.use_tuned_params} (tuning_dir={tuning_dir})')

    for dataset in datasets:
        print(f'\n--- {dataset} ---')
        all_results = {}
        defaults = config['agent_defaults']
        tuned_spo = load_tuned_params(tuning_dir, dataset, 'BayesianSPO') if args.use_tuned_params else None
        tuned_mse = load_tuned_params(tuning_dir, dataset, 'BayesianMSE') if args.use_tuned_params else None
        tuned_linucb = load_tuned_params(tuning_dir, dataset, 'LinUCB') if args.use_tuned_params else None
        spo_hp = resolve_base_hp(defaults, tuned_spo, include_imputation=True)
        mse_hp = resolve_base_hp(defaults, tuned_mse, include_imputation=False)
        linucb_hp = resolve_linucb_hp(defaults, tuned_linucb)
        print(
            f'  BayesianSPO hp: hidden_dim={spo_hp["hidden_dim"]}, lr={spo_hp["lr"]}, '
            f'lambda={spo_hp["lambda_reg"]}, beta={spo_hp["beta"]}, '
            f'M={spo_hp.get("imputation_samples", 1)}'
        )
        print(
            f'  BayesianMSE hp: hidden_dim={mse_hp["hidden_dim"]}, lr={mse_hp["lr"]}, '
            f'lambda={mse_hp["lambda_reg"]}, beta={mse_hp["beta"]}'
        )
        print(f'  LinUCB hp: alpha={linucb_hp["alpha"]}, lambda={linucb_hp["lambda_reg"]}')

        for nw in noise_widths:
            print(f'\n  noise_width={nw}:')

            for seed_idx, seed in enumerate(seeds):
                print(f'    Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):')

                dc = make_dataset_config(dataset, config, noise_width=nw)
                env = create_env(dataset, seed=seed, **dc)
                co_solver = make_co_solver(env)
                episode = pregenerate_episode(env, n_rounds)

                agents = make_agents(env, seed, spo_hp, mse_hp, linucb_hp)

                for agent_name, agent in agents.items():
                    print(f'      {agent_name}...', end=' ', flush=True)
                    result = run_agent_on_episode(
                        agent, episode, n_rounds,
                        lr=lr, full_feedback=False,
                        diagnostics=False, co_solver=co_solver,
                    )
                    all_results[(agent_name, nw, seed)] = compact_result(result, keep_trajectory=False)
                    print(f'regret={result["final_regret"]:.0f}', flush=True)

        ds_dir = os.path.join(output_dir, dataset)
        save_results(all_results, os.path.join(ds_dir, 'results.pkl'))
        if not args.no_plot:
            plot_results(all_results, noise_widths, seeds, ds_dir)


if __name__ == '__main__':
    main()

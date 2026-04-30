import argparse
import importlib
import importlib.util
import json
import os
import sys
import numpy as np

sys.path.insert(0, '.')

plt = importlib.import_module('matplotlib.pyplot') if importlib.util.find_spec('matplotlib.pyplot') is not None else None

from experiments.ablations.common import (
    load_config, make_co_solver, make_dataset_config,
    save_results, get_quick_params, compact_result,
)
from src.agents.contextual import BayesianSPOAgent
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.envs import create_env


def load_tuned_bayesian_spo_params(tuning_dir, dataset, feedback_type='semi_bandit'):
    path = os.path.join(tuning_dir, f'best_hyperparams_{dataset}_{feedback_type}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    entry = data.get('BayesianSPO')
    if entry is None:
        return None
    if isinstance(entry, dict) and 'params' in entry:
        return dict(entry['params'])
    return dict(entry)


def resolve_base_hp(env, defaults, tuned_params):
    if tuned_params:
        hp = {
            'lr': tuned_params.get('init_lr', tuned_params.get('lr', defaults['lr'])),
            'beta': tuned_params.get('beta', defaults['beta']),
            'hidden_dim': tuned_params.get('hidden_dim', defaults['hidden_dim']),
            'lambda_reg': tuned_params.get('lambda_reg', defaults['lambda_reg']),
            'imputation_samples': tuned_params.get('imputation_samples', 1),
            'imputation_scale': tuned_params.get('imputation_scale', 1.0),
            'use_posterior_predictive': tuned_params.get('use_posterior_predictive', False),
            'noise_var': tuned_params.get('noise_var', 1.0),
        }
    else:
        hp = {
            'lr': defaults['lr'],
            'beta': defaults['beta'],
            'hidden_dim': defaults['hidden_dim'],
            'lambda_reg': defaults['lambda_reg'],
            'imputation_samples': 1,
            'imputation_scale': 1.0,
            'use_posterior_predictive': False,
            'noise_var': 1.0,
        }
    return hp


def make_agents(env, seed, base_hp, m_values):
    minimize = getattr(env, 'minimize', False)
    solver = env.solve

    common_base = dict(
        context_dim=env.context_dim, n_arms=env.n_arms,
        weights=env.weights, capacity=env.capacity,
        hidden_dim=base_hp['hidden_dim'], lambda_reg=base_hp['lambda_reg'],
        lr=base_hp['lr'], solver=solver, seed=seed,
        beta=base_hp['beta'], mask_grad=True, minimize=minimize,
        imputation_scale=base_hp.get('imputation_scale', 1.0),
        use_posterior_predictive=base_hp.get('use_posterior_predictive', False),
        noise_var=base_hp.get('noise_var', 1.0),
    )

    return {
        f'M{m}': BayesianSPOAgent(**common_base, imputation_samples=m)
        for m in m_values
    }


def plot_results(all_results, m_values, seeds, output_dir, dataset):
    if plt is None:
        print('matplotlib not available, skipping plots')
        return

    means, stds = [], []
    for m in m_values:
        name = f'M{m}'
        regrets = [all_results[(name, s)]['final_regret']
                   for s in seeds if (name, s) in all_results]
        means.append(np.mean(regrets) if regrets else np.nan)
        stds.append(np.std(regrets) if regrets else 0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(m_values, means, yerr=stds, marker='o', capsize=3,
                color='mediumpurple')
    ax.set_xlabel('Imputation samples $M$')
    ax.set_ylabel('Final cumulative regret')
    ax.set_title(f'BayesianSPO: regret vs.\\ $M$ ({dataset})')
    ax.set_xscale('log', base=2)
    ax.set_xticks(m_values)
    ax.set_xticklabels([str(m) for m in m_values])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'imputation_samples.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'imputation_samples.png'), dpi=150, bbox_inches='tight')
    print(f'Plots saved to {output_dir}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Imputation-samples (M) sweep')
    parser.add_argument('--config', default='experiments/configs/experiment.yaml')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--output', default=None)
    parser.add_argument('--no-plot', action='store_true', help='Save data only, skip plotting')
    parser.add_argument('--use-tuned-params', action='store_true',
                        help='Load per-dataset tuned BayesianSPO HPs from <tuning_dir>/best_hyperparams_*.json '
                             'instead of agent_defaults (M is still swept).')
    parser.add_argument('--tuning-dir', default=None,
                        help='Override tuning dir (default: ablations.imputation_samples.tuning_dir).')
    args = parser.parse_args()

    config = load_config(args.config)
    abl_cfg = config['ablations']['imputation_samples']

    if args.mode == 'quick':
        abl_cfg = get_quick_params(abl_cfg)

    datasets = abl_cfg.get('datasets', ['knapsack_32'])
    m_values = abl_cfg['m_values']
    n_rounds = abl_cfg['n_rounds']
    n_seeds = abl_cfg['n_seeds']
    seeds = list(range(n_seeds))
    output_dir = args.output or abl_cfg['output_dir']
    tuning_dir = args.tuning_dir or abl_cfg.get('tuning_dir', 'results/tuning')
    defaults = config['agent_defaults']

    print(f'M values: {m_values}')
    print(f'Datasets: {datasets}')
    print(f'Use tuned params: {args.use_tuned_params} (tuning_dir={tuning_dir})')

    for dataset in datasets:
        print(f'\n=== Imputation-samples sweep: {dataset} ===')
        dc = make_dataset_config(dataset, config)

        tuned = load_tuned_bayesian_spo_params(tuning_dir, dataset) if args.use_tuned_params else None
        if args.use_tuned_params and tuned is None:
            print(f'  warning: no tuned BayesianSPO params for {dataset}, falling back to agent_defaults')

        all_results = {}
        for seed_idx, seed in enumerate(seeds):
            print(f'  Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):')

            env = create_env(dataset, seed=seed, **dc)
            co_solver = make_co_solver(env)
            episode = pregenerate_episode(env, n_rounds)

            base_hp = resolve_base_hp(env, defaults, tuned)
            agents = make_agents(env, seed, base_hp, m_values)
            for agent_name, agent in agents.items():
                print(f'    {agent_name}...', end=' ', flush=True)
                result = run_agent_on_episode(
                    agent, episode, n_rounds,
                    lr=base_hp['lr'], full_feedback=False,
                    diagnostics=True, co_solver=co_solver,
                )
                all_results[(agent_name, seed)] = compact_result(result, keep_trajectory=False)
                print(f'regret={result["final_regret"]:.0f}', flush=True)

        ds_dir = os.path.join(output_dir, dataset)
        save_results(all_results, os.path.join(ds_dir, 'results.pkl'))
        if not args.no_plot:
            plot_results(all_results, m_values, seeds, ds_dir, dataset)


if __name__ == '__main__':
    main()

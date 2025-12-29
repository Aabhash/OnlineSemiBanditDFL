import argparse
import numpy as np
import json
import os
import yaml
from datetime import datetime
import sys

sys.path.insert(0, '.')

from src.envs.fantasy import FantasyEnv
from src.agents.per_arm_context import (
    CNUCBAgent,
    CNTSAgent,
    PerArmNeuralUCBAgent,
    PerArmBayesianSPOAgent,
    PerArmLinUCBAgent,
    PerArmThompsonSamplingAgent,
)


DEFAULT_CONFIG_PATH = 'experiments/configs/experiment.yaml'


def load_config(config_path: str = DEFAULT_CONFIG_PATH):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_agent(name, env, seed, config, n_rounds):
    defaults = config['agent_defaults']
    hidden_dim = defaults['hidden_dim_small']
    lr = defaults['lr']
    lambda_reg = defaults['lambda_reg']
    gamma = defaults['gamma']
    nu = defaults['nu']
    alpha = defaults['alpha']
    beta = defaults['beta']

    common = dict(
        context_dim=env.context_dim,
        n_arms=env.n_arms,
        weights=env.weights,
        capacity=env.capacity,
        seed=seed,
    )

    if name == 'CN-UCB':
        return CNUCBAgent(
            **common, hidden_dim=hidden_dim, lambda_reg=lambda_reg, gamma=gamma, lr=lr
        )
    elif name == 'CN-TS':
        return CNTSAgent(
            **common, hidden_dim=hidden_dim, lambda_reg=lambda_reg, nu=nu, lr=lr
        )
    elif name == 'PerArmNeuralUCB':
        return PerArmNeuralUCBAgent(
            **common, hidden_dim=hidden_dim, alpha=alpha, lambda_reg=lambda_reg, lr=lr
        )
    elif name == 'PerArmBayesianSPO':
        return PerArmBayesianSPOAgent(
            **common, hidden_dim=hidden_dim, beta=beta, alpha=0.0,
            lambda_reg=lambda_reg, lr=lr, decay_rounds=n_rounds, solver=env.solve
        )
    elif name == 'PerArmLinUCB':
        return PerArmLinUCBAgent(
            context_dim=env.context_dim, n_arms=env.n_arms,
            weights=env.weights, capacity=env.capacity,
            alpha=alpha, lambda_reg=lambda_reg, solver=env.solve
        )
    elif name == 'PerArmTS':
        return PerArmThompsonSamplingAgent(
            context_dim=env.context_dim, n_arms=env.n_arms,
            weights=env.weights, capacity=env.capacity,
            lambda_reg=lambda_reg, sigma=1.0, solver=env.solve, seed=seed
        )
    else:
        raise ValueError(f"Unknown agent: {name}")


def run_single_experiment(n_players, agent_name, n_rounds, seed, config):
    env = FantasyEnv(n_players=n_players, seed=seed)
    agent = get_agent(agent_name, env, seed, config, n_rounds)

    test_selection = env.solve(env.current_values)
    obs_rate = test_selection.sum() / env.n_arms

    regrets = []
    for t in range(n_rounds):
        context = env.current_context
        selection = agent.act(context)
        observed, reward, info = env.step(selection, full_feedback=False)
        regrets.append(info['regret'])
        agent.update(context, selection, observed, full_feedback=False)

    early_regret = np.mean(regrets[:n_rounds//5])
    late_regret = np.mean(regrets[-n_rounds//5:])
    total_regret = sum(regrets)
    improvement = (early_regret - late_regret) / early_regret * 100 if early_regret > 0 else 0

    return {
        'n_players': n_players,
        'obs_rate': obs_rate,
        'agent': agent_name,
        'seed': seed,
        'early_regret': early_regret,
        'late_regret': late_regret,
        'total_regret': total_regret,
        'improvement_pct': improvement,
        'regrets': regrets,
    }


def run_ablation(
    player_counts,
    agents,
    n_rounds,
    n_seeds,
    config,
    verbose=True,
):
    results = []

    for n_players in player_counts:
        if verbose:
            env = FantasyEnv(n_players=n_players, seed=0)
            test_sel = env.solve(env.current_values)
            obs_rate = test_sel.sum() / env.n_arms * 100
            print(f"\nn_players={n_players} (obs_rate={obs_rate:.1f}%)")
            print("-" * 50)

        for agent_name in agents:
            seed_results = []
            for seed in range(n_seeds):
                result = run_single_experiment(
                    n_players, agent_name, n_rounds, seed, config
                )
                seed_results.append(result)
                results.append(result)

            mean_early = np.mean([r['early_regret'] for r in seed_results])
            mean_late = np.mean([r['late_regret'] for r in seed_results])
            mean_impr = np.mean([r['improvement_pct'] for r in seed_results])
            std_late = np.std([r['late_regret'] for r in seed_results])

            if verbose:
                ablation_config = config['ablations']['observation_rate']
                ntk_methods = ablation_config['ntk_methods']
                family = "NTK" if agent_name in ntk_methods else "Bayesian"
                print(f"  [{family}] {agent_name}: "
                      f"early={mean_early:.0f}, late={mean_late:.0f}±{std_late:.0f}, "
                      f"impr={mean_impr:.1f}%")

    return results


def compute_summary_table(results, config):
    from collections import defaultdict

    ablation_config = config['ablations']['observation_rate']
    ntk_methods = ablation_config['ntk_methods']

    grouped = defaultdict(list)
    for r in results:
        key = (r['n_players'], r['agent'])
        grouped[key].append(r)

    summary = []
    for (n_players, agent), runs in grouped.items():
        obs_rate = runs[0]['obs_rate']
        family = "NTK" if agent in ntk_methods else "Bayesian"

        summary.append({
            'n_players': n_players,
            'obs_rate': obs_rate,
            'obs_rate_pct': obs_rate * 100,
            'agent': agent,
            'family': family,
            'early_regret_mean': np.mean([r['early_regret'] for r in runs]),
            'early_regret_std': np.std([r['early_regret'] for r in runs]),
            'late_regret_mean': np.mean([r['late_regret'] for r in runs]),
            'late_regret_std': np.std([r['late_regret'] for r in runs]),
            'improvement_pct_mean': np.mean([r['improvement_pct'] for r in runs]),
            'improvement_pct_std': np.std([r['improvement_pct'] for r in runs]),
            'total_regret_mean': np.mean([r['total_regret'] for r in runs]),
            'total_regret_std': np.std([r['total_regret'] for r in runs]),
        })

    return summary


def print_summary_table(summary):
    print("\n" + "=" * 80)
    print("OBSERVATION RATE ABLATION SUMMARY")
    print("=" * 80)

    obs_rates = sorted(set(s['obs_rate_pct'] for s in summary), reverse=True)

    print(f"\n{'Obs Rate':<10} {'Agent':<20} {'Family':<10} {'Late Regret':<15} {'Improvement':<12}")
    print("-" * 70)

    for obs_rate in obs_rates:
        rows = [s for s in summary if abs(s['obs_rate_pct'] - obs_rate) < 0.5]
        rows = sorted(rows, key=lambda x: x['late_regret_mean'])

        for i, row in enumerate(rows):
            obs_str = f"{row['obs_rate_pct']:.1f}%" if i == 0 else ""
            print(f"{obs_str:<10} {row['agent']:<20} {row['family']:<10} "
                  f"{row['late_regret_mean']:.0f}±{row['late_regret_std']:.0f}{'':>5} "
                  f"{row['improvement_pct_mean']:.1f}%")
        print()


def find_crossover_point(summary):
    obs_rates = sorted(set(s['obs_rate_pct'] for s in summary))

    print("\n" + "=" * 80)
    print("CROSSOVER ANALYSIS: NTK vs Bayesian")
    print("=" * 80)

    for obs_rate in obs_rates:
        rows = [s for s in summary if abs(s['obs_rate_pct'] - obs_rate) < 0.5]

        ntk_best = min([r['late_regret_mean'] for r in rows if r['family'] == 'NTK'], default=float('inf'))
        bayes_best = min([r['late_regret_mean'] for r in rows if r['family'] == 'Bayesian'], default=float('inf'))

        winner = "NTK" if ntk_best < bayes_best else "Bayesian"
        gap = abs(ntk_best - bayes_best)

        print(f"Obs Rate {obs_rate:>5.1f}%: {winner:<10} wins (gap={gap:.0f})")

    print("\nConclusion: NTK methods require >10% observation rate to compete with Bayesian methods.")


def save_results(results, summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    results_clean = [{k: v for k, v in r.items() if k != 'regrets'} for r in results]
    with open(f"{output_dir}/ablation_obs_rate_raw.json", 'w') as f:
        json.dump(results_clean, f, indent=2)

    with open(f"{output_dir}/ablation_obs_rate_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def plot_results(summary, output_dir, config):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib not available")

    ablation_config = config['ablations']['observation_rate']
    ntk_methods = ablation_config['ntk_methods']
    bayesian_methods = ablation_config['bayesian_methods']
    all_methods = ntk_methods + bayesian_methods

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for agent in all_methods:
        rows = sorted([s for s in summary if s['agent'] == agent], key=lambda x: x['obs_rate_pct'])
        if not rows:
            continue

        obs_rates = [r['obs_rate_pct'] for r in rows]
        late_regrets = [r['late_regret_mean'] for r in rows]

        style = '-o' if agent in ntk_methods else '--s'
        ax.plot(obs_rates, late_regrets, style, label=agent, markersize=6)

    ax.set_xlabel('Observation Rate (%)')
    ax.set_ylabel('Late Regret (per round)')
    ax.set_title('Method Performance vs Observation Rate')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    obs_rates = sorted(set(s['obs_rate_pct'] for s in summary))

    ntk_best = []
    bayes_best = []
    for obs_rate in obs_rates:
        rows = [s for s in summary if abs(s['obs_rate_pct'] - obs_rate) < 0.5]
        ntk_best.append(min([r['late_regret_mean'] for r in rows if r['family'] == 'NTK'], default=float('nan')))
        bayes_best.append(min([r['late_regret_mean'] for r in rows if r['family'] == 'Bayesian'], default=float('nan')))

    ax.plot(obs_rates, ntk_best, '-o', label='Best NTK Method', linewidth=2, markersize=8)
    ax.plot(obs_rates, bayes_best, '--s', label='Best Bayesian Method', linewidth=2, markersize=8)

    ax.axvline(x=10, color='gray', linestyle=':', alpha=0.7, label='~10% threshold')
    ax.fill_betweenx([0, max(ntk_best + bayes_best) * 1.1], 0, 10, alpha=0.1, color='red', label='Sparse regime')

    ax.set_xlabel('Observation Rate (%)')
    ax.set_ylabel('Late Regret (per round)')
    ax.set_title('NTK vs Bayesian: Best Method in Each Family')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_obs_rate.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_dir}/ablation_obs_rate.pdf", bbox_inches='tight')
    print(f"Plots saved to {output_dir}/ablation_obs_rate.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))

    for agent in all_methods:
        rows = sorted([s for s in summary if s['agent'] == agent], key=lambda x: x['obs_rate_pct'])
        if not rows:
            continue

        obs_rates = [r['obs_rate_pct'] for r in rows]
        improvements = [r['improvement_pct_mean'] for r in rows]

        style = '-o' if agent in ntk_methods else '--s'
        ax.plot(obs_rates, improvements, style, label=agent, markersize=6)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Observation Rate (%)')
    ax.set_ylabel('Improvement (Early to Late) %')
    ax.set_title('Learning Progress vs Observation Rate')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_obs_rate_improvement.png", dpi=150, bbox_inches='tight')
    print(f"Plots saved to {output_dir}/ablation_obs_rate_improvement.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--n_rounds', type=int, default=None)
    parser.add_argument('--n_seeds', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--quick', action='store_true')

    args = parser.parse_args()

    config = load_config(args.config)
    ablation_config = config['ablations']['observation_rate']

    if args.full:
        n_rounds = args.n_rounds or ablation_config['n_rounds_full']
        n_seeds = args.n_seeds or ablation_config['n_seeds_full']
        player_counts = ablation_config['player_counts_full']
    elif args.quick:
        n_rounds = args.n_rounds or ablation_config['n_rounds_quick']
        n_seeds = args.n_seeds or ablation_config['n_seeds_quick']
        player_counts = ablation_config['player_counts_quick']
    else:
        n_rounds = args.n_rounds or ablation_config['n_rounds']
        n_seeds = args.n_seeds or ablation_config['n_seeds']
        player_counts = ablation_config['player_counts']

    output_dir = args.output_dir or ablation_config['output_dir']

    agents = ablation_config['ntk_methods'] + ablation_config['bayesian_methods']

    print("=" * 80)
    print("OBSERVATION RATE ABLATION STUDY")
    print("=" * 80)
    print(f"Player counts: {player_counts}")
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = run_ablation(
        player_counts=player_counts,
        agents=agents,
        n_rounds=n_rounds,
        n_seeds=n_seeds,
        config=config,
        verbose=True,
    )

    summary = compute_summary_table(results, config)

    print_summary_table(summary)
    find_crossover_point(summary)

    save_results(results, summary, output_dir)

    plot_results(summary, output_dir, config)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

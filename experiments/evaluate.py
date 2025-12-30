import argparse
import numpy as np
import sys
import os
import yaml
import time
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.insert(0, '.')

from src.envs import create_env
from src.agents import run_experiment

from src.agents.shared_context import (
    BayesianSPOAgent,
    LinUCBAgent,
    NeuralUCBAgent,
    ThompsonSamplingAgent,
    UCBAgent,
    EpsilonGreedyAgent,
    RandomAgent,
)
from src.agents.per_arm_context import (
    PerArmBayesianSPOAgent,
    CNUCBAgent,
    CNTSAgent,
    PerArmLinUCBAgent,
    PerArmThompsonSamplingAgent,
    PerArmMSENeuralAgent,
    PerArmEpsilonGreedyAgent,
    PerArmRandomAgent,
)
from src.models import BayesianLinearModel


DEFAULT_CONFIG_PATH = 'experiments/configs/experiment.yaml'
DEFAULT_TUNING_DIR = 'results/tuning'


def load_config(config_path: str = DEFAULT_CONFIG_PATH):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tuned_params(
    tuning_dir: str,
    dataset: str,
    feedback_type: str,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    from experiments.tuning.results_store import ResultsStore
    from experiments.tuning.objectives import uses_holdout_for_selection

    results_store = ResultsStore(tuning_dir)
    use_holdout = uses_holdout_for_selection(config, dataset)

    best_params_file = results_store.output_dir / f"best_hyperparams_{dataset}_{feedback_type}.json"
    if best_params_file.exists():
        import json
        with open(best_params_file) as f:
            data = json.load(f)
        return {agent: info['params'] for agent, info in data.items()}

    tuned_params = {}
    trials = results_store.load_all_trials(dataset=dataset, feedback_type=feedback_type)

    agent_names = set(t.agent_name for t in trials)

    for agent_name in agent_names:
        try:
            params = results_store.get_best_params(
                agent_name, dataset, feedback_type, use_holdout=use_holdout
            )
            tuned_params[agent_name] = params
        except ValueError:
            pass

    return tuned_params


def create_agent_with_tuned_params(
    agent_name: str,
    env,
    tuned_params: Dict[str, Any],
    seed: int,
    n_rounds: int,
    config: Dict[str, Any],
):
    from experiments.tuning.objectives import create_agent_from_params

    lr = tuned_params.get('lr', config['agent_defaults']['lr'])

    return create_agent_from_params(
        agent_name=agent_name,
        env=env,
        params=tuned_params,
        seed=seed,
        n_rounds=n_rounds,
        lr=lr,
    )


def get_agents(env, seed, config, n_rounds):
    defaults = config['agent_defaults']
    lr = defaults['lr']
    hidden_dim = defaults['hidden_dim'] if env.context_dim > 20 else defaults['hidden_dim_small']
    lambda_reg = defaults['lambda_reg']
    alpha = defaults['alpha']
    beta = defaults['beta']
    epsilon = defaults['epsilon']
    decay_epsilon = defaults['decay_epsilon']

    solver = env.solve
    minimize = getattr(env, 'minimize', False)

    return {
        'Random': RandomAgent(
            env.n_arms, env.weights, env.capacity, seed=seed
        ),
        'EpsilonGreedy': EpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, epsilon=epsilon, decay_epsilon=decay_epsilon,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'UCB-Bayesian': UCBAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, beta=beta, minimize=minimize, solver=solver
        ),
        'TS': ThompsonSamplingAgent(
            BayesianLinearModel(env.context_dim, env.n_arms),
            env.weights, env.capacity, minimize=minimize, solver=solver
        ),
        'LinUCB': LinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=alpha, lambda_reg=lambda_reg, minimize=minimize, solver=solver
        ),
        'NeuralUCB': NeuralUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, alpha=alpha, lambda_reg=lambda_reg, lr=lr,
            minimize=minimize, solver=solver, seed=seed
        ),
        'BayesianSPO': BayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=beta, decay_rounds=n_rounds,
            alpha=0.0, lambda_reg=lambda_reg, solver=solver, seed=seed
        ),
    }


def get_per_arm_agents(env, seed, config, n_rounds):
    defaults = config['agent_defaults']
    lr = defaults['lr']
    hidden_dim = defaults['hidden_dim'] if env.context_dim > 20 else defaults['hidden_dim_small']
    lambda_reg = defaults['lambda_reg']
    alpha = defaults['alpha']
    beta = defaults['beta']
    gamma = defaults['gamma']
    nu = defaults['nu']
    epsilon = defaults['epsilon']
    decay_epsilon = defaults['decay_epsilon']

    solver = env.solve

    return {
        'PerArmRandom': PerArmRandomAgent(
            env.n_arms, env.weights, env.capacity, seed=seed
        ),
        'PerArmGreedy': PerArmEpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, epsilon=0.0, decay_epsilon=False,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'PerArmLinUCB': PerArmLinUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            alpha=alpha, lambda_reg=lambda_reg, solver=solver
        ),
        'PerArmTS': PerArmThompsonSamplingAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            lambda_reg=lambda_reg, sigma=1.0, solver=solver, seed=seed
        ),
        'CN-UCB': CNUCBAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, lambda_reg=lambda_reg, gamma=gamma, lr=lr,
            per_arm_context=True, solver=solver, seed=seed
        ),
        'CN-TS': CNTSAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, lambda_reg=lambda_reg, nu=nu, lr=lr,
            per_arm_context=True, solver=solver, seed=seed
        ),
        'PerArmBayesianSPO': PerArmBayesianSPOAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=beta, alpha=0.0, lambda_reg=lambda_reg,
            lr=lr, decay_rounds=n_rounds, solver=solver, seed=seed
        ),
        'PerArmMSENeural': PerArmMSENeuralAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, beta=beta, lambda_reg=lambda_reg,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
        'PerArmEpsilonGreedy': PerArmEpsilonGreedyAgent(
            env.context_dim, env.n_arms, env.weights, env.capacity,
            hidden_dim=hidden_dim, epsilon=epsilon, decay_epsilon=decay_epsilon,
            decay_rounds=n_rounds, lr=lr, solver=solver, seed=seed
        ),
    }


def run_evaluation(
    dataset: str,
    n_rounds: int,
    n_seeds: int,
    config: dict,
    output_dir: str,
    noise: float = 0.0,
    verbose: bool = True,
    tuned_params: Optional[Dict[str, Dict[str, Any]]] = None,
    feedback_types: Optional[list] = None,
    agents_filter: Optional[list] = None,
    wandb_enabled: bool = False,
    use_tuned_params_flag: bool = False,
):
    dataset_config = config['datasets'][dataset].copy()
    dataset_config.pop('env_type', None)

    if dataset == 'synthetic' and noise != 0.0:
        dataset_config['noise_width'] = noise

    if feedback_types is None:
        feedback_types = ['semi_bandit', 'full']

    results = {ft: {} for ft in feedback_types}
    trajectories = {}

    if verbose:
        print(f'\n{"=" * 60}')
        print(f'DATASET: {dataset.upper()}')
        print(f'{"=" * 60}')
        noise_info = f', Noise: {dataset_config.get("noise_width", 0.0)}' if dataset == 'synthetic' else ''
        tuned_info = ' (using tuned HPs)' if tuned_params else ' (using default HPs)'
        print(f'Rounds: {n_rounds}, Seeds: {n_seeds}{noise_info}{tuned_info}')

    feedback_map = {'semi_bandit': False, 'full': True}

    for feedback_mode in feedback_types:
        full_feedback = feedback_map[feedback_mode]
        if verbose:
            print(f'\n--- {feedback_mode.replace("_", " ").title()} Feedback ---')

        for seed in range(n_seeds):
            if verbose:
                print(f'Seed {seed + 1}/{n_seeds}:', flush=True)

            env_base = create_env(dataset, seed=seed, **dataset_config)
            test_set = env_base.generate_test_set(100)

            # Create agents - use tuned params if available, else defaults
            if tuned_params:
                # Only create agents that have tuned params
                agents = {}
                for agent_name, params in tuned_params.items():
                    if agents_filter and agent_name not in agents_filter:
                        continue
                    try:
                        agent = create_agent_with_tuned_params(
                            agent_name, env_base, params, seed, n_rounds, config
                        )
                        agents[agent_name] = agent
                    except Exception as e:
                        if verbose:
                            print(f'  Warning: Could not create {agent_name}: {e}')
            else:
                # Use default hyperparameters
                if env_base.per_arm_context:
                    agents = get_per_arm_agents(env_base, seed, config, n_rounds)
                else:
                    agents = get_agents(env_base, seed, config, n_rounds)

                if agents_filter:
                    agents = {k: v for k, v in agents.items() if k in agents_filter}

            for agent_name, agent in agents.items():
                if verbose:
                    print(f'  {agent_name}...', end=' ', flush=True)

                env = create_env(dataset, seed=seed, **dataset_config)

                # Get LR and params from tuned params or default
                if tuned_params and agent_name in tuned_params:
                    lr = tuned_params[agent_name].get('lr', config['agent_defaults']['lr'])
                    agent_params = tuned_params[agent_name]
                else:
                    lr = config['agent_defaults']['lr']
                    agent_params = {}

                # Time the experiment
                start_time = time.perf_counter()

                result = run_experiment(
                    env, agent, n_rounds,
                    lr=lr, full_feedback=full_feedback,
                    test_set=test_set
                )

                wall_time = time.perf_counter() - start_time
                result['wall_time_seconds'] = wall_time
                result['hyperparameters'] = agent_params

                if agent_name not in results[feedback_mode]:
                    results[feedback_mode][agent_name] = []
                results[feedback_mode][agent_name].append(result['final_regret'])

                trajectories[(agent_name, dataset, feedback_mode, seed)] = {
                    'cumulative_regret': result['cumulative_regret'],
                    'regrets': result['regrets'],
                    'rewards': result['rewards'],
                    'optimal_rewards': result['optimal_rewards'],
                    'holdout_evals': result.get('holdout_evals', []),
                    'mse_losses': result.get('mse_losses', []),
                    'spo_losses': result.get('spo_losses', []),
                    'wall_time_seconds': result.get('wall_time_seconds', 0.0),
                    'hyperparameters': result.get('hyperparameters', {}),
                }

                # Log to wandb if enabled
                if wandb_enabled:
                    agent_params = tuned_params.get(agent_name, {}) if tuned_params else {}
                    wandb_logger = EvaluationWandbLogger(
                        config=config,
                        agent_name=agent_name,
                        dataset=dataset,
                        feedback_type=feedback_mode,
                        seed=seed,
                        use_tuned_params=use_tuned_params_flag,
                        n_rounds=n_rounds,
                        params=agent_params,
                    )
                    wandb_logger.start()
                    wandb_logger.log_trajectory(result)

                    holdout_regret = None
                    holdout_evals = result.get('holdout_evals', [])
                    if holdout_evals:
                        holdout_regret = holdout_evals[-1][1].get('regret')

                    wandb_logger.log_final_summary(
                        result['final_regret'],
                        holdout_regret,
                        wall_time_seconds=wall_time,
                    )
                    wandb_logger.finish()

                if verbose:
                    holdout_evals = result.get('holdout_evals', [])
                    time_str = f'{wall_time:.1f}s'
                    if holdout_evals:
                        final_holdout = holdout_evals[-1][1]
                        holdout_regret = final_holdout['regret']
                        print(f'regret={result["final_regret"]:.0f}, holdout={holdout_regret:.0f}, time={time_str}', flush=True)
                    else:
                        print(f'regret={result["final_regret"]:.0f}, time={time_str}', flush=True)

    return results, trajectories


def print_summary(results, dataset, trajectories=None):
    print(f'\n{"=" * 70}')
    print(f'RESULTS SUMMARY - {dataset.upper()}')
    print(f'{"=" * 70}')

    has_holdout = False
    if trajectories:
        for key, data in trajectories.items():
            if key[1] == dataset and data.get('holdout_evals'):
                has_holdout = True
                break

    for feedback_mode in ['semi_bandit', 'full']:
        print(f'\n{feedback_mode.replace("_", " ").title()} Feedback:')

        if has_holdout and feedback_mode == 'semi_bandit':
            print(f'{"Agent":<25} {"Regret":>10} {"Std":>8} {"Holdout":>10} {"Rank":>6}')
            print('-' * 65)
        else:
            print(f'{"Agent":<25} {"Regret":>10} {"Std":>8} {"Rank":>6}')
            print('-' * 55)

        agents = results[feedback_mode]
        sorted_agents = sorted(agents.items(), key=lambda x: np.mean(x[1]))

        holdout_data = {}
        if trajectories and has_holdout and feedback_mode == 'semi_bandit':
            for key, data in trajectories.items():
                agent_name, ds, mode, seed = key
                if ds == dataset and mode == feedback_mode:
                    holdout_evals = data.get('holdout_evals', [])
                    if holdout_evals:
                        final_regret = holdout_evals[-1][1]['regret']
                        if agent_name not in holdout_data:
                            holdout_data[agent_name] = []
                        holdout_data[agent_name].append(final_regret)

        for rank, (agent_name, regrets) in enumerate(sorted_agents, 1):
            mean = np.mean(regrets)
            std = np.std(regrets)
            marker = ' **' if rank == 1 else ''

            if has_holdout and feedback_mode == 'semi_bandit' and agent_name in holdout_data:
                holdout_regret = np.mean(holdout_data[agent_name])
                print(f'{agent_name:<25} {mean:>10.0f} {std:>8.0f} {holdout_regret:>10.0f} {rank:>5}{marker}')
            else:
                print(f'{agent_name:<25} {mean:>10.0f} {std:>8.0f} {rank:>5}{marker}')

    semi_bandit_best = min(results['semi_bandit'].items(), key=lambda x: np.mean(x[1]))
    full_best = min(results['full'].items(), key=lambda x: np.mean(x[1]))

    print(f'\n{"=" * 70}')
    print('WINNERS:')
    print(f'  Semi-bandit: {semi_bandit_best[0]} ({np.mean(semi_bandit_best[1]):.0f})')
    print(f'  Full feedback: {full_best[0]} ({np.mean(full_best[1]):.0f})')


def plot_trajectories(trajectories, dataset, output_dir, feedback_mode='semi_bandit'):
    from experiments.plot_results import plot_trajectories as _plot_trajectories
    _plot_trajectories(trajectories, dataset, output_dir, feedback_mode)


def save_results(results, trajectories, output_dir):
    import json
    import pickle

    os.makedirs(f'{output_dir}/metrics', exist_ok=True)

    results_json = {}
    for key, value in results.items():
        if isinstance(value, dict) and any(isinstance(v, dict) for v in value.values()):
            results_json[key] = {}
            for mode, agents in value.items():
                results_json[key][mode] = {
                    agent: [float(r) for r in regrets]
                    for agent, regrets in agents.items()
                }
        else:
            results_json[key] = {
                agent: [float(r) for r in regrets]
                for agent, regrets in value.items()
            }

    with open(f'{output_dir}/metrics/results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    with open(f'{output_dir}/metrics/trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

    # Compute and save results summary (mean/std, timing, hyperparameters per agent)
    results_summary = {}
    holdout_summary = {}

    for key, data in trajectories.items():
        agent_name, dataset, feedback_mode, seed = key
        run_key = f"{agent_name}_{dataset}_{feedback_mode}"

        # Initialize if needed
        if run_key not in results_summary:
            results_summary[run_key] = {
                'agent': agent_name,
                'dataset': dataset,
                'feedback_type': feedback_mode,
                'final_regrets': [],
                'wall_times': [],
            }
            holdout_summary[run_key] = []

        # Collect final regret
        cumulative_regret = data.get('cumulative_regret', [])
        if cumulative_regret:
            results_summary[run_key]['final_regrets'].append(cumulative_regret[-1])

        # Collect wall time
        results_summary[run_key]['wall_times'].append(data.get('wall_time_seconds', 0.0))

        # Collect holdout regret if available
        holdout_evals = data.get('holdout_evals', [])
        if holdout_evals:
            final_holdout = holdout_evals[-1][1]
            holdout_summary[run_key].append(final_holdout.get('regret', 0.0))

    # Compute statistics
    for run_key, summary in results_summary.items():
        regrets = summary['final_regrets']
        times = summary['wall_times']

        # Rename to cumulative_regret for clarity
        summary['cumulative_regrets'] = regrets
        del summary['final_regrets']

        summary['mean_cumulative_regret'] = float(np.mean(regrets)) if regrets else 0.0
        summary['std_cumulative_regret'] = float(np.std(regrets)) if regrets else 0.0
        summary['min_cumulative_regret'] = float(np.min(regrets)) if regrets else 0.0
        summary['max_cumulative_regret'] = float(np.max(regrets)) if regrets else 0.0
        summary['n_seeds'] = len(regrets)

        summary['wall_times_seconds'] = [float(t) for t in times]
        del summary['wall_times']

        summary['mean_wall_time_seconds'] = float(np.mean(times)) if times else 0.0
        summary['total_wall_time_seconds'] = float(np.sum(times)) if times else 0.0

        # Add holdout stats if available
        if holdout_summary.get(run_key):
            holdout_regrets = holdout_summary[run_key]
            summary['mean_cumulative_holdout_regret'] = float(np.mean(holdout_regrets))
            summary['std_cumulative_holdout_regret'] = float(np.std(holdout_regrets))

        # Add hyperparameters if available (from first seed)
        if summary.get('hyperparameters'):
            pass  # Already set from timing_summary logic
        else:
            # Find from trajectories
            for tkey, tdata in trajectories.items():
                if f"{tkey[0]}_{tkey[1]}_{tkey[2]}" == run_key:
                    summary['hyperparameters'] = tdata.get('hyperparameters', {})
                    break

    with open(f'{output_dir}/metrics/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Compute overall timing
    total_time = sum(data.get('wall_time_seconds', 0.0) for data in trajectories.values())
    print(f'\nSaved results to {output_dir}/metrics/')
    print(f'Total evaluation time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)')


def run_evaluation_analysis(
    results: Dict[str, Dict[str, list]],
    trajectories: Dict,
    output_dir: str,
    dataset: str,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
):
    from scipy import stats
    import json

    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)

    for feedback_type, agents_data in results.items():
        if not agents_data:
            continue

        print(f"\n--- {feedback_type.replace('_', ' ').title()} Feedback ---")

        # 1. Bootstrap confidence intervals
        print("\n1. Bootstrap Confidence Intervals:")
        print(f"{'Agent':<25} {'Mean':>10} {'95% CI':>20} {'N':>5}")
        print("-" * 65)

        ci_results = {}
        for agent_name, regrets in sorted(agents_data.items(), key=lambda x: np.mean(x[1])):
            regrets_arr = np.array(regrets)
            n = len(regrets_arr)
            mean = np.mean(regrets_arr)

            # Bootstrap CI
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = rng.choice(regrets_arr, size=n, replace=True)
                bootstrap_means.append(np.mean(sample))

            alpha = 1 - confidence
            ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

            ci_results[agent_name] = {
                'mean': float(mean),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'std': float(np.std(regrets_arr)),
                'n_seeds': n,
            }

            print(f"{agent_name:<25} {mean:>10.1f} [{ci_lower:>8.1f}, {ci_upper:>8.1f}] {n:>5}")

        # 2. Pairwise significance tests (Wilcoxon signed-rank)
        print("\n2. Pairwise Significance Tests (Wilcoxon signed-rank, p < 0.05):")

        agents = list(agents_data.keys())
        significance_results = {}

        if len(agents) >= 2:
            # Find best agent
            best_agent = min(agents, key=lambda a: np.mean(agents_data[a]))
            best_regrets = np.array(agents_data[best_agent])

            print(f"Comparing all agents to best: {best_agent}")
            print(f"{'Agent':<25} {'p-value':>10} {'Significant':>12}")
            print("-" * 50)

            for agent_name in sorted(agents, key=lambda a: np.mean(agents_data[a])):
                if agent_name == best_agent:
                    print(f"{agent_name:<25} {'--':>10} {'(baseline)':>12}")
                    continue

                agent_regrets = np.array(agents_data[agent_name])

                # Ensure same length (use min)
                n_compare = min(len(best_regrets), len(agent_regrets))
                if n_compare < 2:
                    continue

                try:
                    stat, p_value = stats.wilcoxon(
                        agent_regrets[:n_compare],
                        best_regrets[:n_compare],
                        alternative='greater'  # Test if agent is worse than best
                    )
                    is_sig = bool(p_value < 0.05)
                    significance_results[agent_name] = {
                        'p_value': float(p_value),
                        'significant': is_sig,
                        'vs': best_agent,
                    }
                    sig_str = "Yes *" if is_sig else "No"
                    print(f"{agent_name:<25} {p_value:>10.4f} {sig_str:>12}")
                except Exception as e:
                    print(f"{agent_name:<25} {'error':>10} {str(e)[:12]:>12}")

        # Save analysis results
        analysis_output = {
            'dataset': dataset,
            'feedback_type': feedback_type,
            'confidence_intervals': ci_results,
            'significance_tests': significance_results,
            'n_bootstrap': n_bootstrap,
            'confidence': confidence,
        }

        output_file = os.path.join(analysis_dir, f'analysis_{dataset}_{feedback_type}.json')
        with open(output_file, 'w') as f:
            json.dump(analysis_output, f, indent=2)

        print(f"\nSaved analysis to {output_file}")

    # 3. Generate comparison bar plot
    _generate_comparison_plot(results, dataset, analysis_dir)


class EvaluationWandbLogger:
    """Logger for a single evaluation run (one agent, one seed)."""

    def __init__(
        self,
        config: Dict[str, Any],
        agent_name: str,
        dataset: str,
        feedback_type: str,
        seed: int,
        use_tuned_params: bool,
        n_rounds: int,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.agent_name = agent_name
        self.dataset = dataset
        self.feedback_type = feedback_type
        self.seed = seed
        self.use_tuned_params = use_tuned_params
        self.n_rounds = n_rounds
        self.params = params or {}

        self._wandb = None
        self._run = None
        self._enabled = config.get('wandb', {}).get('enabled', False)

        if self._enabled:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                self._enabled = False

    def start(self):
        if not self._enabled:
            return

        wandb_config = self.config.get('wandb', {})
        eval_config = wandb_config.get('evaluation', {})

        project = eval_config.get('project', wandb_config.get('project', 'constrained-bandits-eval'))
        base_tags = eval_config.get('tags', wandb_config.get('tags', ['evaluation']))

        run_name = f"{self.dataset}_{self.feedback_type}_{self.agent_name}_seed{self.seed}"
        tags = base_tags + [self.agent_name, self.dataset, self.feedback_type, f'seed_{self.seed}']
        if self.use_tuned_params:
            tags.append('tuned-params')

        group = f"{self.dataset}_{self.feedback_type}_{self.agent_name}"

        run_config = {
            'agent': self.agent_name,
            'dataset': self.dataset,
            'feedback_type': self.feedback_type,
            'seed': self.seed,
            'n_rounds': self.n_rounds,
            'use_tuned_params': self.use_tuned_params,
            **{f'hp/{k}': v for k, v in self.params.items()},
        }

        self._run = self._wandb.init(
            project=project,
            entity=wandb_config.get('entity'),
            name=run_name,
            group=group,
            job_type='evaluation',
            config=run_config,
            tags=tags,
            reinit=True,
        )

    def log_trajectory(self, result: Dict[str, Any]):
        """Log full trajectory after experiment completes."""
        if not self._enabled or not self._run:
            return

        # Log per-round metrics
        cumulative_regret = result.get('cumulative_regret', [])
        regrets = result.get('regrets', [])
        rewards = result.get('rewards', [])
        optimal_rewards = result.get('optimal_rewards', [])

        log_interval = self.config.get('wandb', {}).get('log_interval', 10)

        for i in range(len(cumulative_regret)):
            if i % log_interval == 0 or i == len(cumulative_regret) - 1:
                metrics = {
                    'round': i,
                    'train/cumulative_regret': cumulative_regret[i],
                }
                if i < len(regrets):
                    metrics['train/regret'] = regrets[i]
                if i < len(rewards):
                    metrics['train/reward'] = rewards[i]
                if i < len(optimal_rewards):
                    metrics['train/optimal_reward'] = optimal_rewards[i]

                self._wandb.log(metrics, step=i)

        # Log holdout metrics if available
        holdout_evals = result.get('holdout_evals', [])
        for round_num, holdout_data in holdout_evals:
            self._wandb.log({
                'holdout/regret': holdout_data.get('regret', 0),
                'holdout/cumulative_regret': holdout_data.get('cumulative_regret', holdout_data.get('regret', 0)),
            }, step=round_num)

    def log_final_summary(
        self,
        final_regret: float,
        holdout_regret: Optional[float] = None,
        wall_time_seconds: Optional[float] = None,
    ):
        """Log final summary metrics."""
        if not self._enabled or not self._run:
            return

        self._wandb.summary['final_regret'] = final_regret
        if holdout_regret is not None:
            self._wandb.summary['holdout_regret'] = holdout_regret
        if wall_time_seconds is not None:
            self._wandb.summary['wall_time_seconds'] = wall_time_seconds

    def finish(self):
        if self._run:
            self._run.finish()
            self._run = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


class EvaluationSummaryLogger:
    """Logger for evaluation summary across all agents and seeds."""

    def __init__(self, config: Dict[str, Any], dataset: str, feedback_type: str, use_tuned_params: bool):
        self.config = config
        self.dataset = dataset
        self.feedback_type = feedback_type
        self.use_tuned_params = use_tuned_params
        self._wandb = None
        self._run = None
        self._enabled = config.get('wandb', {}).get('enabled', False)

        if self._enabled:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                self._enabled = False

    def log_summary(self, results: Dict[str, list]):
        """Log summary table comparing all agents."""
        if not self._enabled:
            return

        wandb_config = self.config.get('wandb', {})
        eval_config = wandb_config.get('evaluation', {})

        project = eval_config.get('project', wandb_config.get('project', 'constrained-bandits-eval'))
        base_tags = eval_config.get('tags', wandb_config.get('tags', ['evaluation']))

        run_name = f"{self.dataset}_{self.feedback_type}_summary"
        tags = base_tags + [self.dataset, self.feedback_type, 'summary']
        if self.use_tuned_params:
            tags.append('tuned-params')

        self._run = self._wandb.init(
            project=project,
            entity=wandb_config.get('entity'),
            name=run_name,
            group=f"{self.dataset}_{self.feedback_type}",
            job_type='summary',
            config={
                'dataset': self.dataset,
                'feedback_type': self.feedback_type,
                'use_tuned_params': self.use_tuned_params,
                'type': 'summary',
            },
            tags=tags,
            reinit=True,
        )

        # Log summary statistics for each agent
        for agent_name, regrets in results.items():
            mean_regret = np.mean(regrets)
            std_regret = np.std(regrets)
            self._wandb.summary[f'{agent_name}/mean_regret'] = mean_regret
            self._wandb.summary[f'{agent_name}/std_regret'] = std_regret
            self._wandb.summary[f'{agent_name}/n_seeds'] = len(regrets)

        # Log comparison table
        table_data = [[agent, np.mean(r), np.std(r), len(r)]
                      for agent, r in sorted(results.items(), key=lambda x: np.mean(x[1]))]
        table = self._wandb.Table(
            columns=['Agent', 'Mean Regret', 'Std Regret', 'N Seeds'],
            data=table_data
        )
        self._wandb.log({'results_table': table})

        # Log best agent
        best_agent = min(results.items(), key=lambda x: np.mean(x[1]))
        self._wandb.summary['best_agent'] = best_agent[0]
        self._wandb.summary['best_mean_regret'] = np.mean(best_agent[1])

        self._run.finish()
        self._run = None


def _generate_comparison_plot(
    results: Dict[str, Dict[str, list]],
    dataset: str,
    output_dir: str,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping comparison plot")
        return

    for feedback_type, agents_data in results.items():
        if not agents_data:
            continue

        # Sort by mean regret
        sorted_agents = sorted(agents_data.items(), key=lambda x: np.mean(x[1]))

        agents = [a[0] for a in sorted_agents]
        means = [np.mean(a[1]) for a in sorted_agents]
        stds = [np.std(a[1]) for a in sorted_agents]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(agents))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)

        # Highlight best agent
        bars[0].set_color('forestgreen')

        ax.set_xlabel('Agent')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title(f'{dataset} - {feedback_type.replace("_", " ").title()} Feedback\n(Mean ± Std across seeds)')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right')

        plt.tight_layout()

        plot_file = os.path.join(output_dir, f'comparison_{dataset}_{feedback_type}.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()

        print(f"Saved comparison plot to {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Evaluate bandit agents on various datasets.',
        epilog="""
Two-Stage Workflow Example:
  # Stage 1: Coarse HP search (3 seeds, 1k rounds, 30 trials)
  python experiments/tune_hyperparams.py --all --dataset fantasy --n-seeds 3 --n-rounds 1000 --n-trials 30

  # Stage 2: Full evaluation with tuned HPs (10 seeds, full rounds)
  python experiments/evaluate.py --datasets fantasy --use-tuned-params --full --n_seeds 10

Examples:
  python experiments/evaluate.py --datasets synthetic --quick
  python experiments/evaluate.py --datasets fantasy --full
  python experiments/evaluate.py --datasets fantasy --use-tuned-params --full
  python experiments/evaluate.py --datasets fantasy --use-tuned-params --agents PerArmBayesianSPO,CN-UCB
        """
    )
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--quick', action='store_true', help='Quick evaluation (fewer rounds/seeds)')
    parser.add_argument('--full', action='store_true', help='Full evaluation (more rounds/seeds)')
    parser.add_argument('--n_rounds', type=int, default=None, help='Number of rounds')
    parser.add_argument('--n_seeds', type=int, default=None, help='Number of seeds')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no_save', action='store_true', help='Do not save results')
    parser.add_argument('--datasets', type=str, default='synthetic', help='Comma-separated dataset names')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise width for synthetic dataset')
    parser.add_argument('--feedback', type=str, default='both',
                        choices=['semi_bandit', 'full', 'both'],
                        help='Feedback type to evaluate')
    parser.add_argument('--agents', type=str, default=None,
                        help='Comma-separated agent names to evaluate (default: all)')

    # Tuned hyperparameters options
    parser.add_argument('--use-tuned-params', action='store_true',
                        help='Use tuned hyperparameters from HP search instead of defaults')
    parser.add_argument('--tuning-dir', type=str, default=DEFAULT_TUNING_DIR,
                        help='Directory containing tuning results (default: results/tuning)')

    # Analysis options
    parser.add_argument('--analyze', action='store_true',
                        help='Run statistical analysis (bootstrap CIs, significance tests)')

    # Wandb options
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging for evaluation')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging (overrides config)')

    args = parser.parse_args()

    config = load_config(args.config)

    datasets = [d.strip() for d in args.datasets.split(',')]
    for d in datasets:
        if d not in config['datasets']:
            raise ValueError(f"Unknown dataset: {d}. Available: {list(config['datasets'].keys())}")

    eval_config = config['evaluation']
    if args.quick:
        default_n_rounds = args.n_rounds or eval_config['quick']['n_rounds']
        n_seeds = args.n_seeds or eval_config['quick']['n_seeds']
    elif args.full:
        default_n_rounds = args.n_rounds or eval_config['full']['n_rounds']
        n_seeds = args.n_seeds or eval_config['full']['n_seeds']
    else:
        default_n_rounds = args.n_rounds or eval_config['quick']['n_rounds']
        n_seeds = args.n_seeds or eval_config['quick']['n_seeds']

    output_dir = args.output_dir or eval_config['output_dir']

    # Determine feedback types
    if args.feedback == 'both':
        feedback_types = ['semi_bandit', 'full']
    else:
        feedback_types = [args.feedback]

    # Parse agents filter
    agents_filter = None
    if args.agents:
        agents_filter = [a.strip() for a in args.agents.split(',')]

    print('=' * 70)
    print('CONSTRAINED BANDITS EVALUATION')
    print('=' * 70)
    print(f'Mode: {"quick" if args.quick else "full" if args.full else "custom"}')
    print(f'Datasets: {datasets}')
    print(f'Feedback: {args.feedback}')
    print(f'Default rounds: {default_n_rounds} (may be overridden by dataset config)')
    print(f'Seeds: {n_seeds}')
    print(f'Noise: {args.noise}')
    print(f'Tuned params: {args.use_tuned_params} (from {args.tuning_dir})' if args.use_tuned_params else 'Tuned params: No (using defaults)')
    if agents_filter:
        print(f'Agents: {agents_filter}')
    # Handle wandb configuration
    if args.no_wandb:
        config['wandb'] = {'enabled': False}
    elif args.wandb:
        if 'wandb' not in config:
            config['wandb'] = {}
        config['wandb']['enabled'] = True

    wandb_enabled = config.get('wandb', {}).get('enabled', False) and not args.no_wandb
    print(f'Wandb: {"enabled" if wandb_enabled else "disabled"}')
    print(f'Output: {output_dir}')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    all_results = {}
    all_trajectories = {}

    for dataset in datasets:
        # Determine n_rounds for this dataset (dataset config overrides default)
        dataset_config = config['datasets'][dataset]
        if args.n_rounds:
            # Explicit CLI argument always wins
            n_rounds = args.n_rounds
        elif 'n_rounds' in dataset_config:
            # Dataset-specific n_rounds
            n_rounds = dataset_config['n_rounds']
        else:
            # Fall back to evaluation config default
            n_rounds = default_n_rounds

        print(f'\n[{dataset}] Using n_rounds={n_rounds}')

        # Load tuned parameters if requested
        tuned_params = None
        if args.use_tuned_params:
            for ft in feedback_types:
                try:
                    tuned_params = load_tuned_params(
                        args.tuning_dir, dataset, ft, config
                    )
                    if tuned_params:
                        print(f'Loaded tuned params for {len(tuned_params)} agents on {dataset} ({ft})')
                        for agent_name, params in tuned_params.items():
                            print(f'  {agent_name}: {params}')
                        break
                except Exception as e:
                    print(f'Warning: Could not load tuned params for {dataset} ({ft}): {e}')

            if not tuned_params:
                print(f'Warning: No tuned params found for {dataset}, using defaults')

        results, trajectories = run_evaluation(
            dataset=dataset,
            n_rounds=n_rounds,
            n_seeds=n_seeds,
            config=config,
            output_dir=output_dir,
            noise=args.noise,
            tuned_params=tuned_params,
            feedback_types=feedback_types,
            agents_filter=agents_filter,
            wandb_enabled=wandb_enabled,
            use_tuned_params_flag=args.use_tuned_params,
        )

        print_summary(results, dataset, trajectories)

        all_results[dataset] = results
        all_trajectories.update(trajectories)

        # Log summary to wandb if enabled
        if wandb_enabled:
            for ft in feedback_types:
                if ft in results and results[ft]:
                    summary_logger = EvaluationSummaryLogger(
                        config=config,
                        dataset=dataset,
                        feedback_type=ft,
                        use_tuned_params=args.use_tuned_params,
                    )
                    summary_logger.log_summary(results[ft])
                    print(f'Logged {dataset}/{ft} summary to wandb')

    if not args.no_save:
        save_results(all_results, all_trajectories, output_dir)

        for dataset in datasets:
            for ft in feedback_types:
                plot_trajectories(all_trajectories, dataset, output_dir, ft)

    # Run statistical analysis if requested
    if args.analyze:
        for dataset in datasets:
            if dataset in all_results:
                run_evaluation_analysis(
                    results=all_results[dataset],
                    trajectories=all_trajectories,
                    output_dir=output_dir,
                    dataset=dataset,
                )

    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

from experiments.tuning import BanditHyperparameterTuner
from experiments.tuning.objectives import (
    load_search_spaces,
    get_n_rounds_for_dataset,
    uses_holdout_for_selection,
    has_tunable_params,
    canonicalize_agent_name,
    preferred_agent_name,
)
from experiments.tuning.optuna_tuner import load_config
from experiments.tuning.results_store import ResultsStore
from experiments.analysis.bootstrap import compute_all_confidence_intervals, print_ci_table
from experiments.analysis.significance import generate_significance_report
from experiments.analysis.sensitivity import generate_sensitivity_report


DEFAULT_CONFIG = 'experiments/configs/experiment.yaml'
DEFAULT_SEARCH_SPACE_DIR = 'experiments/configs/search_spaces_online.yaml'
DEFAULT_OUTPUT_DIR = 'results/tuning'


def get_agents_for_dataset(
    search_spaces: dict,
    config: dict,
    dataset: str,
    tunable_only: bool = False,
) -> list:
    agents = list(search_spaces.keys())

    if tunable_only:
        agents = [a for a in agents if has_tunable_params(search_spaces[a])]

    return agents


def get_skipped_agents_for_dataset(search_spaces: dict, config: dict, dataset: str) -> list:
    all_agents = get_agents_for_dataset(search_spaces, config, dataset, tunable_only=False)
    tunable_agents = get_agents_for_dataset(search_spaces, config, dataset, tunable_only=True)
    return [a for a in all_agents if a not in tunable_agents]


def run_analysis(
    output_dir: str,
    agents: list,
    dataset: str,
    search_spaces: dict,
    config: dict,
    feedback_type: str,
):
    print(f"\n{'='*70}")
    print("RUNNING TUNING ANALYSIS")
    print(f"{'='*70}")

    results_store = ResultsStore(output_dir)
    analysis_dir = Path(output_dir) / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    bootstrap_config = config['analysis']['bootstrap']
    significance_config = config['analysis']['significance']
    use_holdout = uses_holdout_for_selection(config, dataset)

    print(f"Selection metric: {'holdout cumulative regret' if use_holdout else 'tuning/train cumulative regret'}")

    print("\n1. Computing bootstrap confidence intervals...")
    all_cis = compute_all_confidence_intervals(
        results_store, agents, dataset, feedback_type,
        n_bootstrap=bootstrap_config['n_bootstrap'],
        confidence=bootstrap_config['confidence'],
        seed=bootstrap_config['seed'],
        output_path=str(analysis_dir / f'bootstrap_ci_{dataset}.json'),
        use_holdout=use_holdout
    )
    print_ci_table(all_cis)

    print("\n2. Running significance tests...")
    report = generate_significance_report(
        results_store, agents, dataset, feedback_type,
        alpha=significance_config['alpha'],
        output_dir=str(analysis_dir),
        use_holdout=use_holdout
    )
    print(report)

    print("\n3. Generating sensitivity report...")
    generate_sensitivity_report(
        results_store, agents, search_spaces, dataset,
        str(analysis_dir / f'sensitivity_report_{dataset}.json'),
        feedback_type
    )

    print(f"\nAnalysis complete. Results saved to {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/tune_hyperparams.py --agent BayesianSPO --dataset knapsack_32
  python experiments/tune_hyperparams.py --all --dataset knapsack_32
  python experiments/tune_hyperparams.py --agent BayesianSPO --dataset knapsack_32 --quick
  python experiments/tune_hyperparams.py --all --dataset knapsack_32 --analyze
        """
    )

    parser.add_argument('--agent', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('--search-spaces', type=str, default=DEFAULT_SEARCH_SPACE_DIR)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--n-trials', type=int, default=None)
    parser.add_argument('--n-seeds', type=int, default=None)
    parser.add_argument('--n-rounds', type=int, default=None)
    parser.add_argument('--feedback', type=str, default='semi_bandit', choices=['semi_bandit', 'full'])
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--parallel', type=int, default=None)
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', action='store_true')

    args = parser.parse_args()

    if args.agent:
        canonical_agent = canonicalize_agent_name(args.agent)
        if canonical_agent != args.agent:
            print(f"Using alias {args.agent} -> {preferred_agent_name(canonical_agent)}")
            args.agent = canonical_agent

    if not args.all and not args.agent and not args.analyze_only:
        parser.error("Specify --agent <name> or --all")

    config = load_config(args.config)
    search_spaces = load_search_spaces(args.search_spaces)



    dataset_config = config['datasets'].get(args.dataset, {})
    if dataset_config.get('skip_tuning', False):
        print(f"Dataset '{args.dataset}' has skip_tuning=true. "
              f"Use default HPs and run evaluation directly.")
        sys.exit(0)


    effective_n_rounds = get_n_rounds_for_dataset(config, args.dataset)

    if args.quick:
        config['tuning']['n_trials'] = 10
        config['tuning_seeds'] = [0, 1, 2]

        if 'n_rounds' in config['datasets'].get(args.dataset, {}):
            config['datasets'][args.dataset]['n_rounds'] = 100
        else:
            config['tuning']['n_rounds'] = 100
        effective_n_rounds = 100
        print("Quick mode baseline applied")

    if args.n_trials:
        config['tuning']['n_trials'] = args.n_trials
    if args.n_seeds:
        config['tuning_seeds'] = list(range(args.n_seeds))
    if args.n_rounds:
        if 'n_rounds' in config['datasets'].get(args.dataset, {}):
            config['datasets'][args.dataset]['n_rounds'] = args.n_rounds
        else:
            config['tuning']['n_rounds'] = args.n_rounds
        effective_n_rounds = args.n_rounds

    if args.parallel:
        config['compute']['max_parallel_trials'] = args.parallel


    skipped_agents = get_skipped_agents_for_dataset(search_spaces, config, args.dataset)

    if args.all:
        agents = get_agents_for_dataset(search_spaces, config, args.dataset, tunable_only=True)
    elif args.agent:
        if args.agent not in search_spaces:
            print(f"Error: Unknown agent '{args.agent}'")
            print(f"Available agents: {list(search_spaces.keys())}")
            sys.exit(1)

        compatible_agents = get_agents_for_dataset(search_spaces, config, args.dataset)
        if args.agent not in compatible_agents:
            print(f"Error: Agent '{args.agent}' is not compatible with dataset '{args.dataset}'")
            print(f"Compatible agents: {compatible_agents}")
            sys.exit(1)

        if args.agent in skipped_agents:
            print(f"Warning: Agent '{args.agent}' has no tunable hyperparameters. Skipping HP tuning.")
            print("This agent will still be included in final evaluation with default params.")
            agents = []
        else:
            agents = [args.agent]
    else:
        agents = get_agents_for_dataset(search_spaces, config, args.dataset, tunable_only=True)


    print("=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Feedback: {args.feedback}")
    print(f"Agents to tune: {agents}")
    if skipped_agents:
        print(f"Skipped (no tunable HPs): {skipped_agents}")
    print(f"Trials: {config['tuning']['n_trials']}")
    print(f"Tuning seeds: {config.get('tuning_seeds', [0, 1, 2, 3, 4])}")
    print(f"Max rounds: {effective_n_rounds}")
    print(f"Output: {args.output}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not args.analyze_only:
        tuner = BanditHyperparameterTuner(
            config_path=args.config,
            search_space_dir=args.search_spaces,
            output_dir=args.output,
        )

        tuner.config = config

        if args.all or len(agents) > 1:
            tuner.tune_all_agents(
                dataset=args.dataset,
                agents=agents,
                feedback_type=args.feedback,
            )
        else:
            tuner.tune_agent(
                agent_name=agents[0],
                dataset=args.dataset,
                feedback_type=args.feedback,
                resume=not args.no_resume,
            )

    if args.analyze or args.analyze_only:
        run_analysis(
            output_dir=args.output,
            agents=agents,
            dataset=args.dataset,
            search_spaces=search_spaces,
            config=config,
            feedback_type=args.feedback,
        )

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

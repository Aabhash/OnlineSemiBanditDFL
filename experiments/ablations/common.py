"""Shared infrastructure for ablation experiments."""
import os
import sys
import pickle
import copy
import yaml
import numpy as np
from typing import Callable, Dict, List, Any, Optional

sys.path.insert(0, '.')

from src.envs import create_env
from src.agents.base import pregenerate_episode, run_agent_on_episode
from src.solvers.knapsack import KnapsackSolver
from src.solvers.shortest_path import ShortestPathSolver
from src.solvers.energy import EnergySolver


def load_config(path: str = 'experiments/configs/experiment.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_co_solver(env):
    """Resolve the combinatorial solver from environment attributes."""
    grid_size = getattr(env, 'grid_size', None)
    min_periods = getattr(env, 'min_periods', None)

    if grid_size is not None:
        return ShortestPathSolver(grid_size)
    elif min_periods is not None:
        _esolver = getattr(env, 'energy_solver', None)
        _max_periods = _esolver.max_periods if _esolver else None
        _max_consec = _esolver.max_consecutive if _esolver else None
        _min_consec = _esolver.min_consecutive if _esolver else None
        return EnergySolver(
            min_periods=min_periods,
            max_periods=_max_periods,
            max_consecutive=_max_consec,
            min_consecutive=_min_consec,
            maximize=not getattr(env, 'minimize', True),
        )
    else:
        return KnapsackSolver(env.weights, env.capacity)


def run_ablation_sweep(
    make_agents_fn: Callable,
    dataset: str,
    dataset_config: dict,
    seeds: List[int],
    n_rounds: int,
    lr: float = 0.001,
    feedback: bool = False,
    diagnostics: bool = True,
    output_dir: str = 'results/ablations',
    verbose: bool = True,
) -> Dict[str, Any]:
    """Core ablation loop: for each seed, create env, pregenerate episode,
    run all agents on same episode, collect results."""
    all_results = {}

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f'  Seed {seed_idx + 1}/{len(seeds)} (seed={seed}):', flush=True)

        env = create_env(dataset, seed=seed, **dataset_config)
        co_solver = make_co_solver(env)
        episode = pregenerate_episode(env, n_rounds, full_feedback=feedback)

        agents = make_agents_fn(env, seed)

        for agent_name, agent in agents.items():
            if verbose:
                print(f'    {agent_name}...', end=' ', flush=True)

            result = run_agent_on_episode(
                agent, episode, n_rounds,
                lr=lr, full_feedback=feedback,
                diagnostics=diagnostics, co_solver=co_solver,
            )

            key = (agent_name, seed)
            all_results[key] = result

            if verbose:
                print(f'regret={result["final_regret"]:.0f}', flush=True)

    return all_results


def recompute_optimals(episode: dict, solver_fn: Callable) -> dict:
    """Copy episode and recompute optimal_action/value per round using a different solver."""
    new_episode = copy.deepcopy(episode)
    for rd in new_episode['rounds_data']:
        true_vals = rd['true_values']
        optimal = solver_fn(true_vals)
        optimal = np.asarray(optimal)
        rd['optimal_action'] = optimal.copy()
        rd['optimal_value'] = float(np.sum(true_vals * optimal))
    return new_episode


def save_results(results: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f'Results saved to {path}')


def load_results(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_quick_params(ablation_config: dict) -> dict:
    """Override config with quick-mode params for fast testing."""
    cfg = ablation_config.copy()
    cfg['n_rounds'] = min(cfg.get('n_rounds', 2500), 500)
    cfg['n_seeds'] = min(cfg.get('n_seeds', 5), 2)
    return cfg


def make_dataset_config(dataset: str, config: dict, **overrides) -> dict:
    """Extract dataset config and apply overrides."""
    dc = config['datasets'][dataset].copy()
    dc.pop('env_type', None)
    dc.pop('skip_tuning', None)
    dc.pop('tuning_year', None)
    dc.update(overrides)
    return dc

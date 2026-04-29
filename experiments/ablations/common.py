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


def load_config(path: str = 'experiments/configs/experiment.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_co_solver(env):
    grid_size = getattr(env, 'grid_size', None)

    if grid_size is not None:
        return ShortestPathSolver(grid_size)
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
    new_episode = dict(episode)
    new_rounds = []

    for rd in episode['rounds_data']:
        new_rd = {}
        for key, value in rd.items():
            if isinstance(value, np.ndarray):
                new_rd[key] = value.copy()
            else:
                try:
                    new_rd[key] = copy.deepcopy(value)
                except TypeError:
                    new_rd[key] = value

        true_vals = rd['true_values']
        optimal = solver_fn(true_vals)
        optimal = np.asarray(optimal)
        new_rd['optimal_action'] = optimal.copy()
        new_rd['optimal_value'] = float(np.sum(true_vals * optimal))
        new_rounds.append(new_rd)

    new_episode['rounds_data'] = new_rounds
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
    cfg = ablation_config.copy()
    cfg['n_rounds'] = min(cfg.get('n_rounds', 2500), 500)
    cfg['n_seeds'] = min(cfg.get('n_seeds', 5), 2)
    return cfg


def make_dataset_config(dataset: str, config: dict, **overrides) -> dict:
    dc = config['datasets'][dataset].copy()
    dc.pop('env_type', None)
    dc.pop('skip_tuning', None)
    dc.pop('tuning_year', None)
    dc.update(overrides)
    return dc

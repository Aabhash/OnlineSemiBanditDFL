import argparse, fcntl, json, sys
from pathlib import Path

import jax
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name('frozen_config.json')
sys.path.insert(0, str(REPO))

from experiments.ablations.common import make_co_solver
from experiments.rebuttal.nonlinear.adapters import (
    GatedResidualBLLAgent, SolverMonitor, action_flags, load_episode, make_legacy_pi_agent, write_json,
)
from experiments.rebuttal.nonlinear.analyze import summarize
from experiments.tuning.objectives import create_agent_from_params
from src.agents.base import run_agent_on_episode
from src.envs import create_env


def subset(available, requested, name):
    if requested is None:
        return available
    if len(requested) != len(set(requested)) or not set(requested) <= set(available):
        raise ValueError(f'{name} must be a unique subset of {available}')
    return [value for value in available if value in requested]


def prepare(args):
    config = json.loads(args.config.read_text())
    config['training_rounds'] = config['rounds']
    config['rounds'] = args.rounds if args.rounds is not None else config['rounds']
    if not 0 < config['rounds'] <= config['training_rounds']:
        raise ValueError('Round count must fit the frozen episodes; the training schedule stays unchanged')
    if config['gradient_mode'] != 'current' or config['feedback'] != 'semi_bandit':
        raise ValueError('This experiment uses current gradients and semi-bandit feedback')
    if config['learning_rate_multiplier'] != 1 or config['beta_multiplier'] != 1:
        raise ValueError('Use the frozen learning rate and exploration settings without multipliers')
    config['dataset_order'] = subset(config['dataset_order'], args.datasets, 'Datasets')
    config['seeds'] = subset(config['seeds'], args.seeds, 'Seeds')
    config['datasets'] = {name: config['datasets'][name] for name in config['dataset_order']}
    for settings in config['datasets'].values():
        settings['method_order'] = subset(settings['method_order'], args.methods, 'Methods')
    if args.methods is not None:
        config['target_methods'] = [method for method in config['target_methods'] if method in args.methods]
    streams = len(config['seeds']) * len(config['passes'])
    config['expected_method_runs'] = streams * sum(len(s['method_order']) for s in config['datasets'].values())
    config['expected_target_cells'] = streams * sum(len(set(s['method_order']) & set(config['target_methods']))
                                                   for s in config['datasets'].values())
    for source in config['episode_sources'].values():
        original = (REPO / source).resolve()
        if args.output.is_relative_to(original) or original.is_relative_to(args.output):
            raise ValueError('Output must be separate from the accepted source archives')
    if args.config.is_relative_to(args.output):
        raise ValueError('Keep the frozen config outside the output directory')
    saved_config = args.output / 'config.json'
    if saved_config.exists():
        # shortcut: resume checks settings only; use a new output after code changes.
        if json.loads(saved_config.read_text()) != config:
            raise ValueError('Existing run has different settings; use a new output directory')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Use an empty output directory')
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(saved_config, config)
    print(f"Prepared {config['expected_method_runs']} method runs, {config['expected_target_cells']} target runs", flush=True)
    return config


def make_agents(env, settings, config, seed, variant):
    agents, effective = {}, {}
    for method in settings['method_order']:
        params = dict(settings['tuned_params'][method])
        if method in config['target_methods'] and method != 'BLL-Online':
            params.update(nonlinear=variant == 'nonlinear', gradient_mode='current')
            if variant == 'nonlinear':
                params.update(settings['architecture'])
        if method == 'BLL-Online' and method in config['target_methods'] and variant == 'nonlinear':
            params.update(settings['architecture'], nonlinear=True, gradient_mode='current')
            agent = GatedResidualBLLAgent(env.context_dim, env.n_arms, env.weights, env.capacity,
                lr=params.get('init_lr', 0.001), decay_rounds=config['training_rounds'],
                minimize=getattr(env, 'minimize', False), solver=env.solve, seed=seed,
                mask_grad=False, posterior_complete_missing=True,
                **{key: value for key, value in params.items() if key not in ('lr', 'init_lr')})
        elif method == 'BayesianSPO-PI':
            agent = make_legacy_pi_agent(env, params, seed, config['training_rounds'])
        else:
            agent = create_agent_from_params(method, env, params, seed, config['training_rounds'],
                                             lr=params.get('lr', config['agent_defaults']['lr']))
        if method == 'BLL-Online' and method in config['target_methods']:
            params.update(nonlinear=variant == 'nonlinear', gradient_mode='current')
        agents[method], effective[method] = agent, params
    return agents, effective


def run_pass(output, config, dataset, seed, variant, episode):
    root = output / 'seeds' / f'seed_{seed}' / 'datasets' / dataset / 'passes' / variant
    if (root / 'complete.json').exists():
        return
    if root.exists():
        raise RuntimeError(f'Incomplete pass at {root}; use a new output to preserve its solver history')
    settings = config['datasets'][dataset]
    env = create_env(dataset, seed=seed, **settings['environment_config'])
    monitor = SolverMonitor(env)
    agents, effective = make_agents(env, settings, config, seed, variant)
    co_solver = make_co_solver(env)
    root.mkdir(parents=True)
    rows = []
    identity = dict(seed=seed, dataset=dataset, variant=variant, rounds=config['rounds'],
                    training_rounds=config['training_rounds'])
    for method, agent in agents.items():
        if method == 'BLL-Online' and variant == 'linear' and settings.get('bll_nonlinear_prefix', False):
            # Archived linear BLL followed nonlinear SPO/MSE on a separate shared solver.
            env = create_env(dataset, seed=seed, **settings['environment_config'])
            monitor = SolverMonitor(env)
            prefix_agents, _ = make_agents(env, settings, config, seed, 'nonlinear')
            co_solver = make_co_solver(env)
            for prefix_method, prefix_agent in prefix_agents.items():
                if prefix_method == method:
                    break
                write_json(output / 'progress.json', {**identity, 'method': prefix_method, 'phase': 'bll_prefix'})
                print(f'replaying {seed} {dataset} {prefix_method} before linear BLL', flush=True)
                run_agent_on_episode(prefix_agent, episode, config['rounds'],
                    lr=settings['tuned_params'][prefix_method].get('lr', config['agent_defaults']['lr']),
                    full_feedback=False, diagnostics=config['diagnostics'],
                    diagnostics_interval=config['diagnostics_interval'], co_solver=co_solver)
            agent.solver = env.solve
        monitor.reset()
        cell = root / 'methods' / method
        cell.mkdir(parents=True)
        progress = {**identity, 'method': method, 'round': 0, 'phase': 'training'}
        write_json(output / 'progress.json', progress)
        update = agent.update

        def updating(*args, _update=update, **kwargs):
            result = _update(*args, **kwargs)
            progress['round'] += 1
            if progress['round'] % 1000 == 0:
                write_json(output / 'progress.json', progress)
                print(f'{seed} {dataset} {variant} {method} {progress["round"]}/{config["rounds"]}', flush=True)
            return result

        agent.update = updating
        update_lr = settings['tuned_params'][method].get('lr', config['agent_defaults']['lr'])
        result = run_agent_on_episode(agent, episode, config['rounds'], lr=update_lr, full_feedback=False,
            diagnostics=config['diagnostics'], diagnostics_interval=config['diagnostics_interval'], co_solver=co_solver)
        trajectory = {key: np.asarray(value) for key, value in result.items() if isinstance(value, (list, np.ndarray))}
        row = {**identity, 'method': method, 'target': method in config['target_methods'],
               'effective_params': effective[method], 'update_lr': update_lr,
               **{key: float(result[key]) for key in ('final_regret', 'final_pseudo_regret', 'final_reward')}}
        row['solver_flags'] = monitor.flags
        row['action_flags'] = action_flags(trajectory['actions'], env)
        row['flagged'] = bool(row['action_flags']['invalid_action_rounds'] or any(
            monitor.flags[key] for key in ('no_incumbent_calls', 'invalid_action_calls', 'nonfinite_objective_calls')))
        if row['target'] and variant == 'nonlinear':
            row['final_gates'] = [float(block['gate']) for block in agent.feature_params['blocks']]
        np.savez_compressed(cell / 'trajectory.npz', **trajectory)
        rows.append(row)
        print(f'done {seed} {dataset} {variant} {method} regret={row["final_regret"]:.9f}', flush=True)
    write_json(root / 'complete.json', {'rows': rows})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', '--datasets', dest='datasets', nargs='+')
    parser.add_argument('--seed', '--seeds', dest='seeds', nargs='+', type=int)
    parser.add_argument('--methods', nargs='+', help='Optional subset, retaining the configured order')
    parser.add_argument('--rounds', type=int, help='Run an episode prefix without changing the learning-rate schedule')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    args.config, args.output = args.config.resolve(), args.output.resolve()
    config = prepare(args)
    if args.prepare_only:
        return
    if jax.default_backend() != 'cpu':
        raise RuntimeError('This frozen experiment runs on CPU')
    with (args.output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in config['seeds']:
            for dataset in config['dataset_order']:
                episode = load_episode(REPO, config, dataset, seed, config['rounds'])
                for variant in config['passes']:
                    run_pass(args.output, config, dataset, seed, variant, episode)
        report = summarize(args.output, config)
        write_json(args.output / 'progress.json', {'phase': 'complete', 'completed_cells': report['completed_cells']})


if __name__ == '__main__':
    main()

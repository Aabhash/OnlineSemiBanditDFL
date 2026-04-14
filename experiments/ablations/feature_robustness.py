
import sys
sys.path.insert(0, '.')

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.envs import create_env
from src.agents.contextual import BayesianSPOAgent, BayesianMSEAgent
from src.agents.contextual.weber_bll import WeberBLLAgent
from src.agents.base import pregenerate_episode, run_agent_on_episode


def compute_feature_relevance(env, n_samples=200):
    data = env.data
    B = data.B
    B_density = B.mean(axis=0)

    solver = env.solve
    rng = np.random.default_rng(9999)
    perturbation_scale = 2.0

    relevance = np.zeros(env.context_dim)

    for _ in range(n_samples):
        x = rng.standard_normal(env.context_dim)
        vals = data._generate_values(x)
        opt_orig = np.asarray(solver(vals)).flatten()

        for j in range(env.context_dim):
            x_pert = x.copy()
            x_pert[j] += perturbation_scale * rng.standard_normal()
            vals_pert = data._generate_values(x_pert)
            opt_pert = np.asarray(solver(vals_pert)).flatten()
            if not np.allclose(opt_orig, opt_pert):
                relevance[j] += 1

    relevance /= n_samples
    return relevance, B_density


def select_feature_groups(relevance, B_density):
    n = len(relevance)
    order = np.argsort(relevance)
    mid = max(1, n // 2)
    irrelevant_dims = order[:mid].tolist()
    relevant_dims = order[mid:].tolist()
    return relevant_dims, irrelevant_dims


def evaluate_perturbation_robustness(
    agent,
    env,
    X_test,
    Y_test,
    irrelevant_dims,
    relevant_dims,
    noise_scales=(0.5, 1.0, 2.0),
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(12345)

    n_test = len(X_test)
    solver = env.solve


    decisions_clean = []
    preds_clean = []
    for i in range(n_test):
        sel = np.asarray(agent.act(X_test[i])).flatten()
        decisions_clean.append(sel)
        pred = np.asarray(agent.predict_mean(X_test[i])).flatten()
        preds_clean.append(pred)
    decisions_clean = np.array(decisions_clean)
    preds_clean = np.array(preds_clean)


    opt_decisions = np.array([np.asarray(solver(Y_test[i])).flatten() for i in range(n_test)])


    clean_values = np.sum(Y_test * decisions_clean, axis=1)
    opt_values = np.sum(Y_test * opt_decisions, axis=1)
    clean_regret = np.mean(opt_values - clean_values)

    metrics = {
        'clean_test_regret': float(clean_regret),
        'clean_decision_accuracy': float(np.mean(np.all(decisions_clean == opt_decisions, axis=1))),
    }

    for scale in noise_scales:
        for group_name, dims in [('irrelevant', irrelevant_dims), ('relevant', relevant_dims)]:
            decision_changes = []
            pred_changes = []
            perturbed_regrets = []

            for i in range(n_test):

                X_pert = X_test[i].copy()
                noise = rng.standard_normal(len(dims)) * scale
                X_pert[dims] += noise


                sel_pert = np.asarray(agent.act(X_pert)).flatten()
                pred_pert = np.asarray(agent.predict_mean(X_pert)).flatten()


                decision_changes.append(np.mean(np.abs(sel_pert - decisions_clean[i])))


                pred_changes.append(np.sqrt(np.mean((pred_pert - preds_clean[i]) ** 2)))


                true_vals = Y_test[i]
                perturbed_regrets.append(
                    np.sum(true_vals * opt_decisions[i]) - np.sum(true_vals * sel_pert)
                )

            key = f'{group_name}_noise{scale}'
            metrics[f'{key}_decision_change'] = float(np.mean(decision_changes))
            metrics[f'{key}_pred_rmse_change'] = float(np.mean(pred_changes))
            metrics[f'{key}_test_regret'] = float(np.mean(perturbed_regrets))

    return metrics


def run_feature_robustness(
    dataset='knapsack_32',
    n_rounds=500,
    n_seeds=3,
    eval_interval=50,
    n_test=100,
    noise_scales=(0.5, 1.0, 2.0),
    output_dir='results/ablations/feature_robustness',
    verbose=True,
):
    with open('experiments/configs/experiment.yaml') as f:
        config = yaml.safe_load(f)

    dataset_config = config['datasets'][dataset].copy()
    dataset_config.pop('env_type', None)

    defaults = config['agent_defaults']
    hidden_dim = defaults['hidden_dim']
    lambda_reg = defaults['lambda_reg']
    lr = defaults['lr']
    beta = defaults['beta']

    all_results = {}

    for seed in range(n_seeds):
        if verbose:
            print(f'\n=== Seed {seed} ===')

        env = create_env(dataset, seed, **dataset_config)
        minimize = getattr(env, 'minimize', False)
        solver = env.solve
        bayes_beta = -beta if minimize else beta


        relevance, B_density = compute_feature_relevance(env)
        relevant_dims, irrelevant_dims = select_feature_groups(relevance, B_density)

        if verbose:
            print(f'Feature relevance scores: {np.round(relevance, 3)}')
            print(f'B density (frac items using feature): {np.round(B_density, 3)}')
            print(f'Relevant dims: {relevant_dims}')
            print(f'Irrelevant dims: {irrelevant_dims}')


        test_rng = np.random.default_rng(seed + 7777)
        X_test = np.array([test_rng.standard_normal(env.context_dim) for _ in range(n_test)])
        Y_test = np.array([env.data._generate_values(X_test[i]) for i in range(n_test)])


        episode = pregenerate_episode(env, n_rounds)


        agents = {
            'BayesianSPO': BayesianSPOAgent(
                env.context_dim, env.n_arms, env.weights, env.capacity,
                hidden_dim=hidden_dim, beta=bayes_beta, decay_rounds=n_rounds,
                lambda_reg=lambda_reg, minimize=minimize, solver=solver, seed=seed,
                mask_grad=True,
            ),
            'BayesianMSE': BayesianMSEAgent(
                env.context_dim, env.n_arms, env.weights, env.capacity,
                hidden_dim=hidden_dim, beta=bayes_beta, lambda_reg=lambda_reg,
                decay_rounds=n_rounds, solver=solver, seed=seed,
            ),
            'WeberBLL-Online': WeberBLLAgent(
                env.context_dim, env.n_arms, env.weights, env.capacity,
                hidden_dim=hidden_dim, beta=bayes_beta, decay_rounds=n_rounds,
                lambda_reg=lambda_reg, minimize=minimize, solver=solver, seed=seed,
                mask_grad=True, phased=False,
            ),
        }

        seed_results = {}

        for agent_name, agent in agents.items():
            if verbose:
                print(f'\n  Agent: {agent_name}')

            eval_rng = np.random.default_rng(seed + 5555)

            rounds_data = episode['rounds_data']
            eval_metrics = []
            cumulative_regret = 0.0

            for t in range(n_rounds):
                rd = rounds_data[t]
                X = rd['context']
                true_vals = rd['true_values']
                optimal_val = rd['optimal_value']

                selection = agent.act(X)
                selection = np.asarray(selection).flatten()

                observed = np.zeros_like(true_vals)
                mask = selection > 0.5
                observed[mask] = true_vals[mask]

                reward = np.sum(true_vals * selection)
                cumulative_regret += optimal_val - reward


                if isinstance(agent, BayesianSPOAgent):
                    agent.update(X, selection, observed, lr, full_feedback=False)
                elif isinstance(agent, BayesianMSEAgent):
                    agent.update(X, selection, observed, full_feedback=False)
                elif isinstance(agent, WeberBLLAgent):
                    agent.update(X, selection, observed, lr, full_feedback=False)


                if (t + 1) % eval_interval == 0:
                    m = evaluate_perturbation_robustness(
                        agent, env, X_test, Y_test,
                        irrelevant_dims, relevant_dims,
                        noise_scales=noise_scales,
                        rng=np.random.default_rng(eval_rng.integers(2**31)),
                    )
                    m['round'] = t + 1
                    m['cumulative_regret'] = float(cumulative_regret)
                    eval_metrics.append(m)

                    if verbose:
                        irr1 = m.get('irrelevant_noise1.0_decision_change', 0)
                        rel1 = m.get('relevant_noise1.0_decision_change', 0)
                        print(f'    t={t+1:4d}  cum_regret={cumulative_regret:.1f}'
                              f'  irr_dec_change={irr1:.3f}  rel_dec_change={rel1:.3f}')

            seed_results[agent_name] = {
                'eval_metrics': eval_metrics,
                'final_cumulative_regret': float(cumulative_regret),
            }

        all_results[seed] = {
            'agents': seed_results,
            'feature_relevance': relevance.tolist(),
            'B_density': B_density.tolist(),
            'relevant_dims': relevant_dims,
            'irrelevant_dims': irrelevant_dims,
        }


    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{dataset}_n{n_rounds}_s{n_seeds}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    if verbose:
        print(f'\nResults saved to {out_path}')


    print('\n' + '=' * 70)
    print('SUMMARY: Feature Robustness (noise_scale=1.0)')
    print('=' * 70)

    for agent_name in agents:
        irr_changes = []
        rel_changes = []
        irr_regrets = []
        rel_regrets = []
        for seed in range(n_seeds):
            last = all_results[seed]['agents'][agent_name]['eval_metrics'][-1]
            irr_changes.append(last.get('irrelevant_noise1.0_decision_change', 0))
            rel_changes.append(last.get('relevant_noise1.0_decision_change', 0))
            irr_regrets.append(last.get('irrelevant_noise1.0_test_regret', 0))
            rel_regrets.append(last.get('relevant_noise1.0_test_regret', 0))

        print(f'\n{agent_name}:')
        print(f'  Irrelevant pert → decision change: {np.mean(irr_changes):.4f} ± {np.std(irr_changes):.4f}')
        print(f'  Relevant   pert → decision change: {np.mean(rel_changes):.4f} ± {np.std(rel_changes):.4f}')
        print(f'  Irrelevant pert → test regret:     {np.mean(irr_regrets):.3f} ± {np.std(irr_regrets):.3f}')
        print(f'  Relevant   pert → test regret:     {np.mean(rel_regrets):.3f} ± {np.std(rel_regrets):.3f}')
        ratio = np.mean(irr_changes) / max(np.mean(rel_changes), 1e-8)
        print(f'  Irrelevant/Relevant change ratio:  {ratio:.3f}  (lower = more focused)')

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature robustness ablation')
    parser.add_argument('--dataset', type=str, default='knapsack_32')
    parser.add_argument('--n_rounds', type=int, default=500)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='results/ablations/feature_robustness')
    args = parser.parse_args()

    run_feature_robustness(
        dataset=args.dataset,
        n_rounds=args.n_rounds,
        n_seeds=args.n_seeds,
        eval_interval=args.eval_interval,
        n_test=args.n_test,
        output_dir=args.output_dir,
    )

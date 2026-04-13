import argparse
import sys
sys.path.insert(0, '.')

import jax
import numpy as np
import jax.numpy as jnp
import yaml
from collections import defaultdict

from src.envs import create_env
import src.losses as losses_module
from src.solver import solve_knapsack
from src.agents.bayesian_utils import (
    init_per_arm_regression,
    compute_ucb_values,
    compute_bayesian_predictions,
    update_bayesian_stats,
)

spo_plus_grad = losses_module.spo_plus_grad
spo_plus_grad_smooth = getattr(losses_module, 'spo_plus_grad_smooth', None)


def analyze_gradients(dataset='knapsack_32', n_rounds=500, seed=42):
    with open('experiments/configs/experiment.yaml') as f:
        config = yaml.safe_load(f)


    dataset_config = config['datasets'][dataset].copy()
    dataset_config.pop('env_type', None)
    env = create_env(dataset, seed, **dataset_config)

    context_dim = env.context_dim
    n_arms = env.n_arms
    hidden_dim = 32
    lambda_reg = 1.0
    beta = 1.0

    key = jax.random.PRNGKey(seed)
    W1 = jax.random.normal(key, (context_dim, hidden_dim)) * 0.1
    b1 = jnp.zeros(hidden_dim)

    A, b_bayes, A_inv = init_per_arm_regression(n_arms, hidden_dim, lambda_reg)

    weights = jnp.asarray(env.weights)
    capacity = env.capacity


    metrics = defaultdict(list)

    X = env.reset()

    for t in range(n_rounds):
        X_jax = jnp.asarray(X)


        z1 = X_jax @ W1 + b1
        h = jnp.maximum(0, z1)


        pred = compute_bayesian_predictions(h, A_inv, b_bayes)
        ucb_values = compute_ucb_values(h, A_inv, b_bayes, beta)


        selection = solve_knapsack(np.asarray(ucb_values), np.asarray(env.weights), capacity)


        observed_values, reward, info = env.step(selection, full_feedback=False)
        true_values = info['true_values']


        selection_jax = jnp.asarray(selection)
        rewards_jax = jnp.asarray(observed_values)
        selection_mask = selection_jax > 0.5

        A, b_bayes, A_inv = update_bayesian_stats(
            A, b_bayes, A_inv, h, rewards_jax, selection_mask
        )


        Y_ucb = jnp.where(selection_jax > 0.5, rewards_jax, ucb_values)

        Y_dm = jnp.where(selection_jax > 0.5, rewards_jax, pred)

        Y_true = jnp.asarray(true_values)


        mse_grad_ucb = 2 * (pred - Y_ucb) / n_arms
        mse_grad_dm = 2 * (pred - Y_dm) / n_arms
        mse_grad_true = 2 * (pred - Y_true) / n_arms

        spo_grad_ucb = spo_plus_grad(pred, Y_ucb, weights, capacity)
        spo_grad_dm = spo_plus_grad(pred, Y_dm, weights, capacity)
        spo_grad_true = spo_plus_grad(pred, Y_true, weights, capacity)


        if spo_plus_grad_smooth is not None:
            smooth_spo_grad_ucb = spo_plus_grad_smooth(
                pred, Y_ucb, weights, capacity, temperature=1.0)
            smooth_spo_grad_true = spo_plus_grad_smooth(
                pred, Y_true, weights, capacity, temperature=1.0)
        else:
            smooth_spo_grad_ucb = None
            smooth_spo_grad_true = None


        metrics['round'].append(t)


        metrics['mse_grad_ucb_norm'].append(float(jnp.linalg.norm(mse_grad_ucb)))
        metrics['mse_grad_dm_norm'].append(float(jnp.linalg.norm(mse_grad_dm)))
        metrics['mse_grad_true_norm'].append(float(jnp.linalg.norm(mse_grad_true)))
        metrics['spo_grad_ucb_norm'].append(float(jnp.linalg.norm(spo_grad_ucb)))
        metrics['spo_grad_dm_norm'].append(float(jnp.linalg.norm(spo_grad_dm)))
        metrics['spo_grad_true_norm'].append(float(jnp.linalg.norm(spo_grad_true)))


        metrics['spo_grad_ucb_sparsity'].append(float(jnp.mean(spo_grad_ucb != 0)))
        metrics['spo_grad_dm_sparsity'].append(float(jnp.mean(spo_grad_dm != 0)))
        metrics['spo_grad_true_sparsity'].append(float(jnp.mean(spo_grad_true != 0)))


        if smooth_spo_grad_ucb is not None and smooth_spo_grad_true is not None:
            metrics['smooth_spo_grad_ucb_norm'].append(float(jnp.linalg.norm(smooth_spo_grad_ucb)))
            metrics['smooth_spo_grad_true_norm'].append(float(jnp.linalg.norm(smooth_spo_grad_true)))
        else:
            metrics['smooth_spo_grad_ucb_norm'].append(np.nan)
            metrics['smooth_spo_grad_true_norm'].append(np.nan)


        def cosine_sim(a, b):
            norm_a = jnp.linalg.norm(a)
            norm_b = jnp.linalg.norm(b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                return 0.0
            return float(jnp.dot(a, b) / (norm_a * norm_b))

        metrics['mse_ucb_vs_true_cosine'].append(cosine_sim(mse_grad_ucb, mse_grad_true))
        metrics['mse_dm_vs_true_cosine'].append(cosine_sim(mse_grad_dm, mse_grad_true))
        metrics['spo_ucb_vs_true_cosine'].append(cosine_sim(spo_grad_ucb, spo_grad_true))
        metrics['spo_dm_vs_true_cosine'].append(cosine_sim(spo_grad_dm, spo_grad_true))
        if smooth_spo_grad_ucb is not None and smooth_spo_grad_true is not None:
            metrics['smooth_spo_ucb_vs_true_cosine'].append(
                cosine_sim(smooth_spo_grad_ucb, smooth_spo_grad_true))
        else:
            metrics['smooth_spo_ucb_vs_true_cosine'].append(np.nan)


        metrics['ucb_imputation_mse'].append(float(jnp.mean((Y_ucb - Y_true) ** 2)))
        metrics['dm_imputation_mse'].append(float(jnp.mean((Y_dm - Y_true) ** 2)))


        metrics['pred_mse'].append(float(jnp.mean((pred - Y_true) ** 2)))


        metrics['regret'].append(info['regret'])

        X = env.get_context()

    return metrics


def print_summary(metrics):

    print("\n" + "="*70)
    print("GRADIENT ANALYSIS SUMMARY")
    print("="*70)


    print("\n--- Gradient Magnitudes (mean ± std) ---")
    print(f"MSE grad (UCB imp):  {np.mean(metrics['mse_grad_ucb_norm']):.4f} ± {np.std(metrics['mse_grad_ucb_norm']):.4f}")
    print(f"MSE grad (DM imp):   {np.mean(metrics['mse_grad_dm_norm']):.4f} ± {np.std(metrics['mse_grad_dm_norm']):.4f}")
    print(f"MSE grad (true):     {np.mean(metrics['mse_grad_true_norm']):.4f} ± {np.std(metrics['mse_grad_true_norm']):.4f}")
    print(f"SPO+ grad (UCB imp): {np.mean(metrics['spo_grad_ucb_norm']):.4f} ± {np.std(metrics['spo_grad_ucb_norm']):.4f}")
    print(f"SPO+ grad (DM imp):  {np.mean(metrics['spo_grad_dm_norm']):.4f} ± {np.std(metrics['spo_grad_dm_norm']):.4f}")
    print(f"SPO+ grad (true):    {np.mean(metrics['spo_grad_true_norm']):.4f} ± {np.std(metrics['spo_grad_true_norm']):.4f}")

    print("\n--- SPO+ Gradient Sparsity (fraction non-zero) ---")
    print(f"SPO+ (UCB imp): {np.mean(metrics['spo_grad_ucb_sparsity']):.4f}")
    print(f"SPO+ (DM imp):  {np.mean(metrics['spo_grad_dm_sparsity']):.4f}")
    print(f"SPO+ (true):    {np.mean(metrics['spo_grad_true_sparsity']):.4f}")

    print("\n--- Gradient Alignment with True Gradient (cosine similarity) ---")
    print(f"MSE (UCB imp) vs true:        {np.mean(metrics['mse_ucb_vs_true_cosine']):.4f}")
    print(f"MSE (DM imp) vs true:         {np.mean(metrics['mse_dm_vs_true_cosine']):.4f}")
    print(f"SPO+ (UCB imp) vs true:       {np.mean(metrics['spo_ucb_vs_true_cosine']):.4f}")
    print(f"SPO+ (DM imp) vs true:        {np.mean(metrics['spo_dm_vs_true_cosine']):.4f}")
    smooth_cos = np.nanmean(metrics['smooth_spo_ucb_vs_true_cosine'])
    if np.isnan(smooth_cos):
        print("Smooth SPO+ (UCB) vs true:    unavailable")
    else:
        print(f"Smooth SPO+ (UCB) vs true:    {smooth_cos:.4f}")

    print("\n--- Smooth SPO+ Gradient Magnitudes ---")
    smooth_ucb = np.nanmean(metrics['smooth_spo_grad_ucb_norm'])
    smooth_true = np.nanmean(metrics['smooth_spo_grad_true_norm'])
    if np.isnan(smooth_ucb) or np.isnan(smooth_true):
        print("Smooth SPO+ metrics: unavailable")
    else:
        print(f"Smooth SPO+ (UCB imp): {smooth_ucb:.4f}")
        print(f"Smooth SPO+ (true):    {smooth_true:.4f}")

    print("\n--- Imputation Error (MSE) ---")
    print(f"UCB imputation: {np.mean(metrics['ucb_imputation_mse']):.4f}")
    print(f"DM imputation:  {np.mean(metrics['dm_imputation_mse']):.4f}")

    print("\n--- Prediction Error ---")
    print(f"Prediction MSE: {np.mean(metrics['pred_mse']):.4f}")


    n = len(metrics['round'])
    early = slice(0, n//4)
    late = slice(3*n//4, n)

    print("\n--- Early vs Late Training ---")
    print(f"SPO+ alignment (early): {np.mean(metrics['spo_ucb_vs_true_cosine'][early]):.4f}")
    print(f"SPO+ alignment (late):  {np.mean(metrics['spo_ucb_vs_true_cosine'][late]):.4f}")
    print(f"MSE alignment (early):  {np.mean(metrics['mse_ucb_vs_true_cosine'][early]):.4f}")
    print(f"MSE alignment (late):   {np.mean(metrics['mse_ucb_vs_true_cosine'][late]):.4f}")
    print(f"Imputation error (early): {np.mean(metrics['ucb_imputation_mse'][early]):.4f}")
    print(f"Imputation error (late):  {np.mean(metrics['ucb_imputation_mse'][late]):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='knapsack_32')
    parser.add_argument('--n_rounds', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Analyzing gradients on {args.dataset} for {args.n_rounds} rounds...")
    metrics = analyze_gradients(args.dataset, args.n_rounds, args.seed)
    print_summary(metrics)

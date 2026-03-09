# Decision-Focused Learning for Combinatorial Semi-Bandits

This repository implements **BayesianSPO**, a decision-focused learning method for combinatorial semi-bandit problems. BayesianSPO combines per-arm Bayesian uncertainty with masked SPO+ gradients to achieve lower cumulative regret compared to traditional bandit baselines (Thompson Sampling, LinUCB, NeuralUCB).

## Problem Setting

**Online Combinatorial Semi-Bandits with Shared Context:**
- At each round, observe shared context $x_t \in \mathbb{R}^d$, predict cost vector $\hat{c}_t = f_\theta(x_t)$
- Solve combinatorial optimization $w^*_t = \arg\min_{w \in \mathcal{W}} \hat{c}_t^\top w$ (knapsack, shortest path)
- Observe costs only for selected arms (semi-bandit feedback, ~5-60% observation rate)
- Goal: Minimize cumulative regret over $T$ rounds

**challenge:** SPO+ requires the full cost vector, but 40-95% is unobserved, the *imputation bottleneck*.

**BayesianSPO solution:** Shared neural feature extractor + per-arm Bayesian linear regression. UCB-based exploration provides stable imputation for unobserved arms; masked SPO+ gradients train the shared features.

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first. Then,

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Quick Start

```bash
# Download Criteo data from HuggingFace
uv run python src/data/download_criteo.py --subsample_rate 1.0

# Run HP tuning for all agents on a dataset
uv run python experiments/tune_hyperparams.py --all --dataset knapsack --feedback semi_bandit --n-rounds 2000 --analyze

# Run full evaluation with tuned parameters
uv run python experiments/evaluate.py --datasets knapsack --feedback semi_bandit --use-tuned-params --full --wandb --analyze

# Plot results
uv run python experiments/plot_eval_results.py --format pdf
```

## Datasets

| Dataset | Arms ($n$) | Selection size | Obs. rate | Description |
|---------|-----------|----------------|-----------|-------------|
| `knapsack` | 32 | ~20 (capacity) | ~62% | Heterogeneous weights, capacity 20 |
| `knapsack_unit_weight` | 32 | 20 | ~62% | Unit weights (cardinality constraint) |
| `shortest_path_8x8` | 112 | ~14 (path) | ~12% | 8x8 grid, shortest path |
| `shortest_path_20x20` | 760 | ~38 (path) | ~5% | 20x20 grid, shortest path |
| `criteo` | 25 | 10 | ~40% | Real-world ad bidding with XGBoost surrogate |

All synthetic datasets use shared context $x_t \sim \mathcal{N}(0, I_5)$ and noise $\epsilon \sim \text{Uniform}(-0.25, 0.25)$.

## Agents

| Agent | Description |
|-------|-------------|
| **BayesianSPO** | Masked SPO+ loss with per-arm Bayesian uncertainty (our method) |
| BayesianMSE | Per-arm Bayesian regression with MSE loss (prediction-focused) |
| LinUCB | Linear UCB baseline |
| NeuralUCB | Neural network with UCB exploration |
| UCB-Bayesian | Bayesian linear model with UCB exploration |
| ThompsonSampling | Thompson Sampling with Bayesian linear model |
| EpsilonGreedy | $\epsilon$-greedy with neural network |
| Greedy | Greedy exploitation with neural network |

## Experiment Pipeline

### 1. Hyperparameter Tuning

Optuna TPE sampler, 50 trials, 2500 rounds, 3 tuning seeds (0, 1, 2). Pruning at 75th percentile with late-stage evaluation.

```bash
uv run python experiments/tune_hyperparams.py --all --dataset knapsack --feedback semi_bandit --n-rounds 2000 --analyze
```

### 2. Full Evaluation

10,000 rounds, 10 disjoint evaluation seeds (100-109), using best HPs from tuning. Results logged to W&B.

```bash
uv run python experiments/evaluate.py --datasets knapsack --feedback semi_bandit --use-tuned-params --full --wandb --analyze
```

### 3. Plotting

```bash
uv run python experiments/plot_eval_results.py --format pdf
```

Generates trajectory plots (cumulative + per-round regret) and bar charts per dataset.

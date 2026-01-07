# Decision-Focused Learning for Combinatorial Semi-Bandits

This repository implements decision-focused learning methods for combinatorial semi-bandit problems, introducing BayesianSPO and comparing with traditional Bayesian exploration methods (Thompson Sampling, UCB).

## Problem Setting

**Combinatorial Semi-Bandit with Constraints:**
- Select a subset of arms subject to a knapsack constraint (budget)
- Observe rewards only for selected arms (semi-bandit feedback)
- Goal: Minimize cumulative regret over T rounds

**Contribution:** Instead of optimizing predictions (MSE), optimize for decision quality using SPO+ loss with UCB-based imputation for unobserved arms.

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
# Download criteo data from HuggingFace
uv run python src/data/download_criteo.py --subsample_rate 1.0

# Run evaluation on default datasets
python experiments/evaluate.py --datasets synthetic,fantasy --n_rounds 500 --n_seeds 3

# Run hyperparameter tuning
python experiments/tune_hyperparams.py --agent PerArmBayesianSPO --dataset fantasy --n_rounds 200 --n_seeds 3

# Plot results
python experiments/plot_results.py --input_dir results --output_dir plots
```

## Main Scripts

### 1. Evaluation (`experiments/evaluate.py`)

Run experiments comparing different agents on various datasets.

```bash
# Basic usage
python experiments/evaluate.py --datasets fantasy --n_rounds 500 --n_seeds 5

# All options
python experiments/evaluate.py \
    --datasets fantasy,synthetic,cn_linear \
    --n_rounds 1000 \
    --n_seeds 10 \
    --lr 0.01 \
    --output_dir results/main \
    --wandb  # enable W&B logging
```

**Available datasets:**
- `synthetic` - Synthetic knapsack problem (shared context)
- `fantasy` - Fantasy sports lineup optimization (per-arm context)
- `shortest_path_8` - Shortest path problems
- `criteo_ads` - Criteo ad selection (shared context)

### 2. Hyperparameter Tuning (`experiments/tune_hyperparams.py`)

Grid search over hyperparameters for each agent.

```bash
python experiments/tune_hyperparams.py \
    --agent PerArmBayesianSPO \
    --dataset fantasy \
    --n_rounds 200 \
    --n_seeds 3

OR

python experiments/tune_hyperparams.py \
    --all \
    --dataset fantasy \
    --n_rounds 200 \
    --n_seeds 3
```

**Output files:**
- `experiments/tuning/results/tuning_<dataset>_semi_bandit.json` - Full results with per-round metrics
- `experiments/tuning/results/best_hyperparams.json` - Best hyperparameters for each agent

### 3. Ablation Studies (`experiments/ablations/`)

```bash
python experiments/ablations/imputation.py --dataset fantasy --n_rounds 200 --n_seeds 3
python experiments/ablations/uncertainty.py --dataset fantasy --n_rounds 200 --n_seeds 3
python experiments/ablations/training_paradigm.py --dataset fantasy --n_rounds 200 --n_seeds 3
python experiments/ablation_observation_rate.py --n_rounds 200 --n_seeds 3
```

## Agents

### Shared Context (use for `synthetic`, `shortest_path`, `criteo_ads`)

| Agent | Description |
|-------|-------------|
| `BayesianSPO` | SPO+ loss with Bayesian uncertainty (main method) |
| `DecisionUCB` | Decision-aware UCB exploration |
| `LinUCB` | Linear UCB baseline |
| `NeuralUCB` | Neural UCB baseline |
| `ThompsonSampling` | Thompson Sampling with Bayesian linear model |
| `MSENeural` | Neural network with MSE loss |

### Per-Arm Context (use for `fantasy`, `cn_*`)

| Agent | Description |
|-------|-------------|
| `PerArmBayesianSPO` | Per-arm Bayesian SPO (main method) |
| `PerArmNTKSPO` | NTK-based uncertainty with SPO+ |
| `CN-UCB` | Combinatorial Neural UCB (Hwang et al. 2023) |
| `CN-TS` | Combinatorial Neural Thompson Sampling |
| `PerArmLinUCB` | Per-arm Linear UCB |
| `PerArmNeuralUCB` | Per-arm Neural UCB |
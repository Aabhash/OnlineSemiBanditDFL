#!/bin/bash
#SBATCH --job-name=dfl_tune_online
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ====================================================================
# Stage 1: Online HP tuning for all agents on each dataset
# ====================================================================

# Shortest path 8x8
uv run python experiments/tune_hyperparams.py --all --dataset shortest_path_8x8 --feedback semi_bandit --analyze --parallel 4

# Shortest path 20x20
uv run python experiments/tune_hyperparams.py --all --dataset shortest_path_20x20 --feedback semi_bandit --analyze --parallel 4

# Portfolio 32
uv run python experiments/tune_hyperparams.py --all --dataset portfolio_32 --feedback semi_bandit --analyze --parallel 4

# Knapsack 32
uv run python experiments/tune_hyperparams.py --all --dataset knapsack_32 --feedback semi_bandit --analyze --parallel 4

# Knapsack 100
uv run python experiments/tune_hyperparams.py --all --dataset knapsack_100 --feedback semi_bandit --analyze --parallel 4

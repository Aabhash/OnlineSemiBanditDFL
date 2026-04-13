#!/bin/bash
#SBATCH --job-name=dfl_tune
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ====================================================================
# HP tuning for all agents on each dataset (no wandb — local only)
# ====================================================================

# Knapsack 32
uv run python experiments/tune_hyperparams.py --all --dataset knapsack_32 --feedback semi_bandit --n-rounds 1000 --analyze

# Shortest path 8x8
uv run python experiments/tune_hyperparams.py --all --dataset shortest_path_8x8 --feedback semi_bandit --n-rounds 1000 --analyze

# Shortest path 20x20
uv run python experiments/tune_hyperparams.py --all --dataset shortest_path_20x20 --feedback semi_bandit --n-rounds 1000 --analyze

# TopK toy datasets — sigmoid (misspecified)
uv run python experiments/tune_hyperparams.py --all --dataset topk_2arm --feedback semi_bandit --n-rounds 500 --analyze
uv run python experiments/tune_hyperparams.py --all --dataset topk_8arm --feedback semi_bandit --n-rounds 500 --analyze
uv run python experiments/tune_hyperparams.py --all --dataset topk_16arm --feedback semi_bandit --n-rounds 1000 --analyze

# TopK toy datasets — linear (well-specified)
uv run python experiments/tune_hyperparams.py --all --dataset topk_2arm_linear --feedback semi_bandit --n-rounds 500 --analyze
uv run python experiments/tune_hyperparams.py --all --dataset topk_8arm_linear --feedback semi_bandit --n-rounds 500 --analyze

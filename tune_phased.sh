#!/bin/bash
#SBATCH --job-name=dfl_tune_phased
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ====================================================================
# Stage 2: Phased agent HP refinement
#
# Backbone HPs (hidden_dim, lambda_reg) inherited from stage-1 online
# tuning in results/tuning/. This stage sweeps phase-specific knobs
# + lr/beta recheck and writes to a SEPARATE output dir to avoid
# overwriting stage-1 results.
#
# Run AFTER tune_online.sh completes.
# ====================================================================

PHASED_SPACES=experiments/configs/search_spaces_phased.yaml
PHASED_OUTPUT=results/tuning_phased
BACKBONE_DIR=results/tuning

# TopK toy datasets — sigmoid
uv run python experiments/tune_hyperparams.py --all --dataset topk_4arm --feedback semi_bandit --n-rounds 1000 --search-spaces $PHASED_SPACES --output $PHASED_OUTPUT --backbone-tuning-dir $BACKBONE_DIR --n-trials 20 --analyze
uv run python experiments/tune_hyperparams.py --all --dataset topk_8arm --feedback semi_bandit --n-rounds 1000 --search-spaces $PHASED_SPACES --output $PHASED_OUTPUT --backbone-tuning-dir $BACKBONE_DIR --n-trials 20 --analyze

# TopK toy datasets — linear
uv run python experiments/tune_hyperparams.py --all --dataset topk_4arm_linear --feedback semi_bandit --n-rounds 1000 --search-spaces $PHASED_SPACES --output $PHASED_OUTPUT --backbone-tuning-dir $BACKBONE_DIR --n-trials 20 --analyze

# Shortest path datasets
uv run python experiments/tune_hyperparams.py --all --dataset shortest_path_8x8 --feedback semi_bandit --n-rounds 1000 --search-spaces $PHASED_SPACES --output $PHASED_OUTPUT --backbone-tuning-dir $BACKBONE_DIR --n-trials 20 --analyze
uv run python experiments/tune_hyperparams.py --all --dataset shortest_path_20x20 --feedback semi_bandit --n-rounds 1000 --search-spaces $PHASED_SPACES --output $PHASED_OUTPUT --backbone-tuning-dir $BACKBONE_DIR --n-trials 20 --analyze

# Knapsack datasets
uv run python experiments/tune_hyperparams.py --all --dataset knapsack_32 --feedback semi_bandit --n-rounds 1000 --search-spaces $PHASED_SPACES --output $PHASED_OUTPUT --backbone-tuning-dir $BACKBONE_DIR --n-trials 20 --analyze
uv run python experiments/tune_hyperparams.py --all --dataset knapsack_100 --feedback semi_bandit --n-rounds 1000 --search-spaces $PHASED_SPACES --output $PHASED_OUTPUT --backbone-tuning-dir $BACKBONE_DIR --n-trials 20 --analyze

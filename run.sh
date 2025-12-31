#!/bin/bash
#SBATCH --job-name=hp_tune_bspo
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd ~/ConstrainedBanditsDFL

# Knapsack with non-stationarity
uv run python experiments/tune_hyperparams.py --all --dataset knapsack_drift_low_noise --feedback semi_bandit --n-rounds 2500 --analyze
# uv run python experiments/evaluate.py --datasets knapsack_drift_low_noise --feedback semi_bandit --use-tuned-params --full --wandb --analyze

uv run python experiments/tune_hyperparams.py --all --dataset knapsack_drift_high_noise --feedback semi_bandit --n-rounds 2500  --analyze
# uv run python experiments/evaluate.py --datasets knapsack_drift_high_noise --feedback semi_bandit --use-tuned-params --full --wandb --analyze

# Shortest path with non-stationarity
# uv run python experiments/tune_hyperparams.py --all --dataset shortest_path_drift_low_noise --feedback semi_bandit --n-rounds 2500  --analyze
# uv run python experiments/evaluate.py --datasets shortest_path_drift_low_noise --feedback semi_bandit --use-tuned-params --full --wandb --analyze

# uv run python experiments/tune_hyperparams.py --all --dataset shortest_path_drift_high_noise --feedback semi_bandit --n-rounds 2500  --analyze
# uv run python experiments/evaluate.py --datasets shortest_path_drift_high_noise --feedback semi_bandit --use-tuned-params --full --wandb --analyze

# Criteo (real-world shared context with XGBoost surrogate)
# uv run python experiments/tune_hyperparams.py --all --dataset criteo --feedback semi_bandit --n-rounds 2500  --analyze
# uv run python experiments/evaluate.py --datasets criteo --feedback semi_bandit --use-tuned-params --full --wandb --analyze

# NFL Fantasy (per-arm context benchmark)
# uv run python experiments/tune_hyperparams.py --all --dataset nfl --feedback semi_bandit --n-rounds 1000  --analyze
# uv run python experiments/evaluate.py --datasets nfl --use-tuned-params --feedback semi_bandit --full --wandb --analyze

EOF
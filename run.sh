#!/bin/bash
#SBATCH --job-name=hp_tune_bspo
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd ~/ConstrainedBanditsDFL
uv run python experiments/tune_hyperparams.py --all --dataset synthetic_drift_low_noise --n-seeds 3 --n-rounds 3000 --n-trials 30 --analyze
uv run python experiments/evaluate.py --datasets synthetic_drift_low_noise --use-tuned-params --full --n_seeds 10 --wandb

# uv run python experiments/tune_hyperparams.py --all --dataset synthetic_drift_high_noise --n-seeds 3 --n-rounds 3000 --n-trials 30  --analyze
# uv run python experiments/evaluate.py --datasets synthetic_drift_high_noise --use-tuned-params --full --n_seeds 10 --wandb

# uv run python experiments/tune_hyperparams.py --all --dataset nfl_online --n-seeds 3 --n-rounds 17 --n-trials 30  --analyze
# uv run python experiments/evaluate.py --datasets nfl_online --use-tuned-params --full --n_seeds 10 --wandb

# uv run python experiments/tune_hyperparams.py --all --dataset nfl_logged --n-seeds 3 --n-rounds 1000 --n-trials 30  --analyze
# uv run python experiments/evaluate.py --datasets nfl_logged --use-tuned-params --full --n_seeds 10 --wandb

EOF
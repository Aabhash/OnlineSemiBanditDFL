#!/bin/bash
#SBATCH --job-name=hp_tune_bspo
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd ~/ConstrainedBanditsDFL
uv run python experiments/tune_hyperparams.py --all --dataset synthetic_no_noise --analyze
uv run python experiments/tune_hyperparams.py --all --dataset synthetic_low_noise --analyze
uv run python experiments/tune_hyperparams.py --all --dataset synthetic_high_noise --analyze
uv run python experiments/tune_hyperparams.py --all --dataset nfl_online --analyze
uv run python experiments/tune_hyperparams.py --all --dataset nfl_logged --analyze
EOF
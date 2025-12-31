#!/bin/bash
#SBATCH --job-name=hp_tune_bspo
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd ~/ConstrainedBanditsDFL
uv run python experiments/tune_hyperparams.py --all --dataset synthetic_drift_low_noise --feedback semi_bandit --n-rounds 2500 --analyze
# uv run python experiments/evaluate.py --datasets synthetic_drift_low_noise --feedback semi_bandit --use-tuned-params --full --wandb --analyze

uv run python experiments/tune_hyperparams.py --all --dataset synthetic_drift_high_noise --feedback semi_bandit --n-rounds 2500  --analyze
# uv run python experiments/evaluate.py --datasets synthetic_drift_high_noise --feedback semi_bandit --use-tuned-params --full --wandb --analyze

# uv run python experiments/tune_hyperparams.py --all --dataset nfl_online --feedback semi_bandit --n-rounds 17 --n-trials 25  --analyze
# uv run python experiments/evaluate.py --datasets nfl_online --use-tuned-params --feedback semi_bandit --full --wandb --n_rounds 17 --analyze

# uv run python experiments/tune_hyperparams.py --all --dataset nfl_logged --feedback semi_bandit --n-rounds 300  --analyze
# uv run python experiments/evaluate.py --datasets nfl_logged --use-tuned-params --feedback semi_bandit --full --wandb --analyze

EOF
#!/bin/bash
#SBATCH --job-name=dfl_eval_phased
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Phased evaluation (stage-2 tuned HPs) ==="

uv run python experiments/evaluate_phased.py --datasets topk_8arm --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_phased --no-wandb
uv run python experiments/evaluate_phased.py --datasets topk_8arm_linear --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_phased --no-wandb
uv run python experiments/evaluate_phased.py --datasets shortest_path_8x8 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_phased --no-wandb
uv run python experiments/evaluate_phased.py --datasets shortest_path_20x20 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_phased --no-wandb
uv run python experiments/evaluate_phased.py --datasets portfolio_32 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_phased --no-wandb
uv run python experiments/evaluate_phased.py --datasets knapsack_32 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_phased --no-wandb
uv run python experiments/evaluate_phased.py --datasets knapsack_100 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_phased --no-wandb

echo ""
echo "=== Phased evaluation complete ==="
echo "  Data: results/eval_phased/"

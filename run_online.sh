#!/bin/bash
#SBATCH --job-name=dfl_eval_online
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Online evaluation (stage-1 tuned HPs) ==="

uv run python experiments/evaluate_online.py --datasets shortest_path_8x8 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online_10k --no-wandb
uv run python experiments/evaluate_online.py --datasets shortest_path_20x20 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online_10k --no-wandb
uv run python experiments/evaluate_online.py --datasets knapsack_32 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online_10k --no-wandb
uv run python experiments/evaluate_online.py --datasets knapsack_100 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online_10k --no-wandb
uv run python experiments/evaluate_online.py --datasets portfolio_32 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online_10k --no-wandb

echo ""
echo "=== Online evaluation complete ==="
echo "  Data: results/eval_online/"

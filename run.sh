#!/bin/bash
#SBATCH --job-name=dfl_eval
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Online evaluation ==="

# Main benchmarks
uv run python experiments/evaluate_online.py --datasets knapsack_32 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online
uv run python experiments/evaluate_online.py --datasets knapsack_100 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online
uv run python experiments/evaluate_online.py --datasets shortest_path_8x8 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online
uv run python experiments/evaluate_online.py --datasets shortest_path_20x20 --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online

# Toy dissection
uv run python experiments/evaluate_online.py --datasets topk_2arm,topk_8arm,topk_16arm --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online
uv run python experiments/evaluate_online.py --datasets topk_2arm_linear,topk_8arm_linear --feedback semi_bandit --use-tuned-params --full --output_dir results/eval_online

echo "=== Step 2: Phased evaluation ==="

# Main benchmarks (phased agents use defaults, not tuned HPs)
uv run python experiments/evaluate_phased.py --datasets knapsack_32 --feedback semi_bandit --full --output_dir results/eval_phased
uv run python experiments/evaluate_phased.py --datasets knapsack_100 --feedback semi_bandit --full --output_dir results/eval_phased
uv run python experiments/evaluate_phased.py --datasets shortest_path_8x8 --feedback semi_bandit --full --output_dir results/eval_phased
uv run python experiments/evaluate_phased.py --datasets shortest_path_20x20 --feedback semi_bandit --full --output_dir results/eval_phased

# Toy dissection
uv run python experiments/evaluate_phased.py --datasets topk_2arm,topk_8arm,topk_16arm --feedback semi_bandit --full --output_dir results/eval_phased
uv run python experiments/evaluate_phased.py --datasets topk_2arm_linear,topk_8arm_linear --feedback semi_bandit --full --output_dir results/eval_phased

echo "=== Step 3: Post-processing ==="

# Compute analysis (wallclock + Pareto)
uv run python experiments/compute_analysis.py --results-dir results/eval_online/metrics results/eval_phased/metrics --output results/compute

# Regret plots + diagnostic plots
uv run python experiments/plot_eval_results.py --input results/eval_online/metrics results/eval_phased/metrics --format pdf

echo ""
echo "=== Complete ==="
echo "  Data:         results/eval_online/  results/eval_phased/"
echo "  Compute:      results/compute/"
echo "  Figures:      results/figures/"

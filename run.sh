#!/bin/bash
#SBATCH --job-name=dfl_eval_all
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Online evaluation ==="
bash "$SCRIPT_DIR/run_online.sh"

echo "=== Step 2: Post-processing ==="

uv run python experiments/compute_analysis.py --results-dir results/eval_online_10k/metrics --output results/compute
uv run python experiments/plot_eval_results.py --input results/eval_online_10k/metrics --format pdf

echo ""
echo "=== Complete ==="
echo "  Data:         results/eval_online_10k/"
echo "  Compute:      results/compute/"
echo "  Figures:      results/figures/"

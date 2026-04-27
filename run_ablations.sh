#!/bin/bash
#SBATCH --job-name=dfl_ablations
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ====================================================================
# Main-text ablations
# ====================================================================

echo "=== Running main-text ablations ==="

uv run python experiments/ablations/kn_ratio_sweep.py --no-plot
uv run python experiments/ablations/mechanism_ablation.py --no-plot
uv run python experiments/ablations/feature_sharing.py --no-plot
uv run python experiments/ablations/flip_set_collapse.py --no-plot

# ====================================================================
# Appendix / supporting ablations
# ====================================================================

echo "=== Running appendix ablations ==="

uv run python experiments/ablations/context_variation.py --no-plot
uv run python experiments/ablations/feature_robustness.py
uv run python experiments/ablations/imputation_samples.py --mode full --use-tuned-params --no-plot

# ====================================================================
# Optional deep-dive sweeps (not part of the default consolidated run)
# ====================================================================

# uv run python experiments/ablations/exploration_strategy.py
# uv run python experiments/ablations/imputation_strategy.py

# ====================================================================
# Later: render figures from the saved ablation results
# ====================================================================

# uv run python experiments/plot_ablation_results.py \
#   --results-dir results/ablations \
#   --output results/figures/ablations \
#   --format pdf

echo "All ablations complete."
echo "  Results saved to: results/ablations/"
uv run python experiments/plot_ablation_results.py --results-dir results/ablations --output results/figures/ablations --format pdf

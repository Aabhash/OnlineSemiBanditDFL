#!/bin/bash
#SBATCH --job-name=dfl_ablations
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ABLATION_MODE="${ABLATION_MODE:-full}"
ABLATION_FIG_FORMAT="${ABLATION_FIG_FORMAT:-pdf}"
ABLATION_TUNING_DIR="${ABLATION_TUNING_DIR:-results/FINAL/tuning}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "Ablation mode: ${ABLATION_MODE}"
echo "Tuning dir: ${ABLATION_TUNING_DIR}"
echo "Figure format: ${ABLATION_FIG_FORMAT}"

# ====================================================================
# Main-text ablations
# ====================================================================

echo "=== Running main-text ablations ==="

uv run python experiments/ablations/kn_ratio_sweep.py --mode "$ABLATION_MODE" --no-plot
uv run python experiments/ablations/mechanism_ablation.py --mode "$ABLATION_MODE" --no-plot
uv run python experiments/ablations/feature_sharing.py --mode "$ABLATION_MODE" --no-plot
uv run python experiments/ablations/flip_set_collapse.py --mode "$ABLATION_MODE" --no-plot

# ====================================================================
# Appendix / supporting ablations
# ====================================================================

echo "=== Running appendix ablations ==="

uv run python experiments/ablations/context_variation.py --mode "$ABLATION_MODE" --no-plot
uv run python experiments/ablations/feature_robustness.py --mode "$ABLATION_MODE"
uv run python experiments/ablations/masking_dose_response.py --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR" --no-plot
uv run python experiments/ablations/beta_sensitivity.py --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR" --no-plot
uv run python experiments/ablations/lambda_reg_sweep.py --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR" --no-plot
uv run python experiments/ablations/imputation_samples.py --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR" --no-plot

# ====================================================================
# Optional deep-dive sweeps (not part of the default consolidated run)
# ====================================================================

# uv run python experiments/ablations/exploration_strategy.py
# uv run python experiments/ablations/imputation_strategy.py

# ====================================================================
# Render figures from the saved ablation results
# ====================================================================

echo "All ablations complete."
echo "  Results saved to: results/ablations/"
uv run python experiments/plot_ablation_results.py --results-dir results/ablations --output results/figures/ablations --format "$ABLATION_FIG_FORMAT"

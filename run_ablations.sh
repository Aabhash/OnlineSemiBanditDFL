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

echo "Ablation mode: ${ABLATION_MODE}"
echo "Tuning dir: ${ABLATION_TUNING_DIR}"
echo "Figure format: ${ABLATION_FIG_FORMAT}"

# ====================================================================
# Main-text ablations
# ====================================================================

echo "=== Running main-text ablations ==="

uv run python experiments/ablations/kn_ratio_sweep.py --mode "$ABLATION_MODE"
uv run python experiments/ablations/posterior_strategy_ablation.py --preset imputation --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR" --no-plot
uv run python experiments/ablations/architecture_ablation.py --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR"
uv run python experiments/ablations/decoupling_ablation.py --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR"

# ====================================================================
# Appendix / supporting ablations
# ====================================================================

echo "=== Running appendix ablations ==="

uv run python experiments/ablations/context_variation.py --mode "$ABLATION_MODE"
uv run python experiments/ablations/beta_sensitivity.py --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR"
uv run python experiments/ablations/posterior_strategy_ablation.py --preset exploration --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR" --no-plot
uv run python experiments/ablations/ts_scale_sweep.py --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR" --no-plot
uv run python experiments/ablations/lambda_reg_sweep.py --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR"
uv run python experiments/ablations/imputation_samples.py --mode "$ABLATION_MODE" --use-tuned-params --tuning-dir "$ABLATION_TUNING_DIR"

# ====================================================================
# Render figures from the saved ablation results
# ====================================================================

echo "All ablations complete."
echo "  Results saved to: results/ablations/"
uv run python experiments/plot_ablation_results.py --results-dir results/ablations --output results/figures/ablations --format "$ABLATION_FIG_FORMAT"

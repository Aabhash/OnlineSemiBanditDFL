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

uv run python experiments/ablations/kn_ratio_sweep.py
uv run python experiments/ablations/mechanism_ablation.py
uv run python experiments/ablations/feature_sharing.py
uv run python experiments/ablations/flip_set_collapse.py

# ====================================================================
# Appendix / supporting ablations
# ====================================================================

echo "=== Running appendix ablations ==="

uv run python experiments/ablations/context_variation.py
uv run python experiments/ablations/feature_robustness.py

# ====================================================================
# Optional deep-dive sweeps (not part of the default consolidated run)
# ====================================================================

# uv run python experiments/ablations/exploration_strategy.py
# uv run python experiments/ablations/imputation_strategy.py

echo "All ablations complete."
echo "  Data + figures: results/ablations/"

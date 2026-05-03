# Paper Result Archive

This directory collects the result artifacts used by the current paper draft.
Files are copied here from the working `results/FINAL` tree so existing scripts
and historical paths remain intact.

## Contents

- `eval_online_10k/`: final online evaluation metrics and saved trajectories
  used for the main regret table, regret trajectories, regret scaling,
  wall-clock plot, and saved-trajectory diagnostics.
- `tuning/`: final hyperparameter tuning outputs, summaries, significance
  analyses, and Optuna database used for the tuning details and selected
  BayesianSPO posterior-completion counts.
- `ablations/`: raw result directories for the ablations currently reported in
  the paper:
  - `architecture`
  - `decoupling`
  - `imputation`
  - `exploration`
  - `ts_scale`
  - `exact2_method_contrast`
  - `imputation_samples`
  - `kn_ratio`
  - `beta_sensitivity`
  - `lambda_reg_sweep`
  - `context_variation`
  - `pmin_coverage`
  - `contested_diagnostics_from_trajectories`
- `figures/result_figures/`: generated result figures currently included from
  `paper_neurips_2026/result_figures`.
- `figures/exact2_m_sweep/`: generated Exact-2 mechanism figures currently
  included from `paper_neurips_2026/figures/exact2_m_sweep`.

## Regeneration Pointers

Core paper result figures can be regenerated from:

```bash
uv run python experiments/plot_paper_core_results.py \
  --trajectories results/paper/eval_online_10k/metrics/trajectories.pkl \
  --canonical-regrets results/paper/eval_online_10k/metrics/final_eval_regrets.json \
  --summary-csv results/paper/eval_online_10k/metrics/final_eval_summary.csv \
  --output paper_neurips_2026/result_figures
```

Ablation figures can be regenerated from:

```bash
uv run python experiments/plot_ablation_results.py \
  --results-dir results/paper/ablations \
  --output paper_neurips_2026/result_figures \
  --format pdf
```

## Exclusions

The archive intentionally excludes stale or currently unused result directories,
including removed feature-sharing ablations, legacy unit-weight knapsack sweeps,
and superseded fixed-grid beta results.

# Figure 10: matched epsilon-greedy control

## Decision, 26 October 2026

Initial decision: add `DecoupledSPO-Eps` directly to Figure 10 as a fifth row, with mean ± sample standard deviation labels. Retain the existing four rows and their archived control values. No experiments were rerun and no table was added.

Current decision: at the user's follow-up request, remove the numerical annotations while retaining the new row and all plotted values. The caption no longer describes standard-deviation labels. The underlying means and sample SDs remain in the adjacent data file.

Layout update: the user subsequently requested `DecoupledSPO-Eps` as the bottom row, below all coupled-head variants. The plot order and the paper's description of the row position now reflect this.

The figure is `paper_neurips_2026/result_figures/ablation_design_summary_linear.pdf`, mirrored in `results/paper/figures/result_figures/`. The adjacent `ablation_design_summary_linear_data.json` records every plotted comparison, source path and SHA-256, seed identity, horizon, final regret, mean, and sample SD.

## Evidence

The three decoupling archives are `results/paper/ablations/decoupling/{knapsack_100,shortest_path_20x20,portfolio_32}/results.pkl`. Each contains `DecoupledSPO-UCB`, `DecoupledSPO-Eps`, and `CoupledSPO-Eps-x{0.5,1,2}` at 10,000 rounds for evaluation seeds 100–109. The metric is final cumulative realized regret; the recorded spread is the sample SD across seeds (`ddof=1`), not a confidence interval or the SD of paired differences.

The DecoupledSPO-Eps summaries, shown as annotations only in the initial version, are:

- Knapsack 100: 83.9 ± 3.7 thousand (unscaled mean 83852.99518099455; SD 3694.633454019872).
- SP 20×20: 31.8 ± 2.5 thousand (unscaled mean 31797.893657351437; SD 2481.210039081455).
- Portfolio: 125.2 ± 6.9 (unscaled mean 125.1941914151317; SD 6.939943858081415).

The matched Portfolio UCB mean is 69.91297308750202, from the same decoupling archive. The main-evaluation mean 68.6300252454 is a separate result and is not substituted into this comparison. Original raw results remain preserved.

## Protocol and limits

The current runner, `experiments/ablations/decoupling_ablation.py`, gives the two analytical-head variants the same tuned parameters and changes UCB to epsilon-greedy for `DecoupledSPO-Eps`. The epsilon-greedy wrapper uses the same epsilon settings as the deterministic coupled variants, with the default schedule 0.1 × max(0, 1 − t/T). The coupled learning-rate multipliers are 0.5, 1, and 2. The paper retains the distinction between this three-point diagnostic sweep and the main method's 50-trial tuning budget.

These compact archives store final metrics and horizons, but no per-run hyperparameter snapshot or source revision. The current runner documents the intended matched protocol; the saved artifacts do not independently establish the complete historical configuration.

## Interpretation and verification

Ratios of mean regret support the existing ranges: DecoupledSPO-Eps incurs 1.4–1.8 times the UCB control's regret; the best coupled variant incurs 2.8–4.3 times DecoupledSPO-Eps regret. The new row makes the comparison under the same exploration rule visible. It supports an advantage for the analytical head within this diagnostic sweep.

All 15 displayed comparisons were checked against their raw archives, including all seed-level values, horizons, source hashes, means, and sample SDs. The new labels and layout were visually checked. Regenerate only this figure with:

```bash
uv run python experiments/plot_ablation_results.py \
  --results-dir results/paper/ablations \
  --output paper_neurips_2026/result_figures \
  --ablations ablation_design_summary --format pdf
```

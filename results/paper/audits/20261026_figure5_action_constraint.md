# Figure 5 and Table 3 action-constraint audit

Recorded 26 October 2026. Repository HEAD at inspection: `90a19726b45a1256b509505e694750058c6e5f36`. The working tree contains other changes; this audit does not treat HEAD as a complete execution snapshot. The [JSON companion](20261026_figure5_action_constraint.json) records settings, file hashes, and the inspected action counts.

**Current publication status, 26 October 2026:** the user subsequently chose the original unrestricted top-two diagnostic and the earlier Figure 5 with accuracy labels. The original seed-0 cache and width study are canonical again. The completed grouped reruns below are **withdrawn from the active paper**, with their results and validation preserved in the [withdrawn archive](../../../archive/20261026_one_per_pair_diagnostic_withdrawn/README.md). This latest decision supersedes both the grouped-rerun authorization as a publication choice and its interim use in Figure 5 and Table 3; it does not invalidate the recorded grouped measurements.

## Evidence

The implemented decision rule selects **any two of the four arms** with largest supplied values. The noiseless true-value oracle selects **one arm from each pair**, but the implemented feasible set does not require that restriction.

At the original inspection, [decision_nuisance_top2_helpers.py](../../../experiments/ablations/decision_nuisance_top2_helpers.py) defined `Top2Solver.solve` at lines 84–91 using `action[np.argsort(values)[-2:]] = 1.0`. The environment installed that solver at line 112, and both learners received its callable at line 180. The only uncommitted helper change then inspected added `gradient_mode` wiring; it did not alter the solver. These line references describe that inspection, before the later grouped experiment and its withdrawal.

Historical source supports the same distinction:

- At `cd19aa0^`, `notebooks/feature_representation_analysis_helpers.py:4255–4263` defines an unrestricted `Exact2Solver`. A separate `compute_oracle_decision_exact2` function at lines 4301–4306 chooses one winner per pair from the context signs.
- Commit `cd19aa0` (3 May 2026) introduces the standalone diagnostic helper with unrestricted top-two selection.
- Commit `472079a` (5 May 2026) adds the figure wording “Pick one from {0,1}. Pick another from {2,3}.” It does not change the solver.
- Commit `466369a` (27 July 2026) adds nuisance-strength support and the row-share helper, leaving the solver unchanged.

At the original history inspection, before the authorized grouped experiment, no grouped-feasibility solver version or corresponding grouped-training result was found in the traced helper, generator, and notebook history. This is a search finding, not a claim that no such artifact could exist elsewhere.

### Table 3: explicit protocol and saved actions

The [width-study configuration](../../exploratory/quadrant_capacity_20261021/config.json) explicitly records: “Existing exact, unrestricted top-two selection; a fresh agent for each cell.”

Settings: widths \(q\in\{2,3,6,16\}\), training seeds 200–209, 10,000 rounds, stream seed 42, test seed 99, 500 held-out contexts, \(\beta=1\), \(\lambda=0.1\), learning rate 0.03, \(M=8\), completion scale 1, noise width 0, nuisance strength 2, affine features, post-observation detached gradients (`gradient_mode=current`), SPO masking enabled and MSE masking disabled. Both full and partial feedback are included. Hyperparameters are not retuned across widths. The four cells within each width and seed share initial feature weights. Probe accuracy uses all \(q\) features and the original five stratified folds; reported variability is across training seeds.

The runner [quadrant_capacity.py](../../../experiments/exploratory/quadrant_capacity.py) records this protocol at lines 33–44, uses the shared diagnostic environment/agent helpers at lines 66–85, and saves actual played actions at lines 123–128.

The audit directly inspected four saved runs at **width 2, seed 200**. For each saved action \(a_t\), a grouped-constraint violation is counted when
\[
\left|a_{t,0}+a_{t,1}-1\right|>10^{-6}
\quad\text{or}\quad
\left|a_{t,2}+a_{t,3}-1\right|>10^{-6}.
\]
These are valid actions for unrestricted exact-two selection, not solver-feasibility violations under the implemented task.

| Saved NPZ | Feedback | Same-pair action rounds / 10,000 |
|---|---|---:|
| [BayesianMSE_full.npz](../../exploratory/quadrant_capacity_20261021/q2/seed_200/BayesianMSE_full.npz) | Full | 1 |
| [BayesianMSE_partial.npz](../../exploratory/quadrant_capacity_20261021/q2/seed_200/BayesianMSE_partial.npz) | Partial | 7,530 |
| [BayesianSPO_full.npz](../../exploratory/quadrant_capacity_20261021/q2/seed_200/BayesianSPO_full.npz) | Full | 1 |
| [BayesianSPO_partial.npz](../../exploratory/quadrant_capacity_20261021/q2/seed_200/BayesianSPO_partial.npz) | Partial | 5,827 |

All four first played actions are \((0,0,1,1)\), at round 1 using one-based indexing. This directly rules out one-per-pair feasibility for those runs. The explicit configuration documents the unrestricted rule for the full width study; the action-count inspection is not an exhaustive count over all 160 runs.

### Figure 5: canonical seed-0 cache and limits

Canonical sources:

- [Learned-feature cache](../ablations/decision_nuisance_top2/cache/decision_nuisance_top2_m8_r10000_seed0.npz)
- [Cache metadata](../ablations/decision_nuisance_top2/cache/decision_nuisance_top2_m8_r10000_seed0.json)

The cache contains 500 test contexts, their true values and oracle labels, four \(500\times2\) learned-feature arrays, and four \(5\times2\) learned weight matrices. Its keys are `x_test`, `decisions_test`, `values_test`, `h_spo_full`, `h_mse_full`, `h_spo`, `h_mse`, `W1_spo_full`, `W1_mse_full`, `W1_spo`, `W1_mse`.

The metadata records seed 0, 10,000 rounds, partial-feedback SPO cumulative regret 2952.1146480526672, and partial-feedback MSE cumulative regret 6190.106784258791. The source and manuscript protocol specify width 2, \(M=8\), and the settings above, but the metadata does not itself contain a complete configuration.

The observed cache modification time is 5 May 2026, 08:34:50. File time does not identify the exact execution revision. There are **no saved training actions, solver identity, or code hash** in this cache. All traced source versions use unrestricted top-two selection, but the cache alone cannot independently certify the historical training solver. Its oracle labels do not resolve the question because the two feasible sets agree on the noiseless oracle for this DGP.

SHA256:

| Source | SHA256 |
|---|---|
| Figure 5 NPZ | `17c700b593ed26fdf250eb1ce7b9ed6d416955d18967cfa27cd384ed6d77e87c` |
| Figure 5 JSON metadata | `59e53a173122d0b9b04bc430952196911cde8ac4dd1bfb540c0b4614782ec9af` |
| Table 3 width-study configuration | `4324e37da7943266872f9b1fad9012302fb33f593fa53a1462a9b48320b7b5b4` |

The JSON companion additionally records SHA256 values for all four inspected action NPZ files.

## Interpretation

With noise width zero, write the common offset as
\[
a(x)=3+\frac{2x_2+2x_3+2x_4}{\sqrt5}.
\]
The true values are
\[
a(x)\mathbf1+\frac{1}{\sqrt5}(x_0,-x_0,x_1,-x_1).
\]
Except at zero-valued decision-coordinate ties, the two largest coordinates are \(a+|x_0|/\sqrt5\) and \(a+|x_1|/\sqrt5\), one from each pair. Continuous Gaussian contexts make the tie exception probability zero.

Thus the phrase “the oracle picks one from each pair” is correct for this noiseless DGP. The phrase “each feasible action must select one from each pair” describes a different problem. Arbitrary learned predictions, optimistic coefficients, completed targets, and SPO-adjusted costs need not retain the true-value structure. Restricting their optimization calls to one per pair can therefore change actions, feedback, gradients, and learned representations. Existing unrestricted training results cannot simply be relabeled as grouped-constraint experiments.

## Dated status and decision

**Superseded status, 26 October 2026 — user clarification pending.** The user was asked whether to retain the existing results with wording that identifies one-per-pair selection as the oracle rule, or to enforce that restriction as a feasible-set constraint and rerun the affected experiments.

**Superseded publication decision, 26 October 2026 — grouped-constraint reruns authorized.** The user explicitly chose to enforce one arm from \(\{0,1\}\) and one arm from \(\{2,3\}\), and requested fresh Figure 5 and Table 3 experiments under that constraint. The old unrestricted results and the evidence above remained preserved; new results were identified separately rather than presented as reproductions of the unrestricted runs. The later decision below withdraws the grouped results from the paper.

At the authorization-stage audit update, this audit agent had not changed solvers or started training. Solver changes, the authorized reruns, and the interim manuscript updates were subsequently coordinated by the primary agent and are recorded below.

## Completed grouped reruns, 26 October 2026 — withdrawn from the paper

All 164 authorized grouped runs are complete: four Figure 5 runs at seed 0 and width 2, and 160 Table 3 runs at widths 2, 3, 6 and 16 with seeds 200–209. The [Figure 5 configuration](../../../archive/20261026_one_per_pair_diagnostic_withdrawn/figure5/config.json) and [width-study configuration](../../../archive/20261026_one_per_pair_diagnostic_withdrawn/width_study/config.json) retain the stated 10,000 rounds, stream seed 42, test seed 99, 500 held-out contexts and hyperparameters, with no retuning. In these grouped runs, every optimization call enforced one arm per pair. The historical unrestricted files and their provenance above remain unchanged.

The [archived report script](../../../archive/20261026_one_per_pair_diagnostic_withdrawn/reporting_source_snapshot/experiments/ablations/grouped_representation_report.py) passed all checks in the [validation report](../../../archive/20261026_one_per_pair_diagnostic_withdrawn/report/validation_report.json). It checked 1,640,000 training actions and 82,000 held-out greedy actions, with zero nonbinary or one-per-pair violations. It also reproduced training regrets from the fixed NumPy data stream and saved actions, checked array-based metrics against saved rows, verified matched initialization in all 41 seed/width blocks and identical held-out data, and checked source snapshots. All 80 learned matrices at widths 6 and 16 have rank 5 under the original tolerance `norm(W1, 2) * 1e-6`. Probe accuracy uses the saved five-fold scores; this report does not refit the probes.

For the Figure 5 seed, semi-bandit decision shares are 55.63% for BayesianSPO and 31.09% for BayesianMSE. Their saved probe accuracies are 50.2% and 53.2%, respectively. Under full feedback the shares are 98.75% and 58.24%, with probe accuracies 97.6% and 60.4%. [Figure 5 metrics](../../../archive/20261026_one_per_pair_diagnostic_withdrawn/report/figure5_metrics.json) include the raw row norms, shares and per-cell result paths.

Across the ten Table 3 seeds at width 2 with semi-bandit feedback, BayesianSPO has decision share 54.4 ± 10.9%, compared with 43.8 ± 12.8% for BayesianMSE. Their mean probe accuracies are similar, 54.9 ± 3.7% and 54.8 ± 2.8%. For BayesianSPO at widths 3, 6 and 16, mean probe accuracy is 93.8%, 90.3% and 94.5%. In the interim grouped-task manuscript, these values replaced the unrestricted-study statement that it exceeds 92% at every width of at least 3. That replacement is now superseded with the grouped results. The full-feedback table entries are unchanged at the displayed precision. All eight new rows are in [summary.csv](../../../archive/20261026_one_per_pair_diagnostic_withdrawn/report/summary.csv) and [table_rows.tex](../../../archive/20261026_one_per_pair_diagnostic_withdrawn/report/table_rows.tex); [summary.json](../../../archive/20261026_one_per_pair_diagnostic_withdrawn/report/summary.json) retains unrounded statistics and all run identities.

Superseded decision on 26 October 2026: use the grouped reruns for Figure 5 and Table 3 and revise their empirical interpretation. Greater decision share does not imply greater probe accuracy. The unrestricted study was temporarily superseded for that grouped-task presentation; the latest decision below restores it.


## Latest user decision, 26 October 2026 — restore unrestricted results

The user reversed the grouped-experiment publication choice and explicitly requested the prior unrestricted results and the old Figure 5 with accuracy labels. This is the controlling publication decision. Figure 5 again uses the original [seed-0 cache](../ablations/decision_nuisance_top2/cache/decision_nuisance_top2_m8_r10000_seed0.npz) and its [metadata](../ablations/decision_nuisance_top2/cache/decision_nuisance_top2_m8_r10000_seed0.json); Table 3 again uses [the original width study](../../exploratory/quadrant_capacity_20261021/config.json). The task permits any two arms. The noiseless oracle selects one arm from each pair because of the true-value construction, not because the feasible set imposes that restriction. The historical cache limitations documented above still apply.

The 164 grouped runs, source snapshots, reports, validation results and interim presentation are retained in [archive/20261026_one_per_pair_diagnostic_withdrawn](../../../archive/20261026_one_per_pair_diagnostic_withdrawn/README.md). They are withdrawn from the active paper and must not supply Figure 5, Table 3 or their reported numbers. Their measurements and validation remain historical evidence for the different, grouped feasible set. The original unrestricted cache and width-study data were not overwritten by the rerun or this reversal. Historical absolute and repository-relative paths inside archived execution manifests remain unchanged as execution provenance; current access uses the archive location linked here.

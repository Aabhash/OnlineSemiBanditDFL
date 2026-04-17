# Feature Representation Analysis Findings

## Goal

This notebook asks three questions on a small controlled problem:

- under `full feedback`, does `BayesianSPO` learn a more decision-useful representation than `BayesianMSE`?
- under `partial feedback`, what exactly breaks?
- can a more structured exploration rule repair that failure?

## Synthetic Setup

The world is intentionally simple:

- 4 items split into 2 symmetric pairs: `(0,1)` and `(2,3)`
- the correct action always picks one winner from each pair
- `x0` controls the winner inside `(0,1)`
- `x1` controls the winner inside `(2,3)`
- `x2,x3,x4` change values a lot but do not change the within-pair winner

So:

- `x0,x1` are the true decision coordinates
- `x2,x3,x4` are predictive nuisance coordinates

## Which Figures Matter Most

The most useful figures are now:

- `action-space heatmap`
- `regret per round`
- `prediction error over training`

The latent scatter and loading plots are still useful, but they are secondary. The main mechanism story is now clearer in action space and in the training curves than in the final 2D latent geometry.

## Full Feedback

This is the cleanest and strongest result.

### Summary

- `BayesianSPO` gets lower regret than `BayesianMSE`
- `BayesianSPO` has better decision-probe accuracy
- `BayesianSPO` is much closer to the oracle decision subspace
- `BayesianMSE` predicts values much better and tracks nuisance/value structure much more strongly

### Key numbers

- final regret
  - `BayesianSPO = 147.8`
  - `BayesianMSE = 197.6`
- decision probe
  - `BayesianSPO = 0.672`
  - `BayesianMSE = 0.580`
- greedy decision accuracy
  - `BayesianSPO = 0.574`
  - `BayesianMSE = 0.530`
- cost correlation
  - `BayesianSPO = 0.540`
  - `BayesianMSE = 0.9999`
- alignment angle
  - `BayesianSPO = 48.5°`
  - `BayesianMSE = 89.9°`

### Figure read

- The `regret per round` figure shows `BayesianSPO` staying lower through training.
- The `prediction error` figure shows the opposite: `BayesianMSE` ends with much lower value-prediction error.
- The `action-space heatmap` shows both methods mostly staying on the oracle-supported cross-pair actions, with `BayesianSPO` a bit cleaner.

### Main conclusion

Under full feedback, the intended mechanism is visible:

- `BayesianSPO` is better for decisions
- `BayesianMSE` is better for prediction

## Partial Feedback (Masked)

This is where the clean representation story weakens.

### Summary

- `BayesianSPO` still gets slightly lower regret than `BayesianMSE`
- but the final learned features are no longer cleanly better in the same way
- both methods start making structurally wrong action choices

### Key numbers

- final regret
  - `BayesianSPO = 382.9`
  - `BayesianMSE = 409.3`
- decision probe
  - `BayesianSPO = 0.450`
  - `BayesianMSE = 0.478`
- greedy decision accuracy
  - `BayesianSPO = 0.024`
  - `BayesianMSE = 0.070`
- pair accuracies
  - `BayesianSPO`: `pair_01 = 0.830`, `pair_23 = 0.512`
  - `BayesianMSE`: `pair_01 = 0.678`, `pair_23 = 0.674`
- cost correlation
  - `BayesianSPO = 0.814`
  - `BayesianMSE = 0.878`
- alignment angle
  - `BayesianSPO = 87.5°`
  - `BayesianMSE = 86.6°`

### Figure read

- The `action-space heatmap` is the most important plot here.
- Under partial feedback, both agents start putting mass on the wrong same-pair actions.
- So the failure is not just “noisier features.” It is also a structural collapse of the policy into the wrong region of action space.

### Main conclusion

Partial feedback breaks more than representation quality:

- it changes what actions are explored
- and it creates a strong pair asymmetry, especially for `BayesianSPO`

## Pair-Balanced Partial Feedback

In this notebook-only probe, both agents are forced to pick one winner from each pair.

### Key numbers

- final regret
  - `BayesianSPO = 408.0`
  - `BayesianMSE = 323.6`
- greedy decision accuracy
  - `BayesianSPO = 0.240`
  - `BayesianMSE = 0.378`
- pair accuracies
  - `BayesianSPO`: `0.480`, `0.534`
  - `BayesianMSE`: `0.502`, `0.708`

### Figure read

- The heatmap shows that the same-pair collapse is removed by construction.
- But the `regret per round` and `prediction error` figures show that this is not a good general fix for `BayesianSPO`.
- It helps `BayesianMSE` more than `BayesianSPO`.

### Main conclusion

`Pair-balanced` is a useful diagnosis:

- it confirms that action-structure matters
- but it is not the best algorithmic direction, especially for SPO

## Disagreement + Coverage Partial Feedback

This is the most promising exploration rule in the notebook.

For each round, the agent:

- samples plausible value vectors from its current mean and uncertainty
- solves the black-box problem on those samples
- measures which components the sampled actions disagree on
- upweights components that are under-observed
- scores candidate actions by predicted value plus disagreement × coverage bonus

This does not hard-code the pair structure into the action itself.

### Key numbers

- final regret
  - `BayesianSPO = 186.7`
  - `BayesianMSE = 411.2`
- greedy decision accuracy
  - `BayesianSPO = 0.432`
  - `BayesianMSE = 0.086`
- pair accuracies
  - `BayesianSPO`: `0.772`, `0.614`
  - `BayesianMSE`: `0.676`, `0.666`
- cost correlation
  - `BayesianSPO = 0.460`
  - `BayesianMSE = 0.878`

### Figure read

- The `action-space heatmap` shows the key effect:
  - `BayesianSPO` is pushed back toward oracle-supported cross-pair actions
  - `BayesianMSE` still spends most of its time on the wrong same-pair actions
- The `regret per round` figure shows a large gain for `BayesianSPO`
- The `prediction error` figure shows something important:
  - `BayesianSPO` gets much better decision performance without getting better value prediction

### Main conclusion

This is the strongest current result:

- `disagreement + coverage` helps `BayesianSPO` a lot
- it does not rescue `BayesianMSE`
- the gain is mostly from better information collection and better action trajectories, not from cleaner final value prediction

## Current Overall Read

The notebook now supports three main claims:

1. Under `full feedback`, `BayesianSPO` learns a more decision-useful representation than `BayesianMSE`, even though `BayesianMSE` predicts values better.
2. Under `partial feedback`, the main failure is not only degraded latent structure. It is structural action collapse into the wrong part of the action space.
3. A more structured exploration rule can repair much of that failure for `BayesianSPO`, and `disagreement + coverage` is the best current candidate.

## Why This Matters

The notebook now points to a real algorithmic contribution:

- the bottleneck in partial-feedback DFL is not just the loss
- it is also how the policy collects information
- a decision-aware exploration rule can substantially improve `BayesianSPO` without requiring solver differentiation or problem-specific pair constraints

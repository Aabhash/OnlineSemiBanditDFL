# BLL-Online, BayesianMSE and the shared-prior interpretation

Reviewed 26 October 2026. This is an audit of saved results and implementation, with no new training runs. The accompanying [JSON](20261026_bll_mse_shared_prior.json) contains seed-level endpoints, settings, source paths and source hashes.

## Evidence

The metric is final cumulative realized regret. Ratios below divide BLL-Online's mean by BayesianMSE's mean. Synthetic results come from `results/paper/eval_online_10k/metrics/trajectories_<dataset>.pkl`, with seeds 100 through 109 and 10,000 rounds. Knapsack and shortest-path values are in thousands of cost units. Portfolio is unscaled. Energy results come from `results/rebuttal/energy_scheduling_task_start/semi_bandit_all_methods_30seed/per_seed.json`, with seeds 0 through 29 on the same 395 days, and are shown in millions of cost units. All uncertainty values in this table are sample standard deviations across learner seeds.

| Benchmark | BLL-Online | BayesianMSE | Ratio |
|---|---:|---:|---:|
| Knapsack 32 | 19.924 +/- 2.444 | 18.464 +/- 1.486 | 1.079 |
| Knapsack 100 | 66.915 +/- 7.208 | 54.619 +/- 4.804 | 1.225 |
| SP 8x8 | 8.584 +/- 0.916 | 6.521 +/- 1.104 | 1.316 |
| SP 20x20 | 51.412 +/- 2.355 | 26.226 +/- 2.568 | 1.960 |
| Portfolio | 341.639 +/- 41.929 | 101.485 +/- 9.755 | 3.366 |
| Energy | 125.167 +/- 5.645 | 101.135 +/- 12.173 | 1.238 |

The implementations share an affine feature-map class, observed-arm Bayesian updates, UCB action selection, and clipped Adam. Both compute their feature gradient after updating the observed heads. They use different initial feature weights even for the same integer seed because they split the initialization key differently.

Let the updated head mean be $\mu_i$, its covariance be $P_i$, the feature vector be $h$, and the observed residual be $e_i=\mu_i^\top h-c_i$. The actual gradients are

$$
g_h^{\mathrm{MSE}}=\frac{2}{n}\sum_{i\in S}e_i\mu_i,
\qquad
g_h^{\mathrm{BLL}}=\frac{1}{\nu}\left(\sum_{i\in S}e_i\mu_i+\sum_{i=1}^{n}P_i h\right).
$$

The BLL factory sets `mask_grad=False` and `posterior_complete_missing=True` in `experiments/tuning/objectives.py`. Missing-arm targets equal their current posterior-mean predictions. Their residuals are zero, but their covariance terms remain in the feature gradient. The gradient implementations are `WeberBLLAgent._marginalized_grad_h` in `src/agents/contextual/weber_bll.py` and `BayesianMSEAgent._compute_mse_grad_h` in `src/agents/contextual/neural_agents.py`.

BLL's `noise_var`, denoted by $\nu$, divides its entire feature gradient. It does not enter the Bayesian sufficient-statistic updates or UCB covariance. MSE normalizes its squared-error gradient by the total number of arms. A fixed multiplier alone need not change Adam's step proportionally, so these formulas do not establish an arm-count multiplier on the effective learning rate.

The methods were tuned separately. The stored Portfolio trajectories record prior precision 0.809912 for BLL versus 0.0104404 for MSE, and initial learning rates 0.0003 versus 0.001. On SP 20x20, BLL uses width 32 and precision 2.104520, while MSE uses width 16 and precision 0.367263. The JSON preserves all recorded settings. The main ablations do not remove BLL's covariance term while holding the remaining settings fixed.

Our BLL baseline adapts the posterior-aware training idea in [Weber et al. (2018)](https://www.bayesiandeeplearning.org/2018/papers/117.pdf) to this Gaussian regression setting. The cited paper studies expectations over the last-layer distribution and a regularizer for past information. The repository baseline should be identified as an adaptation rather than an exact reproduction of that training procedure.

## Interpretation

The BLL covariance term is the gradient of $\tfrac12 h^\top(\sum_i P_i)h$ when the heads are held fixed. It penalizes feature directions with high posterior uncertainty. Unobserved arms can contribute this penalty without providing a new cost label. That pressure can compete with fitting observed costs and can change later UCB decisions. It is a plausible contributor to BLL's larger regret, not an isolated causal explanation.

For the representation interpretation, augment the context and feature matrix as $x^+=(x^\top,1)^\top$ and $B_\psi=[W_1;b_1^\top]$. Independent head priors $\theta_i\sim\mathcal N(0,\lambda^{-1}I)$ imply that the raw affine coefficients $a_i=B_\psi\theta_i$ have common covariance $\lambda^{-1}B_\psi B_\psi^\top$. The arms remain independent conditional on the feature map. Learning the common map changes this prior covariance and the corresponding coefficient regularizer. With a full-row-rank map, the set of affine predictors remains unchanged; with a bottleneck, coefficients are also restricted to a common subspace. This connects to [multi-task feature learning](https://home.ttic.edu/~argyriou/papers/mtl_feat.pdf).

The covariance identity is exact for a fixed map. The learner updates the map with detached SPO+ gradients and retains historical feature coordinates in its head statistics. It does not perform exact marginal-likelihood learning of a prior or refit all historical observations under the latest map. A benefit from decision-focused adaptation of the common prior is therefore an interpretation of the empirical results.

## Manuscript decision

The camera-ready method section now connects decision-focused representation learning to a shared prior covariance. The results paragraph presents this as a possible contributor to the gains and states that the separately tuned comparisons do not isolate it from optimization and exploration. Appendix A gives the covariance and regularizer identities, including the fixed-map limitation. Appendix D.1 records the exact BLL and MSE gradients and the missing-arm covariance contributions. Existing claims that the main methods share the same feature dimension were changed to the same feature-map class, since their widths are tuned separately. No experiment results or learning code were changed.

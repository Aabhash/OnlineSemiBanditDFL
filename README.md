# Decision-Focused Learning for Combinatorial Semi-Bandits

This repository implements **BayesianSPO**, a decision-focused learning method for online combinatorial semi-bandit problems. The main contribution is **Bayesian decoupling for masked decision-focused learning**: an analytically updated per-arm Bayesian last-layer head provides UCB exploration and stable imputations, while masked SPO+ updates train the shared feature representation only through observed arms.

## Problem Setting

**Online Combinatorial Semi-Bandits with Shared Context:**
- At each round, observe shared context $x_t \in \mathbb{R}^d$, predict cost vector $\hat{c}_t = f_\theta(x_t)$
- Solve combinatorial optimization $w^*_t = \arg\min_{w \in \mathcal{W}} \hat{c}_t^\top w$ (knapsack, shortest path)
- Observe costs only for selected arms (semi-bandit feedback, ~5-50% observation rate)
- Goal: Minimize cumulative regret over $T$ rounds

**Challenge:** SPO+ requires the full cost vector, but 50-95% is unobserved — the *imputation bottleneck*.

**Solution:** BayesianSPO separates prediction/imputation from decision-focused representation learning. The Bayesian head is updated from observed semi-bandit feedback and supplies uncertainty-aware action selection plus completions for unobserved arms; the SPO+ gradient is masked before feature updates so imputed missing arms do not directly corrupt the representation. Posterior-imputed variants optionally average the masked SPO+ gradient over sampled completions.

## Installation

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Quick Start

```bash
# Main pipeline: tune → evaluate → plot
bash tune_online.sh     # Online HP tuning (no wandb)
bash run_online.sh      # Main online evaluation
bash run.sh             # Combined wrapper: online evaluation + post-processing

bash run_ablations.sh   # Ablation sweeps
```

Or run individual steps:

```bash
# Tune HPs for one dataset
uv run python experiments/tune_hyperparams.py --all --dataset knapsack_32 --feedback semi_bandit --n-rounds 1000 --analyze

# Evaluate online agents
uv run python experiments/evaluate_online.py --datasets knapsack_32 --feedback semi_bandit --full --output_dir results/eval_online

# Generate plots from saved results (no re-running)
uv run python experiments/plot_eval_results.py --input results/eval_online/metrics --format pdf
uv run python experiments/compute_analysis.py
```

## Datasets

| Dataset | Arms | Selection | Obs. rate | Description |
|---------|------|-----------|-----------|-------------|
| `knapsack_32` | 32 | ~6-7 | ~19-22% | Heterogeneous weights, capacity 20 |
| `knapsack_100` | 100 | ~6-7 | ~19-22% | Heterogeneous weights, larger-scale knapsack |
| `shortest_path_8x8` | 112 | ~14 | ~12% | 8×8 grid shortest path |
| `shortest_path_20x20` | 760 | ~38 | ~5% | 20×20 grid shortest path |
| `portfolio_32` | 32 | 5 | ~16% | Cardinality-constrained binary portfolio |
| `topk_2arm` | 2 | 1 | 50% | 2-arm toy (sigmoid costs) |
| `topk_4arm` | 4 | 1 | 25% | 4-arm toy (sigmoid costs) |
| `topk_8arm` | 8 | 2 | 25% | 8-arm toy (sigmoid costs) |
| `topk_2arm_linear` | 2 | 1 | 50% | 2-arm toy (linear costs, well-specified) |
| `topk_4arm_linear` | 4 | 1 | 25% | 4-arm toy (linear costs, well-specified) |

## Agents

### Online (per-round training)

| Agent | Loss | Description |
|-------|------|-------------|
| **BayesianSPO** | SPO+ | Main method: per-arm Bayesian UCB/imputation with masked SPO+ feature updates; `imputation_samples=1` recovers the deterministic posterior-mean path and `M>1` averages posterior-sampled completions |
| BayesianPFY | PFY | Perturbed Fenchel-Young loss variant |
| BayesianMSE | MSE | Prediction-focused baseline |
| NeuralUCB | MSE | Neural network with NTK uncertainty |
| WeberBLL | Gaussian LL | Weber et al. marginal likelihood |
| LinUCB | — | Linear UCB (no feature learning) |
| TS | — | Thompson Sampling |
| EpsilonGreedy | MSE | ε-greedy with neural network |
| Random | — | Uniform random selection |

## Pipeline

```
tune_online.sh → results/tuning/           (online Optuna DB + best HP JSON)
run_online.sh  → results/eval_online/      (main online agent data)
run.sh         → results/compute/          (online wrapper + compute analysis)
                 results/figures/          (regret + diagnostic plots)
run_ablations.sh → results/ablations/      (ablation data + per-ablation figures)
```

### Re-plot without re-running

```bash
uv run python experiments/plot_eval_results.py --format pdf
uv run python experiments/compute_analysis.py
```

## Ablations

### Main text

| Ablation | Sweeps | Script |
|----------|--------|--------|
| k/n ratio | capacity: [4,8,12,16,20,24,28] | `ablations/kn_ratio_sweep.py` |
| Mechanism ablation | MSE+UCB, SPO+Greedy, SPO+UCB, SPO-unmasked, SPO+DFE, SPO+DI | `ablations/mechanism_ablation.py` |
| Feature sharing | hidden_dim: [1,4,8,16,32,64] | `ablations/feature_sharing.py` |
| Flip set collapse | Fixed vs changing context dynamics | `ablations/flip_set_collapse.py` |

### Appendix / support

| Ablation | Sweeps | Script |
|----------|--------|--------|
| Context variation | noise: [0,0.05,0.1,0.25,0.5,1] | `ablations/context_variation.py` |
| Feature robustness | Perturbation sensitivity | `ablations/feature_robustness.py` |
| Masking dose-response | missing-arm gradient weight: [0,0.05,0.1,0.25,0.5,1] | `ablations/masking_dose_response.py` |
| Beta sensitivity | UCB scale around the tuned BayesianSPO value | `ablations/beta_sensitivity.py` |
| Lambda sensitivity | Bayesian prior precision / regularization sweep | `ablations/lambda_reg_sweep.py` |
| Imputation samples | M: [1,4,8,12] using final tuned BayesianSPO params | `ablations/imputation_samples.py` |

### Optional deep dives

| Ablation | Sweeps | Script |
|----------|--------|--------|
| Exploration strategy | Greedy, UCB, DFE, DI | `ablations/exploration_strategy.py` |
| Imputation strategy | Masked, unmasked, DI-K | `ablations/imputation_strategy.py` |

Default ablation workflow:

```bash
./run_ablations.sh
```

`run_ablations.sh` runs full ablations by default, writes result files under `results/ablations/`, and renders figures under `results/figures/ablations/`. For a smoke run, use `ABLATION_MODE=quick ./run_ablations.sh`.

## Project Structure

```
src/
  agents/
    agent_base.py           # AgentBase ABC (universal interface)
    base.py                 # Episode runners (pregenerate, run_agent_on_episode)
    contextual/             # All agent implementations
  envs/                     # Environments (knapsack, shortest_path, topk)
  losses.py                 # SPO+, PFY, MSE loss gradients
  solver.py                 # Gurobi knapsack + DP shortest path
experiments/
  evaluate_online.py        # Online agent evaluation
  compute_analysis.py       # Wallclock + Pareto analysis
  plot_eval_results.py      # Main evaluation plots
  plot_ablation_results.py  # Ablation plots
  tune_hyperparams.py       # Optuna HP tuning
  ablations/                # Ablation and analysis scripts
notebooks/                  # 4 exploratory notebooks
```

import numpy as np
import jax.numpy as jnp

from src.agents.bayesian_utils import (
    compute_bayesian_predictions,
    compute_uncertainties,
)
from src.agents.contextual.bayesian_spo import BayesianSPOAgent


class BayesianSPOPIAgent(BayesianSPOAgent):
    """Posterior-imputed BayesianSPO.

    Samples multiple completions of the unobserved coordinates and averages
    the SPO+ gradient across them.

    Two sampling modes:

    - ``use_posterior_predictive=False`` (default, legacy): samples with
      std = ``imputation_scale * sigma_parameter`` where sigma_parameter is
      the posterior parameter std. Has a tunable ``imputation_scale`` knob
      and omits observation noise.
    - ``use_posterior_predictive=True`` (principled): samples from the
      posterior predictive distribution
      :math:`\\mathcal{N}(\\mu, \\sigma_{\\text{param}}^2 + \\sigma_{\\text{noise}}^2)`,
      which is an unbiased Monte Carlo estimator of
      :math:`\\mathbb{E}_{c \\mid \\mathcal{D}_t}[\\nabla_h \\ell_{\\text{SPO+}}(\\hat c, c)]`.
      No tunable scale knob.
    """

    def __init__(
        self,
        *args,
        imputation_samples: int = 4,
        imputation_scale: float = 1.0,
        use_posterior_predictive: bool = False,
        noise_var: float = 1.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, seed=seed, **kwargs)
        self.imputation_samples = max(int(imputation_samples), 1)
        self.imputation_scale = float(imputation_scale)
        self.use_posterior_predictive = bool(use_posterior_predictive)
        self.noise_var = float(noise_var)
        self.imputation_rng = np.random.default_rng(seed)

    def _compute_spo_grad_h(self, X, h, Y_target, selection_mask, full_feedback):
        pred = compute_bayesian_predictions(h, self.A_inv, self.b_bayes)
        pred_np = np.asarray(pred, dtype=float).reshape(-1)
        stds_np = np.asarray(compute_uncertainties(h, self.A_inv), dtype=float).reshape(-1)
        target_np = np.asarray(Y_target, dtype=float).reshape(-1)
        observed_mask_np = np.asarray(selection_mask, dtype=bool).reshape(-1)

        targets = [target_np]
        missing_mask = ~observed_mask_np
        if missing_mask.any():
            if self.use_posterior_predictive:
                # Principled: sample from posterior predictive
                # std = sqrt(parameter_var + noise_var). No scale knob.
                sampling_std = np.sqrt(stds_np ** 2 + self.noise_var)
            else:
                # Legacy: parameter posterior only, scaled by imputation_scale.
                sampling_std = self.imputation_scale * stds_np
            for _ in range(self.imputation_samples - 1):
                sampled = target_np.copy()
                sampled[missing_mask] = (
                    pred_np[missing_mask]
                    + sampling_std[missing_mask]
                    * self.imputation_rng.standard_normal(missing_mask.sum())
                )
                targets.append(sampled)

        theta = jnp.einsum("nij,nj->ni", self.A_inv, self.b_bayes)
        grad_h_total = jnp.zeros_like(h)
        for target_vec in targets:
            spo_grad_val = self.loss_grad_fn(pred, jnp.asarray(target_vec))
            if self.mask_grad and not full_feedback:
                spo_grad_val = jnp.where(selection_mask, spo_grad_val, 0.0)
            grad_h_total = grad_h_total + jnp.einsum("n,ni->i", spo_grad_val, theta)

        return grad_h_total / max(len(targets), 1)

import numpy as np
import jax.numpy as jnp

from src.agents.bayesian_utils import (
    compute_bayesian_predictions,
    compute_uncertainties,
)
from src.agents.contextual.bayesian_spo import BayesianSPOAgent


class BayesianSPOPIAgent(BayesianSPOAgent):
    def __init__(
        self,
        *args,
        imputation_samples: int = 8,
        imputation_scale: float = 1.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, seed=seed, **kwargs)
        self.imputation_samples = max(int(imputation_samples), 1)
        self.imputation_scale = float(imputation_scale)
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
            scale = self.imputation_scale * stds_np
            for _ in range(self.imputation_samples - 1):
                sampled = target_np.copy()
                sampled[missing_mask] = (
                    pred_np[missing_mask]
                    + scale[missing_mask] * self.imputation_rng.standard_normal(missing_mask.sum())
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

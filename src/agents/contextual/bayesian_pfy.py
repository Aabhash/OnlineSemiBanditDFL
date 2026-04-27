from src.losses import make_pfy_grad_fn
from src.agents.contextual.bayesian_spo import BayesianSPOAgent


class BayesianPFYAgent(BayesianSPOAgent):
    """PFY with optional posterior-imputation (Monte Carlo over completions of unobserved arms).

    Inherits the M-averaging machinery from `BayesianSPOAgent` and only overrides
    `loss_grad_fn` to use the PFY gradient instead of SPO+. With `imputation_samples=1`
    (the default), the gradient is computed deterministically from posterior-mean
    completion of unobserved arms, matching the previous PFY agent. With `M > 1`,
    the PFY gradient is averaged over `M` posterior samples for the unobserved
    coordinates.
    """

    def __init__(
        self,
        *args,
        pfy_n_samples: int = 1,
        pfy_sigma: float = 1.0,
        pfy_noise: str = 'normal',
        imputation_samples: int = 1,
        imputation_scale: float = 1.0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            *args,
            imputation_samples=imputation_samples,
            imputation_scale=imputation_scale,
            seed=seed,
            **kwargs,
        )
        self.loss_grad_fn = make_pfy_grad_fn(
            self.solver,
            minimize=self.minimize,
            n_samples=pfy_n_samples,
            sigma=pfy_sigma,
            noise=pfy_noise,
            seed=seed,
        )

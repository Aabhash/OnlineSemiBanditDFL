from src.losses import make_pfy_grad_fn
from src.agents.shared_context.bayesian_spo import BayesianSPOAgent


class BayesianPFYAgent(BayesianSPOAgent):

    def __init__(
        self,
        context_dim: int,
        n_arms: int,
        weights,
        capacity: float,
        hidden_dim: int = 32,
        beta: float = 1.0,
        beta_start: float = None,
        beta_end: float = None,
        lambda_reg: float = 1.0,
        lr: float = 0.001,
        decay_rounds: int = 300,
        max_grad_norm: float = 5.0,
        mask_grad: bool = False,
        nonlinear: bool = False,
        minimize: bool = False,
        solver=None,
        seed: int = None,
        # Phased training (Weber-style)
        phased: bool = False,
        phase_length: int = 50,
        phase_epochs: int = 5,
        # PFY-specific
        pfy_n_samples: int = 1,
        pfy_sigma: float = 1.0,
        pfy_noise: str = 'normal'
    ):
        super().__init__(
            context_dim=context_dim,
            n_arms=n_arms,
            weights=weights,
            capacity=capacity,
            hidden_dim=hidden_dim,
            beta=beta,
            beta_start=beta_start,
            beta_end=beta_end,
            lambda_reg=lambda_reg,
            lr=lr,
            decay_rounds=decay_rounds,
            max_grad_norm=max_grad_norm,
            mask_grad=mask_grad,
            nonlinear=nonlinear,
            minimize=minimize,
            solver=solver,
            seed=seed,
            phased=phased,
            phase_length=phase_length,
            phase_epochs=phase_epochs,
        )

        self.loss_grad_fn = make_pfy_grad_fn(
            self.solver,
            minimize=minimize,
            n_samples=pfy_n_samples,
            sigma=pfy_sigma,
            noise=pfy_noise,
            seed=seed,
        )

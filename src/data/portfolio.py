import numpy as np
from typing import Tuple


def genData(
    num_data: int,
    num_features: int,
    num_assets: int,
    deg: int = 4,
    noise_level: float = 0.5,
    signal_scale: float = 0.05,
    baseline_return: float = 0.1,
    seed: int = 42,
):
    if not isinstance(deg, int):
        raise ValueError(f'deg = {deg} should be int.')
    if deg <= 0:
        raise ValueError(f'deg = {deg} should be positive.')

    rnd = np.random.RandomState(seed)
    n = num_data
    p = num_features
    m = num_assets

    B = rnd.binomial(1, 0.5, (m, p))
    L = rnd.uniform(-2.5e-3 * noise_level, 2.5e-3 * noise_level, (m, p))

    x = rnd.normal(0, 1, (n, p))
    r = np.zeros((n, m))
    for i in range(n):
        r[i] = (
            signal_scale * np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p)
            + baseline_return ** (1 / deg)
        ) ** deg
        f = rnd.randn(num_features)
        eps = rnd.randn(num_assets)
        r[i] += L @ f + 0.01 * noise_level * eps

    cov = L @ L.T + (1e-2 * noise_level) ** 2 * np.eye(num_assets)
    return cov, x, r


class PortfolioData:
    def __init__(
        self,
        num_assets: int = 32,
        n_features: int = None,
        context_dim: int = None,
        deg: int = 4,
        noise_width: float = 0.5,
        noise_level: float = None,
        signal_scale: float = 0.05,
        baseline_return: float = 0.1,
        seed: int = 42,
        **kwargs,
    ):
        if context_dim is not None:
            self.n_features = context_dim
        elif n_features is not None:
            self.n_features = n_features
        else:
            self.n_features = 5
        self.context_dim = self.n_features
        self.num_assets = num_assets
        self.n_arms = num_assets
        self.deg = deg
        self.noise_width = noise_width if noise_level is None else noise_level
        self.noise_level = self.noise_width
        self.signal_scale = float(signal_scale)
        self.baseline_return = float(baseline_return)
        self.seed = int(seed)

        init_rng = np.random.default_rng(self.seed)
        self.B = init_rng.binomial(1, 0.5, size=(self.num_assets, self.n_features))
        self.L = init_rng.uniform(
            -2.5e-3 * self.noise_width,
            2.5e-3 * self.noise_width,
            size=(self.num_assets, self.n_features),
        )
        self.idiosyncratic_var = (1e-2 * self.noise_width) ** 2
        self.covariance = (
            self.L @ self.L.T
            + self.idiosyncratic_var * np.eye(self.num_assets)
        )
        self._sample_seed = self.seed + 7777
        self.rng = np.random.default_rng(self._sample_seed)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        context = self.rng.normal(0.0, 1.0, size=self.n_features)
        returns = self._generate_returns(context)
        return context, returns

    def _generate_returns(self, context: np.ndarray) -> np.ndarray:
        p = self.n_features

        mean_returns = (
            self.signal_scale * (self.B @ context) / np.sqrt(p)
            + self.baseline_return ** (1.0 / self.deg)
        ) ** self.deg
        f = self.rng.normal(0.0, 1.0, size=self.n_features)
        eps = self.rng.normal(0.0, 1.0, size=self.num_assets)
        returns = mean_returns + self.L @ f + 0.01 * self.noise_width * eps
        return returns

    def step(self):
        pass

    def reset(self):
        self.rng = np.random.default_rng(self._sample_seed)

    def get_config(self) -> dict:
        return {
            'num_assets': self.num_assets,
            'n_features': self.n_features,
            'context_dim': self.context_dim,
            'deg': self.deg,
            'noise_width': self.noise_width,
            'signal_scale': self.signal_scale,
            'baseline_return': self.baseline_return,
            'idiosyncratic_var': self.idiosyncratic_var,
            'covariance': self.covariance.copy(),
        }

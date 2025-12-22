import numpy as np
import os
from typing import Tuple, Optional


class FamaFrenchData:

    def __init__(
        self,
        n_portfolios: int = 100,
        lookback: int = 12,
        start_date: str = "2000-01",
        end_date: str = "2023-12",
        data_dir: Optional[str] = None,
        seed: int = 42,
    ):
        self.n_portfolios = n_portfolios
        self.lookback = lookback
        self.start_date = start_date
        self.end_date = end_date
        self.rng = np.random.default_rng(seed)

        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

        self.returns, self.factors = self._load_data(data_dir)

        if self.returns is None:
            raise ValueError("Could not load Fama-French data. Run download script first.")

        if self.returns.shape[1] > n_portfolios:
            idx = self.rng.choice(self.returns.shape[1], n_portfolios, replace=False)
            self.returns = self.returns[:, idx]

        self.n_factors = self.factors.shape[1]
        self.context_dim = self.lookback * (self.n_factors + 1)

        self.current_idx = self.lookback
        self.max_idx = len(self.returns) - 1

    def _load_data(self, data_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        portfolio_file = os.path.join(data_dir, '100_portfolios.csv')
        factors_file = os.path.join(data_dir, 'ff_factors.csv')

        if not os.path.exists(portfolio_file) or not os.path.exists(factors_file):
            return None, None

        try:
            returns = self._parse_portfolio_csv(portfolio_file)
            factors = self._parse_factors_csv(factors_file)

            min_len = min(len(returns), len(factors))
            returns = returns[:min_len]
            factors = factors[:min_len]

            return returns, factors

        except Exception as e:
            print(f"Error loading Fama-French data: {e}")
            return None, None

    def _parse_portfolio_csv(self, filepath: str) -> np.ndarray:
        returns_list = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) > 1 and len(parts[0].strip()) == 6:
                try:
                    date_str = parts[0].strip()
                    year = int(date_str[:4])
                    month = int(date_str[4:6])

                    start_year, start_month = map(int, self.start_date.split('-'))
                    end_year, end_month = map(int, self.end_date.split('-'))

                    if (year > start_year or (year == start_year and month >= start_month)) and \
                       (year < end_year or (year == end_year and month <= end_month)):
                        values = []
                        for v in parts[1:101]:
                            try:
                                val = float(v.strip())
                                if val < -90:
                                    val = 0.0
                                values.append(val / 100)
                            except:
                                values.append(0.0)
                        if len(values) == 100:
                            returns_list.append(values)
                except:
                    continue

        return np.array(returns_list)

    def _parse_factors_csv(self, filepath: str) -> np.ndarray:
        factors_list = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) >= 4 and len(parts[0].strip()) == 6:
                try:
                    date_str = parts[0].strip()
                    year = int(date_str[:4])
                    month = int(date_str[4:6])

                    start_year, start_month = map(int, self.start_date.split('-'))
                    end_year, end_month = map(int, self.end_date.split('-'))

                    if (year > start_year or (year == start_year and month >= start_month)) and \
                       (year < end_year or (year == end_year and month <= end_month)):
                        factors = [float(parts[i].strip()) / 100 for i in range(1, 4)]
                        factors_list.append(factors)
                except:
                    continue

        return np.array(factors_list)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.current_idx >= self.max_idx:
            self.current_idx = self.lookback

        context = self._construct_context(self.current_idx)
        values = self.returns[self.current_idx]
        self.current_idx += 1

        return context, values

    def _construct_context(self, idx: int) -> np.ndarray:
        features = []

        for lag in range(1, self.lookback + 1):
            features.extend(self.factors[idx - lag])

        for lag in range(1, self.lookback + 1):
            features.append(np.mean(self.returns[idx - lag]))

        return np.array(features)

    def reset(self):
        self.current_idx = self.lookback

    def get_config(self) -> dict:
        return {
            'n_portfolios': self.n_portfolios,
            'context_dim': self.context_dim,
            'lookback': self.lookback,
            'n_months': len(self.returns),
        }

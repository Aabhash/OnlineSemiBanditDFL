import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


class CriteoData:

    def __init__(
        self,
        n_campaigns: int = 100,
        context_dim: int = 50,
        subsample_rate: float = 0.1,
        data_dir: Optional[str] = None,
        noise_std: float = 0.05,
        env_fraction: float = 0.8,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.n_campaigns = n_campaigns
        self.context_dim = context_dim
        self.subsample_rate = subsample_rate
        self.noise_std = noise_std
        self.env_fraction = env_fraction
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        if data_dir is None:
            data_dir = 'data/criteo'
        self.data_dir = Path(data_dir)

        self._load_or_process_data()

    def _get_cache_path(self) -> Path:
        cache_name = f'criteo_processed_n{self.n_campaigns}_d{self.context_dim}_s{int(self.subsample_rate*100)}_e{int(self.env_fraction*100)}.pkl'
        return self.data_dir / cache_name

    def _load_or_process_data(self):
        cache_path = self._get_cache_path()

        if cache_path.exists():
            self._load_from_cache(cache_path)
        else:
            self._process_and_cache(cache_path)

    def _load_from_cache(self, cache_path: Path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

        self.campaign_costs = cache['campaign_costs']
        self.campaign_conv_rates = cache['campaign_conv_rates']
        self.bandit_contexts = cache['bandit_contexts']
        self.X_encoded_bandit = cache['X_encoded_bandit']
        self.context_coef = cache['context_coef']
        self.campaign_coef = cache['campaign_coef']
        self.intercept = cache['intercept']
        self.projection_matrix = cache['projection_matrix']

        if self.verbose:
            n_env = cache['n_env']
            n_bandit = len(self.bandit_contexts)
            print(f"Loaded Criteo from cache: {n_env + n_bandit:,} impressions, {self.n_campaigns} campaigns")

    def _process_and_cache(self, cache_path: Path):
        print("Processing Criteo data (first time, will be cached)...")

        self._load_data()

        cache = {
            'campaign_costs': self.campaign_costs,
            'campaign_conv_rates': self.campaign_conv_rates,
            'bandit_contexts': self.bandit_contexts,
            'X_encoded_bandit': self.X_encoded_bandit,
            'context_coef': self.context_coef,
            'campaign_coef': self.campaign_coef,
            'intercept': self.intercept,
            'projection_matrix': self.projection_matrix,
            'n_env': self._n_env,
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

        print(f"Cached to {cache_path.name} ({self._n_env + self._n_bandit:,} impressions)")

    def _load_data(self):
        parquet_file = self.data_dir / f'criteo_subsample_{int(self.subsample_rate * 100)}.parquet'

        if not parquet_file.exists():
            raise FileNotFoundError(
                f"Criteo data not found at {parquet_file}\n"
                f"Run: python src/data/download_criteo.py --subsample_rate {self.subsample_rate}"
            )

        df = pd.read_parquet(parquet_file)

        campaign_counts = df['campaign'].value_counts()
        top_campaigns = campaign_counts.head(self.n_campaigns).index.tolist()

        df = df[df['campaign'].isin(top_campaigns)].reset_index(drop=True)

        self.campaign_to_idx = {c: i for i, c in enumerate(top_campaigns)}
        df['campaign_idx'] = df['campaign'].map(self.campaign_to_idx)

        n_total = len(df)
        n_env = int(n_total * self.env_fraction)

        indices = self.rng.permutation(n_total)
        env_indices = indices[:n_env]
        bandit_indices = indices[n_env:]

        df_env = df.iloc[env_indices].reset_index(drop=True)
        df_bandit = df.iloc[bandit_indices].reset_index(drop=True)

        campaign_stats = df_env.groupby('campaign_idx').agg({
            'cost': 'mean',
            'conversion': 'mean',
            'click': 'mean',
        }).reset_index()
        campaign_stats['count'] = df_env.groupby('campaign_idx').size().values

        self.campaign_costs = np.zeros(self.n_campaigns)

        click_rates = campaign_stats['click'].values
        counts = campaign_stats['count'].values

        click_rates_norm = (click_rates - click_rates.min()) / (click_rates.max() - click_rates.min() + 1e-6)
        counts_norm = (np.log1p(counts) - np.log1p(counts).min()) / (np.log1p(counts).max() - np.log1p(counts).min() + 1e-6)

        for i, row in campaign_stats.iterrows():
            idx = int(row['campaign_idx'])
            self.campaign_costs[idx] = 1.0 + 2.0 * click_rates_norm[i] + 2.0 * counts_norm[i]

        self.campaign_conv_rates = np.zeros(self.n_campaigns)
        for _, row in campaign_stats.iterrows():
            idx = int(row['campaign_idx'])
            self.campaign_conv_rates[idx] = row['conversion']

        self._encode_features(df_env, fit=True)
        self._train_model(df_env)
        self._encode_bandit_contexts(df_bandit)

        self._n_env = len(df_env)
        self._n_bandit = len(df_bandit)

    def _encode_features(self, df: pd.DataFrame, fit: bool = True):
        cat_cols = [f'cat{i}' for i in range(1, 10) if f'cat{i}' in df.columns]
        self.cat_cols = cat_cols

        if fit:
            self.label_encoders = {}
            encoded_features = []

            for col in cat_cols:
                le = LabelEncoder()
                df_col = df[col].fillna(-1).astype(int)
                encoded = le.fit_transform(df_col)
                self.label_encoders[col] = le
                encoded_features.append(encoded)

            X_encoded = np.column_stack(encoded_features)

            n_cat_features = X_encoded.shape[1]
            self.projection_matrix = self.rng.standard_normal(
                (n_cat_features, self.context_dim)
            ) / np.sqrt(n_cat_features)

            self.X_encoded_env = X_encoded

    def _encode_bandit_contexts(self, df: pd.DataFrame):
        encoded_features = []

        for col in self.cat_cols:
            le = self.label_encoders[col]
            df_col = df[col].fillna(-1).astype(int)
            encoded = np.zeros(len(df_col), dtype=int)
            for i, val in enumerate(df_col):
                if val in le.classes_:
                    encoded[i] = le.transform([val])[0]
                else:
                    encoded[i] = 0
            encoded_features.append(encoded)

        X_encoded = np.column_stack(encoded_features)
        self.X_encoded_bandit = X_encoded

        contexts = X_encoded @ self.projection_matrix
        norms = np.linalg.norm(contexts, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.bandit_contexts = contexts / norms * 2

    def _train_model(self, df: pd.DataFrame):
        n_samples = len(df)
        campaign_onehot = np.zeros((n_samples, self.n_campaigns))
        for i, idx in enumerate(df['campaign_idx']):
            campaign_onehot[i, idx] = 1

        X = np.hstack([self.X_encoded_env, campaign_onehot])
        y = df['conversion'].values

        self.model = LogisticRegression(
            max_iter=500,
            solver='lbfgs',
            random_state=42,
            n_jobs=-1,
        )

        if n_samples > 100000:
            train_idx = self.rng.choice(n_samples, size=100000, replace=False)
            self.model.fit(X[train_idx], y[train_idx])
        else:
            self.model.fit(X, y)

        self.context_coef = self.model.coef_[0, :self.X_encoded_env.shape[1]]
        self.campaign_coef = self.model.coef_[0, self.X_encoded_env.shape[1]:]
        self.intercept = self.model.intercept_[0]

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.rng.integers(len(self.bandit_contexts))
        context = self.bandit_contexts[idx].copy()

        context += self.rng.standard_normal(self.context_dim) * 0.1

        x_encoded = self.X_encoded_bandit[idx]
        base_logit = self.context_coef @ x_encoded + self.intercept

        logits = base_logit + self.campaign_coef
        probs = 1 / (1 + np.exp(-logits))

        values = probs + self.rng.standard_normal(self.n_campaigns) * self.noise_std
        values = np.clip(values, 0.01, 0.99)

        return context, values

    def get_config(self) -> Dict:
        return {
            'n_campaigns': self.n_campaigns,
            'context_dim': self.context_dim,
            'subsample_rate': self.subsample_rate,
            'env_fraction': self.env_fraction,
            'cost_range': (float(self.campaign_costs.min()), float(self.campaign_costs.max())),
            'mean_conversion_rate': float(self.campaign_conv_rates.mean()),
        }

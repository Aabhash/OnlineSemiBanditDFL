import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


class CriteoData:
    def __init__(
        self,
        n_campaigns: int = 100,
        context_dim: int = 50,
        subsample_rate: float = 0.1,
        data_dir: Optional[str] = None,
        seed: int = 42,
        verbose: bool = False,
        # XGBoost parameters for surrogate model
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        self.n_campaigns = n_campaigns
        self.context_dim = context_dim
        self.subsample_rate = subsample_rate
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # XGBoost surrogate parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        if data_dir is None:
            data_dir = 'data/criteo'
        self.data_dir = Path(data_dir)

        self._load_or_process_data()

    def _get_cache_path(self) -> Path:
        # Cache includes XGBoost params since they affect the surrogate
        cache_name = f'criteo_xgb_n{self.n_campaigns}_d{self.context_dim}_s{int(self.subsample_rate*100)}_est{self.n_estimators}_dep{self.max_depth}.pkl'
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
        self.contexts = cache['contexts']
        self.X_encoded = cache['X_encoded']
        self.campaign_indices = cache['campaign_indices']
        self.projection_matrix = cache['projection_matrix']
        self.model = cache['model']  # XGBoost model

        if self.verbose:
            n_samples = len(self.contexts)
            print(f"Loaded Criteo from cache: {n_samples:,} impressions, {self.n_campaigns} campaigns")

    def _process_and_cache(self, cache_path: Path):
        print("Processing Criteo data with XGBoost surrogate (first time, will be cached)...")

        self._load_data()

        cache = {
            'campaign_costs': self.campaign_costs,
            'campaign_conv_rates': self.campaign_conv_rates,
            'contexts': self.contexts,
            'X_encoded': self.X_encoded,
            'campaign_indices': self.campaign_indices,
            'projection_matrix': self.projection_matrix,
            'model': self.model,
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

        print(f"Cached to {cache_path.name} ({len(self.contexts):,} impressions)")

    def _load_data(self):
        parquet_file = self.data_dir / f'criteo_subsample_{int(self.subsample_rate * 100)}.parquet'

        if not parquet_file.exists():
            raise FileNotFoundError(
                f"Criteo data not found at {parquet_file}\n"
                f"Run: python src/data/download_criteo.py --subsample_rate {self.subsample_rate}"
            )

        df = pd.read_parquet(parquet_file)

        # Select top campaigns by volume
        campaign_counts = df['campaign'].value_counts()
        top_campaigns = campaign_counts.head(self.n_campaigns).index.tolist()

        df = df[df['campaign'].isin(top_campaigns)].reset_index(drop=True)

        self.campaign_to_idx = {c: i for i, c in enumerate(top_campaigns)}
        df['campaign_idx'] = df['campaign'].map(self.campaign_to_idx)

        # Compute campaign costs (varied weights for knapsack constraint)
        campaign_stats = df.groupby('campaign_idx').agg({
            'cost': 'mean',
            'conversion': 'mean',
            'click': 'mean',
        }).reset_index()
        campaign_stats['count'] = df.groupby('campaign_idx').size().values

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

        # Encode features and train XGBoost surrogate on 100% of data
        self._encode_features(df)
        self._train_xgboost_surrogate(df)

        # Store campaign indices for sampling
        self.campaign_indices = df['campaign_idx'].values

    def _encode_features(self, df: pd.DataFrame):
        cat_cols = [f'cat{i}' for i in range(1, 10) if f'cat{i}' in df.columns]
        self.cat_cols = cat_cols

        self.label_encoders = {}
        encoded_features = []

        for col in cat_cols:
            le = LabelEncoder()
            df_col = df[col].fillna(-1).astype(int)
            encoded = le.fit_transform(df_col)
            self.label_encoders[col] = le
            encoded_features.append(encoded)

        self.X_encoded = np.column_stack(encoded_features)

        # Create random projection for context (shared across all campaigns)
        n_cat_features = self.X_encoded.shape[1]
        self.projection_matrix = self.rng.standard_normal(
            (n_cat_features, self.context_dim)
        ) / np.sqrt(n_cat_features)

        # Compute normalized contexts
        contexts = self.X_encoded @ self.projection_matrix
        norms = np.linalg.norm(contexts, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.contexts = contexts / norms * 2

    def _train_xgboost_surrogate(self, df: pd.DataFrame):
        n_samples = len(df)

        # Features: encoded categoricals + campaign one-hot
        campaign_onehot = np.zeros((n_samples, self.n_campaigns))
        for i, idx in enumerate(df['campaign_idx']):
            campaign_onehot[i, idx] = 1

        X = np.hstack([self.X_encoded, campaign_onehot])
        y = df['conversion'].values

        print(f"Training XGBoost surrogate on {n_samples:,} samples...")

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
        )

        self.model.fit(X, y)

        # Compute and report surrogate quality
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        from sklearn.metrics import roc_auc_score, log_loss
        auc = roc_auc_score(y, y_pred_proba)
        logloss = log_loss(y, y_pred_proba)
        print(f"XGBoost surrogate trained: AUC={auc:.4f}, LogLoss={logloss:.4f}")

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a context and get conversion probabilities from XGBoost.

        No artificial noise is added - the XGBoost predictions are the true values.
        """
        idx = self.rng.integers(len(self.contexts))
        context = self.contexts[idx].copy()

        # Get conversion probabilities from XGBoost surrogate for ALL campaigns
        x_encoded = self.X_encoded[idx]

        # Create feature vectors for each campaign
        n_features = len(x_encoded) + self.n_campaigns
        X_pred = np.zeros((self.n_campaigns, n_features))

        for campaign_idx in range(self.n_campaigns):
            X_pred[campaign_idx, :len(x_encoded)] = x_encoded
            X_pred[campaign_idx, len(x_encoded) + campaign_idx] = 1

        # Get probabilities from XGBoost (these ARE the true values)
        values = self.model.predict_proba(X_pred)[:, 1]

        return context, values

    def get_config(self) -> Dict:
        return {
            'n_campaigns': self.n_campaigns,
            'context_dim': self.context_dim,
            'subsample_rate': self.subsample_rate,
            'surrogate': 'xgboost',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'cost_range': (float(self.campaign_costs.min()), float(self.campaign_costs.max())),
            'mean_conversion_rate': float(self.campaign_conv_rates.mean()),
        }

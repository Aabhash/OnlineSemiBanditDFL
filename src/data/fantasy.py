import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union


class FantasyData:

    def __init__(
        self,
        n_players: Optional[int] = None,  # None = all players
        context_dim: int = 14,  # Actual feature dim (no random projection)
        data_dir: Optional[str] = None,
        season: int = 2020,
        seed: int = 42,
        min_appearances: int = 3,  # Minimum weeks a player must appear in
        validation_weeks: int = 0,  # N weeks for HP validation (before holdout)
        holdout_weeks: int = 0,  # Last N weeks held out for final test
        verbose: bool = False,
    ):
        self._n_players_requested = n_players
        self.min_appearances = min_appearances
        self.context_dim = context_dim
        self.season = season
        self.validation_weeks = validation_weeks
        self.holdout_weeks = holdout_weeks
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        if data_dir is None:
            data_dir = 'data/fantasy'
        self.data_dir = Path(data_dir)

        self._load_data()

    def _load_data(self):
        parquet_file = self.data_dir / f'fantasy_{self.season}.parquet'

        if not parquet_file.exists():
            raise FileNotFoundError(
                f"Fantasy data not found at {parquet_file}\n"
                f"Run: python src/data/download_fantasy.py --season {self.season}"
            )

        df = pd.read_parquet(parquet_file)

        # Compute pts_trend: last_1 - last_3 (positive = improving)
        # This must be done before splitting to maintain time ordering
        df = df.sort_values(['player_id', 'week'])
        df['last_1_pts'] = df.groupby('player_id')['fantasy_points'].shift(1)
        df['last_3_pts'] = df.groupby('player_id')['fantasy_points'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df['pts_trend'] = (df['last_1_pts'] - df['last_3_pts']).fillna(0)
        # Normalize pts_trend
        pts_trend_std = df['pts_trend'].std()
        if pts_trend_std > 0:
            df['pts_trend'] = df['pts_trend'] / pts_trend_std

        # Filter to players with enough appearances
        player_counts = df.groupby('player_id').size()
        frequent_players = player_counts[player_counts >= self.min_appearances].index
        df = df[df['player_id'].isin(frequent_players)].copy()

        # Select players
        player_avg = df.groupby('player_id')['fantasy_points'].mean()
        if self._n_players_requested is not None:
            selected_players = player_avg.nlargest(self._n_players_requested).index.tolist()
        else:
            selected_players = player_avg.index.tolist()

        self.n_players = len(selected_players)
        df = df[df['player_id'].isin(selected_players)].copy()

        self.player_to_idx = {p: i for i, p in enumerate(selected_players)}
        self.idx_to_player = {i: p for p, i in self.player_to_idx.items()}
        df['player_idx'] = df['player_id'].map(self.player_to_idx)

        all_weeks = sorted(df['week'].unique())

        # Three-way split: train / validation / holdout (test)
        # Temporal split: train is earliest, validation is middle, holdout is latest
        total_held_out = self.validation_weeks + self.holdout_weeks
        if total_held_out > 0:
            if total_held_out >= len(all_weeks):
                raise ValueError(
                    f"validation_weeks ({self.validation_weeks}) + holdout_weeks ({self.holdout_weeks}) "
                    f"must be less than total weeks ({len(all_weeks)})"
                )
            self.weeks = all_weeks[:-total_held_out]  # Train weeks
            if self.validation_weeks > 0 and self.holdout_weeks > 0:
                self.validation_week_list = all_weeks[-total_held_out:-self.holdout_weeks]
                self.holdout_week_list = all_weeks[-self.holdout_weeks:]
            elif self.validation_weeks > 0:
                self.validation_week_list = all_weeks[-self.validation_weeks:]
                self.holdout_week_list = []
            else:
                self.validation_week_list = []
                self.holdout_week_list = all_weeks[-self.holdout_weeks:]
        else:
            self.weeks = all_weeks
            self.validation_week_list = []
            self.holdout_week_list = []

        self.n_weeks = len(self.weeks)
        self.n_validation_weeks = len(self.validation_week_list)
        self.n_holdout_weeks = len(self.holdout_week_list)

        # CRITICAL: Compute statistics on TRAIN weeks only to prevent data leakage
        train_df = df[df['week'].isin(self.weeks)]
        self._compute_statistics_on_train(train_df)

        # Store full df for context extraction
        self._df = df

        all_weeks = self.weeks + self.validation_week_list + self.holdout_week_list

        self.week_data = {}
        for week in all_weeks:  # Load all weeks (train + holdout)
            week_df = df[df['week'] == week].copy()

            present_players = week_df['player_idx'].values

            contexts = np.zeros((self.n_players, self._get_raw_context_dim()))
            salaries = np.zeros(self.n_players)
            points = np.zeros(self.n_players)
            positions = np.array(['NA'] * self.n_players, dtype=object)
            available = np.zeros(self.n_players, dtype=bool)

            for _, row in week_df.iterrows():
                idx = int(row['player_idx'])
                available[idx] = True
                salaries[idx] = row['salary']
                points[idx] = row['fantasy_points']
                positions[idx] = row['position']

                context = self._extract_context(row)
                contexts[idx] = context

            self.week_data[week] = {
                'contexts': contexts,
                'salaries': salaries,
                'points': points,
                'positions': positions,
                'available': available,
            }

        all_salaries = []
        for week in self.weeks:
            all_salaries.append(self.week_data[week]['salaries'])
        salaries_stacked = np.stack(all_salaries, axis=0)
        self.base_salaries = np.max(salaries_stacked, axis=0)
        median_salary = np.median(self.base_salaries[self.base_salaries > 0])
        self.base_salaries[self.base_salaries == 0] = median_salary

        position_counts = {}
        for week in self.weeks:
            for idx, pos in enumerate(self.week_data[week]['positions']):
                if pos != 'NA':
                    if idx not in position_counts:
                        position_counts[idx] = {}
                    position_counts[idx][pos] = position_counts[idx].get(pos, 0) + 1

        self.player_positions = np.array(['NA'] * self.n_players, dtype=object)
        for idx, counts in position_counts.items():
            if counts:
                self.player_positions[idx] = max(counts, key=counts.get)

        raw_dim = self._get_raw_context_dim()
        if raw_dim != self.context_dim:
            self.projection_matrix = self.rng.standard_normal(
                (raw_dim, self.context_dim)
            ) / np.sqrt(raw_dim)
        else:
            self.projection_matrix = None

        self._week_idx = 0
        self._sequential_idx = 0  # For sequential sampling

        if self.verbose:
            val_info = f", val weeks {self.validation_week_list}" if self.validation_weeks > 0 else ""
            holdout_info = f", holdout weeks {self.holdout_week_list}" if self.holdout_weeks > 0 else ""
            print(f"Loaded Fantasy {self.season}: {self.n_players} players, "
                  f"{self.n_weeks} train weeks{val_info}{holdout_info}, "
                  f"salaries ${self.base_salaries.min():.0f}-${self.base_salaries.max():.0f}")

    def _compute_statistics_on_train(self, train_df: pd.DataFrame):
        # Compute position-specific opponent strength
        # How many points does each opponent allow to each position?
        self.opp_pos_strength = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
            pos_df = train_df[train_df['position'] == pos]
            if len(pos_df) > 0:
                opp_avg = pos_df.groupby('opponent')['fantasy_points'].mean()
                opp_mean = opp_avg.mean()
                opp_std = opp_avg.std() if opp_avg.std() > 0 else 1.0
                self.opp_pos_strength[pos] = (opp_avg - opp_mean) / opp_std
            else:
                self.opp_pos_strength[pos] = pd.Series(dtype=float)

        # Also keep aggregate for fallback
        self.opp_strength = train_df.groupby('opponent')['fantasy_points'].mean()
        opp_mean = self.opp_strength.mean()
        opp_std = self.opp_strength.std() if self.opp_strength.std() > 0 else 1.0
        self.opp_strength_normalized = (self.opp_strength - opp_mean) / opp_std

        # Compute team offensive strength (normalized)
        team_off = train_df.groupby('team')['fantasy_points'].mean()
        team_off_mean = team_off.mean()
        team_off_std = team_off.std() if team_off.std() > 0 else 1.0
        self.team_off_strength = (team_off - team_off_mean) / team_off_std

    def _get_raw_context_dim(self) -> int:
        return 14  # 12 base + pts_trend + team_off_strength

    def _extract_context(self, row: pd.Series) -> np.ndarray:
        # Get position-specific opponent strength (much more predictive!)
        opp = row.get('opponent', '')
        pos = row.get('position', '')
        team = row.get('team', '')

        # Use position-specific if available, else aggregate
        if pos in self.opp_pos_strength and opp in self.opp_pos_strength[pos].index:
            opp_strength = self.opp_pos_strength[pos].get(opp, 0.0)
        else:
            opp_strength = self.opp_strength_normalized.get(opp, 0.0)

        # Team offensive strength
        team_off = self.team_off_strength.get(team, 0.0)

        context = np.array([
            row.get('prev_avg_points', 0) / 20.0,
            row.get('prev_std_points', 0) / 10.0,
            row.get('salary_normalized', 0.5),
            opp_strength,  # Position-specific opponent weakness
            row.get('home', 0),
            int(row.get('position') == 'QB'),
            int(row.get('position') == 'RB'),
            int(row.get('position') == 'WR'),
            int(row.get('position') == 'TE'),
            int(row.get('position') == 'DST'),
            row.get('week_sin', 0),
            row.get('week_cos', 1),
            row.get('pts_trend', 0),  # Recent form trend (last_1 - last_3)
            team_off,  # Team offensive strength
        ])
        return context

    def get_week(self, week: int) -> Dict:
        if week not in self.week_data:
            raise ValueError(f"Week {week} not in data. Available: {self.weeks}")

        data = self.week_data[week]
        contexts = data['contexts'].copy()

        if self.projection_matrix is not None:
            contexts = contexts @ self.projection_matrix

        return {
            'contexts': contexts,
            'salaries': data['salaries'].copy(),
            'points': data['points'].copy(),
            'positions': data['positions'].copy(),
            'available': data['available'].copy(),
        }

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        week = self.rng.choice(self.weeks)
        data = self.get_week(week)

        available = data['available']
        if available.any():
            context = data['contexts'][available].mean(axis=0)
        else:
            context = np.zeros(self.context_dim)

        context += self.rng.standard_normal(self.context_dim) * 0.1

        values = data['points'].copy()
        values[~available] = 0.0

        return context, values

    def sample_per_arm(self, sequential: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if sequential:
            week = self.weeks[self._sequential_idx % self.n_weeks]
            self._sequential_idx += 1
        else:
            week = self.rng.choice(self.weeks)

        data = self.get_week(week)

        contexts = data['contexts'].copy()
        # Removed: contexts += self.rng.standard_normal(contexts.shape) * 0.1

        values = data['points'].copy()

        available = data['available']
        values[~available] = 0.0

        return contexts, values

    def reset_sequential(self):
        self._sequential_idx = 0

    def get_holdout_set(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not self.holdout_week_list:
            return []

        holdout_samples = []
        for week in self.holdout_week_list:
            data = self.get_week(week)
            contexts = data['contexts'].copy()
            values = data['points'].copy()
            available = data['available']
            values[~available] = 0.0
            holdout_samples.append((contexts, values))

        return holdout_samples

    def get_validation_set(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not self.validation_week_list:
            return []

        validation_samples = []
        for week in self.validation_week_list:
            data = self.get_week(week)
            contexts = data['contexts'].copy()
            values = data['points'].copy()
            available = data['available']
            values[~available] = 0.0
            validation_samples.append((contexts, values))

        return validation_samples

    def sample_with_details(self) -> Dict:
        week = self.rng.choice(self.weeks)
        data = self.get_week(week)

        available = data['available']
        if available.any():
            context = data['contexts'][available].mean(axis=0)
        else:
            context = np.zeros(self.context_dim)
        context += self.rng.standard_normal(self.context_dim) * 0.1

        return {
            'week': week,
            'context': context,
            'player_contexts': data['contexts'],
            'values': data['points'],
            'salaries': data['salaries'],
            'positions': data['positions'],
            'available': data['available'],
        }

    def get_config(self) -> Dict:
        return {
            'n_players': self.n_players,
            'context_dim': self.context_dim,
            'season': self.season,
            'n_weeks': self.n_weeks,
            'salary_range': (float(self.base_salaries.min()), float(self.base_salaries.max())),
            'positions': np.unique(self.player_positions[self.player_positions != 'NA']).tolist(),
        }

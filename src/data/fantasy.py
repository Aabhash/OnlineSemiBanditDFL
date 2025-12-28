import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union


class FantasyData:

    def __init__(
        self,
        n_players: Optional[int] = None,  # None = all players
        context_dim: int = 12,  # Actual feature dim (no random projection)
        data_dir: Optional[str] = None,
        season: int = 2020,
        seed: int = 42,
        min_appearances: int = 3,  # Minimum weeks a player must appear in
        holdout_weeks: int = 0,  # Last N weeks held out for evaluation
        verbose: bool = False,
    ):
        self._n_players_requested = n_players
        self.min_appearances = min_appearances
        self.context_dim = context_dim
        self.season = season
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

        # Compute position-specific opponent strength
        # How many points does each opponent allow to each position?
        self.opp_pos_strength = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
            pos_df = df[df['position'] == pos]
            if len(pos_df) > 0:
                opp_avg = pos_df.groupby('opponent')['fantasy_points'].mean()
                opp_mean = opp_avg.mean()
                opp_std = opp_avg.std() if opp_avg.std() > 0 else 1.0
                self.opp_pos_strength[pos] = (opp_avg - opp_mean) / opp_std
            else:
                self.opp_pos_strength[pos] = pd.Series(dtype=float)

        # Also keep aggregate for fallback
        self.opp_strength = df.groupby('opponent')['fantasy_points'].mean()
        opp_mean = self.opp_strength.mean()
        opp_std = self.opp_strength.std()
        self.opp_strength_normalized = (self.opp_strength - opp_mean) / opp_std

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

        # Split into train and holdout
        if self.holdout_weeks > 0:
            self.weeks = all_weeks[:-self.holdout_weeks]  # Train weeks
            self.holdout_week_list = all_weeks[-self.holdout_weeks:]  # Holdout weeks
        else:
            self.weeks = all_weeks
            self.holdout_week_list = []

        self.n_weeks = len(self.weeks)
        self.n_holdout_weeks = len(self.holdout_week_list)

        # Store full df for context extraction
        self._df = df

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
            holdout_info = f", holdout weeks {self.holdout_week_list}" if self.holdout_weeks > 0 else ""
            print(f"Loaded Fantasy {self.season}: {self.n_players} players, "
                  f"{self.n_weeks} train weeks{holdout_info}, "
                  f"salaries ${self.base_salaries.min():.0f}-${self.base_salaries.max():.0f}")

    def _get_raw_context_dim(self) -> int:
        return 12  # Added opponent strength

    def _extract_context(self, row: pd.Series) -> np.ndarray:
        # Get position-specific opponent strength (much more predictive!)
        opp = row.get('opponent', '')
        pos = row.get('position', '')

        # Use position-specific if available, else aggregate
        if pos in self.opp_pos_strength and opp in self.opp_pos_strength[pos].index:
            opp_strength = self.opp_pos_strength[pos].get(opp, 0.0)
        else:
            opp_strength = self.opp_strength_normalized.get(opp, 0.0)

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

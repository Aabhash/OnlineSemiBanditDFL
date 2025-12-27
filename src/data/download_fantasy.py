#!/usr/bin/env python3
import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def download_from_kaggle(data_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "Please install kagglehub: pip install kagglehub"
        )

    dataset_name = "samuellongenbach/historical-nfl-fantasy-2020"

    print(f"Downloading {dataset_name} from Kaggle...")
    cache_path = kagglehub.dataset_download(dataset_name)

    import shutil
    cache_path = Path(cache_path)
    for f in cache_path.glob("*"):
        shutil.copy(f, data_dir / f.name)

    print(f"Downloaded to {data_dir}")
    return data_dir


def load_and_preprocess(data_dir: Path, season: int = 2020) -> pd.DataFrame:
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            f"Run download first or check data directory."
        )

    print(f"Found {len(csv_files)} CSV files")

    all_data = []

    for csv_file in csv_files:
        filename = csv_file.stem.lower()

        if 'dk' not in filename and 'draftkings' not in filename:
            continue

        df = pd.read_csv(csv_file)

        week = None
        for part in filename.split('_'):
            if 'week' in part:
                week = int(''.join(filter(str.isdigit, part)))
                break

        if week is None and 'week' in df.columns:
            pass
        elif week is not None:
            df['week'] = week

        all_data.append(df)

    if not all_data:
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if 'week' in df.columns.str.lower() or 'Week' in df.columns:
                all_data.append(df)
                break

    if not all_data:
        raise ValueError(
            f"Could not find DraftKings fantasy data in {data_dir}. "
            f"Found files: {[f.name for f in csv_files]}"
        )

    df = pd.concat(all_data, ignore_index=True)

    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['week', 'wk']:
            col_map[col] = 'week'
        elif col_lower in ['name', 'player', 'player_name']:
            col_map[col] = 'name'
        elif col_lower in ['pos', 'position']:
            col_map[col] = 'position'
        elif col_lower in ['team', 'tm']:
            col_map[col] = 'team'
        elif col_lower in ['opp', 'opponent', 'oppt']:
            col_map[col] = 'opponent'
        elif col_lower in ['salary', 'sal', 'dk_salary', 'dksalary', 'price']:
            col_map[col] = 'salary'
        elif col_lower in ['pts', 'points', 'fantasy_points', 'fpts', 'dk_points', 'dkpts', 'actual', 'dkp']:
            col_map[col] = 'fantasy_points'
        elif col_lower in ['h/a', 'home_away', 'home']:
            col_map[col] = 'home'

    df = df.rename(columns=col_map)

    required = ['week', 'name', 'position', 'salary', 'fantasy_points']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing}")

    df['week'] = df['week'].astype(int)
    df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
    df['fantasy_points'] = pd.to_numeric(df['fantasy_points'], errors='coerce')

    df = df.dropna(subset=['name', 'position', 'salary', 'fantasy_points'])

    df['player_id'] = df['name'].str.lower().str.replace(' ', '_')

    df['position'] = df['position'].str.upper().str.strip()
    valid_positions = ['QB', 'RB', 'WR', 'TE', 'DST', 'DEF', 'K']
    df = df[df['position'].isin(valid_positions)]
    df['position'] = df['position'].replace('DEF', 'DST')

    if 'home' not in df.columns:
        if 'opponent' in df.columns:
            df['home'] = (~df['opponent'].astype(str).str.startswith('@')).astype(int)
            df['opponent'] = df['opponent'].str.replace('@', '')
        else:
            df['home'] = 0
    else:
        df['home'] = df['home'].apply(
            lambda x: 1 if str(x).lower() in ['h', 'home', '1', 'true'] else 0
        )

    df = df.sort_values(['week', 'name']).reset_index(drop=True)

    print(f"\nPreprocessed data:")
    print(f"  Weeks: {df['week'].min()}-{df['week'].max()}")
    print(f"  Players: {df['player_id'].nunique()}")
    print(f"  Total rows: {len(df)}")
    print(f"  Positions: {df['position'].value_counts().to_dict()}")
    print(f"  Salary range: ${df['salary'].min():.0f} - ${df['salary'].max():.0f}")
    print(f"  Points range: {df['fantasy_points'].min():.1f} - {df['fantasy_points'].max():.1f}")

    return df


def compute_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['player_id', 'week']).copy()

    df['prev_avg_points'] = df.groupby('player_id')['fantasy_points'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['prev_std_points'] = df.groupby('player_id')['fantasy_points'].transform(
        lambda x: x.shift(1).expanding().std()
    )

    position_avg = df.groupby('position')['fantasy_points'].mean()
    position_std = df.groupby('position')['fantasy_points'].std()

    for pos in df['position'].unique():
        mask = (df['position'] == pos) & df['prev_avg_points'].isna()
        df.loc[mask, 'prev_avg_points'] = position_avg[pos]
        df.loc[mask, 'prev_std_points'] = position_std[pos]

    df['prev_std_points'] = df['prev_std_points'].fillna(0)

    df['salary_normalized'] = df.groupby('week')['salary'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1)
    )

    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 17)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 17)

    for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
        df[f'pos_{pos}'] = (df['position'] == pos).astype(int)

    return df


def save_processed_data(df: pd.DataFrame, output_path: Path):
    df.to_parquet(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")


def download_fantasy(
    data_dir: str = 'data/fantasy',
    skip_download: bool = False,
    season: int = 2020,
):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    output_file = data_path / f'fantasy_{season}.parquet'

    if output_file.exists():
        print(f"Processed data already exists at {output_file}")
        return output_file

    csv_files = list(data_path.glob("*.csv"))

    if not csv_files and not skip_download:
        download_from_kaggle(data_path)
        csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_path}. "
            f"Please download the data manually from:\n"
            f"https://www.kaggle.com/datasets/samuellongenbach/historical-nfl-fantasy-2020\n"
            f"and extract to {data_path}"
        )

    df = load_and_preprocess(data_path, season)
    df = compute_context_features(df)
    save_processed_data(df, output_file)

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Download NFL Fantasy Sports dataset')
    parser.add_argument('--data_dir', type=str, default='data/fantasy',
                        help='Directory to save data')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip Kaggle download, use existing files')
    parser.add_argument('--season', type=int, default=2020,
                        help='NFL season year')

    args = parser.parse_args()
    download_fantasy(args.data_dir, args.skip_download, args.season)


if __name__ == '__main__':
    main()

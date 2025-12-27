#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def download_criteo(data_dir: str = 'data/criteo', subsample_rate: float = 0.1, seed: int = 42):
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    output_file = data_path / f'criteo_subsample_{int(subsample_rate * 100)}.parquet'

    if output_file.exists():
        print(f"Data already exists at {output_file}")
        return output_file

    print("Downloading Criteo Attribution Modeling dataset from HuggingFace...")
    print("This may take a few minutes (16M+ rows)...")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the datasets library: pip install datasets"
        )

    dataset = load_dataset("criteo/criteo-attribution-dataset", split="train")

    print(f"Loaded {len(dataset):,} impressions")

    df = dataset.to_pandas()

    rng = np.random.default_rng(seed)
    n_samples = int(len(df) * subsample_rate)
    indices = rng.choice(len(df), size=n_samples, replace=False)
    df = df.iloc[indices].reset_index(drop=True)

    print(f"Subsampled to {len(df):,} impressions ({subsample_rate:.0%})")

    keep_cols = [
        'timestamp', 'campaign', 'conversion', 'click', 'cost',
        'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9'
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    df['cost'] = df['cost'].fillna(0).astype(float)
    df['conversion'] = df['conversion'].fillna(0).astype(int)
    df['click'] = df['click'].fillna(0).astype(int)

    for col in [f'cat{i}' for i in range(1, 10)]:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)

    n_campaigns = df['campaign'].nunique()
    n_conversions = df['conversion'].sum()
    n_clicks = df['click'].sum()

    print(f"\nDataset statistics:")
    print(f"  Impressions: {len(df):,}")
    print(f"  Campaigns: {n_campaigns:,}")
    print(f"  Conversions: {n_conversions:,} ({100 * n_conversions / len(df):.2f}%)")
    print(f"  Clicks: {n_clicks:,} ({100 * n_clicks / len(df):.2f}%)")
    print(f"  Cost range: [{df['cost'].min():.4f}, {df['cost'].max():.4f}]")

    df.to_parquet(output_file, index=False)
    print(f"\nSaved to {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Download Criteo Attribution dataset')
    parser.add_argument('--data_dir', type=str, default='data/criteo',
                        help='Directory to save data')
    parser.add_argument('--subsample_rate', type=float, default=0.1,
                        help='Fraction of data to keep (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subsampling')

    args = parser.parse_args()
    download_criteo(args.data_dir, args.subsample_rate, args.seed)


if __name__ == '__main__':
    main()

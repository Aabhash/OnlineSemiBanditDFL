import os
import argparse
import urllib.request
import zipfile
from pathlib import Path


MOVIELENS_URLS = {
    '100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    '1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
}


def download_movielens(version: str = '1m', data_dir: str = 'data/movielens'):
    if version not in MOVIELENS_URLS:
        raise ValueError(f"Unknown version: {version}. Available: {list(MOVIELENS_URLS.keys())}")

    url = MOVIELENS_URLS[version]
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    zip_filename = f'ml-{version}.zip'
    zip_path = data_path / zip_filename

    extracted_dir = data_path / f'ml-{version}'
    if extracted_dir.exists():
        print(f"MovieLens-{version.upper()} already exists at {extracted_dir}")
        return str(extracted_dir)

    print(f"Downloading MovieLens-{version.upper()} from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
    except Exception as e:
        print(f"Error downloading: {e}")
        raise

    print(f"Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_path)

    os.remove(zip_path)

    print(f"MovieLens-{version.upper()} extracted to {extracted_dir}")

    if version == '1m':
        ratings_file = extracted_dir / 'ratings.dat'
        movies_file = extracted_dir / 'movies.dat'
    else:
        ratings_file = extracted_dir / 'u.data'
        movies_file = extracted_dir / 'u.item'

    if ratings_file.exists():
        with open(ratings_file, 'r', encoding='latin-1') as f:
            n_ratings = sum(1 for _ in f)
        print(f"  Ratings: {n_ratings:,}")

    return str(extracted_dir)


def main():
    parser = argparse.ArgumentParser(description='Download MovieLens dataset')
    parser.add_argument('--version', type=str, default='1m',
                        choices=['100k', '1m'],
                        help='Dataset version (default: 1m)')
    parser.add_argument('--data_dir', type=str, default='data/movielens',
                        help='Directory to save data (default: data/movielens)')

    args = parser.parse_args()
    download_movielens(args.version, args.data_dir)


if __name__ == '__main__':
    main()

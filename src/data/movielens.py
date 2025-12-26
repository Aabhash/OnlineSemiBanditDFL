import numpy as np
import os
from typing import Tuple, Optional, Dict
from pathlib import Path


class MovieLensData:
    def __init__(
        self,
        n_movies: int = 50,
        context_dim: int = 20,
        data_dir: Optional[str] = None,
        version: str = '1m',
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        self.n_movies = n_movies
        self.context_dim = context_dim
        self.version = version
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        if data_dir is None:
            data_dir = 'data/movielens'

        self.data_path = Path(data_dir) / f'ml-{version}'

        self.user_embeddings = None  # (n_users, context_dim)
        self.movie_features = None   # (n_movies, n_genres)
        self.rating_matrix = None    # (n_users, n_movies) sparse
        self.user_ids = None         # List of user IDs
        self.movie_ids = None        # List of movie IDs (top n)

        # Load real data (required)
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"MovieLens data not found at {self.data_path}\n"
                f"Run: python scripts/download_movielens.py --version {version}"
            )
        self._load_data()

    def _load_data(self):
        if self.version == '1m':
            self._load_1m()
        else:
            self._load_100k()

        print(f"Loaded MovieLens-{self.version.upper()}: "
              f"{len(self.user_ids)} users, {self.n_movies} movies")

    def _load_1m(self):
        ratings_file = self.data_path / 'ratings.dat'
        ratings = []
        with open(ratings_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    user_id = int(parts[0])
                    movie_id = int(parts[1])
                    rating = float(parts[2])
                    ratings.append((user_id, movie_id, rating))

        # Load movies (for genre features)
        movies_file = self.data_path / 'movies.dat'
        genres_all = set()
        movie_genres = {}
        with open(movies_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    movie_id = int(parts[0])
                    genres = parts[2].split('|')
                    movie_genres[movie_id] = genres
                    genres_all.update(genres)

        self._process_data(ratings, movie_genres, list(genres_all))

    def _load_100k(self):
        ratings_file = self.data_path / 'u.data'
        ratings = []
        with open(ratings_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    user_id = int(parts[0])
                    movie_id = int(parts[1])
                    rating = float(parts[2])
                    ratings.append((user_id, movie_id, rating))

        # Load genre info
        genres_file = self.data_path / 'u.genre'
        genres_list = []
        if genres_file.exists():
            with open(genres_file, 'r') as f:
                for line in f:
                    if '|' in line:
                        genre = line.strip().split('|')[0]
                        if genre:
                            genres_list.append(genre)

        items_file = self.data_path / 'u.item'
        movie_genres = {}
        if items_file.exists():
            with open(items_file, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 24:  # 5 fields + 19 genres
                        movie_id = int(parts[0])
                        genre_flags = [int(x) for x in parts[5:24]]
                        movie_genre_list = [genres_list[i] for i, g in enumerate(genre_flags) if g == 1]
                        movie_genres[movie_id] = movie_genre_list

        self._process_data(ratings, movie_genres, genres_list)

    def _process_data(self, ratings, movie_genres, genres_list):
        movie_counts = {}
        for user_id, movie_id, rating in ratings:
            movie_counts[movie_id] = movie_counts.get(movie_id, 0) + 1

        sorted_movies = sorted(movie_counts.items(), key=lambda x: -x[1])
        self.movie_ids = [m[0] for m in sorted_movies[:self.n_movies]]
        movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}

        n_genres = len(genres_list)
        genre_to_idx = {g: i for i, g in enumerate(genres_list)}

        self.movie_features = np.zeros((self.n_movies, n_genres))
        for movie_id in self.movie_ids:
            if movie_id in movie_genres:
                idx = movie_to_idx[movie_id]
                for genre in movie_genres[movie_id]:
                    if genre in genre_to_idx:
                        self.movie_features[idx, genre_to_idx[genre]] = 1.0

        row_sums = self.movie_features.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.movie_features = self.movie_features / row_sums

        user_ratings = {}
        for user_id, movie_id, rating in ratings:
            if movie_id in movie_to_idx:
                if user_id not in user_ratings:
                    user_ratings[user_id] = {}
                user_ratings[user_id][movie_to_idx[movie_id]] = rating

        min_ratings = 5
        self.user_ids = [u for u, r in user_ratings.items() if len(r) >= min_ratings]

        n_users = len(self.user_ids)
        user_to_idx = {u: i for i, u in enumerate(self.user_ids)}

        self.rating_matrix = np.zeros((n_users, self.n_movies))
        self.rating_mask = np.zeros((n_users, self.n_movies), dtype=bool)

        for user_id in self.user_ids:
            user_idx = user_to_idx[user_id]
            for movie_idx, rating in user_ratings[user_id].items():
                self.rating_matrix[user_idx, movie_idx] = rating
                self.rating_mask[user_idx, movie_idx] = True

        self._compute_user_embeddings()

        self.movie_means = np.zeros(self.n_movies)
        for j in range(self.n_movies):
            rated = self.rating_matrix[:, j][self.rating_mask[:, j]]
            if len(rated) > 0:
                self.movie_means[j] = np.mean(rated)
            else:
                self.movie_means[j] = 3.0  # Default to middle rating

    def _compute_user_embeddings(self):
        n_users = len(self.user_ids)

        n_genres = self.movie_features.shape[1]

        user_genre_prefs = np.zeros((n_users, n_genres))
        user_genre_counts = np.zeros((n_users, n_genres))

        for i in range(n_users):
            for j in range(self.n_movies):
                if self.rating_mask[i, j]:
                    rating = self.rating_matrix[i, j]
                    user_genre_prefs[i] += self.movie_features[j] * rating
                    user_genre_counts[i] += self.movie_features[j]

        user_genre_counts[user_genre_counts == 0] = 1
        user_genre_prefs = user_genre_prefs / user_genre_counts

        if n_genres < self.context_dim:
            n_extra = self.context_dim - n_genres
            user_stats = np.zeros((n_users, n_extra))
            for i in range(n_users):
                rated = self.rating_matrix[i][self.rating_mask[i]]
                if len(rated) > 0:
                    if n_extra > 0:
                        user_stats[i, 0] = np.mean(rated) / 5.0
                    if n_extra > 1:
                        user_stats[i, 1] = np.std(rated) / 2.0 if len(rated) > 1 else 0.5
                    if n_extra > 2:
                        user_stats[i, 2] = len(rated) / 100.0
            self.user_embeddings = np.hstack([user_genre_prefs, user_stats])
        else:
            proj = self.rng.standard_normal((n_genres, self.context_dim)) * 0.1
            self.user_embeddings = user_genre_prefs @ proj

        norms = np.linalg.norm(self.user_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.user_embeddings = self.user_embeddings / norms

        self.rating_weight = self.rng.standard_normal((self.context_dim, self.movie_features.shape[1])) * 0.5

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        user_idx = self.rng.integers(len(self.user_ids))
        context = self.user_embeddings[user_idx].copy()

        context += self.rng.standard_normal(self.context_dim) * 0.1

        interaction = context @ self.rating_weight @ self.movie_features.T

        values = self.movie_means + interaction

        values += self.rng.standard_normal(self.n_movies) * self.noise_std

        values = np.clip(values, 1.0, 5.0)

        return context, values

    def get_config(self) -> Dict:
        return {
            'n_movies': self.n_movies,
            'context_dim': self.context_dim,
            'version': self.version,
            'n_users': len(self.user_ids) if self.user_ids else 0,
            'rating_range': (1, 5),
        }

"""Memory-efficient data loading for MovieLens 25M dataset."""
import time
import pandas as pd
from typing import Tuple


def load_ratings(path: str) -> pd.DataFrame:
    """Load ratings.csv with optimized dtypes (~60% memory reduction).

    Args:
        path: Absolute path to ratings.csv

    Returns:
        DataFrame with columns [userId, movieId, rating, timestamp]
    """
    dtypes = {
        'userId': 'int32',
        'movieId': 'int32',
        'rating': 'float32',
        'timestamp': 'int32',
    }
    t0 = time.time()
    df = pd.read_csv(path, dtype=dtypes)
    elapsed = time.time() - t0
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f'  Ratings : {len(df):>12,} rows | {mem_mb:.1f} MB | {elapsed:.1f}s')
    return df


def load_movies(path: str) -> pd.DataFrame:
    """Load movies.csv and parse pipe-separated genres into lists.

    Args:
        path: Absolute path to movies.csv

    Returns:
        DataFrame with columns [movieId, title, genres, genres_list]
    """
    df = pd.read_csv(path)
    df['genres_list'] = df['genres'].str.split('|')
    print(f'  Movies  : {len(df):>12,} rows')
    return df


def load_links(path: str) -> pd.DataFrame:
    """Load links.csv for external ID mapping.

    Args:
        path: Absolute path to links.csv

    Returns:
        DataFrame with columns [movieId, imdbId, tmdbId]
    """
    df = pd.read_csv(path, dtype={'movieId': 'int32', 'imdbId': 'str', 'tmdbId': 'float32'})
    print(f'  Links   : {len(df):>12,} rows')
    return df


def load_all(cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and movies using config paths.

    Args:
        cfg: Config object with path attributes

    Returns:
        Tuple of (ratings_df, movies_df)
    """
    print('Loading data...')
    ratings = load_ratings(cfg.RATINGS_PATH)
    movies = load_movies(cfg.MOVIES_PATH)
    return ratings, movies

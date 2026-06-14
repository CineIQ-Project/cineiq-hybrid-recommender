"""Preprocessing: filtering, merging, sparsity analysis for MovieLens data."""
import pandas as pd
import numpy as np
from typing import Tuple


def filter_sparse_users(df: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
    """Remove users with fewer than min_ratings ratings.

    Args:
        df: Ratings DataFrame
        min_ratings: Minimum number of ratings to keep a user

    Returns:
        Filtered DataFrame
    """
    counts = df['userId'].value_counts()
    valid = counts[counts >= min_ratings].index
    filtered = df[df['userId'].isin(valid)]
    print(f'  Users filter (>={min_ratings}): {df["userId"].nunique():,} -> {filtered["userId"].nunique():,}')
    return filtered


def filter_rare_movies(df: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
    """Remove movies with fewer than min_ratings ratings.

    Args:
        df: Ratings DataFrame
        min_ratings: Minimum number of ratings to keep a movie

    Returns:
        Filtered DataFrame
    """
    counts = df['movieId'].value_counts()
    valid = counts[counts >= min_ratings].index
    filtered = df[df['movieId'].isin(valid)]
    print(f'  Movies filter (>={min_ratings}): {df["movieId"].nunique():,} -> {filtered["movieId"].nunique():,}')
    return filtered


def merge_metadata(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Left-join ratings with movie metadata.

    Args:
        ratings: Ratings DataFrame
        movies: Movies DataFrame

    Returns:
        Merged DataFrame with movie title and genres
    """
    merged = ratings.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    return merged


def compute_sparsity(df: pd.DataFrame) -> dict:
    """Compute user-movie matrix sparsity statistics.

    Args:
        df: Ratings DataFrame

    Returns:
        Dict with n_users, n_movies, n_ratings, sparsity, density
    """
    n_users = df['userId'].nunique()
    n_movies = df['movieId'].nunique()
    n_ratings = len(df)
    total_cells = n_users * n_movies
    density = n_ratings / total_cells
    return {
        'n_users': n_users,
        'n_movies': n_movies,
        'n_ratings': n_ratings,
        'total_cells': total_cells,
        'sparsity': 1 - density,
        'density_pct': density * 100,
    }


def preprocess_pipeline(ratings: pd.DataFrame, movies: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline.

    Args:
        ratings: Raw ratings DataFrame
        movies: Movies DataFrame
        cfg: Config object

    Returns:
        Tuple of (filtered_ratings, merged_with_metadata)
    """
    print('Preprocessing...')
    n_before = len(ratings)
    filtered = filter_sparse_users(ratings, cfg.MIN_USER_RATINGS)
    filtered = filter_rare_movies(filtered, cfg.MIN_MOVIE_RATINGS)
    n_after = len(filtered)
    print(f'  Ratings: {n_before:,} -> {n_after:,} ({n_after/n_before*100:.1f}% retained)')

    stats = compute_sparsity(filtered)
    print(f'  Matrix : {stats["n_users"]:,} users x {stats["n_movies"]:,} movies')
    print(f'  Density: {stats["density_pct"]:.4f}%  |  Sparsity: {stats["sparsity"]*100:.4f}%')

    merged = merge_metadata(filtered, movies)
    return filtered, merged

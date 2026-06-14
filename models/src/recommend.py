"""Recommendation functions: top-N, similar users/movies, cold-start."""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def get_top_n_recommendations(algo, user_id: int, movies_df: pd.DataFrame,
                               ratings_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get top-N movie recommendations for a user.

    Predicts ratings for all movies the user hasn't rated,
    returns the top-N by predicted score.

    Args:
        algo: Trained Surprise SVD model
        user_id: Target user ID
        movies_df: Movies DataFrame with movieId, title, genres
        ratings_df: Ratings DataFrame to find already-rated movies
        n: Number of recommendations

    Returns:
        DataFrame with columns [movieId, title, genres, predicted_rating]
    """
    # Movies the user has already rated
    rated_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].values)
    all_movies = set(movies_df['movieId'].values)
    unseen = all_movies - rated_movies

    # Predict ratings for unseen movies
    predictions = []
    for mid in unseen:
        pred = algo.predict(user_id, mid)
        predictions.append((mid, pred.est))

    # Sort by predicted rating descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]

    # Build result DataFrame
    result = pd.DataFrame(top_n, columns=['movieId', 'predicted_rating'])
    result = result.merge(movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')
    result = result[['movieId', 'title', 'genres', 'predicted_rating']]
    result['predicted_rating'] = result['predicted_rating'].round(3)
    result.index = range(1, len(result) + 1)
    result.index.name = 'rank'
    return result


def get_similar_users(algo, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
    """Find most similar users based on latent factor cosine similarity.

    Args:
        algo: Trained Surprise SVD model (must have pu attribute)
        user_id: Target user ID
        n: Number of similar users to return

    Returns:
        List of (userId, similarity_score) tuples, sorted by similarity desc
    """
    trainset = algo.trainset
    try:
        inner_id = trainset.to_inner_uid(user_id)
    except ValueError:
        print(f'  User {user_id} not found in training data.')
        return []

    user_vector = algo.pu[inner_id]
    # Compute cosine similarity against all users
    norms = np.linalg.norm(algo.pu, axis=1)
    user_norm = np.linalg.norm(user_vector)

    if user_norm == 0:
        return []

    similarities = algo.pu @ user_vector / (norms * user_norm + 1e-9)
    # Exclude the user themselves
    similarities[inner_id] = -1

    top_inner = np.argsort(similarities)[::-1][:n]
    results = []
    for iid in top_inner:
        try:
            raw_uid = trainset.to_raw_uid(iid)
            results.append((raw_uid, round(float(similarities[iid]), 4)))
        except ValueError:
            continue
    return results


def get_similar_movies(algo, movie_id: int, movies_df: pd.DataFrame,
                        n: int = 10) -> pd.DataFrame:
    """Find most similar movies based on item latent factors.

    Args:
        algo: Trained SVD model
        movie_id: Target movie ID
        movies_df: Movies DataFrame
        n: Number of similar movies

    Returns:
        DataFrame with movieId, title, genres, similarity
    """
    trainset = algo.trainset
    try:
        inner_id = trainset.to_inner_iid(movie_id)
    except ValueError:
        print(f'  Movie {movie_id} not found in training data.')
        return pd.DataFrame()

    item_vector = algo.qi[inner_id]
    norms = np.linalg.norm(algo.qi, axis=1)
    item_norm = np.linalg.norm(item_vector)

    if item_norm == 0:
        return pd.DataFrame()

    similarities = algo.qi @ item_vector / (norms * item_norm + 1e-9)
    similarities[inner_id] = -1

    top_inner = np.argsort(similarities)[::-1][:n]
    rows = []
    for iid in top_inner:
        try:
            raw_mid = trainset.to_raw_iid(iid)
            rows.append((raw_mid, round(float(similarities[iid]), 4)))
        except ValueError:
            continue

    result = pd.DataFrame(rows, columns=['movieId', 'similarity'])
    result = result.merge(movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')
    result = result[['movieId', 'title', 'genres', 'similarity']]
    result.index = range(1, len(result) + 1)
    result.index.name = 'rank'
    return result


def handle_cold_start(movies_df: pd.DataFrame, ratings_df: pd.DataFrame,
                      strategy: str = 'popular', n: int = 10,
                      genre: Optional[str] = None) -> pd.DataFrame:
    """Provide recommendations for cold-start users (no history).

    Args:
        movies_df: Movies DataFrame
        ratings_df: Ratings DataFrame
        strategy: 'popular' (most rated) or 'top_rated' (highest avg) or 'genre'
        n: Number of recommendations
        genre: Genre filter (required if strategy='genre')

    Returns:
        DataFrame of recommended movies
    """
    if strategy == 'popular':
        popular = (ratings_df.groupby('movieId')
                   .agg(n_ratings=('rating', 'count'), avg_rating=('rating', 'mean'))
                   .reset_index()
                   .sort_values('n_ratings', ascending=False)
                   .head(n))
    elif strategy == 'top_rated':
        popular = (ratings_df.groupby('movieId')
                   .agg(n_ratings=('rating', 'count'), avg_rating=('rating', 'mean'))
                   .reset_index()
                   .query('n_ratings >= 100')
                   .sort_values('avg_rating', ascending=False)
                   .head(n))
    elif strategy == 'genre' and genre:
        genre_movies = movies_df[movies_df['genres'].str.contains(genre, na=False)]
        popular = (ratings_df[ratings_df['movieId'].isin(genre_movies['movieId'])]
                   .groupby('movieId')
                   .agg(n_ratings=('rating', 'count'), avg_rating=('rating', 'mean'))
                   .reset_index()
                   .query('n_ratings >= 50')
                   .sort_values('avg_rating', ascending=False)
                   .head(n))
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    popular = popular.merge(movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')
    popular = popular[['movieId', 'title', 'genres', 'n_ratings', 'avg_rating']]
    popular['avg_rating'] = popular['avg_rating'].round(3)
    popular.index = range(1, len(popular) + 1)
    popular.index.name = 'rank'
    return popular

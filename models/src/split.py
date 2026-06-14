"""Train-test splitting strategies for Surprise-based collaborative filtering."""
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split


def to_surprise_dataset(df: pd.DataFrame, rating_min: float = 0.5, rating_max: float = 5.0) -> Dataset:
    """Convert a pandas DataFrame to a Surprise Dataset.

    Args:
        df: DataFrame with columns [userId, movieId, rating]
        rating_min: Minimum possible rating
        rating_max: Maximum possible rating

    Returns:
        Surprise Dataset object
    """
    reader = Reader(rating_scale=(rating_min, rating_max))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    return data


def random_split(data: Dataset, test_size: float = 0.2, random_state: int = 42):
    """Standard random train-test split.

    Args:
        data: Surprise Dataset
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        Tuple of (trainset, testset)
    """
    trainset, testset = surprise_split(data, test_size=test_size, random_state=random_state)
    print(f'  Random split : train={trainset.n_ratings:,}  test={len(testset):,}')
    return trainset, testset


def temporal_split(df: pd.DataFrame, test_ratio: float = 0.2,
                   rating_min: float = 0.5, rating_max: float = 5.0):
    """Time-aware split: train on older ratings, test on newer.

    Args:
        df: Ratings DataFrame with timestamp column
        test_ratio: Fraction of newest ratings for test
        rating_min: Minimum possible rating
        rating_max: Maximum possible rating

    Returns:
        Tuple of (trainset, testset)
    """
    df_sorted = df.sort_values('timestamp')
    split_idx = int(len(df_sorted) * (1 - test_ratio))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    reader = Reader(rating_scale=(rating_min, rating_max))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()

    testset = list(test_df[['userId', 'movieId', 'rating']].itertuples(index=False, name=None))
    print(f'  Temporal split: train={trainset.n_ratings:,}  test={len(testset):,}')
    return trainset, testset

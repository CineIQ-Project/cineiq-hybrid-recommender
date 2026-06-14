"""Evaluation metrics: RMSE, MAE, Precision@K, Recall@K."""
from collections import defaultdict
from typing import Dict, List, Tuple
from surprise import accuracy


def compute_rmse_mae(predictions) -> Dict[str, float]:
    """Compute RMSE and MAE from Surprise predictions.

    Args:
        predictions: List of Surprise Prediction objects

    Returns:
        Dict with 'rmse' and 'mae' keys
    """
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    print(f'  RMSE: {rmse:.4f}')
    print(f'  MAE:  {mae:.4f}')
    return {'rmse': rmse, 'mae': mae}


def precision_recall_at_k(predictions, k: int = 10,
                          threshold: float = 3.5) -> Tuple[float, float]:
    """Compute macro-averaged Precision@K and Recall@K.

    A relevant item is one whose true rating >= threshold.

    Args:
        predictions: List of Surprise Prediction objects
        k: Number of top recommendations to consider
        threshold: Rating threshold for relevance

    Returns:
        Tuple of (mean_precision, mean_recall)
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []

    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating (descending)
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_relevant = sum(1 for (_, true_r) in user_ratings if true_r >= threshold)
        n_relevant_in_k = sum(1 for (_, true_r) in top_k if true_r >= threshold)

        precision = n_relevant_in_k / k if k > 0 else 0
        recall = n_relevant_in_k / n_relevant if n_relevant > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    mean_p = sum(precisions) / len(precisions) if precisions else 0
    mean_r = sum(recalls) / len(recalls) if recalls else 0

    print(f'  Precision@{k}: {mean_p:.4f}')
    print(f'  Recall@{k}:    {mean_r:.4f}')
    return mean_p, mean_r


def evaluate_pipeline(algo, testset, k: int = 10,
                      threshold: float = 3.5) -> Dict:
    """Full evaluation: rating accuracy + ranking metrics.

    Args:
        algo: Trained Surprise algorithm
        testset: Surprise testset
        k: Top-K for precision/recall
        threshold: Relevance threshold

    Returns:
        Dict with all metrics
    """
    print('Evaluating model...')
    predictions = algo.test(testset)
    metrics = compute_rmse_mae(predictions)
    prec, rec = precision_recall_at_k(predictions, k, threshold)
    metrics['precision_at_k'] = prec
    metrics['recall_at_k'] = rec
    metrics['k'] = k
    metrics['threshold'] = threshold
    return metrics

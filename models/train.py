"""SVD model training and hyperparameter tuning."""
import time
from typing import Dict, Optional, Tuple
from surprise import SVD
from surprise.model_selection import GridSearchCV


def build_svd(n_factors: int = 150, n_epochs: int = 5,
              lr_all: float = 0.005, reg_all: float = 0.02,
              random_state: int = 42) -> SVD:
    """Create an SVD instance with specified hyperparameters.

    Args:
        n_factors: Number of latent factors
        n_epochs: Number of SGD epochs
        lr_all: Learning rate
        reg_all: Regularization term
        random_state: Random seed

    Returns:
        Configured SVD instance
    """
    return SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
        verbose=True,
    )


def train_svd(algo: SVD, trainset) -> SVD:
    """Train an SVD model on the given trainset.

    Args:
        algo: SVD algorithm instance
        trainset: Surprise Trainset object

    Returns:
        Trained SVD model
    """
    print('Training SVD...')
    t0 = time.time()
    algo.fit(trainset)
    elapsed = time.time() - t0
    print(f'  Training complete in {elapsed:.1f}s')
    print(f'  Factors: {algo.n_factors} | Epochs: {algo.n_epochs}')
    print(f'  Users: {trainset.n_users:,} | Items: {trainset.n_items:,}')
    return algo


def tune_svd(data, param_grid: Optional[Dict] = None,
             cv: int = 3, measure: str = 'rmse') -> Dict:
    """Run GridSearchCV for hyperparameter tuning.

    Args:
        data: Surprise Dataset
        param_grid: Dict of param lists to search
        cv: Number of cross-validation folds
        measure: Metric to optimize ('rmse' or 'mae')

    Returns:
        Dict with best_params, best_score, and cv_results
    """
    if param_grid is None:
        param_grid = {
            'n_factors': [100, 150],
            'n_epochs': [20, 30],
            'lr_all': [0.005, 0.01],
            'reg_all': [0.02, 0.1],
        }

    print(f'Tuning SVD with {cv}-fold CV...')
    combos = 1
    for v in param_grid.values():
        combos *= len(v)
    print(f'  Grid size: {combos} combinations')

    t0 = time.time()
    gs = GridSearchCV(SVD, param_grid, measures=[measure, 'mae'], cv=cv, n_jobs=-1)
    gs.fit(data)
    elapsed = time.time() - t0

    print(f'  Tuning complete in {elapsed:.1f}s')
    print(f'  Best {measure.upper()}: {gs.best_score[measure]:.4f}')
    print(f'  Best params: {gs.best_params[measure]}')

    return {
        'best_params': gs.best_params[measure],
        'best_score': gs.best_score[measure],
        'cv_results': gs.cv_results,
    }


def train_pipeline(cfg) -> Tuple:
    """End-to-end training pipeline using config.

    Args:
        cfg: Config object

    Returns:
        Tuple of (trained_algo, trainset, testset)
    """
    from .data_loader import load_all
    from .preprocess import preprocess_pipeline
    from .split import to_surprise_dataset, random_split

    ratings, movies = load_all(cfg)
    filtered, _ = preprocess_pipeline(ratings, movies, cfg)
    data = to_surprise_dataset(filtered, cfg.RATING_MIN, cfg.RATING_MAX)
    trainset, testset = random_split(data, cfg.TEST_SIZE, cfg.RANDOM_STATE)
    algo = build_svd(cfg.N_FACTORS, cfg.N_EPOCHS, cfg.LR_ALL, cfg.REG_ALL, cfg.RANDOM_STATE)
    algo = train_svd(algo, trainset)
    return algo, trainset, testset

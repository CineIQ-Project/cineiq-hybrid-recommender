"""Model I/O: save and load SVD models + ID mappings."""
import json
import os
import pickle
from datetime import datetime
from typing import Dict


def save_model(algo, path: str) -> str:
    """Save trained SVD model using pickle.

    Args:
        algo: Trained Surprise SVD algorithm
        path: Output file path

    Returns:
        Absolute path to saved model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(algo, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f'  Model saved: {path} ({size_mb:.1f} MB)')
    return path


def load_model(path: str):
    """Load a saved SVD model.

    Args:
        path: Path to pickle file

    Returns:
        Loaded SVD algorithm
    """
    with open(path, 'rb') as f:
        algo = pickle.load(f)
    print(f'  Model loaded: {path}')
    return algo


def save_mappings(trainset, path: str) -> str:
    """Save userId <-> innerIdx and movieId <-> innerIdx mappings.

    Args:
        trainset: Surprise Trainset object
        path: Output JSON file path

    Returns:
        Absolute path to saved mappings
    """
    user_map = {str(trainset.to_raw_uid(i)): i for i in range(trainset.n_users)}
    item_map = {str(trainset.to_raw_iid(i)): i for i in range(trainset.n_items)}

    mappings = {
        'user_to_inner': user_map,
        'item_to_inner': item_map,
        'n_users': trainset.n_users,
        'n_items': trainset.n_items,
        'n_ratings': trainset.n_ratings,
        'created_at': datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(mappings, f)
    print(f'  Mappings saved: {path}')
    return path


def load_mappings(path: str) -> Dict:
    """Load saved mappings.

    Args:
        path: Path to JSON mappings file

    Returns:
        Dict with user_to_inner and item_to_inner maps
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_artifacts(algo, trainset, cfg) -> Dict[str, str]:
    """Save all model artifacts (model + mappings).

    Args:
        algo: Trained SVD model
        trainset: Surprise Trainset
        cfg: Config object

    Returns:
        Dict with paths to saved artifacts
    """
    print('Saving artifacts...')
    model_path = save_model(algo, cfg.MODEL_PATH)
    mappings_path = save_mappings(trainset, cfg.MAPPINGS_PATH)
    return {'model': model_path, 'mappings': mappings_path}

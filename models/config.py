"""Central configuration for the CineIQ collaborative filtering pipeline."""
import os
from dataclasses import dataclass, field

# Auto-detect project root
_candidates = [os.getcwd(), os.path.dirname(os.getcwd()), os.path.dirname(os.path.abspath(__file__ if '__file__' in dir() else '.'))]
PROJECT_ROOT = next((p for p in _candidates if os.path.exists(os.path.join(p, 'ratings.csv'))), _candidates[0])


@dataclass
class Config:
    """All tuneable parameters for the pipeline."""

    # ── Paths ──
    PROJECT_ROOT: str = PROJECT_ROOT
    RATINGS_PATH: str = field(default='')
    MOVIES_PATH: str = field(default='')
    LINKS_PATH: str = field(default='')
    ARTIFACTS_DIR: str = field(default='')
    MODEL_PATH: str = field(default='')
    MAPPINGS_PATH: str = field(default='')

    # ── Filtering ──
    MIN_USER_RATINGS: int = 20
    MIN_MOVIE_RATINGS: int = 10

    # ── Model Hyperparameters ──
    N_FACTORS: int = 150
    N_EPOCHS: int = 30
    LR_ALL: float = 0.005
    REG_ALL: float = 0.02

    # ── Evaluation ──
    TOP_K: int = 10
    RELEVANCE_THRESHOLD: float = 3.5

    # ── Split ──
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # ── Rating Scale ──
    RATING_MIN: float = 0.5
    RATING_MAX: float = 5.0

    def __post_init__(self) -> None:
        root = self.PROJECT_ROOT
        self.RATINGS_PATH = self.RATINGS_PATH or os.path.join(root, 'ratings.csv')
        self.MOVIES_PATH = self.MOVIES_PATH or os.path.join(root, 'movies.csv')
        self.LINKS_PATH = self.LINKS_PATH or os.path.join(root, 'links.csv')
        self.ARTIFACTS_DIR = self.ARTIFACTS_DIR or os.path.join(root, 'artifacts')
        self.MODEL_PATH = self.MODEL_PATH or os.path.join(self.ARTIFACTS_DIR, 'svd_model.pkl')
        self.MAPPINGS_PATH = self.MAPPINGS_PATH or os.path.join(self.ARTIFACTS_DIR, 'mappings.json')
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)

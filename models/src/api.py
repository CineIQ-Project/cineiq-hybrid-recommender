"""FastAPI-ready endpoint functions for the recommendation API."""
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class RecommendationRequest:
    """Request schema for recommendations."""
    user_id: int
    n: int = 10
    strategy: str = 'svd'  # 'svd', 'popular', 'top_rated', 'genre'
    genre: Optional[str] = None


@dataclass
class MovieRecommendation:
    """Single movie recommendation."""
    movie_id: int
    title: str
    genres: str
    score: float


@dataclass
class RecommendationResponse:
    """Response schema for recommendations."""
    user_id: int
    strategy: str
    recommendations: List[MovieRecommendation] = field(default_factory=list)


# ── Lazy-loaded global state ──
_model_cache = {}


def _get_model(cfg):
    """Lazy-load model and data (cached after first call)."""
    if 'algo' not in _model_cache:
        from .model_io import load_model
        from .data_loader import load_all
        from .preprocess import preprocess_pipeline

        algo = load_model(cfg.MODEL_PATH)
        ratings, movies = load_all(cfg)
        filtered, _ = preprocess_pipeline(ratings, movies, cfg)
        _model_cache['algo'] = algo
        _model_cache['movies'] = movies
        _model_cache['ratings'] = filtered
    return _model_cache['algo'], _model_cache['movies'], _model_cache['ratings']


def get_recommendations_endpoint(request: RecommendationRequest, cfg) -> RecommendationResponse:
    """Production-ready recommendation function.

    This can be directly wired into a FastAPI route:
        @app.post('/recommend')
        def recommend(request: RecommendationRequest):
            return get_recommendations_endpoint(request, cfg)

    Args:
        request: RecommendationRequest with user_id, n, strategy
        cfg: Config object

    Returns:
        RecommendationResponse with ranked recommendations
    """
    algo, movies_df, ratings_df = _get_model(cfg)

    if request.strategy == 'svd':
        from .recommend import get_top_n_recommendations
        df = get_top_n_recommendations(algo, request.user_id, movies_df, ratings_df, request.n)
    else:
        from .recommend import handle_cold_start
        df = handle_cold_start(movies_df, ratings_df, request.strategy, request.n, request.genre)

    recs = []
    for _, row in df.iterrows():
        score = row.get('predicted_rating', row.get('avg_rating', 0))
        recs.append(MovieRecommendation(
            movie_id=int(row['movieId']),
            title=str(row['title']),
            genres=str(row['genres']),
            score=float(score),
        ))

    return RecommendationResponse(
        user_id=request.user_id,
        strategy=request.strategy,
        recommendations=recs,
    )


def health_check(cfg) -> dict:
    """API health check endpoint.

    Returns:
        Dict with model status
    """
    import os
    model_exists = os.path.exists(cfg.MODEL_PATH)
    return {
        'status': 'healthy' if model_exists else 'model_not_found',
        'model_path': cfg.MODEL_PATH,
        'model_loaded': 'algo' in _model_cache,
    }

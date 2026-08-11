"""
src/api/main.py

FastAPI recommendation service — serves the current champion model.

Endpoints:
    GET /health                   — liveness probe
    GET /model                    — current champion model info
    GET /recommend/{user_id}?k=10 — top-K recommendations for a user
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Movie Recommender API",
    description="MLOps Movie Recommendation System — Champion Model",
    version="2.0.0",
)

# ── State (loaded at startup) ─────────────────────────────────────────────────

_model = None
_movies_df: pd.DataFrame | None = None
_ratings_df: pd.DataFrame | None = None
_model_meta: dict = {}


def _load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@app.on_event("startup")
def startup():
    global _model, _movies_df, _ratings_df, _model_meta
    cfg = _load_config()
    api_cfg = cfg["api"]

    # Load champion model
    model_path = Path(api_cfg["model_path"])
    if model_path.exists():
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        logger.info("Champion model loaded from %s", model_path)
    else:
        logger.warning("No champion model found at %s — /recommend will return 404", model_path)

    # Load champion meta
    meta_path = Path("models/champion_meta.yaml")
    if meta_path.exists():
        with open(meta_path) as f:
            _model_meta = yaml.safe_load(f)

    # Load movies
    movies_path = Path(api_cfg["movies_path"])
    if movies_path.exists():
        _movies_df = pd.read_csv(movies_path)

    # Load ratings (for seen-movie filtering)
    ratings_path = Path(api_cfg["ratings_path"])
    if ratings_path.exists():
        _ratings_df = pd.read_csv(ratings_path)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model")
def model_info():
    return {
        "model_type": _model_meta.get("model_type", "unknown"),
        "metrics": _model_meta.get("metrics", {}),
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    if _model is None:
        raise HTTPException(status_code=503, detail="No champion model loaded.")

    model_type = _model_meta.get("model_type", "svd")

    if model_type == "svd":
        return _recommend_svd(user_id, k)
    elif model_type == "bert4rec":
        return _recommend_bert4rec(user_id, k)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown model type: {model_type}")


def _recommend_svd(user_id: int, k: int) -> dict:
    """Generate top-K recommendations using the SVD model."""
    if _ratings_df is None or _movies_df is None:
        raise HTTPException(status_code=503, detail="Rating data not loaded.")

    seen = set(_ratings_df[_ratings_df["userId"] == user_id]["movieId"].tolist())
    all_movies = _movies_df["movieId"].tolist()
    unseen = [m for m in all_movies if m not in seen]

    predictions = [(_model.predict(user_id, m) if hasattr(_model, "predict") else
                    type("P", (), {"est": 0.0})()) for m in unseen]
    scored = sorted(
        [(m, p.est) for m, p in zip(unseen, predictions)],
        key=lambda x: x[1], reverse=True,
    )[:k]

    movie_info = _movies_df.set_index("movieId")
    return {
        "user_id": user_id,
        "model_type": "svd",
        "recommendations": [
            {
                "movieId": m,
                "predicted_rating": round(score, 3),
                "title": movie_info.loc[m, "title"] if m in movie_info.index else "Unknown",
                "genres": movie_info.loc[m, "genres"] if m in movie_info.index else "Unknown",
            }
            for m, score in scored
        ],
    }


def _recommend_bert4rec(user_id: int, k: int) -> dict:
    """Generate top-K recommendations using the BERT4Rec model."""
    if _ratings_df is None:
        raise HTTPException(status_code=503, detail="Rating data not loaded.")

    user_ratings = _ratings_df[_ratings_df["userId"] == user_id].sort_values("timestamp")
    if user_ratings.empty:
        raise HTTPException(status_code=404, detail=f"No history for user {user_id}.")

    movie_to_idx = _model.movie_to_idx
    sequence = [
        movie_to_idx[m] for m in user_ratings["movieId"].tolist()
        if m in movie_to_idx
    ][-50:]  # last 50 interactions

    recs = _model.recommend(user_id, n=k, user_sequence=sequence)
    movie_info = _movies_df.set_index("movieId") if _movies_df is not None else None

    return {
        "user_id": user_id,
        "model_type": "bert4rec",
        "recommendations": [
            {
                "movieId": r["movieId"],
                "score": round(r["score"], 4),
                "title": (movie_info.loc[r["movieId"], "title"]
                          if movie_info is not None and r["movieId"] in movie_info.index
                          else "Unknown"),
            }
            for r in recs
        ],
    }

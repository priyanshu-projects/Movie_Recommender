"""
src/serving/app.py

FastAPI serving layer — wraps the trained SVD model and exposes:
  GET /recommend/{user_id}?n=10
      → returns top-N movie recommendations for a given user

  GET /health
      → liveness probe for Cloud Run / Docker

The model is loaded once at startup from models/svd_model.pkl.
Movie metadata is loaded from data/raw/.../movies.csv for title lookups.

Run locally:
    uvicorn src.serving.app:app --reload --port 8000

Then test:
    curl http://localhost:8000/recommend/1?n=5
"""

import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/svd_model.pkl"))
MOVIES_PATH = Path(os.getenv("MOVIES_PATH", "data/raw/ml-latest-small/movies.csv"))
RATINGS_PATH = Path(os.getenv("RATINGS_PATH", "data/raw/ml-latest-small/ratings.csv"))

app = FastAPI(
    title="Movie Recommendation API",
    description="SVD collaborative filtering recommendations — MLOps portfolio project",
    version="0.1.0",
)

# ── Startup: load model and data once ──────────────────────────────────────────

_model = None
_movies_df = None
_all_movie_ids = None
_user_seen = None  # dict: userId → set of movieIds already rated


@app.on_event("startup")
def load_artifacts():
    global _model, _movies_df, _all_movie_ids, _user_seen

    if not MODEL_PATH.exists():
        logger.warning("Model not found at %s — /recommend will return 503.", MODEL_PATH)
        return

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    logger.info("Model loaded from %s", MODEL_PATH)

    _movies_df = pd.read_csv(MOVIES_PATH).set_index("movieId")
    _all_movie_ids = list(_movies_df.index)

    ratings = pd.read_csv(RATINGS_PATH)
    _user_seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()
    logger.info("Metadata loaded. %d movies available.", len(_all_movie_ids))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    seen = _user_seen.get(user_id, set())
    candidates = [mid for mid in _all_movie_ids if mid not in seen]

    if not candidates:
        raise HTTPException(status_code=404, detail=f"No unseen movies for user {user_id}.")

    predictions = [_model.predict(user_id, mid) for mid in candidates]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]

    results = []
    for pred in top_n:
        movie_info = _movies_df.loc[pred.iid] if pred.iid in _movies_df.index else {}
        results.append({
            "movieId": pred.iid,
            "predicted_rating": round(pred.est, 3),
            "title": movie_info.get("title", "Unknown"),
            "genres": movie_info.get("genres", ""),
        })

    return JSONResponse(content={"user_id": user_id, "recommendations": results})

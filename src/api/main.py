"""
src/api/main.py

FastAPI recommendation service — serves the current champion model.

Endpoints:
    GET /health                   — liveness probe
    GET /model                    — current champion model info
    GET /recommend/{user_id}?k=10 — top-K recommendations for a user
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException

from src.inference.recommender import Recommender, CHAMPION_META

logger = logging.getLogger(__name__)

recommender: Recommender | None = None
champion_meta: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender, champion_meta
    try:
        recommender = Recommender("configs/config.yaml").load()
        logger.info("Champion model loaded successfully via Recommender.")
    except Exception as e:
        logger.warning("Could not load champion model at startup: %s", e)
        recommender = None

    if CHAMPION_META.exists():
        try:
            with open(CHAMPION_META) as f:
                champion_meta = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Could not load champion metadata: %s", e)

    yield


app = FastAPI(
    title="Movie Recommender API",
    description="MLOps Movie Recommendation System — Champion Model",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": recommender is not None and recommender.model is not None,
    }


@app.get("/model")
def model_info():
    if recommender is None:
        return {"status": "no model loaded", "metrics": {}}
    return {
        "model_type": recommender.model_type,
        "metrics": champion_meta.get("metrics", {}),
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    if recommender is None or recommender.model is None:
        raise HTTPException(status_code=503, detail="No champion model loaded.")

    try:
        recs = recommender.recommend(user_id=user_id, n=k)
        return {
            "user_id": user_id,
            "model_type": recommender.model_type,
            "recommendations": recs,
        }
    except Exception as e:
        logger.error("Error generating recommendations for user %d: %s", user_id, e)
        raise HTTPException(status_code=500, detail=str(e))

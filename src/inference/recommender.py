"""
src/inference/recommender.py

Unified recommender — loads the current champion model (SVD or BERT4Rec)
and provides a single recommend() interface regardless of model type.

The API server (src/api/main.py) uses this class.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CHAMPION_PATH     = Path("models/champion_model.pkl")
CHAMPION_META     = Path("models/champion_meta.yaml")


class Recommender:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.model = None
        self.model_type: str = "unknown"
        self.ratings_df: pd.DataFrame | None = None
        self.movies_df:  pd.DataFrame | None = None

    def load(self) -> "Recommender":
        """Load champion model and supporting data."""
        if not CHAMPION_PATH.exists():
            raise FileNotFoundError("No champion model found. Train and promote a model first.")

        # Load model meta
        if CHAMPION_META.exists():
            with open(CHAMPION_META) as f:
                meta = yaml.safe_load(f)
            self.model_type = meta.get("model_type", "svd")

        # Load champion model
        with open(CHAMPION_PATH, "rb") as f:
            payload = pickle.load(f)

        if self.model_type == "svd":
            from src.models.svd_model import SVDModel
            self.model = SVDModel(self.cfg.get("svd", {}))
            self.model._algo = payload.get("algo") or payload
            self.model._all_movie_ids = payload.get("all_movie_ids", [])
        elif self.model_type == "bert4rec":
            from src.models.bert4rec import BERT4Rec
            self.model = BERT4Rec(self.cfg.get("bert4rec", {}))
            self.model.load(CHAMPION_PATH)

        # Load supporting data
        api_cfg = self.cfg.get("api", {})
        ratings_path = Path(api_cfg.get("ratings_path", "data/raw/ml-latest-small/ratings.csv"))
        movies_path  = Path(api_cfg.get("movies_path",  "data/raw/ml-latest-small/movies.csv"))
        if ratings_path.exists():
            self.ratings_df = pd.read_csv(ratings_path)
        if movies_path.exists():
            self.movies_df = pd.read_csv(movies_path)

        logger.info("Recommender loaded: model_type=%s", self.model_type)
        return self

    def recommend(self, user_id: int, n: int = 10) -> list[dict]:
        """Return top-N recommendations for a user with movie metadata."""
        if self.model is None:
            raise RuntimeError("Call load() first.")

        if self.ratings_df is not None:
            seen = set(self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"].tolist())
        else:
            seen = set()

        if self.model_type == "svd":
            recs = self.model.recommend(user_id, n=n, seen_movie_ids=list(seen))
        elif self.model_type == "bert4rec":
            sequence = self._build_sequence(user_id)
            recs = self.model.recommend(user_id, n=n, user_sequence=sequence)
        else:
            recs = []

        return self._enrich(recs)

    def _build_sequence(self, user_id: int) -> list[int]:
        """Build the recent interaction sequence for BERT4Rec inference."""
        if self.ratings_df is None or self.model.movie_to_idx is None:
            return []
        user_ratings = (
            self.ratings_df[self.ratings_df["userId"] == user_id]
            .sort_values("timestamp")
        )
        return [
            self.model.movie_to_idx[m]
            for m in user_ratings["movieId"].tolist()
            if m in self.model.movie_to_idx
        ][-50:]

    def _enrich(self, recs: list[dict]) -> list[dict]:
        """Add title and genres from movies.csv and sanitize numpy types for JSON serialization."""
        movie_info = self.movies_df.set_index("movieId") if self.movies_df is not None else None
        enriched = []
        for r in recs:
            mid = int(r["movieId"])
            score_val = r.get("score") if "score" in r else r.get("predicted_rating", 0.0)
            score = float(score_val) if score_val is not None else 0.0
            row = movie_info.loc[mid] if (movie_info is not None and mid in movie_info.index) else None
            enriched.append({
                "movieId": mid,
                "score": round(score, 4),
                "title": str(row["title"]) if row is not None else "Unknown",
                "genres": str(row["genres"]) if row is not None else "Unknown",
            })
        return enriched

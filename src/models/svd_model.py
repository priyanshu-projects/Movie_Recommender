"""
src/models/svd_model.py

SVD collaborative filtering model wrapped to implement BaseRecommender.
Uses scikit-surprise under the hood — fast CPU training, simple baseline.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class SVDModel(BaseRecommender):
    """Collaborative Filtering via Matrix Factorization (SVD)."""

    def __init__(self, config: dict):
        self.config = config
        self._algo: SVD | None = None
        self._trainset = None
        self._all_movie_ids: list[int] = []

    @property
    def model_type(self) -> str:
        return "svd"

    def train(self, ratings_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        cfg = self.config
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
        self._trainset = data.build_full_trainset()
        self._all_movie_ids = sorted(ratings_df["movieId"].unique().tolist())

        self._algo = SVD(
            n_factors=cfg.get("n_factors", 50),
            n_epochs=cfg.get("n_epochs", 20),
            lr_all=cfg.get("lr_all", 0.005),
            reg_all=cfg.get("reg_all", 0.02),
        )
        self._algo.fit(self._trainset)
        logger.info("SVD training complete.")
        return {"status": "trained"}

    def evaluate(self, test_df: pd.DataFrame, k: int = 10) -> dict:
        from surprise import accuracy
        from src.evaluation.metrics import precision_at_k, recall_at_k

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(test_df[["userId", "movieId", "rating"]], reader)
        testset = data.build_full_trainset().build_testset()
        predictions = self._algo.test(testset)

        rmse = accuracy.rmse(predictions, verbose=False)
        mae  = accuracy.mae(predictions, verbose=False)

        # Compute Precision@K and Recall@K
        from collections import defaultdict
        user_preds = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_preds[uid].append((est, true_r, iid))

        prec_list, rec_list = [], []
        threshold = 3.5
        for uid, preds in user_preds.items():
            preds.sort(key=lambda x: x[0], reverse=True)
            top_k_ids = [iid for _, _, iid in preds[:k]]
            relevant  = [iid for _, r, iid in preds if r >= threshold]
            prec_list.append(precision_at_k(relevant, top_k_ids, k))
            rec_list.append(recall_at_k(relevant, top_k_ids, k))

        metrics = {
            "test_rmse": float(round(rmse, 4)),
            "test_mae":  float(round(mae, 4)),
            f"precision@{k}": float(round(sum(prec_list) / max(len(prec_list), 1), 4)),
            f"recall@{k}":    float(round(sum(rec_list) / max(len(rec_list), 1), 4)),
        }
        logger.info("SVD evaluation: %s", metrics)
        return metrics

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True,
                  seen_movie_ids: list[int] | None = None) -> list[dict]:
        if self._algo is None:
            raise RuntimeError("Model not trained or loaded.")
        exclude = set(seen_movie_ids or [])
        candidates = [m for m in self._all_movie_ids if m not in exclude]
        preds = [(m, self._algo.predict(user_id, m).est) for m in candidates]
        preds.sort(key=lambda x: x[1], reverse=True)
        return [{"movieId": m, "score": round(s, 4)} for m, s in preds[:n]]

    def predict(self, user_id: int, movie_id: int) -> float:
        """Compatibility helper used by the FastAPI serving layer."""
        return self._algo.predict(user_id, movie_id).est

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "algo": self._algo,
                "all_movie_ids": self._all_movie_ids,
                "config": self.config,
            }, f)
        logger.info("SVD model saved to %s", path)

    def load(self, path: Path) -> "SVDModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._algo = data["algo"]
        self._all_movie_ids = data["all_movie_ids"]
        self.config = data["config"]
        logger.info("SVD model loaded from %s", path)
        return self

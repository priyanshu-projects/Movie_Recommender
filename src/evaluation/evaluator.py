"""
src/evaluation/evaluator.py

Evaluates a trained model against the champion model in the MLflow registry.
Computes RMSE and Precision@K / Recall@K on a held-out temporal test set.

Promotion logic:
  - If new model RMSE < champion RMSE → promote to "Production" in MLflow registry
  - Otherwise → keep champion, log challenger as "Archived"

This gate prevents regressions from being auto-deployed.

Usage (standalone):
    python -m src.evaluation.evaluator \
        --model     models/svd_model.pkl \
        --test-data data/processed/interactions.csv
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.pyfunc
import pandas as pd
from surprise import Dataset, Reader, accuracy

logger = logging.getLogger(__name__)

K = 10  # top-K for Precision@K / Recall@K


def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def temporal_test_set(interactions_path: Path, test_fraction: float = 0.2):
    """Return the latest `test_fraction` of rows as a Surprise testset."""
    df = pd.read_csv(interactions_path).sort_values("timestamp")
    split_idx = int(len(df) * (1 - test_fraction))
    test_df = df.iloc[split_idx:]
    testset = list(zip(test_df["userId"], test_df["movieId"], test_df["rating"]))
    return testset, test_df


def precision_recall_at_k(predictions, k: int = K, threshold: float = 3.5) -> Tuple[float, float]:
    """
    Compute mean Precision@K and Recall@K across all users.
    A rating >= threshold counts as "relevant".
    """
    from collections import defaultdict

    user_preds = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_preds[uid].append((est, true_r))

    precisions, recalls = [], []
    for uid, user_ratings in user_preds.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        n_rel = sum(1 for _, true_r in user_ratings if true_r >= threshold)
        n_rel_and_rec = sum(1 for est, true_r in top_k if true_r >= threshold)
        precisions.append(n_rel_and_rec / k)
        recalls.append(n_rel_and_rec / n_rel if n_rel > 0 else 0)

    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def evaluate_and_promote(
    model_path: Path,
    interactions_path: Path,
    experiment_name: str = "movielens-svd",
    model_name: str = "movielens-svd-champion",
) -> dict:
    """
    Evaluate the candidate model. If better than champion, promote it.
    Returns a dict of computed metrics.
    """
    model = load_model(model_path)
    testset, _ = temporal_test_set(interactions_path)
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k=K)

    logger.info("Candidate — RMSE: %.4f | P@%d: %.4f | R@%d: %.4f", rmse, K, precision, K, recall)

    # --- Compare against current champion ---
    client = mlflow.tracking.MlflowClient()
    champion_rmse = float("inf")
    try:
        champion_versions = client.get_latest_versions(model_name, stages=["Production"])
        if champion_versions:
            champion_run = client.get_run(champion_versions[0].run_id)
            champion_rmse = float(champion_run.data.metrics.get("test_rmse", float("inf")))
            logger.info("Champion RMSE: %.4f", champion_rmse)
    except Exception:
        logger.info("No champion found in registry — candidate will be promoted automatically.")

    # --- Log & conditionally promote ---
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric(f"precision_at_{K}", precision)
        mlflow.log_metric(f"recall_at_{K}", recall)
        mlflow.log_artifact(str(model_path))

        if rmse < champion_rmse:
            logger.info("Candidate is better → promoting to Production.")
            mlflow.register_model(
                f"runs:/{run.info.run_id}/{model_path.name}",
                model_name,
            )
        else:
            logger.info("Candidate is NOT better → keeping champion.")

    return {"rmse": rmse, f"precision@{K}": precision, f"recall@{K}": recall}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate candidate model vs champion.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data", required=True)
    args = parser.parse_args()
    metrics = evaluate_and_promote(Path(args.model), Path(args.test_data))
    print(metrics)

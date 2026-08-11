"""
src/training/train_svd.py

Trains a Matrix Factorization model (SVD) using the Surprise library.
Logs parameters and metrics to MLflow. Saves the trained model as a pickle.

Why Surprise SVD:
- CPU-only, fast even at millions of ratings
- Well-understood algorithm, easy to explain in interviews
- Built-in support for train/test splitting and rating prediction

MLflow run is logged with:
  - params: n_factors, n_epochs, lr_all, reg_all
  - metrics: train_rmse (from cross-val on training fold)
  - artifact: trained model pickle

Usage (standalone):
    python -m src.training.train_svd \
        --interactions data/processed/interactions.csv \
        --model-out    models/svd_model.pkl
"""

import argparse
import logging
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Default hyperparameters — tune via MLflow experiment sweeps
DEFAULT_PARAMS = {
    "n_factors": 50,
    "n_epochs": 20,
    "lr_all": 0.005,
    "reg_all": 0.02,
}


def train(
    interactions_path: Path,
    model_out_path: Path,
    params: dict = None,
    mlflow_experiment: str = "movielens-svd",
) -> float:
    """
    Train SVD on the full interactions dataset using a temporal train/test split.

    Returns the test RMSE of the trained model.
    """
    if params is None:
        params = DEFAULT_PARAMS

    df = pd.read_csv(interactions_path)

    # Temporal split: train on earliest 80%, test on latest 20%
    df_sorted = df.sort_values("timestamp")
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(train_df[["userId", "movieId", "rating"]], reader)
    trainset = train_data.build_full_trainset()

    test_data = Dataset.load_from_df(test_df[["userId", "movieId", "rating"]], reader)
    _, testset = train_test_split(test_data, test_size=1.0)  # all rows as test

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params(params)

        model = SVD(**params)
        model.fit(trainset)

        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mlflow.log_metric("test_rmse", rmse)

        logger.info("Test RMSE: %.4f", rmse)

        model_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_out_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact(str(model_out_path))
        logger.info("Model saved to %s", model_out_path)

    return rmse


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train SVD recommender and log to MLflow.")
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--model-out", required=True)
    args = parser.parse_args()
    train(Path(args.interactions), Path(args.model_out))

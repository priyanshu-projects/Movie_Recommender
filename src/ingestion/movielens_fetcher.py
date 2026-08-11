"""
src/ingestion/movielens_fetcher.py

Downloads and normalizes MovieLens datasets from GroupLens.
Handles both:
  - ml-latest-small / ml-latest  (.csv format, comma separated)
  - ml-1m / ml-10m / ml-25m      (.dat format, '::' separated)

Outputs always-standard CSV files with columns:
    userId, movieId, rating, timestamp  (ratings)
    movieId, title, genres              (movies)

Usage:
    python -m src.ingestion.movielens_fetcher --dataset ml-1m
    python -m src.ingestion.movielens_fetcher --dataset ml-latest-small
"""

import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATASET_URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m":           "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-10m":          "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "ml-25m":          "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-latest":       "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
}

RAW_DATA_DIR = Path("data/raw")


def download_movielens(dataset: str = "ml-1m", dest_dir: Path = RAW_DATA_DIR) -> Path:
    """Download and extract the MovieLens dataset zip."""
    url = DATASET_URLS[dataset]
    logger.info("Downloading %s from %s ...", dataset, url)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(dest_dir)

    extracted = dest_dir / dataset
    logger.info("Extracted to %s", extracted)
    return extracted


def normalize_ratings(dataset_dir: Path, dataset: str) -> pd.DataFrame:
    """
    Load ratings from either .csv or .dat format and return
    a standard DataFrame with columns: userId, movieId, rating, timestamp.
    """
    if dataset in ("ml-1m", "ml-10m"):
        # .dat format: UserID::MovieID::Rating::Timestamp
        ratings_file = dataset_dir / "ratings.dat"
        df = pd.read_csv(
            ratings_file,
            sep="::",
            names=["userId", "movieId", "rating", "timestamp"],
            engine="python",
            encoding="latin-1",
        )
    else:
        # .csv format: userId,movieId,rating,timestamp
        ratings_file = dataset_dir / "ratings.csv"
        df = pd.read_csv(ratings_file)

    df["userId"]    = df["userId"].astype(int)
    df["movieId"]   = df["movieId"].astype(int)
    df["rating"]    = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def normalize_movies(dataset_dir: Path, dataset: str) -> pd.DataFrame:
    """
    Load movies from either .csv or .dat format and return
    a standard DataFrame with columns: movieId, title, genres.
    """
    if dataset in ("ml-1m", "ml-10m"):
        movies_file = dataset_dir / "movies.dat"
        df = pd.read_csv(
            movies_file,
            sep="::",
            names=["movieId", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )
    else:
        movies_file = dataset_dir / "movies.csv"
        df = pd.read_csv(movies_file)

    df["movieId"] = df["movieId"].astype(int)
    return df


def fetch_and_normalize(dataset: str = "ml-1m", dest_dir: Path = RAW_DATA_DIR) -> dict[str, Path]:
    """
    Full pipeline: download → extract → normalize to standard CSVs.
    Returns paths to normalized ratings.csv and movies.csv.
    """
    dataset_dir = download_movielens(dataset, dest_dir)
    out_dir = dest_dir / dataset

    # Normalize ratings
    ratings = normalize_ratings(out_dir, dataset)
    ratings_out = out_dir / "ratings.csv"
    ratings.to_csv(ratings_out, index=False)
    logger.info("Ratings normalized: %d rows → %s", len(ratings), ratings_out)

    # Normalize movies
    movies = normalize_movies(out_dir, dataset)
    movies_out = out_dir / "movies.csv"
    movies.to_csv(movies_out, index=False)
    logger.info("Movies normalized: %d rows → %s", len(movies), movies_out)

    return {"ratings": ratings_out, "movies": movies_out}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download and normalize a MovieLens dataset.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_URLS.keys()),
        default="ml-1m",
        help="Which MovieLens dataset to download (default: ml-1m)",
    )
    args = parser.parse_args()
    paths = fetch_and_normalize(args.dataset)
    print(f"✓ Ratings: {paths['ratings']}")
    print(f"✓ Movies:  {paths['movies']}")

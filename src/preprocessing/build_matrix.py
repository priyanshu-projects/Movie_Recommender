"""
src/preprocessing/build_matrix.py

Loads all accumulated ratings (full dataset up to this cycle) and produces a
clean interaction DataFrame ready for training.

Responsibilities:
- Merge raw ratings with movie metadata
- Re-encode userId / movieId to contiguous integer indices (required by Surprise)
- Save the processed dataset to data/processed/

Usage (standalone):
    python -m src.preprocessing.build_matrix \
        --ratings data/raw/ml-latest-small/ratings.csv \
        --movies  data/raw/ml-latest-small/movies.csv \
        --out     data/processed/interactions.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_interaction_matrix(
    ratings_path: Path,
    movies_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """
    Load ratings + movies, encode IDs, and return a clean interactions DataFrame.

    Columns in output: userId, movieId, rating, timestamp, title, genres,
                       user_idx, movie_idx  (contiguous integer indices)
    """
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    df = ratings.merge(movies, on="movieId", how="left")

    # Encode to contiguous indices (Surprise / implicit need this)
    user_cat = pd.Categorical(df["userId"])
    movie_cat = pd.Categorical(df["movieId"])
    df["user_idx"] = user_cat.codes
    df["movie_idx"] = movie_cat.codes

    logger.info(
        "Interactions built: %d rows | %d unique users | %d unique movies",
        len(df), df["userId"].nunique(), df["movieId"].nunique(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved to %s", out_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build interaction matrix from ratings + movies.")
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--movies", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_interaction_matrix(Path(args.ratings), Path(args.movies), Path(args.out))

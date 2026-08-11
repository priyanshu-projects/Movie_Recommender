"""
src/validation/validate_ratings.py

Validates the incoming diff CSV before it's used for training.
Uses pandera (pure-Python, works on Python 3.14) instead of Great Expectations.

Checks:
  - Required columns present: userId, movieId, rating, timestamp
  - No nulls in key columns
  - Rating range 0.5–5.0
  - No duplicate (userId, movieId) pairs in this diff batch

Usage (standalone):
    python -m src.validation.validate_ratings --input data/diffs/ratings_diff.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)

# --- Schema definition -------------------------------------------------------

ratings_schema = DataFrameSchema(
    columns={
        "userId":    Column(int,   nullable=False),
        "movieId":   Column(int,   nullable=False),
        "rating":    Column(float, nullable=False, checks=[
            Check(lambda s: s.between(0.5, 5.0), element_wise=False,
                  error="rating must be between 0.5 and 5.0"),
        ]),
        "timestamp": Column(int,   nullable=False),
    },
    checks=[
        # No duplicate (userId, movieId) pairs in this diff
        Check(lambda df: ~df.duplicated(subset=["userId", "movieId"]).any(),
              error="Duplicate (userId, movieId) pairs found in diff"),
    ],
    coerce=True,   # try casting types before failing
    strict=False,  # allow extra columns (e.g. title, genres if merged)
)


# --- Entry point -------------------------------------------------------------

def validate_ratings(diff_path: Path) -> bool:
    """
    Run schema + checks against the incoming diff CSV.
    Returns True if all checks pass, raises pandera.errors.SchemaError otherwise.
    """
    df = pd.read_csv(diff_path)
    logger.info("Validating %d rows from %s ...", len(df), diff_path)

    ratings_schema.validate(df, lazy=True)   # lazy=True collects ALL errors before raising

    logger.info("Validation PASSED ✓")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Validate ratings diff CSV.")
    parser.add_argument("--input", required=True, help="Path to the diff CSV to validate")
    args = parser.parse_args()
    validate_ratings(Path(args.input))

"""
src/data/validation.py

Validates incoming batch CSVs before they enter the training pipeline.
Uses pandera (pure Python, works on Python 3.14) for schema + value checks.

Checks:
  - Required columns present: userId, movieId, rating, timestamp
  - No nulls in key columns
  - Rating range 0.5–5.0 (configurable)
  - No duplicate (userId, movieId, timestamp) triples in this batch

Usage (standalone):
    python -m src.data.validation --input data/raw/incoming/batch_001.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)


def _build_schema(min_rating: float = 0.5, max_rating: float = 5.0) -> DataFrameSchema:
    return DataFrameSchema(
        columns={
            "userId":    Column(int,   nullable=False),
            "movieId":   Column(int,   nullable=False),
            "rating":    Column(float, nullable=False, checks=[
                Check(lambda s: s.between(min_rating, max_rating), element_wise=False,
                      error=f"rating must be between {min_rating} and {max_rating}"),
            ]),
            "timestamp": Column(int,   nullable=False),
        },
        checks=[
            Check(
                lambda df: ~df.duplicated(subset=["userId", "movieId", "timestamp"]).any(),
                error="Duplicate (userId, movieId, timestamp) triples found in batch",
            ),
        ],
        coerce=True,
        strict=False,
    )


def validate_batch(
    batch_path: Path,
    config_path: str = "configs/config.yaml",
) -> bool:
    """
    Validate the incoming batch CSV.
    Returns True if valid. Raises pandera.errors.SchemaErrors on failure.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    val_cfg = cfg.get("validation", {})
    min_r = float(val_cfg.get("min_rating", 0.5))
    max_r = float(val_cfg.get("max_rating", 5.0))

    df = pd.read_csv(batch_path)
    logger.info("Validating %d rows from %s ...", len(df), batch_path)

    schema = _build_schema(min_r, max_r)
    schema.validate(df, lazy=True)

    logger.info("Validation PASSED ✓  (%d rows)", len(df))
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Validate incoming batch CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    validate_batch(Path(args.input), args.config)

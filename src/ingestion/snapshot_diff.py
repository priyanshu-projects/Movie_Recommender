"""
src/ingestion/snapshot_diff.py

Computes delta diffs between rating snapshots.
Ensures that only new (userId, movieId, timestamp) triples enter the processing pipeline.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def compute_diff(
    current_csv: Path,
    previous_csv: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Compares current_csv against previous_csv.
    Returns rows present in current_csv that were not in previous_csv.
    Saves to out_path if provided.
    """
    curr_df = pd.read_csv(current_csv)

    if previous_csv is None or not Path(previous_csv).exists():
        logger.info("No previous snapshot found. Entire dataset is treated as new delta.")
        diff_df = curr_df
    else:
        prev_df = pd.read_csv(previous_csv)
        key_cols = ["userId", "movieId", "timestamp"]
        merged = curr_df.merge(prev_df[key_cols], on=key_cols, how="left", indicator=True)
        diff_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        logger.info(
            "Snapshot diff computed: %d previous rows, %d current rows -> %d new delta rows.",
            len(prev_df), len(curr_df), len(diff_df)
        )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        diff_df.to_csv(out_path, index=False)
        logger.info("Saved delta CSV to %s", out_path)

    return diff_df

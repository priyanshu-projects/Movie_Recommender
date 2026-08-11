"""
src/delta/snapshot_diff.py

Compares the freshly downloaded ratings.csv against the last saved snapshot
to extract only genuinely new rows (new ratings added since last run).

This is the core "simulated streaming" mechanic: GroupLens gives us the whole
file every pull, so we diff to isolate what's actually new this cycle.

Snapshots are stored in data/snapshots/ and versioned with DVC.

Usage (standalone):
    python -m src.delta.snapshot_diff \
        --new data/raw/ml-latest-small/ratings.csv \
        --snapshot data/snapshots/ratings_snapshot.csv \
        --out data/diffs/ratings_diff.csv
"""

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path("data/snapshots")
DIFF_DIR = Path("data/diffs")


def compute_diff(
    new_ratings_path: Path,
    snapshot_path: Path,
    diff_out_path: Path,
) -> pd.DataFrame:
    """
    Load the new full ratings file and the previous snapshot.
    Return (and save) only the rows that are new since the last snapshot.

    A row is considered "new" if its (userId, movieId, timestamp) triple
    doesn't exist in the snapshot.
    """
    new_df = pd.read_csv(new_ratings_path)

    if not snapshot_path.exists():
        logger.warning("No snapshot found at %s — treating all rows as new.", snapshot_path)
        diff_df = new_df.copy()
    else:
        snapshot_df = pd.read_csv(snapshot_path)
        # Merge to find rows in new_df that aren't in snapshot_df
        key_cols = ["userId", "movieId", "timestamp"]
        merged = new_df.merge(snapshot_df[key_cols], on=key_cols, how="left", indicator=True)
        diff_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    logger.info("New rows this cycle: %d", len(diff_df))

    diff_out_path.parent.mkdir(parents=True, exist_ok=True)
    diff_df.to_csv(diff_out_path, index=False)
    logger.info("Diff saved to %s", diff_out_path)

    return diff_df


def update_snapshot(new_ratings_path: Path, snapshot_path: Path) -> None:
    """
    Replace the stored snapshot with the latest full file,
    ready for the next cycle's diff.
    """
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(new_ratings_path, snapshot_path)
    logger.info("Snapshot updated at %s", snapshot_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Compute delta between new ratings and snapshot.")
    parser.add_argument("--new", required=True, help="Path to the newly downloaded ratings.csv")
    parser.add_argument("--snapshot", required=True, help="Path to the stored snapshot ratings.csv")
    parser.add_argument("--out", required=True, help="Output path for the diff CSV")
    args = parser.parse_args()

    diff = compute_diff(Path(args.new), Path(args.snapshot), Path(args.out))
    update_snapshot(Path(args.new), Path(args.snapshot))
    print(f"Done. {len(diff)} new ratings written to {args.out}")

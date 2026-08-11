"""
tests/unit/test_snapshot_diff.py

Unit tests for the delta detection logic in src/delta/snapshot_diff.py.
These run on every push via GitHub Actions (no Airflow, no model needed).
"""

import pandas as pd
import pytest
from pathlib import Path
import tempfile

from src.ingestion.snapshot_diff import compute_diff


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def make_ratings(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestComputeDiff:
    """Verify that only genuinely new (userId, movieId, timestamp) triples are returned."""

    def test_no_snapshot_returns_all_rows(self, tmp_path):
        new = make_ratings([
            {"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1000},
            {"userId": 2, "movieId": 20, "rating": 3.5, "timestamp": 1001},
        ])
        new_path = tmp_path / "new.csv"
        _write_csv(new, new_path)

        diff = compute_diff(new_path, tmp_path / "nonexistent.csv", tmp_path / "diff.csv")

        assert len(diff) == 2

    def test_identical_snapshot_returns_empty(self, tmp_path):
        rows = [{"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1000}]
        new = make_ratings(rows)
        snapshot = make_ratings(rows)
        new_path, snap_path, diff_path = tmp_path / "new.csv", tmp_path / "snap.csv", tmp_path / "diff.csv"
        _write_csv(new, new_path)
        _write_csv(snapshot, snap_path)

        diff = compute_diff(new_path, snap_path, diff_path)

        assert len(diff) == 0

    def test_partial_overlap_returns_only_new(self, tmp_path):
        snapshot = make_ratings([
            {"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1000},
        ])
        new = make_ratings([
            {"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1000},  # existing
            {"userId": 2, "movieId": 20, "rating": 3.5, "timestamp": 1999},  # new
        ])
        new_path, snap_path, diff_path = tmp_path / "new.csv", tmp_path / "snap.csv", tmp_path / "diff.csv"
        _write_csv(new, new_path)
        _write_csv(snapshot, snap_path)

        diff = compute_diff(new_path, snap_path, diff_path)

        assert len(diff) == 1
        assert diff.iloc[0]["userId"] == 2

    def test_diff_csv_is_written(self, tmp_path):
        new = make_ratings([{"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1000}])
        new_path, diff_path = tmp_path / "new.csv", tmp_path / "diff.csv"
        _write_csv(new, new_path)

        compute_diff(new_path, tmp_path / "nosnap.csv", diff_path)

        assert diff_path.exists()

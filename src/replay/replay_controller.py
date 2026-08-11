"""
src/replay/replay_controller.py

The Replay Controller is the core of the simulated continuous ingestion system.

How it works:
  1. Reads the immutable master ratings CSV (ml-latest-small or ml-1m).
  2. Loads the current replay state (which batch was last released).
  3. Slices the next chronological chunk (N days of ratings).
  4. Saves the chunk to data/raw/incoming/batch_XXX.csv with metadata.
  5. Updates the replay state so the next call returns the NEXT chunk.

The master dataset is NEVER modified.
Each call is deterministic and idempotent within the same batch window.

Usage (standalone):
    python -m src.replay.replay_controller

    # Or with custom config:
    python -m src.replay.replay_controller --config configs/config.yaml
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.replay.replay_state import ReplayState, load_state, save_state

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_next_batch(config_path: str = "configs/config.yaml") -> Path | None:
    """
    Release the next chronological batch from the master dataset.

    Returns the path to the saved batch CSV, or None if all data is exhausted.
    """
    cfg = load_config(config_path)
    replay_cfg = cfg["replay"]

    master_path = Path(replay_cfg["master_ratings_path"])
    batch_days = int(replay_cfg["batch_days"])
    state_path = Path(replay_cfg["state_file"])
    incoming_dir = Path(cfg["dataset"]["incoming_dir"])

    if not master_path.exists():
        raise FileNotFoundError(
            f"Master ratings not found at {master_path}. "
            "Run src/ingestion/movielens_fetcher.py first."
        )

    # --- Load master dataset and sort chronologically ---
    logger.info("Loading master dataset from %s ...", master_path)
    master = pd.read_csv(master_path)
    master["datetime"] = pd.to_datetime(master["timestamp"], unit="s", utc=True)
    master = master.sort_values("datetime").reset_index(drop=True)

    # --- Load current replay state ---
    state = load_state(state_path)

    # Determine start point for next batch
    if state.last_released_timestamp:
        last_dt = pd.Timestamp(state.last_released_timestamp, tz="UTC")
        remaining = master[master["datetime"] > last_dt]
    else:
        remaining = master  # first batch — start from beginning

    if remaining.empty:
        logger.warning("All data has been replayed. No new batch available.")
        return None

    # --- Slice next N days ---
    batch_start = remaining["datetime"].min()
    batch_end = batch_start + pd.Timedelta(days=batch_days)
    batch = remaining[remaining["datetime"] < batch_end].copy()

    if batch.empty:
        logger.warning("No ratings found in window [%s, %s].", batch_start, batch_end)
        return None

    # Drop helper column before saving
    batch = batch.drop(columns=["datetime"])

    # --- Save batch to incoming dir ---
    batch_num = state.last_released_batch + 1
    incoming_dir.mkdir(parents=True, exist_ok=True)
    batch_path = incoming_dir / f"batch_{batch_num:03d}.csv"
    batch.to_csv(batch_path, index=False)

    # --- Save batch metadata ---
    meta = {
        "batch_id": batch_num,
        "dataset_version": state.dataset_version,
        "earliest_timestamp": int(batch["timestamp"].min()),
        "latest_timestamp": int(batch["timestamp"].max()),
        "num_interactions": len(batch),
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_path": str(batch_path),
    }
    meta_path = incoming_dir / f"batch_{batch_num:03d}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # --- Update replay state ---
    new_state = ReplayState(
        last_released_batch=batch_num,
        last_released_timestamp=pd.Timestamp(batch["timestamp"].max(), unit="s", tz="UTC").isoformat(),
        total_ratings_released=state.total_ratings_released + len(batch),
        dataset_version=state.dataset_version,
    )
    save_state(new_state, state_path)

    logger.info(
        "Batch %03d released: %d ratings | %s → %s → %s",
        batch_num, len(batch), batch_start.date(), batch_end.date(), batch_path,
    )
    return batch_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Release the next chronological replay batch.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    path = get_next_batch(args.config)
    if path:
        print(f"Batch saved to: {path}")
    else:
        print("No more data to replay.")

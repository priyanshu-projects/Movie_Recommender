"""
src/replay/replay_state.py

Manages the persistent replay state stored in data/replay/replay_state.json.

The replay state tracks which chronological batch was last released,
ensuring the replay controller is idempotent — calling it multiple times
on the same schedule tick returns the same batch, never skips or duplicates.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("data/replay/replay_state.json")


@dataclass
class ReplayState:
    last_released_batch: int = 0             # 0 = no batch released yet
    last_released_timestamp: str = ""        # ISO8601 of latest rating in last batch
    total_ratings_released: int = 0
    dataset_version: str = "ml-latest-small"


def load_state(path: Path = DEFAULT_STATE_PATH) -> ReplayState:
    """Load replay state from disk. Returns fresh state if file doesn't exist."""
    if not path.exists():
        logger.info("No replay state found at %s — starting from scratch.", path)
        return ReplayState()

    with open(path) as f:
        data = json.load(f)
    logger.info("Loaded replay state: batch %d", data.get("last_released_batch", 0))
    return ReplayState(**data)


def save_state(state: ReplayState, path: Path = DEFAULT_STATE_PATH) -> None:
    """Persist replay state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2)
    logger.info("Replay state saved: batch %d", state.last_released_batch)

"""
tests/unit/test_replay_controller.py

Unit tests for the replay controller and replay state modules.
Uses a tiny synthetic ratings dataset so no real MovieLens download is needed.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tiny_ratings(tmp_path):
    """Create a tiny ratings CSV spanning 3 months."""
    import time
    data = []
    base_ts = 1_420_070_400  # 2015-01-01 00:00:00 UTC
    for i in range(300):
        data.append({
            "userId": (i % 10) + 1,
            "movieId": (i % 20) + 1,
            "rating": 3.5 + (i % 3) * 0.5,
            "timestamp": base_ts + i * 86400,  # one rating per day
        })
    df = pd.DataFrame(data)
    path = tmp_path / "ratings.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def test_config(tmp_path, tiny_ratings):
    """Write a minimal config.yaml pointing at tmp paths."""
    cfg = {
        "dataset": {
            "incoming_dir": str(tmp_path / "incoming"),
        },
        "replay": {
            "state_file": str(tmp_path / "replay_state.json"),
            "batch_days": 30,
            "master_ratings_path": str(tiny_ratings),
            "schedule_interval_days": 14,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return str(config_path)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_replay_state_fresh_start(tmp_path):
    """load_state on missing file returns default state."""
    from src.replay.replay_state import load_state, ReplayState
    state = load_state(tmp_path / "nonexistent.json")
    assert state.last_released_batch == 0
    assert state.total_ratings_released == 0


def test_replay_state_roundtrip(tmp_path):
    """save_state then load_state returns the same values."""
    from src.replay.replay_state import ReplayState, load_state, save_state
    path = tmp_path / "state.json"
    original = ReplayState(
        last_released_batch=3,
        last_released_timestamp="2015-04-01T00:00:00+00:00",
        total_ratings_released=999,
        dataset_version="ml-1m",
    )
    save_state(original, path)
    loaded = load_state(path)
    assert loaded.last_released_batch == 3
    assert loaded.total_ratings_released == 999
    assert loaded.dataset_version == "ml-1m"


def test_first_batch_released(test_config, tmp_path):
    """First call to get_next_batch should produce batch_001.csv."""
    from src.replay.replay_controller import get_next_batch
    path = get_next_batch(test_config)
    assert path is not None
    assert path.exists()
    assert "batch_001" in path.name


def test_second_batch_is_different(test_config, tmp_path):
    """Second call should release batch_002.csv with newer timestamps."""
    from src.replay.replay_controller import get_next_batch
    p1 = get_next_batch(test_config)
    p2 = get_next_batch(test_config)
    assert p2 is not None
    assert "batch_002" in p2.name

    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    assert df2["timestamp"].min() > df1["timestamp"].max()


def test_replay_is_idempotent_state(test_config):
    """State file is updated and persisted correctly after each call."""
    from src.replay.replay_controller import get_next_batch, load_config
    from src.replay.replay_state import load_state
    from pathlib import Path
    cfg = load_config(test_config)
    state_path = Path(cfg["replay"]["state_file"])

    get_next_batch(test_config)
    state = load_state(state_path)
    assert state.last_released_batch == 1

    get_next_batch(test_config)
    state = load_state(state_path)
    assert state.last_released_batch == 2

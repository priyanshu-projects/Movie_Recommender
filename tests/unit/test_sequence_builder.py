"""
tests/unit/test_sequence_builder.py

Unit tests for BERT4Rec sequence builder.
"""

import pandas as pd
import pytest

from src.data.sequence_builder import build_sequences, MASK_TOKEN, IGNORE_INDEX


@pytest.fixture()
def sample_ratings():
    """Minimal ratings dataframe with 3 users, enough interactions each."""
    rows = []
    for user_id in [1, 2, 3]:
        for i in range(10):
            rows.append({
                "userId": user_id,
                "movieId": (user_id * 10) + i + 1,
                "rating": 4.0,
                "timestamp": 1_420_070_400 + i * 3600,
            })
    return pd.DataFrame(rows)


def test_sequences_built(sample_ratings):
    sequences, vocab = build_sequences(sample_ratings, min_seq_len=5)
    assert len(sequences) == 3
    assert len(vocab) > 0


def test_sequence_contains_masks(sample_ratings):
    sequences, _ = build_sequences(sample_ratings, mask_prob=0.5, min_seq_len=5)
    for seq in sequences:
        assert MASK_TOKEN in seq["masked_sequence"], "Every sequence must have at least one mask"


def test_labels_align_with_masks(sample_ratings):
    sequences, _ = build_sequences(sample_ratings, min_seq_len=5)
    for seq in sequences:
        for pos, (m, lbl) in enumerate(zip(seq["masked_sequence"], seq["labels"])):
            if m == MASK_TOKEN:
                assert lbl != IGNORE_INDEX, "Masked position must have a real label"
            else:
                assert lbl == IGNORE_INDEX, "Unmasked position must have IGNORE_INDEX label"


def test_below_min_seq_len_excluded():
    tiny = pd.DataFrame([
        {"userId": 99, "movieId": 1, "rating": 4.0, "timestamp": 1000},
        {"userId": 99, "movieId": 2, "rating": 4.0, "timestamp": 2000},
    ])
    sequences, _ = build_sequences(tiny, min_seq_len=5)
    user_ids = [s["user_id"] for s in sequences]
    assert 99 not in user_ids

"""
src/data/sequence_builder.py

Builds chronological user interaction sequences for BERT4Rec.

Steps:
  1. Filter to positive interactions (rating >= threshold).
  2. Sort each user's interactions by timestamp.
  3. Build sequences of movieId per user.
  4. Truncate to max_sequence_length.
  5. Create masked training examples (random movie positions masked).

Output format (for training):
    {
        "user_id": 42,
        "sequence": [movieId_1, movieId_2, ..., movieId_N],
        "masked_sequence": [movieId_1, MASK_TOKEN, ..., movieId_N],
        "labels": [-100, movieId_2, ..., -100]   # -100 = not masked (ignored in loss)
    }

Usage (standalone):
    python -m src.data.sequence_builder \
        --ratings data/processed/all_ratings.csv \
        --out data/processed/sequences.jsonl
"""

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

MASK_TOKEN = 0          # Reserved token ID for [MASK]
PAD_TOKEN  = -1         # Reserved token ID for [PAD]
IGNORE_INDEX = -100     # PyTorch CrossEntropyLoss ignore index


def build_sequences(
    ratings: pd.DataFrame,
    min_rating: float = 3.5,
    max_seq_len: int = 50,
    min_seq_len: int = 5,
    mask_prob: float = 0.20,
    seed: int = 42,
) -> list[dict]:
    """
    Build masked sequences for BERT4Rec training.

    Returns a list of dicts (one per training example per user).
    """
    random.seed(seed)

    # Filter to positive interactions only
    positive = ratings[ratings["rating"] >= min_rating].copy()
    positive = positive.sort_values(["userId", "timestamp"])

    # Build a movie ID vocabulary (1-indexed; 0 reserved for MASK)
    all_movies = sorted(positive["movieId"].unique())
    movie_to_idx = {m: i + 1 for i, m in enumerate(all_movies)}  # 1-indexed

    sequences = []

    for uid, group in positive.groupby("userId"):
        movie_ids = group["movieId"].tolist()
        if len(movie_ids) < min_seq_len:
            continue

        # Truncate to max length
        if len(movie_ids) > max_seq_len:
            movie_ids = movie_ids[-max_seq_len:]

        # Convert to indexed sequence
        indexed = [movie_to_idx[m] for m in movie_ids]

        # Create masked sequence for training
        masked = list(indexed)
        labels = [IGNORE_INDEX] * len(indexed)

        for pos in range(len(indexed)):
            if random.random() < mask_prob:
                labels[pos] = indexed[pos]
                masked[pos] = MASK_TOKEN

        # Ensure at least 1 mask per sequence
        if all(l == IGNORE_INDEX for l in labels):
            pos = random.randint(0, len(indexed) - 1)
            labels[pos] = indexed[pos]
            masked[pos] = MASK_TOKEN

        sequences.append({
            "user_id": int(uid),
            "sequence": indexed,
            "masked_sequence": masked,
            "labels": labels,
            "length": len(indexed),
        })

    logger.info(
        "Built %d sequences from %d users (%d positive interactions).",
        len(sequences), positive["userId"].nunique(), len(positive),
    )
    return sequences, movie_to_idx


def save_sequences(sequences: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for seq in sequences:
            f.write(json.dumps(seq) + "\n")
    logger.info("Saved %d sequences to %s", len(sequences), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build BERT4Rec training sequences.")
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--out", default="data/processed/sequences.jsonl")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seq_cfg = cfg.get("sequences", {})

    ratings = pd.read_csv(args.ratings)
    sequences, vocab = build_sequences(
        ratings,
        min_rating=seq_cfg.get("min_rating_threshold", 3.5),
        max_seq_len=seq_cfg.get("max_sequence_length", 50),
        min_seq_len=seq_cfg.get("min_sequence_length", 5),
        mask_prob=seq_cfg.get("mask_probability", 0.20),
    )
    save_sequences(sequences, Path(args.out))
    print(f"Vocabulary size: {len(vocab)} movies | Sequences: {len(sequences)}")

"""
src/data/temporal_split.py

Performs a temporal (time-based) train / validation / test split.

NEVER uses random splitting — that leaks future data into training.

For collaborative filtering (SVD):
    Split the full ratings pool chronologically.

For sequential models (BERT4Rec):
    Per-user split:
        Earlier interactions → TRAIN
        Second-to-last      → VALIDATION
        Last                → TEST

Usage (standalone):
    python -m src.data.temporal_split --ratings data/processed/all_ratings.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def global_temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by timestamp and split into train / val / test globally.
    Used for SVD evaluation.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    logger.info(
        "Global split → train: %d | val: %d | test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


def per_user_temporal_split(
    df: pd.DataFrame,
    min_interactions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Per-user chronological split for BERT4Rec:
        All but last 2 interactions → TRAIN
        Second-to-last              → VALIDATION
        Last                        → TEST

    Users with fewer than min_interactions are excluded.
    """
    df = df.sort_values(["userId", "timestamp"])
    train_rows, val_rows, test_rows = [], [], []

    for uid, group in df.groupby("userId"):
        if len(group) < min_interactions:
            continue
        group = group.sort_values("timestamp")
        train_rows.append(group.iloc[:-2])
        val_rows.append(group.iloc[[-2]])
        test_rows.append(group.iloc[[-1]])

    train = pd.concat(train_rows).reset_index(drop=True)
    val   = pd.concat(val_rows).reset_index(drop=True)
    test  = pd.concat(test_rows).reset_index(drop=True)

    logger.info(
        "Per-user split → train: %d | val: %d | test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["global", "per_user"], default="global")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    split_cfg = cfg.get("split", {})

    df = pd.read_csv(args.ratings)
    if args.mode == "global":
        train, val, test = global_temporal_split(
            df,
            train_frac=split_cfg.get("train_fraction", 0.80),
            val_frac=split_cfg.get("val_fraction", 0.10),
        )
    else:
        train, val, test = per_user_temporal_split(df)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

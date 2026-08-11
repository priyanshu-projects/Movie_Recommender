"""
src/training/train_bert4rec.py

Orchestrates BERT4Rec training end-to-end:
  1. Load sequences from data/processed/sequences.jsonl
  2. Per-user temporal split (train/val)
  3. Build PyTorch DataLoaders
  4. Train BERT4Rec with early stopping
  5. Log to MLflow
  6. Save model artifact

Usage:
    python -m src.training.train_bert4rec \
        --sequences data/processed/sequences.jsonl \
        --vocab-size <N> \
        --model-out models/bert4rec_candidate.pkl
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models.bert4rec import BERT4Rec

logger = logging.getLogger(__name__)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, sequences: list[dict], max_seq_len: int = 50):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        masked = self._pad(seq["masked_sequence"])
        labels = self._pad(seq["labels"], pad_val=-100)
        return {
            "masked_sequence": torch.tensor(masked, dtype=torch.long),
            "labels":          torch.tensor(labels, dtype=torch.long),
        }

    def _pad(self, lst: list[int], pad_val: int = 0) -> list[int]:
        if len(lst) >= self.max_seq_len:
            return lst[-self.max_seq_len:]
        return [pad_val] * (self.max_seq_len - len(lst)) + lst


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sequences(path: Path) -> tuple[list[dict], int]:
    """Load sequences and return (sequences, vocab_size)."""
    sequences = []
    max_idx = 0
    with open(path) as f:
        for line in f:
            seq = json.loads(line)
            sequences.append(seq)
            max_idx = max(max_idx, max(seq["sequence"], default=0))
    vocab_size = max_idx + 1   # +1 because 0 is MASK
    logger.info("Loaded %d sequences | vocab_size=%d", len(sequences), vocab_size)
    return sequences, vocab_size


def temporal_split_sequences(
    sequences: list[dict],
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """80/10/10 split by shuffling with fixed seed (sequences are already per-user temporal)."""
    rng = random.Random(seed)
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    val_end = int(len(shuffled) * val_frac)
    return shuffled[val_end:], shuffled[:val_end]


# ── Main training function ────────────────────────────────────────────────────

def train_bert4rec(
    sequences_path: Path,
    model_out: Path,
    config_path: str = "configs/config.yaml",
    warm_start: bool = False,
    champion_path: Path = Path("models/champion_model.pkl"),
) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bert_cfg   = cfg["bert4rec"]
    mlflow_cfg = cfg["mlflow"]
    max_seq    = bert_cfg.get("max_sequence_length", 50)
    batch_size = bert_cfg.get("batch_size", 32)

    # Warm-start: reduce epochs for fine-tuning
    if warm_start and champion_path.exists():
        logger.info("Warm-start mode: loading champion weights from %s", champion_path)
        epochs_override = max(5, bert_cfg.get("max_epochs", 30) // 4)
        logger.info("Fine-tuning for %d epochs (instead of %d)",
                    epochs_override, bert_cfg.get("max_epochs", 30))
    else:
        epochs_override = None
        if warm_start:
            logger.info("No champion found — falling back to full training.")

    # Load sequences
    sequences, vocab_size = load_sequences(sequences_path)
    train_seqs, val_seqs = temporal_split_sequences(sequences)

    train_loader = DataLoader(
        SequenceDataset(train_seqs, max_seq),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(val_seqs, max_seq),
        batch_size=batch_size,
    )

    # Build model
    # Reconstruct movie_to_idx from sequences (idx → idx mapping for now)
    movie_to_idx = {i: i for i in range(1, vocab_size)}
    model = BERT4Rec(bert_cfg).build(vocab_size, movie_to_idx)

    # Warm-start: load champion weights before fine-tuning
    if warm_start and champion_path.exists():
        try:
            model.load(champion_path)
            logger.info("Champion weights loaded for warm-start fine-tuning.")
            # Override max_epochs for fine-tuning
            if epochs_override:
                bert_cfg = {**bert_cfg, "max_epochs": epochs_override}
                model.config = bert_cfg
        except Exception as e:
            logger.warning("Failed to load champion for warm-start: %s. Training from scratch.", e)

    # Train
    logger.info("Starting BERT4Rec training (CPU) ...")
    train_metrics = model.train(train_loader, val_loader)

    # Save
    model_out = Path(model_out)
    model.save(model_out)

    # MLflow logging
    import mlflow
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["bert4rec_experiment"])
    with mlflow.start_run(run_name="bert4rec_training") as run:
        mlflow.log_params({
            "vocab_size":    vocab_size,
            "embedding_dim": bert_cfg["embedding_dim"],
            "num_layers":    bert_cfg["num_layers"],
            "num_heads":     bert_cfg["num_attention_heads"],
            "max_seq_len":   max_seq,
            "lr":            bert_cfg["learning_rate"],
            "dropout":       bert_cfg["dropout"],
            "batch_size":    batch_size,
            "warm_start":    warm_start,
        })
        mlflow.log_metrics({
            "train_loss":    round(train_metrics.get("train_loss", 0), 4),
            "best_val_loss": round(train_metrics.get("best_val_loss", 0), 4),
        })
        mlflow.log_artifact(str(model_out))

    logger.info("BERT4Rec training complete. Model saved to %s", model_out)
    return train_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences",   default="data/processed/sequences.jsonl")
    parser.add_argument("--model-out",   default="models/bert4rec_candidate.pkl")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--warm-start",  action="store_true",
                        help="Load champion weights and fine-tune (fewer epochs).")
    parser.add_argument("--champion",    default="models/champion_model.pkl",
                        help="Path to champion model for warm-start.")
    args = parser.parse_args()

    metrics = train_bert4rec(
        Path(args.sequences), Path(args.model_out), args.config,
        warm_start=args.warm_start, champion_path=Path(args.champion),
    )
    print(f"Training metrics: {metrics}")

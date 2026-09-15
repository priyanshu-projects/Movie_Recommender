"""
src/models/bert4rec.py

BERT4Rec sequential recommendation model implemented in PyTorch (CPU-friendly).

Architecture:
    Movie ID → Embedding → Position Embedding → Transformer Encoder → Prediction Head

The model learns from the ORDER of user interactions (masked item prediction),
unlike SVD which treats each rating independently.

Reference: Sun et al. 2019 "BERT4Rec: Sequential Recommendation with
Bidirectional Encoder Representations from Transformer"
"""

import logging
import math
import pickle
from pathlib import Path

import torch
import torch.nn as nn

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class BERT4RecModel(nn.Module):
    """
    PyTorch module implementing BERT4Rec architecture.

    vocab_size: number of unique movies + 1 (for MASK token at index 0)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        num_attention_heads: int = 2,
        feed_forward_dim: int = 256,
        max_sequence_length: int = 50,
        dropout: float = 0.2,
        attention_dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        # Embeddings
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=feed_forward_dim,
            dropout=attention_dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN for more stable CPU training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Prediction head: hidden → vocabulary scores
        self.prediction_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization + Weight Tying for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.embedding_dim ** -0.5)

        # Tie item embedding weights with output prediction head weights
        self.prediction_head.weight = self.item_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len) — movie ID indices (0 = MASK/PAD)
        Returns: (batch_size, seq_len, vocab_size) — logits over movie vocabulary
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        item_emb = self.item_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        x = self.embedding_norm(item_emb + pos_emb)
        x = self.embedding_dropout(x)

        # Create padding mask (True = ignore)
        padding_mask = (input_ids == 0)

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        logits = self.prediction_head(x)
        return logits


class BERT4Rec(BaseRecommender):
    """
    High-level wrapper around BERT4RecModel implementing BaseRecommender.
    Handles training loop, evaluation, and recommendation generation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("BERT4Rec device: %s", self.device)
        self.model: BERT4RecModel | None = None
        self.movie_to_idx: dict | None = None
        self.idx_to_movie: dict | None = None

    @property
    def model_type(self) -> str:
        return "bert4rec"

    def build(self, vocab_size: int, movie_to_idx: dict) -> "BERT4Rec":
        """Initialize the model with a known vocabulary size."""
        cfg = self.config
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = {v: k for k, v in movie_to_idx.items()}
        self.model = BERT4RecModel(
            vocab_size=vocab_size,
            embedding_dim=cfg.get("embedding_dim", 64),
            num_layers=cfg.get("num_layers", 2),
            num_attention_heads=cfg.get("num_attention_heads", 2),
            feed_forward_dim=cfg.get("feed_forward_dim", 256),
            max_sequence_length=cfg.get("max_sequence_length", 50),
            dropout=cfg.get("dropout", 0.2),
            attention_dropout=cfg.get("attention_dropout", 0.2),
        ).to(self.device)
        logger.info(
            "BERT4Rec built: vocab=%d | layers=%d | heads=%d | emb=%d | device=%s",
            vocab_size, cfg.get("num_layers", 2),
            cfg.get("num_attention_heads", 2), cfg.get("embedding_dim", 64),
            self.device,
        )
        return self

    def train(self, train_loader, val_loader=None) -> dict:
        """Full training loop with early stopping."""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        cfg = self.config
        optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.get("learning_rate", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.01),
        )
        max_epochs = cfg.get("max_epochs", 30)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        best_val_loss = float("inf")
        patience_counter = 0
        patience = cfg.get("early_stopping_patience", 5)

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                masked_seq = batch["masked_sequence"].to(self.device)
                labels     = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(masked_seq)          # (B, L, V)
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / max(len(train_loader), 1)
            logger.info("Epoch %d/%d | train_loss: %.4f", epoch, max_epochs, avg_loss)

            if val_loader:
                val_loss = self._val_loss(val_loader, criterion)
                logger.info("             | val_loss:   %.4f", val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint("models/bert4rec_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping at epoch %d.", epoch)
                        break

        return {"train_loss": avg_loss, "best_val_loss": best_val_loss}

    def _val_loss(self, val_loader, criterion) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                masked_seq = batch["masked_sequence"].to(self.device)
                labels     = batch["labels"].to(self.device)
                logits = self.model(masked_seq).view(-1, self.model.vocab_size)
                total += criterion(logits, labels.view(-1)).item()
        return total / max(len(val_loader), 1)

    def evaluate(self, test_sequences: list[dict], k: int = 10) -> dict:
        """Evaluate using Hit Rate@K, Recall@K, NDCG@K."""
        from src.evaluation.metrics import ndcg_at_k, hit_rate_at_k, recall_at_k

        self.model.eval()
        hits, recalls, ndcgs = [], [], []

        with torch.no_grad():
            for example in test_sequences:
                seq = torch.tensor([example["sequence"]], dtype=torch.long)
                target = example["labels"][-1]
                if target == -100:
                    continue
                logits = self.model(seq)[0, -1, :]     # last position scores
                scores = logits.cpu().numpy()
                top_k = scores.argsort()[::-1][:k].tolist()
                hits.append(hit_rate_at_k(target, top_k))
                recalls.append(recall_at_k([target], top_k))
                ndcgs.append(ndcg_at_k([target], top_k))

        return {
            f"hit_rate@{k}": float(sum(hits) / max(len(hits), 1)),
            f"recall@{k}":   float(sum(recalls) / max(len(recalls), 1)),
            f"ndcg@{k}":     float(sum(ndcgs) / max(len(ndcgs), 1)),
        }

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True,
                  user_sequence: list[int] | None = None) -> list[dict]:
        """Generate top-N recommendations for a user given their recent sequence."""
        if self.model is None:
            raise RuntimeError("Model not built or loaded.")
        if user_sequence is None:
            return []

        self.model.eval()
        seq = torch.tensor([user_sequence], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(seq)[0, -1, :]
        scores = logits.cpu().numpy()

        if exclude_seen:
            for idx in user_sequence:
                scores[idx] = float("-inf")
        scores[0] = float("-inf")      # exclude MASK token

        top_indices = scores.argsort()[::-1][:n]
        return [
            {"movieId": self.idx_to_movie.get(int(idx), int(idx)), "score": float(scores[idx])}
            for idx in top_indices
        ]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "state_dict": self.model.state_dict(),
                "config": self.config,
                "movie_to_idx": self.movie_to_idx,
                "vocab_size": self.model.vocab_size,
            }, f)
        logger.info("BERT4Rec saved to %s", path)

    def load(self, path: Path) -> "BERT4Rec":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.config = data["config"]
        self.movie_to_idx = data["movie_to_idx"]
        self.idx_to_movie = {v: k for k, v in self.movie_to_idx.items()}
        self.build(data["vocab_size"], self.movie_to_idx)
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()
        logger.info("BERT4Rec loaded from %s", path)
        return self

    def _save_checkpoint(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

"""
src/inference/bert4rec_inference.py

Self-contained BERT4Rec inference engine for the Movie Mind Reader app.
Loads the 32M-trained model from models/bert4rec/bert4rec_candidate.pkl
and scores sequences of movie IDs directly — no champion_model.pkl needed.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

MODEL_PATH = Path("models/bert4rec/bert4rec_candidate.pkl")
MAX_SEQ    = 50   # must match training config


# ── Model Architecture (must match training) ──────────────────────────────────

class BERT4RecModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        emb_dim = cfg["embedding_dim"]
        self.vocab_size = vocab_size

        self.item_embedding     = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(MAX_SEQ, emb_dim)
        self.embedding_norm     = nn.LayerNorm(emb_dim)
        self.embedding_dropout  = nn.Dropout(cfg["dropout"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=cfg["num_attention_heads"],
            dim_feedforward=cfg["feed_forward_dim"],
            dropout=cfg["attention_dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg["num_layers"],
            enable_nested_tensor=False,
        )
        self.prediction_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.prediction_head.weight = self.item_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)
        pad_mask = (input_ids == 0)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.prediction_head(x)


# ── Inference Engine ──────────────────────────────────────────────────────────

class BERT4RecInference:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path    = model_path
        self.model:        Optional[BERT4RecModel] = None
        self.movie_to_idx: dict[int, int] = {}
        self.idx_to_movie: dict[int, int] = {}
        self.vocab_size:   int = 0
        self.cfg:          dict = {}
        self.device        = torch.device("cpu")

    def load(self) -> "BERT4RecInference":
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)

        self.cfg          = payload["config"]
        self.vocab_size   = payload["vocab_size"]
        self.movie_to_idx = {int(k): int(v) for k, v in payload["movie_to_idx"].items()}
        self.idx_to_movie = {int(k): int(v) for k, v in payload["idx_to_movie"].items()}

        self.model = BERT4RecModel(self.vocab_size, self.cfg)
        # Strip DataParallel "module." prefix if present
        cleaned = {k.replace("module.", ""): v for k, v in payload["state_dict"].items()}
        self.model.load_state_dict(cleaned, strict=True)
        self.model.eval()
        self.model.to(self.device)
        return self

    def _encode_sequence(self, movie_id_sequence: list[int]) -> torch.Tensor:
        seq_idx = [self.movie_to_idx[m] for m in movie_id_sequence if m in self.movie_to_idx]
        if not seq_idx:
            return None
        if len(seq_idx) >= MAX_SEQ:
            seq_idx = seq_idx[-MAX_SEQ:]
        else:
            seq_idx = [0] * (MAX_SEQ - len(seq_idx)) + seq_idx
        return torch.tensor([seq_idx], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def score_movies(self, movie_id_sequence: list[int], candidate_ids: list[int]) -> dict[int, float]:
        """Score specific candidate movie IDs given a watch sequence."""
        if self.model is None:
            raise RuntimeError("Call load() first.")
        t = self._encode_sequence(movie_id_sequence)
        if t is None:
            return {mid: 0.0 for mid in candidate_ids}
        probs = torch.softmax(self.model(t)[0, -1, :], dim=-1)
        return {
            mid: float(probs[self.movie_to_idx[mid]].item())
            if mid in self.movie_to_idx else 0.0
            for mid in candidate_ids
        }

    @torch.no_grad()
    def top_n(self, movie_id_sequence: list[int], n: int = 20, exclude_ids: Optional[list[int]] = None) -> list[tuple[int, float]]:
        """Return top-N recommended (movie_id, score) from full vocab."""
        if self.model is None:
            raise RuntimeError("Call load() first.")
        exclude_set = set(exclude_ids or [])
        t = self._encode_sequence(movie_id_sequence)
        if t is None:
            return []
        probs = torch.softmax(self.model(t)[0, -1, :], dim=-1)
        topk_vals, topk_idxs = torch.topk(probs, min(n * 5, self.vocab_size - 1))
        results = []
        for prob_val, vocab_idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
            mid = self.idx_to_movie.get(vocab_idx)
            if mid is None or mid in exclude_set:
                continue
            results.append((mid, prob_val))
            if len(results) >= n:
                break
        return results

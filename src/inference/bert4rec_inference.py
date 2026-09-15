"""
src/inference/bert4rec_inference.py

Self-contained BERT4Rec inference engine for the Movie Mind Reader app.
Loads the 32M-trained model from models/bert4rec/bert4rec_candidate.pkl
and scores sequences of movie IDs directly — no champion_model.pkl needed.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Limit threads on single-vCPU EC2 to avoid RAM bloat
torch.set_num_threads(1)
if hasattr(torch, 'set_num_interop_threads'):
    torch.set_num_interop_threads(1)

MODEL_PATH       = Path("models/bert4rec/bert4rec_candidate.pkl")
PRECOMPUTED_PATH = Path("models/bert4rec/precomputed_recs.pkl")
MAX_SEQ          = 50   # must match training config


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
    def __init__(self, model_path: Path = MODEL_PATH, precomputed_path: Path = PRECOMPUTED_PATH):
        self.model_path       = model_path
        self.precomputed_path = precomputed_path
        self.model:           Optional[BERT4RecModel] = None
        self.movie_to_idx:    dict[int, int] = {}
        self.idx_to_movie:    dict[int, int] = {}
        self.precomputed_recs: dict[int, list[int]] = {}
        self.vocab_size:      int = 0
        self.cfg:             dict = {}
        self.device           = torch.device("cpu")

    def load(self) -> "BERT4RecInference":
        # 1. Load precomputed recommendations if available
        if self.precomputed_path.exists():
            try:
                with open(self.precomputed_path, "rb") as f:
                    self.precomputed_recs = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Could not load precomputed recs: {e}")

        if not self.model_path.exists() and not self.precomputed_recs:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                payload = pickle.load(f)

            self.cfg          = payload["config"]
            self.vocab_size   = payload["vocab_size"]
            self.movie_to_idx = {int(k): int(v) for k, v in payload["movie_to_idx"].items()}
            self.idx_to_movie = {int(k): int(v) for k, v in payload["idx_to_movie"].items()}
            state_dict        = payload.get("state_dict")
            del payload

            if state_dict is not None:
                try:
                    self.model = BERT4RecModel(self.vocab_size, self.cfg)
                    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    del state_dict
                    self.model.load_state_dict(cleaned, strict=True)
                    del cleaned
                    self.model.eval()
                    self.model = self.model.half()
                    self.model.to(self.device)
                except Exception as e:
                    print(f"⚠️ PyTorch model load failed (using precomputed recs): {e}")
                    self.model = None

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

    def score_movies(self, movie_id_sequence: list[int], candidate_ids: list[int]) -> dict[int, float]:
        """Score specific candidate movie IDs given a watch sequence."""
        if self.model is not None:
            with torch.no_grad():
                t = self._encode_sequence(movie_id_sequence)
                if t is None:
                    return {mid: 0.0 for mid in candidate_ids}
                with torch.autocast(device_type="cpu", dtype=torch.float16):
                    logits = self.model(t)[0, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
                return {
                    mid: float(probs[self.movie_to_idx[mid]].item())
                    if mid in self.movie_to_idx else 0.0
                    for mid in candidate_ids
                }

        # Fallback using precomputed recs
        last_mid = movie_id_sequence[-1] if movie_id_sequence else None
        if last_mid and last_mid in self.precomputed_recs:
            recs = self.precomputed_recs[last_mid]
            rec_map = {mid: 1.0 - (i * 0.01) for i, mid in enumerate(recs)}
            return {mid: rec_map.get(mid, 0.0) for mid in candidate_ids}

        return {mid: 0.0 for mid in candidate_ids}

    def top_n(self, movie_id_sequence: list[int], n: int = 20, exclude_ids: Optional[list[int]] = None) -> list[tuple[int, float]]:
        """Return top-N recommended (movie_id, score) from full vocab."""
        exclude_set = set(exclude_ids or [])

        # 1. Check precomputed recommendations first for fast zero-RAM lookup
        if movie_id_sequence:
            last_mid = movie_id_sequence[-1]
            if last_mid in self.precomputed_recs:
                results = []
                for i, mid in enumerate(self.precomputed_recs[last_mid]):
                    if mid not in exclude_set:
                        # Synthetic rank score: 1.0 down to 0.5
                        score = max(0.01, 1.0 - (i * 0.01))
                        results.append((mid, score))
                        if len(results) >= n:
                            break
                if results:
                    return results

        # 2. PyTorch live model inference
        if self.model is not None:
            with torch.no_grad():
                t = self._encode_sequence(movie_id_sequence)
                if t is None:
                    return []
                with torch.autocast(device_type="cpu", dtype=torch.float16):
                    logits = self.model(t)[0, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
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

        return []

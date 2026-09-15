"""
scripts/precompute_bert4rec_recs.py

Precomputes top-50 recommendations for the top N most frequent movies using vectorized BERT4Rec batch inference.
Saves to models/bert4rec/precomputed_recs.pkl.
"""

import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.bert4rec_inference import BERT4RecInference, MAX_SEQ


def main():
    print("🚀 Initializing BERT4Rec model...")
    engine = BERT4RecInference().load()
    if not engine or not engine.model:
        print("❌ Failed to load BERT4Rec model.")
        sys.exit(1)

    print(f"✅ Model loaded. Vocab size: {engine.vocab_size}")

    # Load ratings to get movie popularity order
    possible_ratings = [
        Path("data/raw/ml-1m/ratings.csv"),
        Path("data/raw/ml-latest-small/ratings.csv"),
        Path("data/raw/ml-32m/ratings.csv"),
    ]
    ratings_path = next((p for p in possible_ratings if p.exists()), None)
    if ratings_path:
        print(f"📊 Ranking movies by popularity from {ratings_path}...")
        ratings_df = pd.read_csv(ratings_path, usecols=["movieId"])
        top_mids = ratings_df["movieId"].value_counts().index.tolist()
    else:
        print("⚠️ ratings.csv not found, using vocab movie list order...")
        top_mids = list(engine.movie_to_idx.keys())

    # Filter to movies present in BERT4Rec vocab
    valid_mids = [int(mid) for mid in top_mids if int(mid) in engine.movie_to_idx]
    
    # Precompute top 5,000 movies
    N = min(5000, len(valid_mids))
    target_mids = valid_mids[:N]
    print(f"🎯 Precomputing top {N} most popular movies...")

    batch_size = 128
    precomputed = {}
    
    engine.model.eval()
    t_start = time.time()

    for i in range(0, N, batch_size):
        batch_mids = target_mids[i : i + batch_size]
        curr_batch_size = len(batch_mids)
        indices = [engine.movie_to_idx[m] for m in batch_mids]

        input_seqs = torch.zeros((curr_batch_size, MAX_SEQ), dtype=torch.long)
        # Put token index at last position
        input_seqs[:, -1] = torch.tensor(indices, dtype=torch.long)

        with torch.no_grad():
            pad_mask = (input_seqs == 0)
            positions = torch.arange(MAX_SEQ).unsqueeze(0)
            x = engine.model.item_embedding(input_seqs) + engine.model.position_embedding(positions)
            x = engine.model.embedding_norm(x)
            x = engine.model.embedding_dropout(x)
            x = engine.model.transformer(x, src_key_padding_mask=pad_mask)
            
            # Extract last position representation (B, emb_dim)
            last_x = x[:, -1, :]
            
            # Compute logits for last position (B, vocab_size)
            logits = engine.model.prediction_head(last_x)
            
            # Top 50 indices per movie
            topk_scores, topk_indices = torch.topk(logits, k=55, dim=-1)

        # Convert back to movie IDs
        for idx_in_batch, mid in enumerate(batch_mids):
            rec_indices = topk_indices[idx_in_batch].tolist()
            rec_mids = []
            for r_idx in rec_indices:
                if r_idx in engine.idx_to_movie:
                    rec_mid = engine.idx_to_movie[r_idx]
                    if rec_mid != mid:  # exclude self
                        rec_mids.append(rec_mid)
                if len(rec_mids) >= 50:
                    break
            precomputed[mid] = rec_mids[:50]

        if (i + batch_size) % 1000 < batch_size or (i + batch_size) >= N:
            elapsed = time.time() - t_start
            print(f"  Processed {min(i + batch_size, N)} / {N} movies ({elapsed:.1f}s elapsed)...")

    out_path = Path("models/bert4rec/precomputed_recs.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(precomputed, f)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"🎉 Successfully precomputed recommendations for {len(precomputed)} movies!")
    print(f"💾 Saved to {out_path} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

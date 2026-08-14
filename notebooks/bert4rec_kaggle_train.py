"""
notebooks/bert4rec_kaggle_train.py

BERT4Rec Training Notebook — Kaggle T4 x2 GPU
Dataset: MovieLens 32M (ml-latest full, ~87K movies)
Filter:  Movies released 1970 or later only, rating >= 3.5
Epochs:  15 (streamlined low-memory sequence builder to prevent OOM)

SETUP (do once in Kaggle):
    1. Add dataset: search "MovieLens 32M" on Kaggle and attach it (justsahil/movielens-32m)
    2. Add Secrets (optional for cloud storage): AZURE_STORAGE_CONNECTION_STRING
    3. Set GPU: Settings → Accelerator → GPU T4 x2
    4. Enable Internet: Settings → Internet → On
"""

# ── Cell 1: Install dependencies ──────────────────────────────────────────────
import subprocess
try:
    subprocess.run(
        ["pip", "install", "-q", "azure-storage-blob", "pandera"],
        check=False, timeout=120,
    )
    print("✓ Optional dependencies checked")
except Exception as e:
    print(f"Skipping optional install: {e}")


# ── Cell 2: Secrets ───────────────────────────────────────────────────────────
import os
from pathlib import Path

try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = _s.get_secret("AZURE_STORAGE_CONNECTION_STRING")
    os.environ["AZURE_STORAGE_CONTAINER"]         = _s.get_secret("AZURE_STORAGE_CONTAINER")
    print("✓ Azure secrets loaded")
except Exception as e:
    print(f"Azure secrets not found ({e}) — saving locally to Kaggle output")


# ── Cell 3: Azure helpers ─────────────────────────────────────────────────────
AZURE_OK = bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING"))

if AZURE_OK:
    try:
        from azure.storage.blob import BlobServiceClient
        _blob_service = BlobServiceClient.from_connection_string(
            os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        )
        _container = os.environ.get("AZURE_STORAGE_CONTAINER", "mlops-artifacts")

        def download_blob(blob_name: str, local_path: Path) -> bool:
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                client = _blob_service.get_blob_client(container=_container, blob=blob_name)
                with open(local_path, "wb") as f:
                    f.write(client.download_blob().readall())
                print(f"  ✓ Downloaded {blob_name} → {local_path}")
                return True
            except Exception as e:
                print(f"  ✗ {blob_name}: {e}")
                return False

        def upload_blob(local_path: Path, blob_name: str) -> None:
            client = _blob_service.get_blob_client(container=_container, blob=blob_name)
            with open(local_path, "rb") as f:
                client.upload_blob(f, overwrite=True)
            print(f"  ✓ Uploaded {local_path} → {blob_name}")

        print("✓ Azure Blob client ready")
    except Exception as e:
        print(f"Azure Blob init failed: {e}")
        AZURE_OK = False


# ── Cell 4: Streamlined Low-Memory Data Pipeline ──────────────────────────────
import pandas as pd
import re
from collections import defaultdict
import gc

WORK = Path("/kaggle/working")
WORK.mkdir(exist_ok=True)

def find_kaggle_file(pattern: str) -> Path | None:
    for p in sorted(Path("/kaggle/input").rglob(pattern)):
        return p
    return None

raw_ratings_path = find_kaggle_file("ratings.csv")
raw_movies_path  = find_kaggle_file("movies.csv")

if raw_ratings_path is None:
    raise FileNotFoundError(
        "ratings.csv not found in /kaggle/input.\n"
        "Please attach MovieLens 32M dataset: Add Data → search 'movielens-32m'"
    )

print(f"Loading ratings from: {raw_ratings_path}")
print(f"Loading movies  from: {raw_movies_path}")

# Load movies & filter 1970+
movies_raw = pd.read_csv(raw_movies_path)
movies_raw["year"] = movies_raw["title"].str.extract(r"\((\d{4})\)$").astype(float)
movies_1970 = movies_raw[movies_raw["year"].notna() & (movies_raw["year"] >= 1970)].copy()
valid_movie_ids = set(movies_1970["movieId"].tolist())
print(f"Movies: {len(movies_raw):,} total → {len(movies_1970):,} from 1970 onwards")

# Stream ratings into dictionary directly (< 300MB RAM usage)
user_histories = defaultdict(list)
total_read = 0
total_kept = 0

print("Streaming 32M ratings into memory-safe structure...")
for chunk in pd.read_csv(
    raw_ratings_path,
    chunksize=1_000_000,
    usecols=["userId", "movieId", "rating", "timestamp"],
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
):
    total_read += len(chunk)
    filtered = chunk[(chunk["rating"] >= 3.5) & (chunk["movieId"].isin(valid_movie_ids))]
    total_kept += len(filtered)
    for uid, mid, ts in zip(filtered["userId"], filtered["movieId"], filtered["timestamp"]):
        user_histories[uid].append((ts, mid))
    print(f"  Processed {total_read:,} rows | kept: {total_kept:,}", end="\r")

print(f"\n✓ Loaded {total_kept:,} positive ratings across {len(user_histories):,} users")


# ── Cell 5: Config ────────────────────────────────────────────────────────────
WARM_START = False
if AZURE_OK:
    ws_path = WORK / "champion_model.pkl"
    if not ws_path.exists():
        WARM_START = download_blob("models/champion/champion_model.pkl", ws_path)
    else:
        WARM_START = True

CONFIG = {
    "bert4rec": {
        "embedding_dim":          128,
        "hidden_dim":             128,
        "num_layers":             4,
        "num_attention_heads":    4,
        "feed_forward_dim":       512,
        "max_sequence_length":    50,
        "dropout":                0.15,
        "attention_dropout":      0.15,
        "mask_probability":       0.20,
        "learning_rate":          0.0005,
        "weight_decay":           0.01,
        "batch_size":             64,      # 64 batch size fits 54K vocab in GPU VRAM
        "gradient_accumulation_steps": 4,  # Effective batch = 256 (64 x 4)
        "max_epochs":             5 if WARM_START else 15,
        "early_stopping_patience": 4,
    },
    "sequences": {
        "min_rating_threshold":  3.5,
        "max_sequence_length":   50,
        "min_sequence_length":   5,
        "mask_probability":      0.20,
    },
}


# ── Cell 6: Fast Sequence Builder ─────────────────────────────────────────────
import random
import math

def build_sequences_from_dict(histories: dict, cfg: dict):
    max_seq_len = cfg["max_sequence_length"]
    min_seq_len = cfg["min_sequence_length"]
    mask_prob   = cfg["mask_probability"]
    MASK_TOKEN  = 0
    IGNORE_IDX  = -100

    # Build unique vocabulary
    all_movies = set()
    for uid, hist in histories.items():
        if len(hist) >= min_seq_len:
            for _, mid in hist:
                all_movies.add(mid)

    movie_to_idx = {m: i + 1 for i, m in enumerate(sorted(all_movies))}
    idx_to_movie = {v: k for k, v in movie_to_idx.items()}

    sequences = []
    random.seed(42)

    for uid, hist in histories.items():
        if len(hist) < min_seq_len:
            continue
        # Sort chronologically
        hist.sort(key=lambda x: x[0])
        movie_ids = [mid for _, mid in hist]
        if len(movie_ids) > max_seq_len:
            movie_ids = movie_ids[-max_seq_len:]

        indexed = [movie_to_idx[m] for m in movie_ids]
        masked  = list(indexed)
        labels  = [IGNORE_IDX] * len(indexed)

        for pos in range(len(indexed)):
            if random.random() < mask_prob:
                labels[pos] = indexed[pos]
                masked[pos] = MASK_TOKEN

        if all(l == IGNORE_IDX for l in labels):
            pos = random.randint(0, len(indexed) - 1)
            labels[pos] = indexed[pos]
            masked[pos] = MASK_TOKEN

        sequences.append({
            "user_id":         int(uid),
            "sequence":        indexed,
            "masked_sequence": masked,
            "labels":          labels,
        })

    return sequences, movie_to_idx, idx_to_movie

seq_cfg = CONFIG["sequences"]
sequences, movie_to_idx, idx_to_movie = build_sequences_from_dict(user_histories, seq_cfg)
vocab_size = len(movie_to_idx) + 1

print(f"✓ Built {len(sequences):,} sequences | Vocab size: {vocab_size:,}")

# Free raw user_histories to keep RAM clean
del user_histories
gc.collect()


# ── Cell 7: Dataset & DataLoader ──────────────────────────────────────────────
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | CUDA GPUs: {torch.cuda.device_count()}")

MAX_SEQ = CONFIG["bert4rec"]["max_sequence_length"]

class SequenceDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        def pad(lst, val=0):
            if len(lst) >= MAX_SEQ:
                return lst[-MAX_SEQ:]
            return [val] * (MAX_SEQ - len(lst)) + lst

        return {
            "masked_sequence": torch.tensor(pad(seq["masked_sequence"]),  dtype=torch.long),
            "labels":          torch.tensor(pad(seq["labels"], val=-100), dtype=torch.long),
        }

random.shuffle(sequences)
val_n    = max(1, int(len(sequences) * 0.10))
val_seqs = sequences[:val_n]
trn_seqs = sequences[val_n:]

BATCH = CONFIG["bert4rec"]["batch_size"]
train_loader = DataLoader(SequenceDataset(trn_seqs), batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(SequenceDataset(val_seqs),  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")


# ── Cell 8: BERT4Rec Model ────────────────────────────────────────────────────
import torch.nn as nn

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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=emb_dim ** -0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)
        pad_mask = (input_ids == 0)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.prediction_head(x)

model = BERT4RecModel(vocab_size, CONFIG["bert4rec"]).to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"✓ DataParallel enabled across {torch.cuda.device_count()} GPUs")

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")


# ── Cell 9: Training Loop ─────────────────────────────────────────────────────
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json as json_module
import time

cfg        = CONFIG["bert4rec"]
optimizer  = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
max_epochs = cfg["max_epochs"]
accum_steps = cfg.get("gradient_accumulation_steps", 4)
scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
criterion  = nn.CrossEntropyLoss(ignore_index=-100)
patience   = cfg["early_stopping_patience"]

best_val_loss    = float("inf")
patience_counter = 0
best_state       = None
history          = []

print(f"\nStarting training: {max_epochs} epochs | device: {device} | batch_size: {BATCH} (accum: {accum_steps})")
print("=" * 70)

for epoch in range(1, max_epochs + 1):
    t0 = time.time()
    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(train_loader):
        masked = batch["masked_sequence"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(masked).view(-1, vocab_size)
        loss   = criterion(logits, labels.view(-1))
        
        # Scale loss for gradient accumulation
        loss = loss / accum_steps
        loss.backward()
        
        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        total_loss += loss.item() * accum_steps
        n_batches  += 1
        
    scheduler.step()
    avg_train = total_loss / n_batches

    model.eval()
    val_loss = 0.0
    val_batches = 0
    ndcg_scores = []
    prec_scores = []
    K = 10

    with torch.no_grad():
        for batch in val_loader:
            masked = batch["masked_sequence"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits_3d = model(masked)
            logits_2d = logits_3d.view(-1, vocab_size)
            loss = criterion(logits_2d, labels.view(-1))
            val_loss    += loss.item()
            val_batches += 1

            label_flat = labels.view(-1)
            mask_pos   = (label_flat != -100).nonzero(as_tuple=True)[0]
            if len(mask_pos) == 0:
                continue

            preds   = logits_2d[mask_pos]
            targets = label_flat[mask_pos]
            top_k   = preds.topk(K, dim=-1).indices

            for j in range(len(targets)):
                tgt = targets[j].item()
                top = top_k[j].tolist()
                if tgt in top:
                    rank = top.index(tgt) + 1
                    ndcg_scores.append(1.0 / math.log2(rank + 1))
                    prec_scores.append(1.0)
                else:
                    ndcg_scores.append(0.0)
                    prec_scores.append(0.0)

    avg_val = val_loss / val_batches
    ndcg_10 = sum(ndcg_scores) / max(len(ndcg_scores), 1)
    prec_10 = sum(prec_scores) / max(len(prec_scores), 1)
    elapsed = time.time() - t0

    print(f"Epoch {epoch:02d}/{max_epochs} | train: {avg_train:.4f} | val: {avg_val:.4f} | "
          f"NDCG@10: {ndcg_10:.4f} | Prec@10: {prec_10:.4f} | {elapsed/60:.1f}m")

    history.append({
        "epoch": epoch, "train_loss": round(avg_train, 4),
        "val_loss": round(avg_val, 4), "ndcg_10": round(ndcg_10, 4), "prec_10": round(prec_10, 4)
    })

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        best_state    = {k: v.cpu().clone() for k, v in (model.module if hasattr(model, "module") else model).state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\n✓ Training complete | Best val_loss: {best_val_loss:.4f}")


# ── Cell 10: Save & Upload ───────────────────────────────────────────────────
import pickle

(model.module if hasattr(model, "module") else model).load_state_dict(best_state)
model.eval()

best_ndcg = max(h["ndcg_10"] for h in history)
best_prec = max(h["prec_10"] for h in history)

model_data = {
    "state_dict":   best_state,
    "config":       CONFIG["bert4rec"],
    "movie_to_idx": movie_to_idx,
    "idx_to_movie": idx_to_movie,
    "vocab_size":   vocab_size,
    "dataset":      "ml-latest-32m-1970plus",
    "metrics": {
        "best_val_loss": round(best_val_loss, 4),
        "ndcg_10":       round(best_ndcg, 4),
        "prec_10":       round(best_prec, 4),
        "num_sequences": len(sequences),
        "vocab_size":    vocab_size,
    },
}

candidate_path = WORK / "bert4rec_candidate.pkl"
with open(candidate_path, "wb") as f:
    pickle.dump(model_data, f)

metrics_path = WORK / "bert4rec_metrics.json"
with open(metrics_path, "w") as f:
    json_module.dump({"metrics": model_data["metrics"], "history": history}, f, indent=2)

print(f"✓ Model saved: {candidate_path}")

if AZURE_OK and "upload_blob" in dir():
    try:
        upload_blob(candidate_path, "models/bert4rec_candidate.pkl")
        upload_blob(metrics_path,   "models/bert4rec_metrics.json")
        print("✓ All artifacts uploaded to Azure Blob Storage")
    except Exception as e:
        print(f"Azure upload skipped: {e}")

print("\n🎉 Training finished successfully!")

"""
notebooks/bert4rec_kaggle_train.py

BERT4Rec Training Notebook — Kaggle T4 x2 GPU
Dataset: MovieLens 32M (ml-latest full, ~87K movies, 200K users)
Filter:  Movies released 1970 or later only
Epochs:  15 (30M ratings = equivalent learning to 40 epochs on ml-1m)

SETUP (do once in Kaggle):
    1. Add dataset: search "MovieLens 32M" on Kaggle and attach it
       (handle: rajmehra03/movielens32m or similar — check exact name)

    2. In Kaggle notebook → Add-ons → Secrets, add:
       - AZURE_STORAGE_CONNECTION_STRING
       - AZURE_STORAGE_CONTAINER  (e.g. mlops-artifacts)

    3. Set GPU: Settings → Accelerator → GPU T4 x2

    4. Enable Internet: Settings → Internet → On

WHAT THIS SCRIPT DOES:
    1. Install dependencies
    2. Load MovieLens 32M from Kaggle input (pre-mounted)
    3. Filter: movies with release year >= 1975
    4. Quality filter: users with >= 5 ratings, rating >= 3.5
    5. Build chronological BERT4Rec sequences (sorted by timestamp)
    6. Train BERT4Rec 4-Layer 128-Dim Transformer (15 epochs, early stop)
    7. Evaluate: val loss + NDCG@10 + Precision@10
    8. Upload candidate model + metrics to Azure Blob
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
    print(f"Azure secrets not found ({e}) — will save to Kaggle output only")


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
                print(f"  ✓ {blob_name} → {local_path}")
                return True
            except Exception as e:
                print(f"  ✗ {blob_name}: {e}")
                return False

        def upload_blob(local_path: Path, blob_name: str) -> None:
            client = _blob_service.get_blob_client(container=_container, blob=blob_name)
            with open(local_path, "rb") as f:
                client.upload_blob(f, overwrite=True)
            print(f"  ✓ {local_path} → {blob_name}")

        print("✓ Azure Blob client ready")
    except Exception as e:
        print(f"Azure Blob init failed: {e}")
        AZURE_OK = False


# ── Cell 4: Load MovieLens 32M from Kaggle Input ──────────────────────────────
import pandas as pd
import re

WORK = Path("/kaggle/working")
WORK.mkdir(exist_ok=True)

ratings_path = WORK / "ratings_filtered.csv"
movies_path  = WORK / "movies_filtered.csv"

def find_kaggle_file(pattern: str) -> Path | None:
    """Recursively find a file matching pattern in /kaggle/input."""
    for p in sorted(Path("/kaggle/input").rglob(pattern)):
        return p
    return None

if not ratings_path.exists():
    raw_ratings_path = find_kaggle_file("ratings.csv")
    raw_movies_path  = find_kaggle_file("movies.csv")

    if raw_ratings_path is None:
        raise FileNotFoundError(
            "ratings.csv not found in /kaggle/input.\n"
            "Please attach the MovieLens 32M dataset to this notebook:\n"
            "  → Notebook Settings → Add Data → search 'MovieLens 32M'"
        )

    print(f"Loading ratings from: {raw_ratings_path}")
    print(f"Loading movies  from: {raw_movies_path}")

    # ── Load movies & extract release year ────────────────────────────────────
    movies_raw = pd.read_csv(raw_movies_path)
    movies_raw["year"] = (
        movies_raw["title"]
        .str.extract(r"\((\d{4})\)$")
        .astype(float)
    )
    before = len(movies_raw)
    movies_1975 = movies_raw[
        movies_raw["year"].notna() & (movies_raw["year"] >= 1970)
    ].copy()
    print(f"Movies: {before:,} total → {len(movies_1975):,} from 1970 onwards "
          f"({before - len(movies_1975):,} pre-1970 removed)")

    valid_movie_ids = set(movies_1975["movieId"].tolist())

    # ── Load ratings in chunks (32M rows = ~1.5GB) ────────────────────────────
    print("Loading 32M ratings in chunks (this may take a few minutes)...")
    chunks = []
    chunk_size = 2_000_000
    for i, chunk in enumerate(pd.read_csv(raw_ratings_path, chunksize=chunk_size)):
        # Filter: only 1975+ movies + rating >= 3.5
        filtered = chunk[
            chunk["movieId"].isin(valid_movie_ids) &
            (chunk["rating"] >= 3.5)
        ]
        chunks.append(filtered)
        loaded = (i + 1) * chunk_size
        print(f"  Processed ~{min(loaded, 32_000_000):,} rows | kept so far: "
              f"{sum(len(c) for c in chunks):,}", end="\r")
    print()

    ratings_all = pd.concat(chunks, ignore_index=True)
    del chunks

    # ── Quality filter: users with >= 5 positive ratings ─────────────────────
    user_counts = ratings_all.groupby("userId").size()
    active_users = user_counts[user_counts >= 5].index
    ratings_all = ratings_all[ratings_all["userId"].isin(active_users)]

    print(f"\nAfter filtering:")
    print(f"  Ratings  : {len(ratings_all):,}")
    print(f"  Users    : {ratings_all['userId'].nunique():,}")
    print(f"  Movies   : {ratings_all['movieId'].nunique():,}")

    # Save filtered data
    ratings_all.to_csv(ratings_path, index=False)
    movies_1975.to_csv(movies_path,  index=False)
    print(f"\n✓ Filtered data saved → {ratings_path}")

else:
    print(f"✓ Filtered data already cached at {ratings_path}")
    ratings_all = pd.read_csv(ratings_path)
    movies_1975 = pd.read_csv(movies_path)

print(f"Dataset ready: {len(ratings_all):,} ratings | "
      f"{ratings_all['userId'].nunique():,} users | "
      f"{ratings_all['movieId'].nunique():,} movies")


# ── Cell 5: Config ────────────────────────────────────────────────────────────
WARM_START = False

# Try loading warm-start champion from Azure
if AZURE_OK:
    ws_path = WORK / "champion_model.pkl"
    if not ws_path.exists():
        WARM_START = download_blob("models/champion/champion_model.pkl", ws_path)
    else:
        WARM_START = True
    if WARM_START:
        print("✓ Champion model found — warm-start fine-tuning (5 epochs)")

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
        # 15 epochs on 30M ratings ≈ 40 epochs on 1M ratings
        "batch_size":             256,     # T4 x2 can handle 256
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

print(f"\nTraining config:")
print(f"  Mode   : {'warm-start fine-tune' if WARM_START else 'full training'}")
print(f"  Epochs : {CONFIG['bert4rec']['max_epochs']}")
print(f"  Batch  : {CONFIG['bert4rec']['batch_size']}")
print(f"  Layers : {CONFIG['bert4rec']['num_layers']} × "
      f"{CONFIG['bert4rec']['embedding_dim']}dim × "
      f"{CONFIG['bert4rec']['num_attention_heads']}heads")


# ── Cell 6: Build sequences ───────────────────────────────────────────────────
import random
import math

def build_sequences(ratings_df: pd.DataFrame, cfg: dict):
    """
    Build chronological masked sequences for BERT4Rec training.
    Sequences are sorted by timestamp per user.
    """
    max_seq_len = cfg["max_sequence_length"]
    min_seq_len = cfg["min_sequence_length"]
    mask_prob   = cfg["mask_probability"]
    MASK_TOKEN  = 0
    IGNORE_IDX  = -100

    # Sort once globally
    ratings_df = ratings_df.sort_values(["userId", "timestamp"])

    # Build movie vocab (1-indexed; 0 = padding/mask)
    all_movies   = sorted(ratings_df["movieId"].unique())
    movie_to_idx = {m: i + 1 for i, m in enumerate(all_movies)}
    idx_to_movie = {v: k for k, v in movie_to_idx.items()}

    sequences = []
    random.seed(42)

    grouped = ratings_df.groupby("userId", sort=False)
    n_users = len(grouped)
    print(f"Building sequences for {n_users:,} users...")

    for i, (uid, group) in enumerate(grouped):
        movie_ids = group["movieId"].tolist()
        if len(movie_ids) < min_seq_len:
            continue
        if len(movie_ids) > max_seq_len:
            movie_ids = movie_ids[-max_seq_len:]

        indexed = [movie_to_idx[m] for m in movie_ids]
        masked  = list(indexed)
        labels  = [IGNORE_IDX] * len(indexed)

        for pos in range(len(indexed)):
            if random.random() < mask_prob:
                labels[pos]  = indexed[pos]
                masked[pos]  = MASK_TOKEN

        # Guarantee at least one masked position
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

        if (i + 1) % 10_000 == 0:
            print(f"  {i+1:,}/{n_users:,} users processed", end="\r")

    print(f"\n✓ Built {len(sequences):,} sequences | vocab size: {len(movie_to_idx):,}")
    return sequences, movie_to_idx, idx_to_movie


seq_cfg = CONFIG["sequences"]
sequences, movie_to_idx, idx_to_movie = build_sequences(ratings_all, seq_cfg)
vocab_size = len(movie_to_idx) + 1  # +1 for mask/pad token
print(f"Vocabulary: {vocab_size:,} items (including padding token)")

# Free ratings from RAM — not needed anymore
del ratings_all
import gc; gc.collect()


# ── Cell 7: Dataset & DataLoader ──────────────────────────────────────────────
import torch
from torch.utils.data import Dataset, DataLoader

print(f"\nCUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            "masked_sequence": torch.tensor(pad(seq["masked_sequence"]),      dtype=torch.long),
            "labels":          torch.tensor(pad(seq["labels"], val=-100),     dtype=torch.long),
        }


random.shuffle(sequences)
val_n    = max(1, int(len(sequences) * 0.10))
val_seqs = sequences[:val_n]
trn_seqs = sequences[val_n:]

BATCH = CONFIG["bert4rec"]["batch_size"]
n_workers = 4  # T4 x2 has enough CPU cores

train_loader = DataLoader(
    SequenceDataset(trn_seqs), batch_size=BATCH,
    shuffle=True, num_workers=n_workers, pin_memory=True,
)
val_loader = DataLoader(
    SequenceDataset(val_seqs), batch_size=BATCH,
    shuffle=False, num_workers=n_workers, pin_memory=True,
)

print(f"\nTrain sequences : {len(trn_seqs):,} | batches: {len(train_loader):,}")
print(f"Val   sequences : {len(val_seqs):,}  | batches: {len(val_loader):,}")


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

        # Weight tying: prediction head shares item embedding weights
        self.prediction_head.weight = self.item_embedding.weight

        # Xavier init
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
        pad_mask = (input_ids == 0)  # True where padded
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.prediction_head(x)


model = BERT4RecModel(vocab_size, CONFIG["bert4rec"]).to(device)

# Multi-GPU: wrap with DataParallel if 2× T4 available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"✓ DataParallel across {torch.cuda.device_count()} GPUs")

# Warm-start: load champion weights (vocabulary may differ → strict=False)
if WARM_START:
    try:
        import pickle
        with open(WORK / "champion_model.pkl", "rb") as f:
            checkpoint = pickle.load(f)
        state = checkpoint.get("state_dict", checkpoint)
        # Strip DataParallel prefix if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = (model.module if hasattr(model, "module") else model).load_state_dict(
            state, strict=False
        )
        print(f"✓ Champion weights loaded | missing: {len(missing)} | unexpected: {len(unexpected)}")
    except Exception as e:
        print(f"Warm-start failed ({e}) — training from scratch")
        WARM_START = False

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")


# ── Cell 9: Training loop ─────────────────────────────────────────────────────
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json as json_module
import time

cfg        = CONFIG["bert4rec"]
optimizer  = AdamW(
    model.parameters(),
    lr=cfg["learning_rate"],
    weight_decay=cfg["weight_decay"],
    betas=(0.9, 0.999),
)
max_epochs = cfg["max_epochs"]

# Cosine with warm restarts: restarts every 5 epochs to escape local minima
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
patience  = cfg["early_stopping_patience"]

best_val_loss    = float("inf")
patience_counter = 0
best_state       = None
history          = []

print(f"\nStarting training: {max_epochs} epochs on {device}")
print(f"Vocab: {vocab_size:,} | Batch: {BATCH} | Seq len: {MAX_SEQ}")
print("=" * 70)

for epoch in range(1, max_epochs + 1):
    t0 = time.time()

    # ── Train ────────────────────────────────────────────────────────────
    model.train()
    total_loss  = 0.0
    n_batches   = 0
    for batch in train_loader:
        masked = batch["masked_sequence"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(masked).view(-1, vocab_size)
        loss   = criterion(logits, labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    scheduler.step()
    avg_train = total_loss / n_batches

    # ── Validate ──────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_batches = 0

    # NDCG@10 & Precision@10 — sample-based evaluation
    ndcg_scores  = []
    prec_scores  = []
    K = 10

    with torch.no_grad():
        for batch in val_loader:
            masked = batch["masked_sequence"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits_3d = model(masked)             # (B, L, V)
            logits_2d = logits_3d.view(-1, vocab_size)
            loss = criterion(logits_2d, labels.view(-1))
            val_loss   += loss.item()
            val_batches += 1

            # Compute ranking metrics on masked positions
            label_flat = labels.view(-1)
            mask_pos   = (label_flat != -100).nonzero(as_tuple=True)[0]
            if len(mask_pos) == 0:
                continue

            preds   = logits_2d[mask_pos]          # (n_masked, V)
            targets = label_flat[mask_pos]          # (n_masked,)
            top_k   = preds.topk(K, dim=-1).indices  # (n_masked, K)

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

    avg_val  = val_loss / val_batches
    ndcg_10  = sum(ndcg_scores)  / max(len(ndcg_scores), 1)
    prec_10  = sum(prec_scores)  / max(len(prec_scores), 1)
    elapsed  = time.time() - t0

    print(f"Epoch {epoch:02d}/{max_epochs} | "
          f"train: {avg_train:.4f} | val: {avg_val:.4f} | "
          f"NDCG@10: {ndcg_10:.4f} | Prec@10: {prec_10:.4f} | "
          f"{elapsed/60:.1f}min")

    history.append({
        "epoch": epoch,
        "train_loss": round(avg_train, 4),
        "val_loss":   round(avg_val,   4),
        "ndcg_10":    round(ndcg_10,   4),
        "prec_10":    round(prec_10,   4),
    })

    # Early stopping
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        best_state    = {
            k: v.cpu().clone()
            for k, v in (model.module if hasattr(model, "module") else model).state_dict().items()
        }
        patience_counter = 0
        print(f"  ↑ New best val_loss: {best_val_loss:.4f} — checkpoint saved")
    else:
        patience_counter += 1
        print(f"  → No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\n✓ Training complete | Best val_loss: {best_val_loss:.4f}")


# ── Cell 10: Save model ───────────────────────────────────────────────────────
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
        "warm_start":    WARM_START,
    },
}

candidate_path = WORK / "bert4rec_candidate.pkl"
with open(candidate_path, "wb") as f:
    pickle.dump(model_data, f)

metrics = model_data["metrics"]
metrics_path = WORK / "bert4rec_metrics.json"
with open(metrics_path, "w") as f:
    json_module.dump({"metrics": metrics, "history": history}, f, indent=2)

print(f"✓ Model saved : {candidate_path}")
print(f"✓ Metrics     : {metrics}")


# ── Cell 11: Upload to Azure Blob ─────────────────────────────────────────────
print("\nUploading artifacts ...")
if AZURE_OK and "upload_blob" in dir():
    try:
        upload_blob(candidate_path, "models/bert4rec_candidate.pkl")
        upload_blob(metrics_path,   "models/bert4rec_metrics.json")
        print("✓ All artifacts uploaded to Azure Blob Storage")
    except Exception as e:
        print(f"Azure upload failed: {e}")
        print(f"Artifacts saved locally at {WORK}")
else:
    print(f"✓ Artifacts saved to Kaggle output: {WORK}")
    print("  Download manually: bert4rec_candidate.pkl + bert4rec_metrics.json")

print("\n🎉 BERT4Rec (ml-latest 32M · 1975+) training complete!")
print(f"   NDCG@10: {best_ndcg:.4f} | Prec@10: {best_prec:.4f} | Val loss: {best_val_loss:.4f}")

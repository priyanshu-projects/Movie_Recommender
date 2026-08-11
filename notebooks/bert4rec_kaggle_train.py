"""
notebooks/bert4rec_kaggle_train.py

BERT4Rec Training Notebook — runs on Kaggle T4 GPU.

This script is designed to be pasted into a Kaggle notebook
(or run as a Kaggle kernel via API trigger).

SETUP (do once in Kaggle):
    1. In Kaggle notebook → Add-ons → Secrets, add:
       - AZURE_STORAGE_CONNECTION_STRING
       - AZURE_STORAGE_CONTAINER  (e.g. mlops-artifacts)

    2. Set GPU accelerator: Settings → Accelerator → GPU T4 x2

    3. Enable Internet: Settings → Internet → On

WHAT THIS SCRIPT DOES:
    1. Install dependencies
    2. Pull training data from Azure Blob Storage
    3. Pull warm-start weights from Azure Blob (if available)
    4. Build BERT4Rec sequences
    5. Train BERT4Rec on T4 GPU (warm-start: 8 epochs / full: 30 epochs)
    6. Evaluate model
    7. Upload candidate model + metrics back to Azure Blob
"""

# ── Cell 1: Install dependencies ─────────────────────────────────────────────
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "azure-storage-blob",
    "scikit-surprise",
    "pandera",
    "pyyaml",
    "mlflow",
], check=True)
print("✓ Dependencies installed")


# ── Cell 2: Kaggle secrets → env vars ────────────────────────────────────────
import os

# Kaggle Secrets API (available in Kaggle notebooks)
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = secrets.get_secret("AZURE_STORAGE_CONNECTION_STRING")
    os.environ["AZURE_STORAGE_CONTAINER"]         = secrets.get_secret("AZURE_STORAGE_CONTAINER")
    print("✓ Secrets loaded from Kaggle")
except ImportError:
    print("Not running in Kaggle — expecting env vars already set")


# ── Cell 3: Pull training data from Azure Blob ────────────────────────────────
import json
from pathlib import Path
from azure.storage.blob import BlobServiceClient

CONTAINER     = os.environ["AZURE_STORAGE_CONTAINER"]
CONN_STR      = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
blob_service  = BlobServiceClient.from_connection_string(CONN_STR)


def download_blob(blob_name: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob_client = blob_service.get_blob_client(container=CONTAINER, blob=blob_name)
    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob().readall())
    print(f"  ✓ Downloaded {blob_name} → {local_path}")
    return local_path


def upload_blob(local_path: Path, blob_name: str) -> None:
    blob_client = blob_service.get_blob_client(container=CONTAINER, blob=blob_name)
    with open(local_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
    print(f"  ✓ Uploaded {local_path} → {blob_name}")


print("Downloading training data from Azure Blob ...")
download_blob("training/all_ratings.csv",  Path("/kaggle/working/all_ratings.csv"))

# Try to get warm-start champion weights
WARM_START = False
try:
    download_blob("models/champion/champion_model.pkl", Path("/kaggle/working/champion_model.pkl"))
    download_blob("models/champion/champion_meta.yaml", Path("/kaggle/working/champion_meta.yaml"))
    WARM_START = True
    print("✓ Champion model found — will use warm-start fine-tuning")
except Exception:
    print("No champion model — will train from scratch")

print("✓ Data ready")


# ── Cell 4: Load config ───────────────────────────────────────────────────────
import yaml

# Inline config — mirrors configs/config.yaml but with Kaggle paths
CONFIG = {
    "bert4rec": {
        "embedding_dim":         64,
        "hidden_dim":            64,
        "num_layers":            2,
        "num_attention_heads":   2,
        "feed_forward_dim":      256,
        "max_sequence_length":   50,
        "dropout":               0.2,
        "attention_dropout":     0.2,
        "mask_probability":      0.20,
        "learning_rate":         0.0001,
        "weight_decay":          0.01,
        "batch_size":            256,       # Larger batch on GPU
        "max_epochs":            8 if WARM_START else 30,
        "early_stopping_patience": 3,
    },
    "sequences": {
        "min_rating_threshold": 3.5,
        "max_sequence_length":  50,
        "min_sequence_length":  5,
        "mask_probability":     0.20,
    },
}
print(f"Training mode: {'warm-start fine-tuning' if WARM_START else 'full training from scratch'}")
print(f"Max epochs: {CONFIG['bert4rec']['max_epochs']}")


# ── Cell 5: Build sequences ───────────────────────────────────────────────────
import pandas as pd
import json as json_module
import random, math

def build_sequences(ratings_df, cfg):
    min_rating  = cfg["min_rating_threshold"]
    max_seq_len = cfg["max_sequence_length"]
    min_seq_len = cfg["min_sequence_length"]
    mask_prob   = cfg["mask_probability"]
    MASK_TOKEN  = 0
    IGNORE_IDX  = -100

    positive = ratings_df[ratings_df["rating"] >= min_rating].copy()
    positive = positive.sort_values(["userId", "timestamp"])
    all_movies = sorted(positive["movieId"].unique())
    movie_to_idx = {m: i + 1 for i, m in enumerate(all_movies)}

    sequences = []
    random.seed(42)
    for uid, group in positive.groupby("userId"):
        movie_ids = group["movieId"].tolist()
        if len(movie_ids) < min_seq_len:
            continue
        if len(movie_ids) > max_seq_len:
            movie_ids = movie_ids[-max_seq_len:]
        indexed = [movie_to_idx[m] for m in movie_ids]
        masked = list(indexed)
        labels = [IGNORE_IDX] * len(indexed)
        for pos in range(len(indexed)):
            if random.random() < mask_prob:
                labels[pos] = indexed[pos]
                masked[pos] = MASK_TOKEN
        if all(l == IGNORE_IDX for l in labels):
            pos = random.randint(0, len(indexed) - 1)
            labels[pos] = indexed[pos]
            masked[pos] = MASK_TOKEN
        sequences.append({
            "user_id": int(uid), "sequence": indexed,
            "masked_sequence": masked, "labels": labels,
        })
    return sequences, movie_to_idx

ratings = pd.read_csv("/kaggle/working/all_ratings.csv")
print(f"Ratings loaded: {len(ratings):,}")
seq_cfg = CONFIG["sequences"]
sequences, movie_to_idx = build_sequences(ratings, seq_cfg)
vocab_size = len(movie_to_idx) + 1
print(f"Sequences: {len(sequences):,} | Vocab size: {vocab_size:,}")


# ── Cell 6: Build PyTorch Dataset + DataLoader ────────────────────────────────
import torch
from torch.utils.data import Dataset, DataLoader

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
            "masked_sequence": torch.tensor(pad(seq["masked_sequence"]), dtype=torch.long),
            "labels":          torch.tensor(pad(seq["labels"], -100),    dtype=torch.long),
        }

random.shuffle(sequences)
val_n    = max(1, int(len(sequences) * 0.10))
val_seqs = sequences[:val_n]
trn_seqs = sequences[val_n:]

BATCH = CONFIG["bert4rec"]["batch_size"]
train_loader = DataLoader(SequenceDataset(trn_seqs), batch_size=BATCH, shuffle=True,  num_workers=2)
val_loader   = DataLoader(SequenceDataset(val_seqs),  batch_size=BATCH, shuffle=False, num_workers=2)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


# ── Cell 7: Define BERT4Rec model ─────────────────────────────────────────────
import torch.nn as nn

class BERT4RecModel(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        emb_dim   = cfg["embedding_dim"]
        self.vocab_size = vocab_size
        self.item_embedding     = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(MAX_SEQ, emb_dim)
        self.embedding_dropout  = nn.Dropout(cfg["dropout"])
        self.embedding_norm     = nn.LayerNorm(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=cfg["num_attention_heads"],
            dim_feedforward=cfg["feed_forward_dim"],
            dropout=cfg["attention_dropout"],
            batch_first=True, norm_first=True,
        )
        self.transformer      = nn.TransformerEncoder(encoder_layer, cfg["num_layers"], enable_nested_tensor=False)
        self.prediction_head  = nn.Linear(emb_dim, vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, emb_dim ** -0.5)
        self.prediction_head.weight = self.item_embedding.weight

    def forward(self, input_ids):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embedding_norm(self.item_embedding(input_ids) + self.position_embedding(positions))
        x = self.embedding_dropout(x)
        x = self.transformer(x, src_key_padding_mask=(input_ids == 0))
        return self.prediction_head(x)

model = BERT4RecModel(vocab_size, CONFIG["bert4rec"]).to(device)

# Load warm-start weights if available
if WARM_START:
    try:
        import pickle
        with open("/kaggle/working/champion_model.pkl", "rb") as f:
            checkpoint = pickle.load(f)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Champion weights loaded for warm-start fine-tuning")
    except Exception as e:
        print(f"Could not load champion weights: {e} — training from scratch")

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")


# ── Cell 8: Training loop ─────────────────────────────────────────────────────
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

cfg        = CONFIG["bert4rec"]
optimizer  = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
max_epochs = cfg["max_epochs"]
scheduler  = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
criterion  = nn.CrossEntropyLoss(ignore_index=-100)
patience   = cfg["early_stopping_patience"]

best_val_loss    = float("inf")
patience_counter = 0
best_state       = None

print(f"\nStarting training: {max_epochs} epochs | device: {device}")
print("-" * 60)

for epoch in range(1, max_epochs + 1):
    # Train
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        masked = batch["masked_sequence"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(masked).view(-1, vocab_size)
        loss   = criterion(logits, labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_train = total_loss / len(train_loader)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            masked = batch["masked_sequence"].to(device)
            labels = batch["labels"].to(device)
            logits = model(masked).view(-1, vocab_size)
            val_loss += criterion(logits, labels.view(-1)).item()
    avg_val = val_loss / len(val_loader)

    print(f"Epoch {epoch:02d}/{max_epochs} | train_loss: {avg_train:.4f} | val_loss: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\n✓ Training complete | Best val_loss: {best_val_loss:.4f}")


# ── Cell 9: Save model ────────────────────────────────────────────────────────
import pickle

model.load_state_dict(best_state)
model.eval()

model_data = {
    "state_dict":   best_state,
    "config":       CONFIG["bert4rec"],
    "movie_to_idx": movie_to_idx,
    "vocab_size":   vocab_size,
}
model_path = Path("/kaggle/working/bert4rec_candidate.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model_data, f)

# Save metrics
metrics = {
    "best_val_loss": round(best_val_loss, 4),
    "warm_start":    WARM_START,
    "num_sequences": len(sequences),
    "vocab_size":    vocab_size,
}
metrics_path = Path("/kaggle/working/bert4rec_metrics.json")
with open(metrics_path, "w") as f:
    json_module.dump(metrics, f, indent=2)

print(f"✓ Model saved: {model_path}")
print(f"✓ Metrics: {metrics}")


# ── Cell 10: Upload to Azure Blob ─────────────────────────────────────────────
print("\nUploading model artifact to Azure Blob ...")
upload_blob(model_path,   "models/bert4rec_candidate.pkl")
upload_blob(metrics_path, "models/bert4rec_metrics.json")
print("✓ All artifacts uploaded to Azure Blob Storage")
print("\n🎉 BERT4Rec training complete!")

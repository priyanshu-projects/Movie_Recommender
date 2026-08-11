# MLOps Movie Recommendation System

> **Production-grade MLOps portfolio project** — Dual-model sequential recommendation architecture (SVD + BERT4Rec) built on 1,000,000 MovieLens ratings, with automated continuous replay, Kaggle T4 GPU fine-tuning, Azure Blob artifact storage, and Azure Container App serving.

---

## 🏗️ Architecture Overview

```
                               ┌───────────────────────────┐
                               │   Immutable Master Data   │
                               │  data/raw/ml-1m/ (1M rows)│
                               └─────────────┬─────────────┘
                                             │
                                   Replay Controller
                             (releases N-day batches)
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     Data Validation       │
                               │   (pandera schema checks) │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             ┌──────────────────┐                        ┌──────────────────┐
             │  SVD Retraining  │                        │ Sequence Builder │
             │  (scikit-surprise│                        │ (BERT4Rec masks) │
             │   local / CPU)   │                        └────────┬─────────┘
             └────────┬─────────┘                                 │
                      │                                    Kaggle T4 GPU
                      │                              (warm-start fine-tuning)
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                               ┌───────────────────────────┐
                               │ Champion/Challenger Gate  │
                               │  (MLflow + model_registry)│
                               └────────────┬──────────────┘
                                            │
                                            ▼
                               ┌───────────────────────────┐
                               │   Azure Container Apps    │
                               │   FastAPI /recommend API  │
                               └───────────────────────────┘
```

---

## ✨ Key Features

- **Continuous Chronological Replay**: Simulates real production data streams by periodically releasing timestamped chunks from an immutable master dataset (`ml-1m`).
- **Dual-Model Engine**:
  - **SVD (Matrix Factorization)**: Fast collaborative filtering baseline.
  - **BERT4Rec (Transformer Encoder)**: Sequential recommendation model with bidirectional attention, masked item prediction, and automatic CUDA/CPU detection.
- **Warm-Start GPU Fine-Tuning**: BERT4Rec fine-tunes on Kaggle T4 GPU (`slavery786/bert4rec-movie-recommender-fine-tuning`) for 5–8 epochs per batch instead of retraining from scratch.
- **Champion / Challenger Promotion**: Models compete on held-out temporal evaluation metrics (`NDCG@10`, `Recall@10`, `RMSE`). Only models exceeding promotion thresholds become Champion.
- **Automated Cloud Retraining**: GitHub Actions workflow (`.github/workflows/retrain.yml`) runs on the 1st & 15th of every month: Replay → Validate → SVD → Kaggle API → Promote → Azure Container App redeploy.
- **Azure Cloud Integration**: Model artifacts and dataset snapshots stored in Azure Blob Storage; API served via Azure Container Apps.

---

## 📁 Project Structure

```text
.
├── .github/workflows/
│   └── retrain.yml              # 12-step bi-weekly automated retraining pipeline
├── configs/
│   └── config.yaml              # Central single-source-of-truth configuration
├── azure/
│   └── container-app.yml        # Azure Container App deployment specification
├── notebooks/
│   ├── bert4rec_kaggle_train.py # Kaggle GPU T4 training notebook script
│   └── kernel-metadata.json    # Kaggle CLI metadata configuration
├── dags/
│   └── movie_retraining_dag.py  # 13-task Airflow Retraining DAG
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI server (serving champion model)
│   ├── data/
│   │   ├── validation.py        # pandera schema and boundary checks
│   │   ├── temporal_split.py    # Time-aware train/val/test splitter
│   │   └── sequence_builder.py  # Masked interaction sequence builder for BERT4Rec
│   ├── evaluation/
│   │   └── metrics.py           # NDCG@K, Recall@K, Precision@K, Hit Rate@K, MRR@K
│   ├── inference/
│   │   └── recommender.py       # Unified Recommender class (SVD + BERT4Rec)
│   ├── ingestion/
│   │   ├── movielens_fetcher.py # MovieLens ml-1m downloader & normalizer
│   │   └── snapshot_diff.py     # Rating snapshot diff calculator
│   ├── models/
│   │   ├── base_model.py        # Abstract BaseRecommender interface
│   │   ├── svd_model.py         # SVD Matrix Factorization wrapper
│   │   └── bert4rec.py          # PyTorch BERT4Rec Transformer model
│   ├── monitoring/
│   │   └── performance.py       # Performance history logger & degradation detector
│   ├── replay/
│   │   ├── replay_controller.py # Chronological batch release engine
│   │   └── replay_state.py      # Idempotent state management
│   ├── storage/
│   │   └── azure_blob.py        # Azure Blob Storage client
│   ├── tracking/
│   │   ├── mlflow_tracker.py    # MLflow experiment tracking helpers
│   │   └── model_registry.py    # Champion/Challenger promotion gate
│   └── training/
│       ├── train_bert4rec.py    # BERT4Rec training/fine-tuning script
│       └── kaggle_trigger.py    # Kaggle API trigger & polling script
├── tests/unit/                  # Pytest unit test suite (16 tests)
├── Dockerfile                   # Production container definition
└── requirements.txt
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone & Install Environment
```bash
git clone https://github.com/priyanshu-projects/Movie_Recommender.git
cd Movie_Recommender
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch & Normalize `ml-1m` Dataset
```bash
python -m src.ingestion.movielens_fetcher --dataset ml-1m
```

### 3. Release Replay Batch & Train Baseline
```bash
# Release chronological batch 001
python -m src.replay.replay_controller

# Validate batch data
python -m src.data.validation --input data/raw/incoming/batch_001.csv

# Train SVD model and generate recommendations
python -m src.models.svd_model
```

### 4. Run FastAPI Server
```bash
uvicorn src.api.main:app --reload --port 8000
```

Query recommendations:
```bash
curl "http://localhost:8000/recommend/1?k=5"
```

Response:
```json
{
  "user_id": 1,
  "model_type": "svd",
  "recommendations": [
    {
      "movieId": 2019,
      "score": 4.7669,
      "title": "Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954)",
      "genres": "Action|Drama"
    },
    {
      "movieId": 3307,
      "score": 4.6931,
      "title": "City Lights (1931)",
      "genres": "Comedy|Drama|Romance"
    },
    {
      "movieId": 318,
      "score": 4.6552,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Drama"
    }
  ]
}
```

---

## 🧪 Running Unit Tests

Run the full pytest suite:
```bash
pytest tests/unit/ -v
```

Output:
```text
16 passed in 3.75s
```

---

## 🛠️ Tech Stack & Tooling

| Layer | Technology |
|---|---|
| **Master Dataset** | MovieLens `ml-1m` (1,000,209 ratings, 3,883 movies) |
| **Recommendation Models** | SVD (`scikit-surprise`) + BERT4Rec (`PyTorch`) |
| **GPU Acceleration** | Kaggle T4 GPU (`slavery786/bert4rec-movie-recommender-fine-tuning`) |
| **Cloud Storage** | Azure Blob Storage (`azure-storage-blob`) |
| **Cloud Serving** | Azure Container Apps (`Dockerfile` + `FastAPI` + `uvicorn`) |
| **Data Validation** | `pandera` |
| **Experiment Tracking** | MLflow (`mlflow`) |
| **Orchestration / CI/CD** | GitHub Actions (`.github/workflows/retrain.yml`) & Apache Airflow |
| **Testing** | `pytest` + `FastAPI TestClient` |

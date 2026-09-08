# 🎬 Movie Mind Reader & Recommender Engine

[![Live Demo](https://img.shields.io/badge/Live_App-Streamlit-E50914?style=for-the-badge&logo=streamlit)](http://3.111.33.59:8501)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-BERT4Rec-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![AWS](https://img.shields.io/badge/AWS-S3_%26_EC2-FF9900?style=for-the-badge&logo=amazon-aws)](https://aws.amazon.com)

An interactive movie recommender app and MLOps system that uses deep learning to guess which movie you're secretly thinking of based on your watch history.

👉 **[Try the Live Web App Here](http://3.111.33.59:8501)**

---

## 🎮 The "Movie Mind Reader" Game

Instead of a plain list of movie recommendations, this app turns recommendation into an interactive mind-reading game powered by a 4-layer **BERT4Rec Transformer**:

1. **Pick your top movies**: Choose 3 to 5 movies you recently loved.
2. **AI generates candidates**: BERT4Rec scans 54,000+ movies and picks 4 candidates based on your sequence.
3. **Think of one**: Secretly pick one of the 4 candidates in your head without telling the app.
4. **Reveal**: Hit **Reveal Mind Read**—BERT4Rec uses sequential attention probabilities to predict which movie you chose!

---

## 🏗️ System Architecture

```text
 ┌─────────────────────────────────────────────────────────────┐
 │                    MovieLens Master Data                     │
 │          (ml-1m local dev / ml-32m on Kaggle T4 GPU)        │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                      Replay Controller
                (Simulates 30-day stream data)
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                Pandera Data Validation                      │
 └──────────────┬──────────────────────────────┬───────────────┘
                ▼                               ▼
 ┌─────────────────────────────┐ ┌─────────────────────────────┐
 │      Baseline SVD Model     │ │     BERT4Rec Transformer    │
 │   (scikit-surprise local)   │ │   (PyTorch 4-layer on     │
 └──────────────┬──────────────┘ │    Kaggle Nvidia T4 GPU)    │
                │                └──────────────┬──────────────┘
                └──────────────┬────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────┐
 │             Champion / Challenger Promotion                 │
 │            (MLflow Registry + S3 Artifacts)                 │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │              Streamlit "Mind Reader" Game                   │
 │                 (Live on AWS EC2)                           │
 └─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Highlights

- **BERT4Rec Transformer**: Implemented a 4-layer Transformer Encoder with bidirectional self-attention in PyTorch to capture dynamic context shifts in what users watch next.
- **Kaggle GPU Training Pipeline**: Trained on **32+ million MovieLens ratings** using Kaggle Nvidia T4 GPUs, with warm-start fine-tuning triggered via API.
- **Continuous Stream Replay**: Simulates real production data drift by chronologically releasing 30-day interaction chunks from an immutable master dataset.
- **Automated Retraining**: GitHub Actions workflow (`.github/workflows/retrain.yml`) runs every Sunday to pull data from AWS S3, validate schema with Pandera, retrain SVD & BERT4Rec, gate-check models via MLflow, and upload the winning champion model to S3.
- **AWS Deployment**: Hosted on an AWS EC2 instance (`http://3.111.33.59:8501`) connected to AWS S3 storage.

---

## 📁 Project Structure

```text
.
├── .github/workflows/
│   ├── ci.yml                   # Automated test suite execution
│   └── retrain.yml              # Weekly retraining pipeline
├── configs/
│   └── config.yaml              # Central settings configuration
├── dags/
│   └── movielens_pipeline.py    # Local Airflow orchestration DAG
├── notebooks/
│   └── bert4rec_kaggle_train.py # Kaggle GPU training script
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI recommendation API
│   ├── dashboard/
│   │   └── app.py              # Streamlit "Movie Mind Reader" game app
│   ├── data/
│   │   ├── sequence_builder.py  # BERT4Rec sequence mask generator
│   │   ├── temporal_split.py    # Time-aware train/val/test splitter
│   │   └── validation.py        # Pandera schema validator
│   ├── evaluation/
│   │   ├── evaluator.py         # Evaluation pipeline runner
│   │   └── metrics.py           # Precision@K, NDCG@K, Recall@K, MRR@K
│   ├── inference/
│   │   ├── bert4rec_inference.py# Model inference & prediction logic
│   │   └── recommender.py       # Unified recommender interface
│   ├── ingestion/
│   │   ├── movielens_fetcher.py # MovieLens dataset fetcher
│   │   └── snapshot_diff.py     # Rating snapshot diff calculator
│   ├── models/
│   │   ├── bert4rec.py          # PyTorch BERT4Rec Transformer model
│   │   └── svd_model.py         # Collaborative filtering baseline
│   ├── monitoring/
│   │   ├── drift_monitor.py     # Data & concept drift monitor
│   │   └── performance.py       # Runtime performance tracking
│   ├── replay/
│   │   ├── replay_controller.py # Chronological stream replay engine
│   │   └── replay_state.py      # Idempotent state manager
│   ├── storage/
│   │   └── s3_storage.py        # AWS S3 storage interface
│   ├── tracking/
│   │   ├── mlflow_tracker.py    # MLflow experiment tracking
│   │   └── model_registry.py    # Champion/Challenger promotion gate
│   └── training/
│       ├── kaggle_trigger.py    # Kaggle API GPU trigger
│       ├── train_bert4rec.py    # PyTorch training loop
│       └── train_svd.py         # SVD training script
├── tests/unit/                  # Pytest unit tests (16/16 passing)
├── requirements.txt
└── Dockerfile
```

---

## 🏃 Quickstart (Local Run)

### 1. Setup Environment
```bash
git clone https://github.com/priyanshu-projects/Movie_Recommender.git
cd Movie_Recommender
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data & Train Baseline
```bash
# Download MovieLens dataset
python -m src.ingestion.movielens_fetcher --dataset ml-1m

# Release a 30-day batch
python -m src.replay.replay_controller

# Train baseline SVD model
python -m src.models.svd_model
```

### 3. Launch App Locally
```bash
streamlit run src/dashboard/app.py
```

### 4. Run Unit Tests
```bash
pytest tests/unit/
```

---

## 📊 Evaluation Results

| Model | Architecture | Training Dataset | Precision@10 | NDCG@10 |
|---|---|---|---|---|
| **SVD Baseline** | Collaborative Filtering (Matrix Factorization) | MovieLens 1M | 14.82% | 0.0821 |
| **BERT4Rec (Champion)** | 4-Layer Transformer Encoder | MovieLens 32M | **20.57%** | **0.1084** |

---

## 📜 License
MIT

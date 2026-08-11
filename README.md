# MLOps Movie Recommendation System

> **Portfolio project** — end-to-end MLOps pipeline on MovieLens data.  
> Target: placement interviews at SaaS / e-commerce / OTT companies.

## What This Is

A production-style recommendation system using collaborative filtering (SVD matrix factorization), with a full MLOps pipeline:

- **Weekly automated ingestion** of MovieLens data, with snapshot diffing to isolate new ratings
- **Data validation** with Great Expectations before any training
- **Temporal train/test splitting** — not random, mimicking real production evaluation
- **MLflow model registry** with a champion vs challenger evaluation gate — new model only deploys if it's better
- **Evidently AI drift monitoring** — flags distribution shifts in incoming data
- **Airflow orchestration** — ties all steps into a single, schedulable DAG
- **FastAPI serving** with `/recommend/{user_id}` endpoint
- **Docker + GitHub Actions CI/CD** — builds and deploys on push to main

## Project Structure

```
.
├── dags/                        # Airflow DAG
│   └── movielens_pipeline.py
├── src/
│   ├── ingestion/               # Download MovieLens zip from GroupLens
│   ├── delta/                   # Diff new file vs last snapshot
│   ├── validation/              # Great Expectations checks
│   ├── preprocessing/           # Build user-item interaction matrix
│   ├── training/                # SVD training + MLflow logging
│   ├── evaluation/              # Temporal eval + MLflow promotion
│   ├── serving/                 # FastAPI app
│   └── monitoring/              # Evidently drift detection
├── tests/unit/                  # Pytest unit tests (run in CI)
├── data/
│   ├── raw/                     # Downloaded MovieLens files
│   ├── snapshots/               # Previous-cycle snapshot for diffing
│   ├── diffs/                   # New ratings this cycle
│   └── processed/               # Interaction matrix for training
├── models/                      # Trained model pickles
├── mlflow_tracking/             # Local MLflow DB + artifacts
├── .github/workflows/
│   ├── ci.yml                   # Lint + test on every push
│   └── cd.yml                   # Build + deploy to Cloud Run on main
├── Dockerfile
├── docker-compose.yml           # Local dev: API + MLflow + Airflow
├── dvc.yaml                     # DVC pipeline stages
└── requirements.txt
```

## Quick Start (Local)

```bash
# 1. Clone and install
git clone <your-repo-url>
cd movielens-recsys
pip install -r requirements.txt

# 2. Download data
python -m src.ingestion.movielens_fetcher

# 3. Run the pipeline manually (without Airflow)
python -m src.delta.snapshot_diff \
    --new data/raw/ml-latest-small/ratings.csv \
    --snapshot data/snapshots/ratings_snapshot.csv \
    --out data/diffs/ratings_diff.csv

python -m src.preprocessing.build_matrix \
    --ratings data/raw/ml-latest-small/ratings.csv \
    --movies  data/raw/ml-latest-small/movies.csv \
    --out     data/processed/interactions.csv

python -m src.training.train_svd \
    --interactions data/processed/interactions.csv \
    --model-out    models/svd_model.pkl

# 4. Serve
uvicorn src.serving.app:app --reload --port 8000
# → curl http://localhost:8000/recommend/1?n=5

# 5. Or run everything with Docker Compose
docker compose up
# API:    http://localhost:8000
# MLflow: http://localhost:5000
# Airflow: http://localhost:8080  (admin/admin)
```

## Pipeline (Airflow DAG)

```
download_data → diff_snapshot → validate_data → preprocess → train_model → evaluate_and_promote
```

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | Apache Airflow |
| Data validation | Great Expectations |
| Data versioning | DVC |
| Experiment tracking + registry | MLflow |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Serving | FastAPI |
| Drift monitoring | Evidently AI |
| Model | SVD via `scikit-surprise` (CPU-only) |
| Data | MovieLens `ml-latest-small` → `ml-latest` |

## Running Tests

```bash
pytest tests/unit/ -v
```

## Deployment

Set these GitHub secrets before pushing to main:

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_SA_KEY` | JSON key for a service account with Cloud Run + Artifact Registry permissions |

⚠️ Set a GCP billing alert at $1 immediately — everything fits in free tier but it's easy to overshoot.

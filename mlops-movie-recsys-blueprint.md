# Project Blueprint: MLOps Movie Recommendation System

## Context / Goal
College placement portfolio project targeting **product/tech companies (SaaS, e-commerce)**. Goal: build a **simple but complete MLOps pipeline** that's easy to explain end-to-end in interviews — covering continuous data ingestion, CI/CD, orchestration, versioning, and monitoring. Already have: RAG-based ESG claim auditor, AI credit risk system. This project should NOT be finance-related, and should demonstrate MLOps skills (not deep ML research).

## Core Idea
A **movie recommendation system** using collaborative filtering, with a pipeline that simulates **continuous weekly/monthly data ingestion** from a real (not synthetic) dataset, triggering automatic retraining, evaluation, and deployment.

## Why This Project
- Recommendation systems are one of the most common real-world ML systems at e-commerce/SaaS/OTT companies (Amazon, Netflix, Flipkart, etc.)
- Natural "why retrain regularly" story: user behavior and catalog change constantly
- No GPU required — classical collaborative filtering trains fast on CPU
- Focus is on the **pipeline/MLOps mechanics**, not the model sophistication — model choice is secondary

## Data Source
- **MovieLens** dataset from **GroupLens** (research lab, University of Minnesota) — https://grouplens.org/datasets/movielens/
- Not an API — it's a **direct zip file download** (stable URL), containing `ratings.csv`, `movies.csv`, `tags.csv`
- No permission needed for personal/college project use (permission form is only for redistribution/publishing research)
- Use **`ml-latest`** dataset — GroupLens states this updates over time (continuously), unlike the versioned/stable datasets (25M, 32M etc.) which are frozen snapshots
- **Important**: GroupLens gives you the *whole current file* each pull, not just new rows — you must diff against your last saved snapshot to isolate "new" data
- Alternative for getting started fast: `ml-latest-small` (~100K ratings) for quick dev/testing before scaling to the full dataset

## How the Model Works (for interview explanation)
- Data forms a sparse **user-item matrix** (rows = users, columns = movies, cells = ratings), mostly empty
- **Collaborative filtering**: learns patterns across ALL users/items simultaneously — not just one user's history in isolation
- **Matrix Factorization** (classical approach, e.g., SVD/ALS): decomposes the matrix into two smaller matrices — User factors (U) and Item factors (V) — each row is a vector of "latent factors" (hidden taste dimensions the model learns, not human-labeled)
- Predicted rating = dot product of a user's vector and an item's vector
- Training = adjust U and V via gradient descent so predictions match known ratings; once trained, use U/V to fill in unknown cells → these become recommendations
- No GPU needed — runs on CPU in seconds-to-minutes even at millions of ratings, using libraries like `Surprise` or `implicit`
- (Optional stretch goal) Neural/two-tower model — same idea but with small neural nets instead of a simple dot product; more flexible, closer to what companies like YouTube/Amazon use today. Not required for this project.

## Handling New Users/Items (Cold Start)
- New data each cycle = mix of: existing users adding new ratings, brand-new users (no history), brand-new movies (no ratings yet)
- Model doesn't need the *same* individuals repeatedly — just needs the matrix to keep growing over time
- Cold start (new user/item with no data) is a known, real limitation — worth mentioning as an "aware of but out of scope" point in interviews, not something to fully solve

## Evaluation / Accuracy
- Use **temporal train/test split** (not random) — train on all data up to time T, test on the next batch of real ratings that arrive after T. This mimics real production evaluation ("did we correctly predict what people would rate next").
- **Metrics**:
  - **RMSE** (Root Mean Squared Error) — how far off predicted ratings are from actual ratings, in star-rating units. Classic recsys metric.
  - **Precision@K / Recall@K** — of the top-K recommended items, how many did the user actually rate highly. More realistic for "would they like this recommendation."
- No need to manually remove/store reviews separately — train/test split is done programmatically each pipeline run on the growing dataset.

## Full Pipeline Architecture

**1. Data Ingestion (Airflow task)**
- Script downloads the GroupLens `ml-latest` zip via HTTP GET, unzips into `/data/raw/`
- Runs on schedule (weekly/monthly) via Airflow

**2. Diff / Delta Detection**
- Compare newly downloaded `ratings.csv` against last saved snapshot
- Extract only new (user_id, movie_id, timestamp) rows → this is "this cycle's incoming batch"
- This diffing logic is a legitimate, explainable piece of engineering (real companies do similar snapshot-diffing when no streaming API exists)

**3. Data Validation**
- Use **Great Expectations** — check for nulls, valid rating ranges (0.5–5.0), schema consistency, no duplicate (user, movie) pairs

**4. Preprocessing**
- Build/update the user-item interaction matrix
- Handle new user/item IDs (cold start entries)

**5. Model Training**
- Matrix Factorization (SVD/ALS via `Surprise` or `implicit`), CPU-only, fast
- Train on all data up to current cycle

**6. Evaluation**
- Temporal train/test split, compute RMSE and Precision@K/Recall@K
- Compare new model's metrics against current "champion" model in registry

**7. Model Registry / Versioning**
- **MLflow** — experiment tracking (log metrics, params per run) + model registry (promote to "Production" if better than champion)
- **DVC** — version the dataset snapshots and diffs alongside Git-tracked code

**8. CI (Continuous Integration)**
- **GitHub Actions** — on every push: lint code, run unit tests (e.g., test the diffing logic, data validation rules, preprocessing functions)

**9. CD (Continuous Deployment)**
- On merge to main / on new model promotion: build Docker image, push to a container registry, redeploy

**10. Serving**
- **FastAPI** app wrapping the trained model — endpoint takes a user_id, returns top-N recommended movies

**11. Monitoring**
- **Evidently AI** — track data drift (are new ratings statistically different from training distribution?) and prediction drift
- Feeds back into "should we trigger retraining" decision logic

## Orchestration Flow (Airflow DAG)

```
download_data
   → diff_against_previous_snapshot
      → validate_data (Great Expectations)
         → preprocess (build/update interaction matrix)
            → train_model (matrix factorization)
               → evaluate (temporal split, RMSE, Precision@K)
                  → register_if_better (MLflow registry)
                     → trigger_deploy (CI/CD via GitHub Actions)
```

## Tech Stack Summary

| Purpose | Tool |
|---|---|
| Orchestration | Apache Airflow |
| Data validation | Great Expectations |
| Data versioning | DVC |
| Experiment tracking / model registry | MLflow |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Serving | FastAPI |
| Drift monitoring | Evidently AI |
| Model | Matrix Factorization (SVD/ALS) via `Surprise` or `implicit` (CPU-only) |
| Data source | MovieLens `ml-latest` (GroupLens) |

## Key Interview Talking Points
- "User-level interaction data isn't publicly available anywhere in real-time (companies keep that private), so I built an automated ingestion step that pulls GroupLens' continuously-updating MovieLens dataset and diffs it against the previous snapshot to isolate genuinely new records — similar to how teams handle sources without a push/streaming API."
- "I used temporal train/test splitting instead of random splitting to mimic real production evaluation — training on data up to a point in time and testing on what actually happened next."
- "The pipeline auto-retrains, evaluates the new model against the current production model, and only promotes it if it's better — preventing regressions."
- "I'm aware of the cold-start problem for new users/items and it's a known limitation of collaborative filtering — a natural next step would be a hybrid model incorporating content-based features."
- Runs fully on CPU (no GPU dependency) — relevant since matrix factorization is lightweight by design.

## Linux / OS Note
- Recommended to build this on a native Linux setup (dual-boot preferred over full wipe) or WSL2 as a fallback
- Benefit: Airflow, Docker are Linux-native; avoids path/permission friction; builds familiarity with SSH, systemd, bash — common in DevOps/MLOps-adjacent interview questions
- Budget 1-2 weeks of setup friction before any placement crunch, not during one

## Open Decisions / Next Steps
- [ ] Confirm dataset size to start with (`ml-latest-small` for fast iteration vs `ml-25m`/`ml-latest` full for realism)
- [ ] Set up Airflow locally (Docker Compose is simplest) and build the DAG skeleton
- [ ] Write the diffing script for snapshot comparison
- [ ] Decide on final metrics/thresholds for "promote new model" logic
- [ ] (Optional) Explore two-tower neural model as a stretch goal after the classical pipeline works end-to-end

# Project Blueprint: MLOps Movie Recommendation System

## 1. Context / Goal

College placement portfolio project targeting product/tech companies
(SaaS, e-commerce, OTT, consumer technology).

Goal: build a complete, explainable MLOps recommendation system covering:

- periodic data ingestion
- data validation
- temporal preprocessing
- model training
- automatic retraining
- experiment tracking
- model versioning
- orchestration
- CI/CD
- model serving
- monitoring
- champion/challenger model promotion

Already have:
- RAG-based ESG claim auditor
- AI credit risk system

This project should NOT be finance-related.

The primary focus is MLOps and production ML engineering, not deep
recommender-system research.

---

# 2. Core Idea

Build a movie recommendation platform using real MovieLens data.

MovieLens provides historical real-world user/movie interactions, but it
should NOT be treated as a guaranteed real-time streaming source.

Instead:

1. Download a complete MovieLens dataset.
2. Upload the complete dataset once to remote storage/server.
3. Keep the master dataset immutable.
4. Build a replay service that periodically releases the next chronological
   portion of the dataset.
5. Treat each released portion as a newly arrived production data batch.
6. Airflow automatically ingests the new batch.
7. The system validates, preprocesses, retrains, evaluates, registers and
   potentially deploys a new model.

This creates a controlled but realistic continuous-data-ingestion
environment using real historical interactions rather than synthetic
ratings.

IMPORTANT:

The periodic release schedule is a controlled replay/simulation.
MovieLens itself is NOT being claimed to provide fortnightly or real-time
updates.

---

# 3. Why This Project

Recommendation systems are common in:

- Netflix
- Amazon
- YouTube
- Spotify
- Flipkart
- e-commerce
- SaaS personalization

They naturally require continuous ML lifecycle management because:

- user behavior changes
- new interactions arrive
- new users appear
- new movies appear
- recommendation quality can change
- models need periodic retraining
- new models must be evaluated before deployment

The project therefore demonstrates real MLOps concepts without requiring
GPU-heavy infrastructure.

---

# 4. Data Source

Primary source:

MovieLens from GroupLens / University of Minnesota.

Development dataset:

    ml-latest-small
    ~100K ratings

First realistic-scale dataset:

    ml-1m
    ~1M ratings

Optional larger experiments:

    ml-10m
    ml-25m
    ml-32m
    ml-latest

Development machine:

- Linux
- CPU only
- approximately 8 GB RAM

Recommended progression:

    ml-latest-small
          ↓
    development/debugging
          ↓
    ml-1m
          ↓
    realistic MLOps experiment

Do NOT jump to 33M+ ratings on the 8 GB machine unless there is a specific
reason.

---

# 5. MovieLens Data

ratings.csv:

    userId
    movieId
    rating
    timestamp

movies.csv:

    movieId
    title
    genres

The timestamp is critical for:

- chronological replay
- temporal train/test splitting
- sequential recommendation
- realistic data-arrival simulation

---

# 6. Remote Data / Replay Architecture

The complete MovieLens dataset is uploaded once to remote storage.

Possible storage for development:

- S3-compatible object storage
- Cloud storage
- Supabase Storage
- another simple object-storage service

The exact provider should remain configurable.

Architecture:

    COMPLETE MOVIELENS DATASET
              ↓
        Remote Storage
              ↓
       Replay Controller
              ↓
    Every configured period
              ↓
       Next chronological batch
              ↓
        Incoming Data API
              ↓
          Airflow
              ↓
       MLOps pipeline

The master dataset must never be modified.

---

# 7. Replay Controller

Create:

    src/replay/replay_controller.py

Purpose:

Expose the next chronological section of the master dataset.

Maintain replay state:

    last_released_timestamp
    last_released_batch
    total_batches_released

Example:

    Master dataset
    2015 ───────────────────────── 2018

    Batch 01:
    2015-01 → 2015-03

    Batch 02:
    2015-03 → 2015-05

    Batch 03:
    2015-05 → 2015-07

    ...

The actual batch boundaries should be configurable.

The demonstration schedule can be:

    every 2 weeks

But the data itself should remain chronological.

IMPORTANT:

The replay service should not modify the ratings.

It only controls when historical data becomes available to the downstream
pipeline.

---

# 8. Replay State

Store something like:

    replay_state.json

Example:

    {
        "last_released_batch": 7,
        "last_released_timestamp": "...",
        "dataset_version": "ml-1m-v1"
    }

This prevents:

- sending the same batch twice
- skipping batches
- losing replay position

The replay controller should be idempotent.

---

# 9. Data Ingestion

Tool:

    Apache Airflow

Airflow runs on the configured schedule.

Example:

    Every 2 weeks

Airflow calls the replay endpoint:

    GET /next-batch

The response contains the next chronological batch.

Store the received batch locally:

    data/raw/incoming/
        batch_001.csv
        batch_002.csv
        batch_003.csv

Each batch should have metadata:

- batch ID
- source dataset version
- earliest timestamp
- latest timestamp
- number of interactions
- ingestion timestamp

---

# 10. Snapshot / Delta Detection

Create:

    src/ingestion/snapshot_diff.py

The system should verify that the incoming batch contains genuinely new
records relative to the already-ingested data.

Interaction identity:

    (userId, movieId, timestamp)

Do NOT simply use:

    (userId, movieId)

because timestamp makes the interaction identity more robust.

Output:

    data/processed/deltas/
        batch_001.csv
        batch_002.csv

Track:

- previous batch
- current batch
- number of records
- number of new records
- duplicate count
- earliest timestamp
- latest timestamp

---

# 11. Data Validation

Use:

    Great Expectations

Validate:

- required columns exist
- userId is non-null
- movieId is non-null
- rating is non-null
- timestamp is non-null
- rating range is 0.5–5.0
- valid IDs
- valid timestamps
- schema consistency
- duplicate interactions

Critical validation failure:

    STOP PIPELINE

Do not train on invalid data.

---

# 12. Data Versioning

Use:

    DVC

Version:

- original/master dataset reference
- replay batches
- important processed datasets

Git tracks:

- source code
- configs
- Airflow DAGs
- tests

DVC tracks:

- data versions
- important data artifacts

Do not unnecessarily version every generated file.

---

# 13. Data Accumulation

Each new batch is added to the historical training pool.

Example:

    Initial:
    1,000,000 ratings

    Batch 01:
    +20,000

    Training dataset:
    1,020,000

    Batch 02:
    +18,000

    Training dataset:
    1,038,000

The model is retrained using all available historical interactions up to
the current replay point.

This is intentionally a batch-retraining architecture.

IMPORTANT:

Continuous data ingestion does NOT mean continuous model training.

Data can arrive continuously/periodically while retraining happens on a
scheduled cadence.

---

# 14. Model Strategy

Use two models.

## Model 1: SVD

Classical collaborative-filtering baseline.

## Model 2: BERT4Rec

Advanced sequential recommendation model.

Architecture:

    Historical interactions
            ↓
       SVD baseline

and:

    Chronological sequences
            ↓
        BERT4Rec

Both should use a common model interface so the MLOps pipeline can treat
them as interchangeable candidate models.

---

# 15. SVD Model

Use:

    Surprise SVD

Initial configuration:

    n_factors = 50
    n_epochs = 20
    lr_all = 0.005
    reg_all = 0.02

Concept:

    predicted rating
        =
    global average
    + user bias
    + movie bias
    + user latent factors × movie latent factors

Purpose:

- simple baseline
- fast CPU training
- easy interview explanation
- benchmark for BERT4Rec

---

# 16. BERT4Rec Model

Use:

    PyTorch

BERT4Rec learns from chronological user interaction sequences.

Example:

    User 42:

    Iron Man
        ↓
    Avengers
        ↓
    Thor
        ↓
    Black Panther
        ↓
    Guardians

During training:

    Iron Man
        ↓
    Avengers
        ↓
    [MASK]
        ↓
    Black Panther
        ↓
    Guardians

Target:

    Thor

This is masked-item prediction.

The model learns relationships between movies from actual user behavior.

---

# 17. Preparing MovieLens for BERT4Rec

Input:

    userId
    movieId
    rating
    timestamp

Initial approach:

    rating >= 3.5
        ↓
    positive interaction

Then:

1. Sort by userId.
2. Sort each user's interactions by timestamp.
3. Group by user.
4. Build chronological sequences.
5. Limit sequence length.
6. Create masked training examples.

Example:

    A → B → C → D → E

becomes training examples such as:

    A → B → MASK → D → E
                  ↓
                  C

and:

    A → MASK → C → D → E
        ↓
        B

---

# 18. BERT4Rec Architecture

    Movie ID
        ↓
    Movie Embedding
        ↓
    Position Embedding
        ↓
    Transformer Encoder
        ↓
    Prediction Head
        ↓
    Scores for movie catalogue
        ↓
    Top-K recommendations

Initial configuration:

    embedding_dim = 64
    hidden_dim = 64
    num_layers = 2
    num_attention_heads = 2
    feed_forward_dim = 256
    max_sequence_length = 50
    dropout = 0.2
    attention_dropout = 0.2
    mask_probability = 0.20

CPU-friendly starting configuration.

---

# 19. BERT4Rec Training

Optimizer:

    AdamW

Initial:

    learning_rate = 1e-4
    weight_decay = 0.01
    batch_size = 32
    max_epochs = 30
    early_stopping_patience = 5

Loss:

    CrossEntropyLoss

Calculate loss only for masked movie positions.

Do not calculate loss for padding tokens.

Use CPU automatically.

The model should save the best validation checkpoint rather than simply
the final epoch.

---

# 20. Hyperparameter Experimentation

Do not run a huge hyperparameter search.

Use MLflow to compare controlled experiments.

Embedding:

    32
    64
    128

Layers:

    1
    2

Attention heads:

    2
    4

Sequence length:

    30
    50
    100

Learning rate:

    1e-4
    5e-4

Dropout:

    0.1
    0.2

Initial baseline:

    embedding = 64
    layers = 2
    heads = 2
    sequence = 50
    learning_rate = 1e-4
    dropout = 0.2

---

# 21. Temporal Train / Validation / Test Split

Never use random splitting for the sequential model.

Example:

    User:

    A → B → C → D → E

Training:

    A → B → C

Validation:

    D

Test:

    E

More generally:

    Earlier interactions → TRAIN
    Later interactions   → VALIDATION
    Latest interactions  → TEST

No future interaction may appear in training.

This prevents temporal leakage.

---

# 22. Evaluation

## SVD

Metrics:

- RMSE
- MAE
- Precision@10
- Recall@10

RMSE answers:

    How accurately did the model predict ratings?

## BERT4Rec

Metrics:

- Hit Rate@10
- Recall@10
- NDCG@10
- MRR@10

Primary metric:

    NDCG@10

Secondary:

    Recall@10

Do not use RMSE as the primary BERT4Rec metric.

---

# 23. Champion / Challenger

Every retraining cycle produces a candidate model.

Example:

    Current production model:
        BERT4Rec v3

    New candidate:
        BERT4Rec v4

Evaluate candidate.

If:

    NDCG@10 improves sufficiently

then:

    candidate → champion

Otherwise:

    candidate → rejected

This prevents model regressions.

The promotion threshold must be configurable.

---

# 24. MLflow

Use MLflow for:

## Experiment tracking

Track:

- model type
- dataset version
- replay batch
- embedding size
- transformer layers
- attention heads
- sequence length
- learning rate
- dropout
- batch size
- training duration
- evaluation metrics

## Model registry

Store:

    SVD v1
    SVD v2
    BERT4Rec v1
    BERT4Rec v2
    BERT4Rec v3

Store:

- model artifact
- model version
- configuration
- dataset version
- training timestamp

---

# 25. Airflow DAG

Main DAG:

    get_next_replay_batch
            ↓
    detect_delta
            ↓
    validate_data
            ↓
    update_training_dataset
            ↓
    build_sequences
            ↓
    train_candidate
            ↓
    evaluate_candidate
            ↓
    register_model
            ↓
    compare_with_champion
            ↓
    promote_if_better
            ↓
    trigger_deployment
            ↓
    update_monitoring

Airflow schedule:

    Every 2 weeks

The replay controller and Airflow schedule should be independently
configurable.

---

# 26. Retraining Strategy

Retraining is intentionally periodic.

Example:

    Week 0:
    Initial model

    Week 2:
    New batch
        ↓
    Retrain

    Week 4:
    New batch
        ↓
    Retrain

    Week 6:
    New batch
        ↓
    Retrain

This demonstrates continuous MLOps behavior without requiring a live
streaming platform.

Retraining may also be triggered by:

    scheduled interval
    OR
    sufficient new data
    OR
    significant data drift
    OR
    recommendation performance degradation

---

# 27. Monitoring

Use:

    Evidently AI

Monitor data:

- rating distribution
- number of new interactions
- active users
- new users
- new movies
- sequence lengths
- interaction volume

Monitor recommendation/model behavior:

- NDCG@10
- Recall@10
- Hit Rate@10
- recommendation coverage
- ranking behavior

Monitor drift:

    incoming batch
          vs
    training/reference distribution

Drift is a signal.

Do NOT automatically assume drift means the model must be retrained.

Use configurable thresholds.

---

# 28. FastAPI Serving

Create:

    src/api/main.py

Endpoints:

    GET /health

    GET /recommend/{user_id}?k=10

    GET /model

Example:

    /recommend/42?k=10

Returns:

    movieId
    title
    score

The recommendation service loads the current champion model.

---

# 29. Inference

For BERT4Rec:

    user_id
        ↓
    retrieve recent interaction sequence
        ↓
    create sequence
        ↓
    BERT4Rec
        ↓
    score candidate movies
        ↓
    remove already-interacted movies
        ↓
    rank
        ↓
    Top-K

For SVD:

    user_id
        ↓
    score candidate movies
        ↓
    remove already-rated movies
        ↓
    rank
        ↓
    Top-K

---

# 30. Cold Start

Known limitation.

## New user

No history.

Fallback:

    popular movies
    global recommendations

## New movie

No interactions.

Future improvement:

    content-based features
    genres
    metadata
    hybrid recommendation

Do not over-engineer cold start in V1.

---

# 31. CI

Use:

    GitHub Actions

On push / pull request:

- install dependencies
- lint
- run unit tests
- test replay controller
- test snapshot/delta detection
- test validation
- test sequence construction
- test model input/output shapes
- test inference
- test FastAPI health endpoint

Do NOT train the full BERT4Rec model in CI.

Use a tiny dataset for tests.

---

# 32. CD

Use:

    Docker
    GitHub Actions

Flow:

    merge to main
          ↓
    tests
          ↓
    Docker build
          ↓
    push image
          ↓
    deploy FastAPI

Model promotion should remain logically separate from normal application
code deployment.

---

# 33. Docker

Containerize:

- FastAPI recommendation service
- Airflow development environment if useful

Use Docker Compose locally for:

- Airflow
- MLflow
- API
- supporting services

Do not introduce Kubernetes.

---

# 34. Project Structure

movie-recommender/

    src/
        ingestion/
            snapshot_diff.py
            batch_ingestor.py

        replay/
            replay_controller.py
            replay_state.py

        data/
            validation.py
            sequence_builder.py
            temporal_split.py
            dataset.py

        models/
            base_model.py
            svd_model.py
            bert4rec.py

        training/
            train_svd.py
            train_bert4rec.py

        evaluation/
            metrics.py
            evaluate_svd.py
            evaluate_bert4rec.py

        inference/
            recommender.py
            candidate_generator.py

        tracking/
            mlflow_tracker.py
            model_registry.py

        monitoring/
            drift.py
            performance.py

        api/
            main.py

    dags/
        movie_retraining_dag.py

    configs/
        config.yaml

    data/
        raw/
        processed/
        replay/

    models/

    tests/

    Dockerfile
    docker-compose.yml
    requirements.txt
    .gitignore
    README.md

---

# 35. Development Order

Do NOT build everything simultaneously.

PHASE 1:
    MovieLens ingestion

PHASE 2:
    Upload master dataset to remote storage

PHASE 3:
    Replay controller

PHASE 4:
    Batch ingestion + state management

PHASE 5:
    Data validation

PHASE 6:
    Temporal preprocessing / sequence generation

PHASE 7:
    SVD baseline

PHASE 8:
    BERT4Rec

PHASE 9:
    Evaluation

PHASE 10:
    MLflow tracking

PHASE 11:
    Champion/challenger logic

PHASE 12:
    Airflow orchestration

PHASE 13:
    FastAPI serving

PHASE 14:
    Docker

PHASE 15:
    GitHub Actions CI/CD

PHASE 16:
    Evidently monitoring

PHASE 17:
    Automated retraining triggers

After every phase:

    run tests
    verify outputs
    fix issues
    only then proceed

---

# 36. Important Engineering Rules

1. MovieLens must NOT be described as a real-time streaming source.

2. The replay system must be explicitly described as controlled temporal
   replay of real historical data.

3. Never fabricate ratings.

4. Keep the master dataset immutable.

5. Never randomly shuffle chronological user interactions.

6. Prevent future-data leakage.

7. Never create a dense user × movie matrix for BERT4Rec.

8. Keep SVD as the classical baseline.

9. Keep BERT4Rec as the advanced sequential model.

10. Keep the training pipeline model-agnostic.

11. Keep hyperparameters configurable.

12. Never hardcode storage URLs or credentials.

13. Use environment variables for secrets.

14. Do not train full BERT4Rec inside CI.

15. Do not require a GPU.

16. Do not add Kafka, Spark, Kubernetes, or other heavy infrastructure unless
    a real requirement emerges.

17. Prioritize a working end-to-end MLOps pipeline over model complexity.

18. The replay mechanism must be deterministic and reproducible.

19. Every training run must be traceable to a dataset/replay-batch version.

20. Every deployed model must have a corresponding MLflow version.

---

# 37. Final Architecture

                         REAL MOVIELENS DATA
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Immutable Master Data  │
                     │    Remote Storage      │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │    Replay Controller    │
                     │  Chronological batches  │
                     └────────────┬───────────┘
                                  │
                        Every 2 weeks
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │    Incoming Batch       │
                     └────────────┬───────────┘
                                  │
                                  ▼
                           Apache Airflow
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             Delta Detection              Validation
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                         Data / Sequence Prep
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
                  SVD                       BERT4Rec
               Baseline                  Sequential Model
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                              Evaluation
                                  │
                                  ▼
                              MLflow
                                  │
                                  ▼
                       Champion / Challenger
                                  │
                         ┌────────┴────────┐
                         │                 │
                      Better            Worse
                         │                 │
                         ▼                 ▼
                     Promote           Reject
                         │
                         ▼
                       Docker
                         │
                         ▼
                      FastAPI
                         │
                         ▼
                  Recommendations
                         │
                         ▼
                     Monitoring
                         │
                         ▼
                  Retraining Trigger
                         │
                         └───────────────→ NEXT BATCH

---

# 38. Final Interview Explanation

"I built an MLOps movie recommendation platform using real MovieLens
interaction data.

Because MovieLens is a historical dataset rather than a guaranteed
real-time stream, I separated the immutable data source from the data
delivery layer. I uploaded the complete dataset to remote storage and
built a chronological replay service that releases the next batch every
two weeks.

Airflow treats each released batch as newly arrived production data. The
pipeline validates the batch, detects new interactions, updates the
training data, constructs temporal user sequences, retrains the
recommendation models, evaluates the candidate against the current
champion, tracks the experiment and model in MLflow, and deploys the new
model only when it meets the promotion criteria.

I implemented SVD as the classical collaborative-filtering baseline and
BERT4Rec as a sequential Transformer model that learns from the order of
user interactions.

The system is containerized with Docker, served through FastAPI,
orchestrated with Airflow, monitored with Evidently, versioned with DVC,
and automated through GitHub Actions."

---

# 39. Final Project Positioning

This project is primarily:

    MLOps + Recommendation Systems

NOT:

    Deep Learning Research

The strongest demonstration is:

    Real historical data
          ↓
    Controlled data arrival
          ↓
    Automated pipeline
          ↓
    Retraining
          ↓
    Evaluation
          ↓
    Model registry
          ↓
    Champion promotion
          ↓
    Deployment
          ↓
    Monitoring
          ↓
    Next cycle

The model is important, but the MLOps lifecycle is the main project.

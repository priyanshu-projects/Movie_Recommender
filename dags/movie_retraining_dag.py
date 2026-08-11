"""
dags/movie_retraining_dag.py

Airflow DAG for the MLOps Movie Recommendation pipeline.

Schedule: Every 2 weeks (configurable in configs/config.yaml).

Tasks:
    1.  get_next_replay_batch      — Replay controller releases next chronological chunk
    2.  detect_delta               — Verify batch has genuinely new (userId, movieId, timestamp) triples
    3.  validate_data              — pandera schema + value checks
    4.  update_training_dataset    — Append delta to cumulative training pool
    5.  build_sequences            — Construct BERT4Rec masked sequences
    6.  train_svd                  — Retrain SVD on full accumulated data
    7.  train_bert4rec             — Retrain BERT4Rec on updated sequences
    8.  evaluate_svd               — RMSE, MAE, Precision@K, Recall@K
    9.  evaluate_bert4rec          — NDCG@K, Recall@K, Hit Rate@K
    10. register_and_promote       — Champion/challenger gate via MLflow + model_registry
    11. trigger_deployment         — Restart API server with new champion (local: write flag file)
    12. run_drift_report           — Evidently data drift report on incoming batch vs reference
    13. update_performance_log     — Log cycle metrics to performance_log.jsonl
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/config.yaml"

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="movie_recommender_retraining",
    default_args=default_args,
    description="Periodic retraining of SVD + BERT4Rec movie recommender",
    schedule_interval=timedelta(weeks=2),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "recommendation", "retraining"],
) as dag:

    # ── Task 1: Get next replay batch ─────────────────────────────────────────

    def task_get_next_batch(**ctx):
        from src.replay.replay_controller import get_next_batch
        path = get_next_batch(CONFIG_PATH)
        if path is None:
            raise ValueError("No more replay data available.")
        ctx["ti"].xcom_push(key="batch_path", value=str(path))
        logger.info("Batch released: %s", path)

    get_next_batch_task = PythonOperator(
        task_id="get_next_replay_batch",
        python_callable=task_get_next_batch,
    )

    # ── Task 2: Delta detection ───────────────────────────────────────────────

    def task_detect_delta(**ctx):
        import pandas as pd
        batch_path = Path(ctx["ti"].xcom_pull(key="batch_path", task_ids="get_next_replay_batch"))
        history_path = Path("data/processed/all_ratings.csv")

        batch = pd.read_csv(batch_path)
        if history_path.exists():
            history = pd.read_csv(history_path)
            key = ["userId", "movieId", "timestamp"]
            merged = batch.merge(history[key], on=key, how="left", indicator=True)
            new_rows = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        else:
            new_rows = batch

        if new_rows.empty:
            raise ValueError("No new interactions detected in this batch.")

        delta_path = Path("data/processed/deltas") / batch_path.name
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        new_rows.to_csv(delta_path, index=False)
        ctx["ti"].xcom_push(key="delta_path", value=str(delta_path))
        logger.info("Delta: %d new rows", len(new_rows))

    detect_delta_task = PythonOperator(
        task_id="detect_delta",
        python_callable=task_detect_delta,
    )

    # ── Task 3: Validate data ─────────────────────────────────────────────────

    def task_validate(**ctx):
        from src.data.validation import validate_batch
        delta_path = Path(ctx["ti"].xcom_pull(key="delta_path", task_ids="detect_delta"))
        validate_batch(delta_path, CONFIG_PATH)

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate,
    )

    # ── Task 4: Update training dataset ──────────────────────────────────────

    def task_update_dataset(**ctx):
        import pandas as pd
        delta_path = Path(ctx["ti"].xcom_pull(key="delta_path", task_ids="detect_delta"))
        all_path = Path("data/processed/all_ratings.csv")
        delta = pd.read_csv(delta_path)
        if all_path.exists():
            history = pd.read_csv(all_path)
            combined = pd.concat([history, delta], ignore_index=True)
        else:
            combined = delta
        combined.to_csv(all_path, index=False)
        logger.info("Training pool updated: %d total ratings", len(combined))

    update_dataset_task = PythonOperator(
        task_id="update_training_dataset",
        python_callable=task_update_dataset,
    )

    # ── Task 5: Build sequences ───────────────────────────────────────────────

    def task_build_sequences(**ctx):
        import pandas as pd
        import yaml
        from src.data.sequence_builder import build_sequences, save_sequences
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        seq_cfg = cfg["sequences"]
        ratings = pd.read_csv("data/processed/all_ratings.csv")
        sequences, vocab = build_sequences(
            ratings,
            min_rating=seq_cfg["min_rating_threshold"],
            max_seq_len=seq_cfg["max_sequence_length"],
            min_seq_len=seq_cfg["min_sequence_length"],
            mask_prob=seq_cfg["mask_probability"],
        )
        out_path = Path("data/processed/sequences.jsonl")
        save_sequences(sequences, out_path)
        ctx["ti"].xcom_push(key="vocab_size", value=len(vocab) + 1)
        logger.info("Sequences built: %d | vocab: %d", len(sequences), len(vocab))

    build_sequences_task = PythonOperator(
        task_id="build_sequences",
        python_callable=task_build_sequences,
    )

    # ── Task 6: Train SVD ─────────────────────────────────────────────────────

    def task_train_svd(**ctx):
        import pandas as pd
        import yaml
        from src.models.svd_model import SVDModel
        from src.data.temporal_split import global_temporal_split
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        ratings = pd.read_csv("data/processed/all_ratings.csv")
        train, val, _ = global_temporal_split(ratings)
        model = SVDModel(cfg["svd"])
        model.train(train)
        model.save(Path("models/svd_candidate.pkl"))

    train_svd_task = PythonOperator(
        task_id="train_svd",
        python_callable=task_train_svd,
    )

    # ── Task 7: Train BERT4Rec ────────────────────────────────────────────────

    def task_train_bert4rec(**ctx):
        from src.training.train_bert4rec import train_bert4rec
        train_bert4rec(
            Path("data/processed/sequences.jsonl"),
            Path("models/bert4rec_candidate.pkl"),
            CONFIG_PATH,
        )

    train_bert4rec_task = PythonOperator(
        task_id="train_bert4rec",
        python_callable=task_train_bert4rec,
    )

    # ── Task 8: Evaluate SVD ──────────────────────────────────────────────────

    def task_evaluate_svd(**ctx):
        import pandas as pd
        from src.models.svd_model import SVDModel
        from src.data.temporal_split import global_temporal_split
        import yaml
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        ratings = pd.read_csv("data/processed/all_ratings.csv")
        _, _, test = global_temporal_split(ratings)
        model = SVDModel(cfg["svd"])
        model.load(Path("models/svd_candidate.pkl"))
        metrics = model.evaluate(test, k=cfg["evaluation"]["k"])
        ctx["ti"].xcom_push(key="svd_metrics", value=metrics)

    evaluate_svd_task = PythonOperator(
        task_id="evaluate_svd",
        python_callable=task_evaluate_svd,
    )

    # ── Task 9: Evaluate BERT4Rec ─────────────────────────────────────────────

    def task_evaluate_bert4rec(**ctx):
        import json, yaml
        from src.models.bert4rec import BERT4Rec
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        model = BERT4Rec(cfg["bert4rec"])
        model.load(Path("models/bert4rec_candidate.pkl"))
        sequences = []
        with open("data/processed/sequences.jsonl") as f:
            for line in f:
                sequences.append(json.loads(line))
        test_seqs = sequences[-max(1, len(sequences) // 10):]
        metrics = model.evaluate(test_seqs, k=cfg["evaluation"]["k"])
        ctx["ti"].xcom_push(key="bert4rec_metrics", value=metrics)

    evaluate_bert4rec_task = PythonOperator(
        task_id="evaluate_bert4rec",
        python_callable=task_evaluate_bert4rec,
    )

    # ── Task 10: Register and promote ────────────────────────────────────────

    def task_register_promote(**ctx):
        from src.tracking.model_registry import evaluate_and_promote
        svd_metrics     = ctx["ti"].xcom_pull(key="svd_metrics",     task_ids="evaluate_svd")
        bert4rec_metrics = ctx["ti"].xcom_pull(key="bert4rec_metrics", task_ids="evaluate_bert4rec")
        # Try BERT4Rec first (primary model), fall back to SVD
        if bert4rec_metrics:
            promoted = evaluate_and_promote(Path("models/bert4rec_candidate.pkl"),
                                            bert4rec_metrics, "bert4rec", CONFIG_PATH)
        if svd_metrics and not bert4rec_metrics:
            promoted = evaluate_and_promote(Path("models/svd_candidate.pkl"),
                                            svd_metrics, "svd", CONFIG_PATH)
        ctx["ti"].xcom_push(key="promoted", value=promoted)

    promote_task = PythonOperator(
        task_id="register_and_promote",
        python_callable=task_register_promote,
    )

    # ── Task 11: Trigger deployment ───────────────────────────────────────────

    def task_trigger_deploy(**ctx):
        promoted = ctx["ti"].xcom_pull(key="promoted", task_ids="register_and_promote")
        if promoted:
            # Signal the API to reload champion model (write a flag file)
            Path("models/.reload_flag").touch()
            logger.info("Deployment flag set — API will reload champion model.")
        else:
            logger.info("No promotion — keeping current champion.")

    deploy_task = PythonOperator(
        task_id="trigger_deployment",
        python_callable=task_trigger_deploy,
    )

    # ── Task 12: Drift report ─────────────────────────────────────────────────

    def task_drift_report(**ctx):
        from src.monitoring.drift_monitor import run_drift_report
        delta_path = ctx["ti"].xcom_pull(key="delta_path", task_ids="detect_delta")
        run_drift_report(
            reference_path="data/processed/all_ratings.csv",
            current_path=delta_path,
        )

    drift_task = PythonOperator(
        task_id="run_drift_report",
        python_callable=task_drift_report,
    )

    # ── Task 13: Log performance ──────────────────────────────────────────────

    def task_log_performance(**ctx):
        from src.monitoring.performance import log_cycle_metrics
        from src.replay.replay_state import load_state
        state = load_state()
        svd_metrics     = ctx["ti"].xcom_pull(key="svd_metrics",     task_ids="evaluate_svd") or {}
        bert4rec_metrics = ctx["ti"].xcom_pull(key="bert4rec_metrics", task_ids="evaluate_bert4rec") or {}
        promoted = ctx["ti"].xcom_pull(key="promoted", task_ids="register_and_promote") or False
        all_metrics = {**svd_metrics, **bert4rec_metrics}
        log_cycle_metrics(state.last_released_batch, "dual", all_metrics, promoted)

    perf_task = PythonOperator(
        task_id="update_performance_log",
        python_callable=task_log_performance,
    )

    # ── DAG dependency chain ──────────────────────────────────────────────────
    (
        get_next_batch_task
        >> detect_delta_task
        >> validate_task
        >> update_dataset_task
        >> build_sequences_task
        >> [train_svd_task, train_bert4rec_task]
    )
    train_svd_task     >> evaluate_svd_task
    train_bert4rec_task >> evaluate_bert4rec_task
    [evaluate_svd_task, evaluate_bert4rec_task] >> promote_task
    promote_task >> [deploy_task, drift_task, perf_task]

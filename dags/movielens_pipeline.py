"""
dags/movielens_pipeline.py

Airflow DAG: the full MLOps pipeline, running weekly.

Task order:
  download_data
    → diff_snapshot
      → validate_data
        → preprocess
          → train_model
            → evaluate_and_promote
              → (optional) trigger_deploy

Each task is a PythonOperator calling the corresponding src/ module.
This makes it easy to test tasks in isolation without Airflow.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# --- Task functions (thin wrappers around src/ modules) ----------------------

def _download(**context):
    from pathlib import Path
    from src.ingestion.movielens_fetcher import download_movielens
    extracted = download_movielens()
    # Push extracted path so downstream tasks can use it
    context["ti"].xcom_push(key="extracted_path", value=str(extracted))


def _diff(**context):
    from pathlib import Path
    from src.delta.snapshot_diff import compute_diff, update_snapshot

    ti = context["ti"]
    extracted = Path(ti.xcom_pull(task_ids="download_data", key="extracted_path"))
    new_ratings = extracted / "ratings.csv"
    snapshot = Path("data/snapshots/ratings_snapshot.csv")
    diff_out = Path("data/diffs/ratings_diff.csv")

    diff_df = compute_diff(new_ratings, snapshot, diff_out)
    update_snapshot(new_ratings, snapshot)
    ti.xcom_push(key="diff_rows", value=len(diff_df))


def _validate(**context):
    from pathlib import Path
    from src.validation.validate_ratings import validate_ratings
    validate_ratings(Path("data/diffs/ratings_diff.csv"))


def _preprocess(**context):
    from pathlib import Path
    from src.preprocessing.build_matrix import build_interaction_matrix

    ti = context["ti"]
    extracted = Path(ti.xcom_pull(task_ids="download_data", key="extracted_path"))
    build_interaction_matrix(
        ratings_path=extracted / "ratings.csv",
        movies_path=extracted / "movies.csv",
        out_path=Path("data/processed/interactions.csv"),
    )


def _train(**context):
    from pathlib import Path
    from src.training.train_svd import train
    train(
        interactions_path=Path("data/processed/interactions.csv"),
        model_out_path=Path("models/svd_model.pkl"),
    )


def _evaluate(**context):
    from pathlib import Path
    from src.evaluation.evaluator import evaluate_and_promote
    metrics = evaluate_and_promote(
        model_path=Path("models/svd_model.pkl"),
        interactions_path=Path("data/processed/interactions.csv"),
    )
    context["ti"].xcom_push(key="metrics", value=metrics)


# --- DAG definition ----------------------------------------------------------

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="movielens_recsys_pipeline",
    description="Weekly MovieLens ingestion → diff → validate → train → evaluate",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "recsys", "movielens"],
) as dag:

    download = PythonOperator(task_id="download_data", python_callable=_download)
    diff = PythonOperator(task_id="diff_snapshot", python_callable=_diff)
    validate = PythonOperator(task_id="validate_data", python_callable=_validate)
    preprocess = PythonOperator(task_id="preprocess", python_callable=_preprocess)
    train = PythonOperator(task_id="train_model", python_callable=_train)
    evaluate = PythonOperator(task_id="evaluate_and_promote", python_callable=_evaluate)
    done = EmptyOperator(task_id="pipeline_complete")

    download >> diff >> validate >> preprocess >> train >> evaluate >> done

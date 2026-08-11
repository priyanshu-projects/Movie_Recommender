"""
src/tracking/mlflow_tracker.py

MLflow logging helpers — wraps raw MLflow calls so all pipeline steps
log consistently without repeating boilerplate.
"""

import logging
from pathlib import Path

import mlflow
import yaml

logger = logging.getLogger(__name__)


def init_mlflow(config_path: str = "configs/config.yaml") -> None:
    """Set the MLflow tracking URI from config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    uri = cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI: %s", uri)


def log_training_run(
    experiment_name: str,
    run_name: str,
    params: dict,
    metrics: dict,
    model_path: Path | None = None,
    tags: dict | None = None,
    config_path: str = "configs/config.yaml",
) -> str:
    """
    Log a training run to MLflow.
    Returns the run_id.
    """
    init_mlflow(config_path)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        if model_path and Path(model_path).exists():
            mlflow.log_artifact(str(model_path))
        run_id = run.info.run_id

    logger.info("MLflow run logged: %s (experiment: %s)", run_id[:8], experiment_name)
    return run_id

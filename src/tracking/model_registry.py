"""
src/tracking/model_registry.py

Champion / Challenger promotion logic.

Every retraining cycle produces a candidate model.
This module compares it against the current production model
and promotes only if it meets the threshold improvement.

Promotion rules (configurable in configs/config.yaml):
    SVD:       new RMSE < champion RMSE - promotion_min_rmse_delta
    BERT4Rec:  new NDCG@K > champion NDCG@K + promotion_min_ndcg_delta
"""

import logging
import pickle
from pathlib import Path

import mlflow
import yaml

logger = logging.getLogger(__name__)

CHAMPION_PATH = Path("models/champion_model.pkl")
CHAMPION_META_PATH = Path("models/champion_meta.yaml")


def load_champion_meta() -> dict | None:
    """Load champion model metadata. Returns None if no champion exists yet."""
    if not CHAMPION_META_PATH.exists():
        return None
    with open(CHAMPION_META_PATH) as f:
        return yaml.safe_load(f)


def _to_native(val):
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_native(v) for v in val]
    return val


def save_champion(model_path: Path, metrics: dict, model_type: str) -> None:
    """Copy candidate model to champion path and save metadata."""
    import shutil
    CHAMPION_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, CHAMPION_PATH)

    clean_metrics = _to_native(metrics)
    meta = {"model_type": model_type, "metrics": clean_metrics}
    with open(CHAMPION_META_PATH, "w") as f:
        yaml.dump(meta, f)

    logger.info("New champion promoted: %s | metrics: %s", model_type, clean_metrics)


def evaluate_and_promote(
    candidate_model_path: Path,
    candidate_metrics: dict,
    model_type: str,
    config_path: str = "configs/config.yaml",
) -> bool:
    """
    Compare candidate metrics against current champion.
    Promote candidate if it's sufficiently better.

    Returns True if candidate was promoted, False otherwise.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    eval_cfg = cfg.get("evaluation", {})

    champion_meta = load_champion_meta()

    if champion_meta is None:
        logger.info("No champion exists. Promoting first candidate automatically.")
        save_champion(candidate_model_path, candidate_metrics, model_type)
        return True

    champ_metrics = champion_meta.get("metrics", {})

    if model_type == "svd":
        delta = eval_cfg.get("promotion_min_rmse_delta", 0.005)
        cand_rmse = candidate_metrics.get("test_rmse", float("inf"))
        champ_rmse = champ_metrics.get("test_rmse", float("inf"))
        promoted = cand_rmse < champ_rmse - delta
        logger.info(
            "SVD champion/challenger: champion RMSE=%.4f | candidate RMSE=%.4f | Δ=%.4f | promoted=%s",
            champ_rmse, cand_rmse, champ_rmse - cand_rmse, promoted,
        )
    elif model_type == "bert4rec":
        k = eval_cfg.get("k", 10)
        delta = eval_cfg.get("promotion_min_ndcg_delta", 0.001)
        metric_key = f"ndcg@{k}"
        cand_ndcg = candidate_metrics.get(metric_key, 0.0)
        champ_ndcg = champ_metrics.get(metric_key, 0.0)
        promoted = cand_ndcg > champ_ndcg + delta
        logger.info(
            "BERT4Rec champion/challenger: champion NDCG=%.4f | candidate NDCG=%.4f | Δ=%.4f | promoted=%s",
            champ_ndcg, cand_ndcg, cand_ndcg - champ_ndcg, promoted,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if promoted:
        save_champion(candidate_model_path, candidate_metrics, model_type)

    return promoted

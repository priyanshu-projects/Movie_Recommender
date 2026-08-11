"""
src/monitoring/performance.py

Tracks recommendation model performance over time.
Logs per-cycle metrics to a JSONL file for trend analysis.
Complements Evidently drift reports with model quality signals.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PERF_LOG = Path("data/drift_reports/performance_log.jsonl")


def log_cycle_metrics(
    batch_id: int,
    model_type: str,
    metrics: dict,
    promoted: bool,
) -> None:
    """Append a performance record for this retraining cycle."""
    PERF_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "model_type": model_type,
        "metrics": metrics,
        "promoted": promoted,
    }
    with open(PERF_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Cycle %d metrics logged: %s | promoted=%s", batch_id, metrics, promoted)


def load_performance_history() -> list[dict]:
    """Load all historical cycle metrics."""
    if not PERF_LOG.exists():
        return []
    with open(PERF_LOG) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_performance_degradation(
    metric_key: str = "test_rmse",
    window: int = 3,
    threshold: float = 0.05,
) -> bool:
    """
    Returns True if the model metric has degraded consistently
    over the last `window` cycles (signals need for investigation).
    """
    history = load_performance_history()
    if len(history) < window:
        return False

    recent = history[-window:]
    values = [h["metrics"].get(metric_key) for h in recent if metric_key in h.get("metrics", {})]
    if len(values) < window:
        return False

    # For RMSE: degradation = consistently increasing
    # For NDCG: degradation = consistently decreasing
    if "rmse" in metric_key.lower():
        degraded = all(values[i] > values[i - 1] for i in range(1, len(values)))
    else:
        degraded = all(values[i] < values[i - 1] for i in range(1, len(values)))

    if degraded:
        logger.warning(
            "Performance degradation detected on '%s' over last %d cycles: %s",
            metric_key, window, values,
        )
    return degraded

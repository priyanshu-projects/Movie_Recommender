"""
src/monitoring/drift_monitor.py

Uses Evidently AI to detect data drift between the training distribution
and the incoming batch of new ratings (diff CSV).

What we monitor:
  - Rating distribution drift (is the new batch skewed vs training?)
  - Timestamp drift (are new ratings recent — sanity check)

If drift is detected, this script exits with code 1, which the Airflow task
can use to trigger the retraining GitHub Actions workflow.

Usage (standalone):
    python -m src.monitoring.drift_monitor \
        --reference data/processed/interactions.csv \
        --current   data/diffs/ratings_diff.csv \
        --report    data/drift_report.html
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.5  # fraction of drifted features that triggers alert


def detect_drift(
    reference_path: Path,
    current_path: Path,
    report_out: Path,
) -> bool:
    """
    Compare current batch against reference (training) data.

    Returns True if drift is detected (triggers retraining), False otherwise.
    Saves an HTML report to report_out.
    """
    reference = pd.read_csv(reference_path)[["rating", "timestamp"]].dropna()
    current = pd.read_csv(current_path)[["rating", "timestamp"]].dropna()

    if len(current) < 10:
        logger.warning("Current batch too small (%d rows) — skipping drift check.", len(current))
        return False

    column_mapping = ColumnMapping(numerical_features=["rating", "timestamp"])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_out))
    logger.info("Drift report saved to %s", report_out)

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    if drift_detected:
        logger.warning("DRIFT DETECTED — retraining should be triggered.")
    else:
        logger.info("No drift detected — pipeline continues normally.")

    return drift_detected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Detect data drift in incoming ratings batch.")
    parser.add_argument("--reference", required=True, help="Path to training interactions CSV")
    parser.add_argument("--current", required=True, help="Path to incoming diff CSV")
    parser.add_argument("--report", default="data/drift_report.html", help="Output report path")
    args = parser.parse_args()

    drift = detect_drift(Path(args.reference), Path(args.current), Path(args.report))
    sys.exit(1 if drift else 0)

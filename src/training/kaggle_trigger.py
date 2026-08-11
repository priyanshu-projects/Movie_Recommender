"""
src/training/kaggle_trigger.py

Triggers a Kaggle notebook kernel via the Kaggle CLI and polls until complete.
Used by GitHub Actions to kick off BERT4Rec GPU fine-tuning on Kaggle T4.

Required env vars (from GitHub Secrets / environment):
    KAGGLE_API_TOKEN or (KAGGLE_USERNAME and KAGGLE_KEY)

Usage:
    python -m src.training.kaggle_trigger \
        --kernel slavery786/bert4rec-movie-recommender-fine-tuning \
        --timeout 3600
"""

import argparse
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def trigger_kernel(kernel_dir: Path = Path("notebooks")) -> str:
    """Push kernel code to Kaggle to trigger a new GPU run."""
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    env = os.environ.copy()
    if token:
        env["KAGGLE_API_TOKEN"] = token

    logger.info("Pushing kernel from %s to Kaggle...", kernel_dir)
    res = subprocess.run(
        ["kaggle", "kernels", "push"],
        cwd=kernel_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        logger.error("Failed to push kernel: %s", res.stderr)
        raise RuntimeError(f"Kaggle push failed: {res.stderr}")

    logger.info("✓ Kernel pushed successfully: %s", res.stdout.strip())
    return res.stdout.strip()


def poll_kernel(kernel_slug: str, timeout_seconds: int = 3600, poll_interval: int = 20) -> bool:
    """Poll Kaggle kernel status until COMPLETE or timeout."""
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    env = os.environ.copy()
    if token:
        env["KAGGLE_API_TOKEN"] = token

    deadline = time.time() + timeout_seconds
    logger.info("Polling kernel status for %s (timeout: %ds)...", kernel_slug, timeout_seconds)

    while time.time() < deadline:
        res = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            env=env,
            capture_output=True,
            text=True,
        )
        output = res.stdout.strip()
        logger.info("  Status output: %s", output)

        if "complete" in output.lower() or "complete" in res.stderr.lower():
            logger.info("✓ Kernel execution completed successfully!")
            return True
        elif "error" in output.lower() or "failed" in output.lower():
            logger.error("✗ Kernel execution failed: %s", output)
            return False

        time.sleep(poll_interval)

    logger.error("✗ Timeout waiting for kernel %s", kernel_slug)
    return False


def trigger_and_wait(kernel_slug: str, kernel_dir: Path = Path("notebooks"), timeout_seconds: int = 3600) -> bool:
    trigger_kernel(kernel_dir)
    return poll_kernel(kernel_slug, timeout_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="slavery786/bert4rec-movie-recommender-fine-tuning")
    parser.add_argument("--dir",    default="notebooks")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    success = poll_kernel(args.kernel, timeout_seconds=args.timeout)
    exit(0 if success else 1)

"""
src/training/kaggle_trigger.py

Triggers a Kaggle notebook kernel via the Kaggle API and polls until complete.
Used by GitHub Actions to kick off BERT4Rec GPU fine-tuning on Kaggle T4.

Required env vars (from GitHub Secrets):
    KAGGLE_USERNAME  — your Kaggle username
    KAGGLE_KEY       — your Kaggle API key

Usage:
    python -m src.training.kaggle_trigger \
        --kernel priyanshu-projects/bert4rec-finetune \
        --timeout 3600
"""

import argparse
import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

KAGGLE_API = "https://www.kaggle.com/api/v1"


def _get_request_kwargs() -> dict:
    """Return requests kwargs (headers and/or auth) for Kaggle API calls."""
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")

    if key.startswith("KGAT_"):
        return {"headers": {"Authorization": f"Bearer {key}"}}
    return {"auth": (username, key)}


def trigger_kernel(kernel_slug: str) -> str:
    """
    Push (trigger) a Kaggle kernel run.
    kernel_slug format: "username/kernel-name"

    Returns the new kernel run version number.
    """
    owner, kernel = kernel_slug.split("/")
    kwargs = _get_request_kwargs()

    # Get current kernel metadata to build push payload
    meta_url = f"{KAGGLE_API}/kernels/{owner}/{kernel}"
    resp = requests.get(meta_url, timeout=30, **kwargs)
    resp.raise_for_status()
    meta = resp.json()

    # Trigger a new run
    push_url = f"{KAGGLE_API}/kernels/push"
    payload = {
        "slug": kernel,
        "newTitle": meta.get("title", kernel),
        "source": meta.get("source", ""),
        "language": meta.get("language", "python"),
        "kernelType": meta.get("kernelType", "notebook"),
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": True,
        "datasetDataSources": meta.get("datasetDataSources", []),
        "kernelDataSources": meta.get("kernelDataSources", []),
        "categoryIds": [],
    }
    resp = requests.post(push_url, json=payload, timeout=30, **kwargs)
    resp.raise_for_status()
    version = resp.json().get("currentRunningVersion", "unknown")
    logger.info("Kaggle kernel triggered: %s v%s", kernel_slug, version)
    return str(version)


def poll_kernel(kernel_slug: str, timeout_seconds: int = 3600, poll_interval: int = 30) -> bool:
    """
    Poll Kaggle kernel status until COMPLETE or timeout.
    Returns True if completed successfully, False otherwise.
    """
    owner, kernel = kernel_slug.split("/")
    kwargs = _get_request_kwargs()
    status_url = f"{KAGGLE_API}/kernels/{owner}/{kernel}"
    deadline = time.time() + timeout_seconds

    logger.info("Polling kernel %s (timeout: %ds) ...", kernel_slug, timeout_seconds)
    while time.time() < deadline:
        resp = requests.get(status_url, timeout=30, **kwargs)
        resp.raise_for_status()
        status = resp.json().get("currentRunningStatus", "unknown")
        logger.info("  Kernel status: %s", status)

        if status == "complete":
            logger.info("✓ Kernel completed successfully.")
            return True
        elif status in ("error", "cancelAcknowledged", "cancel"):
            logger.error("✗ Kernel failed with status: %s", status)
            return False

        time.sleep(poll_interval)

    logger.error("✗ Timeout waiting for kernel %s", kernel_slug)
    return False


def trigger_and_wait(kernel_slug: str, timeout_seconds: int = 3600) -> bool:
    """Trigger kernel and wait for completion. Returns True if successful."""
    trigger_kernel(kernel_slug)
    return poll_kernel(kernel_slug, timeout_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel",  required=True, help="username/kernel-name")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    success = trigger_and_wait(args.kernel, args.timeout)
    exit(0 if success else 1)

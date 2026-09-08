"""
src/training/kaggle_trigger.py

Triggers a Kaggle kernel via the Kaggle Python API (not CLI subprocess).
Uses the Python API acc parameter which maps directly to machine_shape
in the kernel push request, reliably forcing NvidiaTeslaT4 instead of
the random GPU Kaggle assigns via plain kaggle kernels push.

Required env vars:
    KAGGLE_API_TOKEN  -- your Kaggle API token (KGAT_xxx...)

Usage:
    python -m src.training.kaggle_trigger \\
        --kernel slavery786/bert4rec-movie-recommender-fine-tuning \\
        --timeout 7200
"""

import argparse
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Valid machine_shape values per kagglesdk.kernels.types.kernels_api_service:
#   "NvidiaTeslaT4"   -> single T4 GPU
#   "NvidiaTeslaP100" -> P100 GPU
# T4x2 is a web-UI only option, not exposed via the API.
KAGGLE_T4_SHAPE = "NvidiaTeslaT4"


def _get_api():
    """Write token to access_token file and return authenticated KaggleApi."""
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    if token:
        token_path = Path.home() / ".kaggle" / "access_token"
        token_path.parent.mkdir(exist_ok=True)
        token_path.write_text(token)
        token_path.chmod(0o600)

    from kaggle import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _inject_timestamp(kernel_dir: Path) -> None:
    """Prepend a timestamp comment so Kaggle always detects a code change."""
    script_path = kernel_dir / "bert4rec_kaggle_train.py"
    if not script_path.exists():
        return
    original = script_path.read_text()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ts_comment = f"# AUTO-TRIGGERED: {ts}\n"
    if original.startswith("# AUTO-TRIGGERED:"):
        lines = original.splitlines(keepends=True)
        patched = ts_comment + "".join(lines[1:])
    else:
        patched = ts_comment + original
    script_path.write_text(patched)
    logger.info("Timestamp injected into training script.")


def trigger_kernel(kernel_dir: Path = Path("notebooks")) -> str:
    """Push kernel via Python API with acc=NvidiaTeslaT4.

    The acc param maps to machine_shape in the API request body,
    reliably selecting T4 instead of Kaggle's random default.
    """
    _inject_timestamp(Path(kernel_dir))

    api = _get_api()
    logger.info("Pushing kernel from %s with machine_shape=%s", kernel_dir, KAGGLE_T4_SHAPE)

    result = api.kernels_push(folder=str(kernel_dir), acc=KAGGLE_T4_SHAPE)

    if result is None:
        raise RuntimeError("Kaggle push returned None -- check API credentials.")
    if result.error:
        raise RuntimeError(f"Kaggle push failed: {result.error}")

    version = getattr(result, "versionNumber", "?")
    url = getattr(result, "url", "")
    logger.info("Kernel version %s pushed with T4. See: %s", version, url)
    return url


def poll_kernel(kernel_slug: str, timeout_seconds: int = 7200, poll_interval: int = 30) -> bool:
    """Poll kernel status via Python API until COMPLETE or timeout."""
    from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

    api = _get_api()
    deadline = time.time() + timeout_seconds
    logger.info("Polling %s (timeout: %ds)...", kernel_slug, timeout_seconds)

    while time.time() < deadline:
        try:
            status_obj = api.kernels_status(kernel_slug)
            status = getattr(status_obj, "status", None)
            logger.info("  Status: %s", status)

            if status == KernelWorkerStatus.COMPLETE:
                logger.info("Kernel completed successfully!")
                return True
            if status in (
                KernelWorkerStatus.ERROR,
                KernelWorkerStatus.CANCEL_ACKNOWLEDGED,
                KernelWorkerStatus.CANCEL_REQUESTED,
            ):
                logger.error("Kernel failed with status: %s", status)
                return False
        except Exception as exc:
            logger.warning("Poll error (will retry): %s", exc)

        time.sleep(poll_interval)

    logger.error("Timeout waiting for kernel %s", kernel_slug)
    return False


def trigger_and_wait(
    kernel_slug: str,
    kernel_dir: Path = Path("notebooks"),
    timeout_seconds: int = 7200,
) -> bool:
    trigger_kernel(kernel_dir)
    return poll_kernel(kernel_slug, timeout_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Trigger Kaggle BERT4Rec GPU training on T4.")
    parser.add_argument("--kernel",    default="slavery786/bert4rec-movie-recommender-fine-tuning")
    parser.add_argument("--dir",       default="notebooks")
    parser.add_argument("--timeout",   type=int, default=7200)
    parser.add_argument("--poll-only", action="store_true", help="Skip push, only poll status")
    args = parser.parse_args()

    if args.poll_only:
        success = poll_kernel(args.kernel, timeout_seconds=args.timeout)
    else:
        success = trigger_and_wait(args.kernel, kernel_dir=Path(args.dir), timeout_seconds=args.timeout)

    exit(0 if success else 1)

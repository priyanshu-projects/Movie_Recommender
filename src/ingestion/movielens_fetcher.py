
"""
src/ingestion/movielens_fetcher.py

Downloads the latest MovieLens dataset from GroupLens and stores it in data/raw/.
GroupLens provides the entire file each time — not just new rows.
Delta detection is handled separately in src/delta/.

Usage (standalone):
    python -m src.ingestion.movielens_fetcher
"""

import io
import logging
import os
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Swap to "ml-latest" once you're done with fast iteration on small data
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RAW_DATA_DIR = Path("data/raw")


def download_movielens(url: str = MOVIELENS_URL, dest_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Download and unzip the MovieLens dataset into dest_dir.

    Returns the path to the unzipped folder (e.g. data/raw/ml-latest-small/).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MovieLens dataset from %s ...", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(dest_dir)
        # The top-level folder name inside the zip (e.g. "ml-latest-small")
        top_level = zf.namelist()[0].split("/")[0]

    extracted_path = dest_dir / top_level
    logger.info("Extracted to %s", extracted_path)
    return extracted_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = download_movielens()
    print(f"Downloaded to: {path}")

"""
src/storage/s3_storage.py

AWS S3 Storage integration for model artifacts and datasets.

Environment variables / GitHub Secrets:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION (default: us-east-1)
    AWS_S3_BUCKET (default: movie-recommender-mlops)

Usage:
    from src.storage.s3_storage import S3Storage
    store = S3Storage()
    store.upload_champion()
    store.download_champion()
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. S3 storage features disabled.")


class S3Storage:
    """Thin wrapper around AWS S3 for model artifacts and replay state."""

    def __init__(
        self,
        bucket_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str | None = None,
    ):
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 is not installed. Run: pip install boto3")

        self.bucket_name = (
            bucket_name
            or os.environ.get("AWS_S3_BUCKET")
            or "movie-recommender-mlops-745600"
        )
        self.region = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "ap-south-1"
        )

        session_kwargs = {"region_name": self.region}
        key_id = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        secret = aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        if key_id and secret:
            session_kwargs["aws_access_key_id"] = key_id
            session_kwargs["aws_secret_access_key"] = secret

        self.s3 = boto3.client("s3", **session_kwargs)
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Check if bucket exists; create if missing (handles us-east-1 constraint)."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code in ("404", "403", "NoSuchBucket"):
                logger.info("Bucket %s not found. Creating...", self.bucket_name)
                create_args = {"Bucket": self.bucket_name}
                if self.region != "us-east-1":
                    create_args["CreateBucketConfiguration"] = {
                        "LocationConstraint": self.region
                    }
                try:
                    self.s3.create_bucket(**create_args)
                    logger.info("Created S3 bucket: %s", self.bucket_name)
                except Exception as create_exc:
                    logger.warning("Could not create bucket %s: %s", self.bucket_name, create_exc)

    def upload(self, local_path: Path, object_name: str) -> str:
        """Upload a local file to S3. Returns S3 URI."""
        local_path = Path(local_path)
        self.s3.upload_file(str(local_path), self.bucket_name, object_name)
        s3_uri = f"s3://{self.bucket_name}/{object_name}"
        logger.info("Uploaded %s -> %s", local_path, s3_uri)
        return s3_uri

    def download(self, object_name: str, local_path: Path) -> Path:
        """Download an object from S3 to a local path."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket_name, object_name, str(local_path))
        logger.info("Downloaded %s -> %s", object_name, local_path)
        return local_path

    def upload_champion(self, local_path: Path = Path("models/champion_model.pkl")) -> str:
        return self.upload(local_path, "models/champion/champion_model.pkl")

    def download_champion(self, local_path: Path = Path("models/champion_model.pkl")) -> Path:
        return self.download("models/champion/champion_model.pkl", local_path)

    def upload_champion_meta(self, local_path: Path = Path("models/champion_meta.yaml")) -> str:
        return self.upload(local_path, "models/champion/champion_meta.yaml")

    def download_champion_meta(self, local_path: Path = Path("models/champion_meta.yaml")) -> Path:
        return self.download("models/champion/champion_meta.yaml", local_path)

    def upload_replay_state(self, local_path: Path = Path("data/replay/replay_state.json")) -> str:
        return self.upload(local_path, "replay/replay_state.json")

    def download_replay_state(self, local_path: Path = Path("data/replay/replay_state.json")) -> Path:
        return self.download("replay/replay_state.json", local_path)

    def list_objects(self, prefix: str = "") -> list[str]:
        res = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        return [obj["Key"] for obj in res.get("Contents", [])]

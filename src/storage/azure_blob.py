"""
src/storage/azure_blob.py

Azure Blob Storage integration for model artifacts and datasets.

All connection details come from environment variables / GitHub Secrets:
    AZURE_STORAGE_CONNECTION_STRING  — full connection string to storage account
    AZURE_STORAGE_CONTAINER          — container name (default: mlops-artifacts)

Usage:
    from src.storage.azure_blob import AzureBlobStorage
    store = AzureBlobStorage()
    store.upload_model("models/champion_model.pkl", "champion/champion_model.pkl")
    store.download_model("champion/champion_model.pkl", "models/champion_model.pkl")
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazily imported so local runs without azure SDK installed don't crash
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False
    logger.warning("azure-storage-blob not installed. Azure storage features disabled.")


class AzureBlobStorage:
    """Thin wrapper around Azure Blob Storage for model artifacts."""

    def __init__(
        self,
        connection_string: str | None = None,
        container: str | None = None,
    ):
        if not AZURE_SDK_AVAILABLE:
            raise RuntimeError(
                "azure-storage-blob is not installed. "
                "Run: pip install azure-storage-blob"
            )
        conn_str = connection_string or os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        self.container = container or os.environ.get("AZURE_STORAGE_CONTAINER", "mlops-artifacts")
        self.client = BlobServiceClient.from_connection_string(conn_str)
        self._ensure_container()

    def _ensure_container(self) -> None:
        container_client = self.client.get_container_client(self.container)
        if not container_client.exists():
            container_client.create_container()
            logger.info("Created Azure Blob container: %s", self.container)

    def upload(self, local_path: Path, blob_name: str, overwrite: bool = True) -> str:
        """Upload a file to Azure Blob. Returns the blob URL."""
        local_path = Path(local_path)
        blob_client = self.client.get_blob_client(container=self.container, blob=blob_name)
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=overwrite)
        url = blob_client.url
        logger.info("Uploaded %s → %s", local_path, blob_name)
        return url

    def download(self, blob_name: str, local_path: Path) -> Path:
        """Download a blob to a local file."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob_client = self.client.get_blob_client(container=self.container, blob=blob_name)
        with open(local_path, "wb") as f:
            data = blob_client.download_blob()
            data.readinto(f)
        logger.info("Downloaded %s → %s", blob_name, local_path)
        return local_path

    def upload_model(self, local_path: Path, version_tag: str = "latest") -> str:
        """Upload a model artifact. Saves under models/<version_tag>/."""
        blob_name = f"models/{version_tag}/{Path(local_path).name}"
        return self.upload(local_path, blob_name)

    def download_champion(self, local_path: Path = Path("models/champion_model.pkl")) -> Path:
        """Download the current champion model."""
        return self.download("models/champion/champion_model.pkl", local_path)

    def upload_champion(self, local_path: Path = Path("models/champion_model.pkl")) -> str:
        """Upload the champion model (overwrites current champion)."""
        return self.upload(local_path, "models/champion/champion_model.pkl")

    def upload_champion_meta(self, local_path: Path = Path("models/champion_meta.yaml")) -> str:
        return self.upload(local_path, "models/champion/champion_meta.yaml")

    def download_champion_meta(self, local_path: Path = Path("models/champion_meta.yaml")) -> Path:
        return self.download("models/champion/champion_meta.yaml", local_path)

    def upload_replay_state(self, local_path: Path = Path("data/replay/replay_state.json")) -> str:
        return self.upload(local_path, "replay/replay_state.json")

    def download_replay_state(self, local_path: Path = Path("data/replay/replay_state.json")) -> Path:
        return self.download("replay/replay_state.json", local_path)

    def list_blobs(self, prefix: str = "") -> list[str]:
        container_client = self.client.get_container_client(self.container)
        return [b.name for b in container_client.list_blobs(name_starts_with=prefix)]

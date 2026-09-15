"""
src/models/base_model.py

Abstract base class that both SVD and BERT4Rec implement.
Keeps the MLOps pipeline model-agnostic — the training, evaluation,
and serving layers interact with this interface, not concrete models.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseRecommender(ABC):
    """Common interface for all recommendation models."""

    @abstractmethod
    def train(self, train_data, val_data=None) -> dict:
        """
        Train the model.
        Returns a dict of training metrics (e.g. {"train_loss": 0.42}).
        """

    @abstractmethod
    def evaluate(self, test_data) -> dict:
        """
        Evaluate the model on held-out test data.
        Returns a dict of evaluation metrics.
        """

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> list[dict]:
        """
        Return top-N recommendations for a user.
        Each dict contains at minimum: {"movieId": int, "score": float}.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: Path) -> "BaseRecommender":
        """Load model from disk. Returns self for chaining."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return a string identifier e.g. 'svd' or 'bert4rec'."""

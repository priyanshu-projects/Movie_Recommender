"""
tests/unit/test_api.py

Integration tests for FastAPI endpoints (/health, /model, /recommend).
"""

from fastapi.testclient import TestClient
from src.api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


def test_model_endpoint():
    with TestClient(app) as client:
        response = client.get("/model")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data


def test_recommend_endpoint():
    with TestClient(app) as client:
        response = client.get("/recommend/1?k=3")
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert data["user_id"] == 1
            assert len(data["recommendations"]) <= 3

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_spam(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"message": "Congratulations! You won a free prize. Call now to claim."},
    )

    assert response.status_code == 200
    assert response.json() == {"label": "Spam"}
    assert response.headers["content-type"].startswith("application/json")


def test_predict_ham(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"message": "Hi, are we still meeting for lunch today?"},
    )

    assert response.status_code == 200
    assert response.json() == {"label": "Not Spam"}
    assert response.headers["content-type"].startswith("application/json")


def test_rejects_blank_message(client: TestClient) -> None:
    response = client.post("/predict", json={"message": "   "})

    assert response.status_code == 422

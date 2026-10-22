from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    """Verify that root endpoint returns operational status and service name."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["service"] == "AI Document Assistant"


def test_health_endpoint() -> None:
    """Verify that health check endpoint returns 200 OK."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "AI Document Assistant"

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


def test_parse_uploaded_document_success() -> None:
    """Verify that uploading a valid text file parses successfully."""
    content = b"Line 1\nLine 2 content."
    response = client.post(
        "/documents/parse",
        files={"file": ("sample.txt", content, "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["source_name"] == "sample.txt"
    assert data["file_type"] == "txt"
    assert data["extraction_method"] == "text_reader"
    assert "Line 1" in data["preview"]
    assert data["character_count"] == len(content)


def test_parse_uploaded_document_unsupported_type() -> None:
    """Verify that unsupported extensions return 415 Unsupported Media Type."""
    response = client.post(
        "/documents/parse",
        files={"file": ("image.png", b"image bytes", "image/png")},
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_parse_uploaded_document_empty_file() -> None:
    """Verify that empty uploads return 400 Bad Request."""
    response = client.post(
        "/documents/parse",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

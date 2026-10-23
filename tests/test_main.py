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


def test_index_uploaded_document_success(monkeypatch) -> None:
    """Verify that uploading a valid document indexes successfully."""
    monkeypatch.setattr(
        "backend.main.index_parsed_document",
        lambda parsed, session_id=None: {
            "source_id": "test-source-id-12345",
            "source_name": parsed.source_name,
            "file_type": parsed.file_type,
            "extraction_method": parsed.extraction_method,
            "chunk_count": 2,
            "content_hash": "hash123",
            "collection_name": "ai_document_assistant",
            "persist_directory": "./chroma_db",
            "session_id": session_id,
        },
    )

    content = b"Paragraph 1.\n\nParagraph 2."
    response = client.post(
        "/documents/index",
        files={"file": ("notes.txt", content, "text/plain")},
        data={"session_id": "test-session-123"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "indexed"
    assert data["session_id"] == "test-session-123"
    assert data["source_name"] == "notes.txt"
    assert data["source_id"] == "test-source-id-12345"
    assert data["chunk_count"] == 2


def test_index_uploaded_document_indexing_failure(monkeypatch) -> None:
    """Verify that indexing failure returns 500 Internal Server Error."""

    def fail_index(parsed, *, session_id=None):
        raise RuntimeError("Vector store failure")

    monkeypatch.setattr("backend.main.index_parsed_document", fail_index)

    response = client.post(
        "/documents/index",
        files={"file": ("notes.txt", b"Valid content", "text/plain")},
        data={"session_id": "test-session-123"},
    )

    assert response.status_code == 500
    assert "indexing failed" in response.json()["detail"].lower()


def test_index_uploaded_document_empty_session_id() -> None:
    """Verify that an empty or whitespace session_id returns 422 Unprocessable Content."""
    response = client.post(
        "/documents/index",
        files={"file": ("notes.txt", b"Valid content", "text/plain")},
        data={"session_id": "   "},
    )

    assert response.status_code == 422
    assert "session_id" in response.json()["detail"].lower()


def test_index_uploaded_document_missing_session_id() -> None:
    """Verify that omitting session_id returns 422 Unprocessable Content."""
    response = client.post(
        "/documents/index",
        files={"file": ("notes.txt", b"Valid content", "text/plain")},
    )

    assert response.status_code == 422

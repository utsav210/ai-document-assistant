from backend import vector_store
from backend.document_parser import ParsedDocument


class FakeVectorStore:
    """In-memory substitute for Chroma in unit tests."""

    def __init__(self):
        self.documents = {}

    def get(self, *, where, include):
        source_id = where.get("source_id")
        session_id = where.get("session_id")
        if "$and" in where:
            for item in where["$and"]:
                if "source_id" in item:
                    source_id = item["source_id"]
                if "session_id" in item:
                    session_id = item["session_id"]

        matching = {
            document_id: document
            for document_id, document in self.documents.items()
            if (source_id is None or document.metadata.get("source_id") == source_id)
            and (
                session_id is None or document.metadata.get("session_id") == session_id
            )
        }

        return {
            "ids": list(matching),
            "metadatas": [document.metadata for document in matching.values()],
        }

    def add_documents(self, *, documents, ids):
        for document, document_id in zip(documents, ids, strict=True):
            self.documents[document_id] = document

    def delete(self, *, ids):
        for document_id in ids:
            self.documents.pop(document_id, None)


def test_indexing_replaces_old_chunks_and_removes_stale_chunks(
    monkeypatch,
):
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        vector_store,
        "get_vector_store",
        lambda: fake_store,
    )

    original = ParsedDocument(
        source_name="notes.txt",
        file_type="txt",
        content=(
            "Authentication is required for access. "
            "Authorization determines what users may do. "
            "Monitoring records suspicious security activity. "
        )
        * 10,
        extraction_method="text_reader",
    )

    first_result = vector_store.index_parsed_document(original)

    assert first_result["chunk_count"] > 1
    assert len(fake_store.documents) == first_result["chunk_count"]

    updated = ParsedDocument(
        source_name="notes.txt",
        file_type="txt",
        content="Updated security notes.",
        extraction_method="text_reader",
    )

    second_result = vector_store.index_parsed_document(updated)

    assert second_result["chunk_count"] == 1
    assert len(fake_store.documents) == 1

    stored_document = next(iter(fake_store.documents.values()))
    assert stored_document.page_content == "Updated security notes."


def test_indexing_with_session_id_isolates_metadata(monkeypatch):
    fake_store = FakeVectorStore()
    monkeypatch.setattr(vector_store, "get_vector_store", lambda: fake_store)

    doc = ParsedDocument(
        source_name="session_doc.txt",
        file_type="txt",
        content="User confidential session notes.",
        extraction_method="text_reader",
    )

    result = vector_store.index_parsed_document(doc, session_id="user-session-42")

    assert result["session_id"] == "user-session-42"
    assert len(fake_store.documents) == 1
    stored = next(iter(fake_store.documents.values()))
    assert stored.metadata.get("session_id") == "user-session-42"


def test_semantic_relevance_score_fn():
    fn = vector_store._semantic_relevance_score_fn
    assert fn(0.0) == 1.0  # Identical vectors
    assert fn(2.0) == 0.0  # Orthogonal vectors
    assert fn(1.0) == 0.5  # Cosine similarity 0.5
    assert fn(4.0) == 0.0  # Clamped minimum
    assert fn(-1.0) == 1.0  # Clamped maximum

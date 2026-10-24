from __future__ import annotations

from threading import Lock
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.chunking import create_chunks
from backend.config import get_settings
from backend.document_parser import ParsedDocument

_INDEX_LOCK = Lock()
_STORE_LOCK = Lock()
_EMBEDDINGS_LOCK = Lock()

_CACHED_EMBEDDINGS: HuggingFaceEmbeddings | None = None
_CACHED_STORE: Chroma | None = None
_CHROMA_CLIENT: Any = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load and reuse the local embedding model in a thread-safe manner."""
    global _CACHED_EMBEDDINGS

    if _CACHED_EMBEDDINGS is None:
        with _EMBEDDINGS_LOCK:
            if _CACHED_EMBEDDINGS is None:
                settings = get_settings()
                _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
                    model_name=settings.embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )

    return _CACHED_EMBEDDINGS


def _semantic_relevance_score_fn(distance: float) -> float:
    """Convert Chroma squared-L2 distance into true [0, 1] cosine semantic similarity.

    Since HuggingFace embeddings are unit-normalized (norm = 1):
    squared_distance = 2 - 2 * cosine_similarity
    cosine_similarity = 1.0 - distance / 2.0
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def get_vector_store() -> Chroma:
    """Create or reopen the persistent Chroma collection with cosine relevance scoring."""
    global _CACHED_STORE, _CHROMA_CLIENT

    if _CACHED_STORE is None:
        with _STORE_LOCK:
            if _CACHED_STORE is None:
                import chromadb

                settings = get_settings()
                settings.chroma_persist_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if _CHROMA_CLIENT is None:
                    _CHROMA_CLIENT = chromadb.PersistentClient(
                        path=str(settings.chroma_persist_directory)
                    )

                _CACHED_STORE = Chroma(
                    client=_CHROMA_CLIENT,
                    collection_name=settings.chroma_collection_name,
                    embedding_function=get_embeddings(),
                    relevance_score_fn=_semantic_relevance_score_fn,
                )

    return _CACHED_STORE


def index_parsed_document(
    parsed: ParsedDocument,
    *,
    session_id: str | None = None,
) -> dict:
    """Index a parsed document within an optional owning session."""
    chunks = create_chunks(
        parsed,
        session_id=session_id,
    )

    source_id = chunks[0].metadata["source_id"]
    chunk_ids = [f"{source_id}-{index:06d}" for index in range(len(chunks))]

    store = get_vector_store()

    if session_id:
        where_filter = {
            "$and": [
                {"session_id": session_id},
                {"source_id": source_id},
            ]
        }
    else:
        where_filter = {"source_id": source_id}

    with _INDEX_LOCK:
        # Only inspect chunks belonging to this session and source.
        existing = store.get(
            where=where_filter,
            include=["metadatas"],
        )

        existing_ids = set(existing.get("ids", []))

        # Add new chunks or update existing chunks with the same IDs.
        store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        # Remove obsolete chunks only from this session and source.
        stale_ids = list(existing_ids - set(chunk_ids))

        if stale_ids:
            store.delete(ids=stale_ids)

    settings = get_settings()

    result = {
        "source_id": source_id,
        "source_name": parsed.source_name,
        "file_type": parsed.file_type,
        "extraction_method": parsed.extraction_method,
        "chunk_count": len(chunks),
        "content_hash": chunks[0].metadata["content_hash"],
        "collection_name": settings.chroma_collection_name,
        "persist_directory": str(settings.chroma_persist_directory),
    }

    if session_id:
        result["session_id"] = session_id

    return result

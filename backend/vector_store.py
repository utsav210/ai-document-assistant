from __future__ import annotations

from functools import lru_cache
from threading import Lock

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.chunking import create_chunks
from backend.config import get_settings
from backend.document_parser import ParsedDocument

_INDEX_LOCK = Lock()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load and reuse the local embedding model."""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Create or reopen the persistent Chroma collection."""
    settings = get_settings()
    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_persist_directory),
    )


def index_parsed_document(parsed: ParsedDocument) -> dict:
    """Embed and persist a parsed document's chunks."""
    chunks = create_chunks(parsed)

    source_id = chunks[0].metadata["source_id"]
    chunk_ids = [f"{source_id}-{index:06d}" for index in range(len(chunks))]

    store = get_vector_store()

    with _INDEX_LOCK:
        existing = store.get(
            where={"source_id": source_id},
            include=["metadatas"],
        )
        existing_ids = set(existing.get("ids", []))

        # Add or update the current chunks.
        store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        # Remove obsolete chunks if the document became shorter.
        stale_ids = list(existing_ids - set(chunk_ids))

        if stale_ids:
            store.delete(ids=stale_ids)

    settings = get_settings()
    return {
        "source_id": source_id,
        "source_name": parsed.source_name,
        "file_type": parsed.file_type,
        "extraction_method": parsed.extraction_method,
        "chunk_count": len(chunks),
        "content_hash": chunks[0].metadata["content_hash"],
        "collection_name": settings.chroma_collection_name,
        "persist_directory": str(settings.chroma_persist_directory),
    }

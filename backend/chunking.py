from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings
from backend.document_parser import ParsedDocument

PAGE_MARKER = re.compile(r"(?m)^<!-- Page (\d+) -->[ \t]*\n?")


def _split_by_page(content: str) -> list[tuple[int | None, str]]:
    """Split extracted Markdown into page segments when markers exist."""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(PAGE_MARKER.finditer(content))

    if not matches:
        text = content.strip()
        return [(None, text)] if text else []

    segments = []

    prefix = content[: matches[0].start()].strip()
    if prefix:
        segments.append((None, prefix))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)

        text = content[match.end() : end].strip()

        if text:
            segments.append((int(match.group(1)), text))

    return segments


def create_chunks(
    parsed: ParsedDocument,
    *,
    session_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split parsed content into searchable chunks with source metadata."""
    if session_id is not None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        session_id = session_id.strip()
        if len(session_id) > 128:
            raise ValueError("session_id must not exceed 128 characters.")
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if overlap < 0 or overlap >= size:
        raise ValueError(
            "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE."
        )

    if not parsed.content.strip():
        raise ValueError("Cannot chunk an empty document.")

    session_prefix = f"{session_id}:" if session_id else ""
    source_identity = (
        f"{session_prefix}{parsed.file_type}:{parsed.source_name.casefold()}"
    )
    source_id = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()[:24]

    content_hash = hashlib.sha256(parsed.content.encode("utf-8")).hexdigest()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        add_start_index=True,
        keep_separator="start",
        separators=["\n\n", "\n", " ", ""],
    )

    documents: list[Document] = []

    for page_number, text in _split_by_page(parsed.content):
        metadata = {
            "source_id": source_id,
            "source_name": parsed.source_name,
            "file_type": parsed.file_type,
            "extraction_method": parsed.extraction_method,
            "content_hash": content_hash,
        }

        if session_id is not None:
            metadata["session_id"] = session_id

        if page_number is not None:
            metadata["page_number"] = page_number

        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )

        for chunk in chunks:
            if not chunk.page_content.strip():
                continue

            chunk.metadata["chunk_index"] = len(documents)
            documents.append(chunk)

    if not documents:
        raise ValueError("No usable chunks were produced.")

    return documents

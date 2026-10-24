from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.config import get_settings
from backend.document_parser import (
    SUPPORTED_EXTENSIONS,
    DocumentParsingError,
)
from backend.document_parser import (
    parse_document as parse_document_file,
)
from backend.logging_config import configure_logging
from backend.rag_service import (
    ask_question,
    clear_conversation_history,
    get_conversation_history,
)
from backend.vector_store import (
    get_embeddings,
    get_vector_store,
    index_parsed_document,
)

settings = get_settings()
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = settings.max_upload_mb
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm up heavy ML models and vector store client at server startup."""
    logger.info("Pre-warming embedding model and vector store...")
    try:
        await run_in_threadpool(get_embeddings)
        await run_in_threadpool(get_vector_store)
        logger.info("Warmup complete. Embedding model and vector store are ready.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Startup pre-warm failed: %s", exc)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A document-grounded conversational AI assistant.",
    lifespan=lifespan,
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    """Return service metadata and operational status."""
    return {
        "service": settings.app_name,
        "status": "running",
    }


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return service health status."""
    logger.debug("Health check requested")

    return {
        "service": settings.app_name,
        "status": "ok",
    }


@asynccontextmanager
async def _stage_upload(
    file: UploadFile,
) -> AsyncIterator[tuple[str, Path]]:
    """Validate and temporarily store an uploaded document."""
    try:
        # Normalize both Windows and POSIX-style filename paths.
        original_name = (file.filename or "").replace("\\", "/")
        original_name = Path(original_name).name

        if not original_name:
            raise HTTPException(
                status_code=400,
                detail="A filename is required.",
            )

        extension = Path(original_name).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Upload a PDF, DOCX, or TXT document.",
            )

        with tempfile.TemporaryDirectory(prefix="ai-document-assistant-") as directory:
            temporary_path = Path(directory) / f"upload{extension}"
            total_bytes = 0

            with temporary_path.open("wb") as destination:
                while True:
                    chunk = await file.read(UPLOAD_CHUNK_SIZE)

                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"File exceeds the {MAX_UPLOAD_MB} MB upload limit."
                            ),
                        )

                    destination.write(chunk)

            if total_bytes == 0:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded file is empty.",
                )

            yield original_name, temporary_path

    finally:
        await file.close()


class DocumentParseResponse(BaseModel):
    source_name: str
    file_type: str
    extraction_method: str
    page_count: int | None
    character_count: int
    preview: str
    warnings: list[str]


@app.post(
    "/documents/parse",
    response_model=DocumentParseResponse,
    tags=["documents"],
)
async def parse_uploaded_document(
    file: Annotated[
        UploadFile,
        File(description="PDF, DOCX or TXT document"),
    ],
) -> DocumentParseResponse:
    """Extract document content and return a preview."""
    async with _stage_upload(file) as (original_name, temporary_path):
        try:
            parsed = await run_in_threadpool(
                parse_document_file,
                temporary_path,
            )
        except DocumentParsingError as exc:
            logger.warning(
                "Parsing failed for %s: %s",
                original_name,
                exc,
            )
            raise HTTPException(
                status_code=422,
                detail="Unable to extract readable document content.",
            ) from exc

        # Restore the uploaded filename instead of keeping upload.pdf.
        parsed = replace(parsed, source_name=original_name)

        return DocumentParseResponse(
            source_name=parsed.source_name,
            file_type=parsed.file_type,
            extraction_method=parsed.extraction_method,
            page_count=parsed.page_count,
            character_count=len(parsed.content),
            preview=parsed.content[:1200],
            warnings=list(parsed.warnings),
        )


class DocumentIndexResponse(BaseModel):
    status: str
    session_id: str
    source_id: str
    source_name: str
    file_type: str
    extraction_method: str
    page_count: int | None
    chunk_count: int
    content_hash: str
    collection_name: str
    persist_directory: str
    warnings: list[str]


@app.post(
    "/documents/index",
    response_model=DocumentIndexResponse,
    tags=["documents"],
)
async def index_uploaded_document(
    file: Annotated[
        UploadFile,
        File(description="PDF, DOCX or TXT document"),
    ],
    session_id: Annotated[
        str,
        Form(
            min_length=1,
            max_length=128,
            description="Identifier of the chat session that owns this document",
        ),
    ],
) -> DocumentIndexResponse:
    """Parse, chunk, embed, and persist an uploaded document."""
    session_id = session_id.strip()

    if not session_id:
        raise HTTPException(
            status_code=422,
            detail="session_id must not be empty.",
        )
    async with _stage_upload(file) as (original_name, temporary_path):
        try:
            parsed = await run_in_threadpool(
                parse_document_file,
                temporary_path,
            )
        except DocumentParsingError as exc:
            logger.warning(
                "Parsing failed for %s: %s",
                original_name,
                exc,
            )
            raise HTTPException(
                status_code=422,
                detail="Unable to extract readable document content.",
            ) from exc

        # This must happen BEFORE chunking and indexing.
        parsed = replace(parsed, source_name=original_name)

        try:
            result = await run_in_threadpool(
                index_parsed_document,
                parsed,
                session_id=session_id,
            )
        except Exception as exc:
            logger.exception(
                "Indexing failed for %s",
                original_name,
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "Document parsing succeeded, but indexing failed. "
                    "Check the application logs."
                ),
            ) from exc

        return DocumentIndexResponse(
            status="indexed",
            session_id=session_id,
            source_id=result["source_id"],
            source_name=original_name,
            file_type=parsed.file_type,
            extraction_method=parsed.extraction_method,
            page_count=parsed.page_count,
            chunk_count=result["chunk_count"],
            content_hash=result["content_hash"],
            collection_name=result["collection_name"],
            persist_directory=result["persist_directory"],
            warnings=list(parsed.warnings),
        )


class ChatRequest(BaseModel):
    """Validated request for a document-grounded chat interaction."""

    session_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    source_name: str | None = Field(default=None, max_length=255)


class SourceCitation(BaseModel):
    """A source reference supporting a generated answer."""

    source_name: str
    file_type: str | None = None
    page_number: int | None = None
    extraction_method: str | None = None
    chunk_index: int | None = None


class ChatResponse(BaseModel):
    """Response containing an answer and its supporting sources."""

    answer: str
    sources: list[SourceCitation]
    standalone_question: str


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using documents indexed for the session."""
    session_id = request.session_id.strip()
    question = request.question.strip()
    source_name = request.source_name.strip() or None if request.source_name else None

    if not session_id:
        raise HTTPException(
            status_code=422,
            detail="session_id must not be empty.",
        )

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question must not be empty.",
        )

    try:
        result = await run_in_threadpool(
            ask_question,
            session_id,
            question,
            source_name=source_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception(
            "Chat request failed for session %s",
            session_id,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to answer the question at this time. "
                "Check the application logs."
            ),
        ) from exc

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        standalone_question=result["standalone_question"],
    )


@app.get(
    "/chat/history/{session_id}",
    tags=["chat"],
)
async def get_session_chat_history(session_id: str) -> dict[str, Any]:
    """Retrieve the recent conversational exchanges (up to 3 turns) for a session."""
    clean_id = session_id.strip()
    if not clean_id:
        raise HTTPException(
            status_code=422,
            detail="session_id must not be empty.",
        )
    history = await run_in_threadpool(get_conversation_history, clean_id)
    return {"session_id": clean_id, "turns": history}


@app.delete(
    "/chat/history/{session_id}",
    tags=["chat"],
)
async def clear_session_chat_history(session_id: str) -> dict[str, str]:
    """Clear conversational memory for a session without modifying indexed documents."""
    clean_id = session_id.strip()
    if not clean_id:
        raise HTTPException(
            status_code=422,
            detail="session_id must not be empty.",
        )
    await run_in_threadpool(clear_conversation_history, clean_id)
    return {"status": "cleared", "session_id": clean_id}

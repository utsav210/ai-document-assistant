import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
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

settings = get_settings()
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = settings.max_upload_mb
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024

# defined the app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A document-grounded conversational AI assistant.",
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    """Return top-level API metadata and operational status."""
    return {
        "service": settings.app_name,
        "status": "running",
    }


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return service health status for container and monitoring probes."""
    logger.debug("Health check requested")

    return {
        "service": settings.app_name,
        "status": "ok",
    }


class DocumentParseResponse(BaseModel):
    source_name: str
    file_type: str
    extraction_method: str
    page_count: int | None
    character_count: int
    preview: str
    warnings: list[str]


@app.post("/documents/parse", response_model=DocumentParseResponse, tags=["documents"])
async def parse_uploaded_document(
    file: Annotated[UploadFile, File(description="PDF, DOCX or TXT document")],
) -> DocumentParseResponse:
    try:
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
                detail="Unsupported file type. Upload a PDF, DOCX or TXT file.",
            )

        with tempfile.TemporaryDirectory(
            prefix="ai-document-assistant-"
        ) as temporary_directory:
            temporary_path = Path(temporary_directory) / f"upload{extension}"

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

            try:
                parsed = await run_in_threadpool(
                    parse_document_file,
                    temporary_path,
                )
            except DocumentParsingError as exc:
                logger.warning(
                    "Document parsing failed for %s: %s",
                    original_name,
                    exc,
                )
                raise HTTPException(
                    status_code=422,
                    detail="Unable to extract readable content from this document.",
                ) from exc

            return DocumentParseResponse(
                source_name=original_name,
                file_type=parsed.file_type,
                extraction_method=parsed.extraction_method,
                page_count=parsed.page_count,
                character_count=len(parsed.content),
                preview=parsed.content[:1200],
                warnings=list(parsed.warnings),
            )

    finally:
        await file.close()

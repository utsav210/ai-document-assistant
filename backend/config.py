import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _read_int(name: str, default: int, minimum: int = 1) -> int:
    """Parse an integer environment variable with boundary validation.

    Args:
        name: Environment variable name.
        default: Fallback integer value if variable is unset.
        minimum: Lowest allowed value.

    Returns:
        Validated integer value.

    Raises:
        ValueError: If value cannot be parsed or falls below minimum.
    """
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")

    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings container."""

    app_name: str
    app_version: str
    environment: str
    log_level: str
    groq_api_key: str
    groq_model: str
    embedding_model: str
    max_upload_mb: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    chroma_persist_directory: Path
    chroma_collection_name: str = "ai_document_assistant"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load, validate, and cache application settings from the environment.

    Returns:
        Cached immutable Settings instance.

    Raises:
        ValueError: If any configuration value violates validation rules.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    if log_level not in valid_log_levels:
        raise ValueError("LOG_LEVEL must be a valid logging level.")

    chunk_size = _read_int("CHUNK_SIZE", 900)
    chunk_overlap = _read_int("CHUNK_OVERLAP", 150, minimum=0)

    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chroma_path = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"))

    if not chroma_path.is_absolute():
        chroma_path = PROJECT_ROOT / chroma_path

    return Settings(
        app_name="AI Document Assistant",
        app_version="0.1.0",
        environment=os.getenv("APP_ENV", "development"),
        log_level=log_level,
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=(os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-120b"),
        embedding_model=(
            os.getenv("EMBEDDING_MODEL", "").strip()
            or "sentence-transformers/all-MiniLM-L6-v2"
        ),
        max_upload_mb=_read_int("MAX_UPLOAD_MB", 20),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_top_k=_read_int("RETRIEVAL_TOP_K", 4),
        chroma_persist_directory=chroma_path,
        chroma_collection_name=(
            os.getenv("CHROMA_COLLECTION_NAME", "").strip() or "ai_document_assistant"
        ),
    )

import logging

from fastapi import FastAPI

from backend.config import get_settings
from backend.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A document-grounded conversational AI assistant.",
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "status": "running",
    }


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    logger.debug("Health check requested")

    return {
        "service": settings.app_name,
        "status": "ok",
    }
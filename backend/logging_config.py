import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure centralized application logging format and severity level.

    Args:
        level: String representation of the logging level (e.g. 'DEBUG', 'INFO').
    """
    numeric_level = getattr(logging, level.strip().upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        force=True,
    )
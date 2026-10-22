import logging


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper())

    logging.basicConfig(
        level=numeric_level,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )
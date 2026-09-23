import logging
import os
import sys

from typing import Union


def setup_logger(
    log_file: str | None = None, level: Union[int, str] = logging.INFO
) -> logging.Logger:
    """Configure console logging and optional file logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    log_format = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    if log_file is None:
        log_file = os.environ.get(
            "DOCMIND_LOG_FILE", os.path.join(os.getcwd(), "docmind.log")
        )

    try:
        parent = os.path.dirname(os.path.abspath(log_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except OSError:
        pass

    return logger


log = setup_logger()

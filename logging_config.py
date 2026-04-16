"""
logging_config.py - Shared Logging Configuration
=================================================
Centralizes logging setup so that all pipeline stages write to
the same log file with consistent formatting. Import get_logger()
in any module that needs to log messages.

The log file is written to logs/pipeline.log in the project root

Usage:
    from logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Stage started")
    logger.warning("Something unexpected but not fatal")
    logger.error("Something broke", exc_info=True)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Path setup - this file is in the project root
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"


def get_logger(name):
    """
    Return a configured logger for the given module name.

    Parameters
    ----------
    name : str
        Name of the logger, typically __name__ from the calling
        module. Appears in log messages to identify the source.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Ensure log directory exists before creating the file handler
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # If the logger already has handlers, it's been configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Formatter includes timestamp, log level, module name, and message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler: captures everything at DEBUG level and above
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger